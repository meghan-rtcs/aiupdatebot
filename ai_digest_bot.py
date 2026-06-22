#!/usr/bin/env python3
"""
RTCS AI Digest Bot
==================
Scans for brand-new AI features/news in the last ~24h, has Claude filter and
summarize the signal (dedupe, rank, drop noise), and emails a clean digest to
your team.

Focus: new capabilities/integrations from Claude (Anthropic), ChatGPT (OpenAI),
Gemini (Google), Grok (xAI), plus agentic/MCP/API news relevant to ops automation.

Two layers, by design:
  1) RSS base layer  -> guarantees first-party announcements are never missed.
  2) Web-search layer -> Claude runs its own searches to catch everything else
                         (smaller integrations, X announcements, etc.).
The web-search layer is optional (USE_WEB_SEARCH=false to disable).

Run:
  python3 ai_digest_bot.py            # gather -> synthesize -> email
  python3 ai_digest_bot.py --dry-run  # gather -> render to digest_preview.html, NO email, NO LLM needed

Config: environment variables (see .env.example). On the VPS, point at
/etc/rtcs/ai-digest.env per your secrets convention.
"""

from __future__ import annotations

import argparse
import datetime as dt
import html
import json
import os
import re
import smtplib
import sys
import time
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.utils import formataddr
from pathlib import Path

import feedparser
from dotenv import load_dotenv

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #

# Load secrets. Prefer the VPS path, fall back to a local .env for dev.
for env_path in ("/etc/rtcs/ai-digest.env", ".env"):
    if Path(env_path).exists():
        load_dotenv(env_path)
        break

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6")
USE_WEB_SEARCH = os.getenv("USE_WEB_SEARCH", "true").lower() == "true"
WEB_SEARCH_MAX_USES = int(os.getenv("WEB_SEARCH_MAX_USES", "6"))

# Email
SMTP_HOST = os.getenv("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER = os.getenv("SMTP_USER", "")
SMTP_PASS = os.getenv("SMTP_PASS", "")
EMAIL_FROM = os.getenv("EMAIL_FROM", SMTP_USER)
EMAIL_FROM_NAME = os.getenv("EMAIL_FROM_NAME", "RTCS AI Digest")
EMAIL_TO = [a.strip() for a in os.getenv("EMAIL_TO", "").split(",") if a.strip()]

# Behavior
LOOKBACK_HOURS = int(os.getenv("LOOKBACK_HOURS", "30"))  # a little >24 to cover overnight + tz drift
MAX_CANDIDATES = int(os.getenv("MAX_CANDIDATES", "60"))  # cap items fed to the LLM (cost control)
SEND_WHEN_EMPTY = os.getenv("SEND_WHEN_EMPTY", "true").lower() == "true"

# Who this digest is for — drives Claude's relevance scoring. Edit freely.
AUDIENCE_CONTEXT = os.getenv(
    "AUDIENCE_CONTEXT",
    "Real-Time Consulting Services (RTCS), a property-management operations & "
    "automation consultancy. The team builds API integrations, KPI dashboards, "
    "and workflow automations for property managers using tools like AppFolio, "
    "monday.com, Buildium, Aptly, LeadSimple, plus Python/Node and LLM/agentic "
    "tooling. They care most about: new Claude/ChatGPT/Gemini capabilities, new "
    "integrations & connectors (MCP, Zapier-style, native API), agentic/automation "
    "features, API changes, and anything that could speed up building client "
    "integrations or dashboards.",
)

# --------------------------------------------------------------------------- #
# Sources (RSS base layer). Add/remove freely — these are high-signal feeds.
# --------------------------------------------------------------------------- #

FEEDS = [
    # First-party (most important — primary announcements)
    ("Anthropic News", "https://www.anthropic.com/rss.xml"),
    ("OpenAI News", "https://openai.com/news/rss.xml"),
    ("Google AI Blog", "https://blog.google/technology/ai/rss/"),
    ("Google DeepMind", "https://deepmind.google/blog/rss.xml"),
    ("Microsoft AI", "https://blogs.microsoft.com/ai/feed/"),
    ("Hugging Face Blog", "https://huggingface.co/blog/feed.xml"),
    # High-signal press
    ("TechCrunch AI", "https://techcrunch.com/category/artificial-intelligence/feed/"),
    ("The Verge AI", "https://www.theverge.com/rss/ai-artificial-intelligence/index.xml"),
    ("VentureBeat AI", "https://venturebeat.com/category/ai/feed/"),
    ("Ars Technica AI", "https://arstechnica.com/ai/feed/"),
    ("MIT Tech Review AI", "https://www.technologyreview.com/topic/artificial-intelligence/feed"),
    # Community signal (often earliest)
    ("Hacker News (AI front page)",
     "https://hnrss.org/newest?q=AI+OR+LLM+OR+Claude+OR+OpenAI+OR+Anthropic&points=50"),
    ("r/LocalLLaMA", "https://www.reddit.com/r/LocalLLaMA/top/.rss?t=day"),
]

USER_AGENT = "RTCS-AI-Digest/1.0 (+https://realtimecs.com)"


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _strip_html(text: str) -> str:
    text = re.sub(r"<[^>]+>", " ", text or "")
    text = html.unescape(text)
    return re.sub(r"\s+", " ", text).strip()


def _normalize_title(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (title or "").lower()).strip()


def _entry_datetime(entry) -> dt.datetime | None:
    for key in ("published_parsed", "updated_parsed"):
        st = getattr(entry, key, None) or entry.get(key)
        if st:
            return dt.datetime.fromtimestamp(time.mktime(st), tz=dt.timezone.utc)
    return None


# --------------------------------------------------------------------------- #
# 1) Gather
# --------------------------------------------------------------------------- #

def gather_rss(lookback_hours: int = LOOKBACK_HOURS, max_items: int = MAX_CANDIDATES) -> list[dict]:
    """Pull recent items from all feeds, dedupe by title, return newest-first."""
    cutoff = dt.datetime.now(tz=dt.timezone.utc) - dt.timedelta(hours=lookback_hours)
    seen_titles: set[str] = set()
    seen_links: set[str] = set()
    items: list[dict] = []

    for source_name, url in FEEDS:
        try:
            feed = feedparser.parse(url, agent=USER_AGENT)
        except Exception as e:  # never let one bad feed kill the run
            print(f"  ! feed error ({source_name}): {e}", file=sys.stderr)
            continue

        for entry in feed.entries:
            published = _entry_datetime(entry)
            if published and published < cutoff:
                continue  # too old

            title = (getattr(entry, "title", "") or "").strip()
            link = (getattr(entry, "link", "") or "").strip()
            if not title or not link:
                continue

            norm = _normalize_title(title)
            if norm in seen_titles or link in seen_links:
                continue
            seen_titles.add(norm)
            seen_links.add(link)

            summary = _strip_html(getattr(entry, "summary", "") or getattr(entry, "description", ""))
            items.append({
                "source": source_name,
                "title": title,
                "link": link,
                "published": published.isoformat() if published else "",
                "summary": summary[:600],
            })

    items.sort(key=lambda x: x["published"], reverse=True)
    print(f"  gathered {len(items)} candidate items from {len(FEEDS)} feeds "
          f"(last {lookback_hours}h)")
    return items[:max_items]


# --------------------------------------------------------------------------- #
# 2) Synthesize with Claude (filter, dedupe, rank, summarize)
# --------------------------------------------------------------------------- #

SYNTHESIS_SYSTEM = f"""You are the editor of a daily AI-news digest for {AUDIENCE_CONTEXT}

Your job: turn raw, noisy AI news into a short, high-signal digest of what is
GENUINELY NEW in the last ~24 hours.

WHAT COUNTS AS SIGNAL (include):
- New model releases or new capabilities in Claude, ChatGPT/OpenAI, Gemini/Google, Grok/xAI, Llama, etc.
- New product features, especially anything users can actually do today.
- New integrations, connectors, MCP servers, APIs, or platform changes (Zapier/Make, native APIs, SDKs).
- Agentic / automation features, coding-agent updates, computer-use, tool-calling changes.
- Pricing, access, or availability changes that affect real usage.
- Anything that could help a property-management automation shop build integrations or dashboards faster.

WHAT IS NOISE (exclude):
- Generic listicles ("10 best AI tools"), opinion/think-pieces, vague "AI is changing X" articles.
- Pure funding/valuation news UNLESS it ships a concrete product or capability.
- Rehashed or week-old news; rumors with no primary source.
- Duplicate coverage of the same event — collapse into ONE item and cite the best source.

RULES:
- Ground every item in the material provided to you (RSS items) and/or your own web-search results. NEVER invent a feature, quote, or URL. Every source_url must be a real link you actually saw.
- Prefer the primary/official source over secondary coverage when both exist.
- Rank by relevance to the audience above; most useful first.
- Be concise and concrete. "what_changed" = the factual update in 1-2 sentences. "why_it_matters" = 1 sentence tying it to the audience's work.
- Aim for 3-8 items on a normal day. If truly nothing new and relevant happened, return an empty items array.

OUTPUT FORMAT — return ONLY valid JSON, no markdown, no prose, no code fences:
{{
  "date": "YYYY-MM-DD",
  "tldr": "One or two sentences summarizing the day at a glance.",
  "items": [
    {{
      "headline": "Short punchy headline",
      "category": "one of: Claude | ChatGPT/OpenAI | Gemini/Google | Grok/xAI | Integrations | Agents/Automation | API/Dev | Other",
      "what_changed": "1-2 factual sentences.",
      "why_it_matters": "1 sentence on relevance to the team.",
      "source_name": "Publication or company",
      "source_url": "https://real-url",
      "date": "YYYY-MM-DD"
    }}
  ]
}}"""


def _build_candidates_block(candidates: list[dict]) -> str:
    lines = []
    for i, c in enumerate(candidates, 1):
        lines.append(
            f"{i}. [{c['source']}] {c['title']}\n"
            f"   url: {c['link']}\n"
            f"   when: {c['published']}\n"
            f"   summary: {c['summary']}"
        )
    return "\n\n".join(lines) if lines else "(no RSS items found in the window)"


def _extract_text(content_blocks) -> str:
    """Concatenate all text blocks from a Messages API response."""
    out = []
    for block in content_blocks:
        btype = getattr(block, "type", None)
        if btype == "text":
            out.append(getattr(block, "text", "") or "")
    return "\n".join(out).strip()


def _parse_json(raw: str) -> dict:
    raw = raw.strip()
    raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.IGNORECASE).strip()
    # Be forgiving: grab the outermost {...} if the model added stray text.
    start, end = raw.find("{"), raw.rfind("}")
    if start != -1 and end != -1:
        raw = raw[start:end + 1]
    return json.loads(raw)


def synthesize(candidates: list[dict], today: str) -> dict:
    """Call Claude to filter/dedupe/rank/summarize. Falls back to RSS passthrough on error."""
    if not ANTHROPIC_API_KEY:
        print("  ! ANTHROPIC_API_KEY not set — using RSS passthrough (no LLM filtering)",
              file=sys.stderr)
        return _passthrough(candidates, today)

    try:
        import anthropic
    except ImportError:
        print("  ! `anthropic` not installed — RSS passthrough. (pip install anthropic)",
              file=sys.stderr)
        return _passthrough(candidates, today)

    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    user_prompt = (
        f"Today is {today}. Build today's digest.\n\n"
        f"Here are RSS candidate items from the last {LOOKBACK_HOURS} hours "
        f"(use these as your base, and verify/expand with web search if enabled):\n\n"
        f"{_build_candidates_block(candidates)}\n\n"
        f"Produce the digest JSON now."
    )

    tools = []
    if USE_WEB_SEARCH:
        # Server-side web search tool: Anthropic runs the searches and returns
        # the final answer in one call. Cap uses to control cost.
        tools.append({
            "type": "web_search_20250305",
            "name": "web_search",
            "max_uses": WEB_SEARCH_MAX_USES,
        })

    try:
        resp = client.messages.create(
            model=ANTHROPIC_MODEL,
            max_tokens=4000,
            system=SYNTHESIS_SYSTEM,
            tools=tools or anthropic.NOT_GIVEN,
            messages=[{"role": "user", "content": user_prompt}],
        )
        text = _extract_text(resp.content)
        digest = _parse_json(text)
        digest.setdefault("date", today)
        digest.setdefault("items", [])
        digest.setdefault("tldr", "")
        print(f"  synthesized {len(digest['items'])} digest item(s) "
              f"(web_search={'on' if USE_WEB_SEARCH else 'off'})")
        return digest
    except Exception as e:
        print(f"  ! synthesis failed ({e}) — falling back to RSS passthrough", file=sys.stderr)
        return _passthrough(candidates, today)


def _passthrough(candidates: list[dict], today: str) -> dict:
    """No-LLM fallback so the bot still ships something useful."""
    items = []
    for c in candidates[:8]:
        items.append({
            "headline": c["title"],
            "category": "Other",
            "what_changed": c["summary"][:240] or "(see source)",
            "why_it_matters": "",
            "source_name": c["source"],
            "source_url": c["link"],
            "date": (c["published"][:10] if c["published"] else today),
        })
    return {
        "date": today,
        "tldr": "Unfiltered feed (LLM synthesis unavailable). Top recent AI items below.",
        "items": items,
    }


# --------------------------------------------------------------------------- #
# 3) Render HTML email
# --------------------------------------------------------------------------- #

CATEGORY_COLORS = {
    "Claude": "#c96442",
    "ChatGPT/OpenAI": "#10a37f",
    "Gemini/Google": "#4285f4",
    "Grok/xAI": "#111827",
    "Integrations": "#7c3aed",
    "Agents/Automation": "#d97706",
    "API/Dev": "#0891b2",
    "Other": "#6b7280",
}


def render_html(digest: dict, pretty_date: str) -> str:
    items = digest.get("items", [])
    tldr = html.escape(digest.get("tldr", "") or "")

    if items:
        cards = []
        for it in items:
            cat = it.get("category", "Other")
            color = CATEGORY_COLORS.get(cat, "#6b7280")
            headline = html.escape(it.get("headline", "Untitled"))
            what = html.escape(it.get("what_changed", ""))
            why = html.escape(it.get("why_it_matters", ""))
            src_name = html.escape(it.get("source_name", "source"))
            src_url = it.get("source_url", "#")
            why_block = (
                f'<p style="margin:8px 0 0;font-size:13px;line-height:1.5;color:#6b7280;">'
                f'<strong style="color:#374151;">Why it matters:</strong> {why}</p>'
                if why else ""
            )
            cards.append(f"""
            <tr><td style="padding:0 0 14px;">
              <table role="presentation" width="100%" cellpadding="0" cellspacing="0"
                     style="background:#ffffff;border:1px solid #e5e7eb;border-radius:10px;">
                <tr><td style="padding:18px 20px;">
                  <span style="display:inline-block;font-size:11px;font-weight:700;letter-spacing:.04em;
                               text-transform:uppercase;color:#ffffff;background:{color};
                               padding:3px 9px;border-radius:999px;">{html.escape(cat)}</span>
                  <h2 style="margin:10px 0 6px;font-size:17px;line-height:1.35;color:#111827;">{headline}</h2>
                  <p style="margin:0;font-size:14px;line-height:1.55;color:#374151;">{what}</p>
                  {why_block}
                  <p style="margin:12px 0 0;font-size:13px;">
                    <a href="{html.escape(src_url)}" style="color:{color};text-decoration:none;font-weight:600;">
                      {src_name} &nbsp;&rarr;</a></p>
                </td></tr>
              </table>
            </td></tr>""")
        body_rows = "".join(cards)
    else:
        body_rows = """
            <tr><td style="padding:30px 20px;text-align:center;background:#ffffff;
                           border:1px solid #e5e7eb;border-radius:10px;">
              <p style="margin:0;font-size:15px;color:#6b7280;">
                No major new AI features or releases surfaced in the last 24 hours.
                Quiet day. ☕</p>
            </td></tr>"""

    tldr_block = (
        f"""<tr><td style="padding:0 0 18px;">
              <table role="presentation" width="100%" cellpadding="0" cellspacing="0"
                     style="background:#fff7ed;border:1px solid #fed7aa;border-radius:10px;">
                <tr><td style="padding:14px 18px;">
                  <p style="margin:0;font-size:11px;font-weight:700;letter-spacing:.05em;
                            text-transform:uppercase;color:#c2410c;">TL;DR</p>
                  <p style="margin:6px 0 0;font-size:14px;line-height:1.55;color:#7c2d12;">{tldr}</p>
                </td></tr></table></td></tr>"""
        if tldr else ""
    )

    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>RTCS AI Digest</title></head>
<body style="margin:0;padding:0;background:#f3f4f6;
             font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Helvetica,Arial,sans-serif;">
  <table role="presentation" width="100%" cellpadding="0" cellspacing="0" style="background:#f3f4f6;">
    <tr><td align="center" style="padding:24px 12px;">
      <table role="presentation" width="600" cellpadding="0" cellspacing="0"
             style="max-width:600px;width:100%;">

        <!-- header -->
        <tr><td style="padding:0 0 18px;">
          <table role="presentation" width="100%" cellpadding="0" cellspacing="0"
                 style="background:#111827;border-radius:12px;">
            <tr><td style="padding:22px 24px;">
              <p style="margin:0;font-size:20px;font-weight:800;color:#ffffff;letter-spacing:-.02em;">
                🛰️ AI Digest</p>
              <p style="margin:4px 0 0;font-size:13px;color:#9ca3af;">
                What's new in AI &middot; {pretty_date}</p>
            </td></tr></table></td></tr>

        {tldr_block}
        {body_rows}

        <!-- footer -->
        <tr><td style="padding:14px 8px 0;text-align:center;">
          <p style="margin:0;font-size:12px;line-height:1.6;color:#9ca3af;">
            Auto-generated daily by the RTCS AI Digest Bot.<br>
            Sources: official AI blogs, tech press, and live web search, filtered by Claude.</p>
        </td></tr>

      </table>
    </td></tr></table>
</body></html>"""


# --------------------------------------------------------------------------- #
# 4) Send
# --------------------------------------------------------------------------- #

def send_email(subject: str, html_body: str, recipients: list[str]) -> None:
    if not recipients:
        raise ValueError("EMAIL_TO is empty — set recipients in your env file.")
    if not (SMTP_HOST and SMTP_USER and SMTP_PASS):
        raise ValueError("SMTP_HOST/SMTP_USER/SMTP_PASS not fully configured.")

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = formataddr((EMAIL_FROM_NAME, EMAIL_FROM))
    msg["To"] = ", ".join(recipients)
    msg.attach(MIMEText("Your email client does not support HTML. See the web version.", "plain"))
    msg.attach(MIMEText(html_body, "html"))

    with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
        server.starttls()
        server.login(SMTP_USER, SMTP_PASS)
        server.sendmail(EMAIL_FROM, recipients, msg.as_string())
    print(f"  sent to {len(recipients)} recipient(s): {', '.join(recipients)}")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main() -> int:
    parser = argparse.ArgumentParser(description="RTCS AI Digest Bot")
    parser.add_argument("--dry-run", action="store_true",
                        help="Gather + render to digest_preview.html. No email, no LLM required.")
    args = parser.parse_args()

    now = dt.datetime.now(tz=dt.timezone.utc)
    today = now.strftime("%Y-%m-%d")
    pretty_date = now.strftime("%A, %B %-d, %Y")

    print(f"[{today}] RTCS AI Digest Bot — starting")
    print("  step 1/4: gathering RSS …")
    candidates = gather_rss()

    if args.dry_run:
        # In dry-run we skip the LLM (passthrough) so it works with zero keys.
        print("  step 2/4: (dry-run) skipping LLM, using passthrough …")
        digest = _passthrough(candidates, today)
        print("  step 3/4: rendering …")
        out = Path("digest_preview.html")
        out.write_text(render_html(digest, pretty_date), encoding="utf-8")
        print(f"  step 4/4: wrote {out.resolve()} (open in a browser to preview)")
        return 0

    print("  step 2/4: synthesizing with Claude …")
    digest = synthesize(candidates, today)

    if not digest.get("items") and not SEND_WHEN_EMPTY:
        print("  no items and SEND_WHEN_EMPTY=false — skipping send. Done.")
        return 0

    print("  step 3/4: rendering …")
    html_body = render_html(digest, pretty_date)

    n = len(digest.get("items", []))
    subject = (f"RTCS AI Digest — {n} new thing{'s' if n != 1 else ''} ({pretty_date})"
               if n else f"RTCS AI Digest — quiet day ({pretty_date})")

    print("  step 4/4: sending …")
    send_email(subject, html_body, EMAIL_TO)
    print("  done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
