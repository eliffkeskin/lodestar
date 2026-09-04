"""
Lodestar — L1 support chatbot, Streamlit app (with documentation RAG)
===============================================================

Answering strategy (in order):
  1. Known error signature? -> answer from the local error catalog (fast, exact).
  2. Otherwise -> retrieve relevant doc chunks (RAG) and answer GROUNDED in them,
     with sources shown.
  3. If neither the catalog matches nor the docs contain a confident answer ->
     escalate to a human engineer instead of guessing.

Input: pasted error text OR an uploaded screenshot (LLM transcribes the image).

Run:
    pip install -r requirements.txt
    export ANTHROPIC_API_KEY=sk-...
    python rag.py --index            # build the doc index once (and after doc changes)
    streamlit run app.py
"""

from dotenv import load_dotenv
load_dotenv()                     

from langfuse import get_client, observe
langfuse = get_client()

print(langfuse.auth_check())     

import base64
import csv
import json
import os
import time
import uuid
from datetime import datetime

import streamlit as st
import streamlit.components.v1 as components

import error_catalog as ec

# Where interaction metrics are logged (for later analysis in Excel, etc.).
METRICS_FILE = os.getenv("LODESTAR_METRICS_FILE", "metrics.csv")
METRICS_FIELDS = ["timestamp", "turn_id", "question_preview", "route",
                  "num_sources", "latency_seconds", "answer_chars",
                  "input_tokens", "output_tokens", "cache_read_tokens",
                  "cache_write_tokens", "est_cost_usd", "feedback"]


def log_metric(row: dict):
    """Append one interaction row to the metrics CSV (creates header if new)."""
    try:
        new = not os.path.exists(METRICS_FILE)
        with open(METRICS_FILE, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=METRICS_FIELDS)
            if new:
                w.writeheader()
            w.writerow({k: row.get(k, "") for k in METRICS_FIELDS})
    except Exception:
        pass  # never let metrics logging break the app


def update_feedback(turn_id: str, value: str):
    """Rewrite the feedback column for a given turn_id in the metrics CSV."""
    try:
        if not os.path.exists(METRICS_FILE):
            return
        rows = []
        with open(METRICS_FILE, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        for r in rows:
            if r.get("turn_id") == turn_id:
                r["feedback"] = value
        with open(METRICS_FILE, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=METRICS_FIELDS)
            w.writeheader()
            w.writerows(rows)
    except Exception:
        pass


# --------------------------------------------------------------------------- #
# Conversation persistence — multiple chats, saved to disk so history survives
# restarts (like the "previous conversations" panel in polished assistants).
# --------------------------------------------------------------------------- #
CONV_DIR = os.getenv("LODESTAR_CONV_DIR", "conversations")


def _conv_path(cid):
    return os.path.join(CONV_DIR, f"{cid}.json")


def save_conversation(cid, title, messages):
    """Persist one conversation to disk as JSON."""
    try:
        os.makedirs(CONV_DIR, exist_ok=True)
        data = {"id": cid, "title": title,
                "updated": datetime.now().isoformat(timespec="seconds"),
                "messages": messages}
        with open(_conv_path(cid), "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
    except Exception:
        pass


def load_conversation(cid):
    try:
        with open(_conv_path(cid), encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def list_conversations():
    """Return saved conversations, newest first: [{id,title,updated}, ...]."""
    out = []
    if not os.path.isdir(CONV_DIR):
        return out
    for fn in os.listdir(CONV_DIR):
        if fn.endswith(".json"):
            try:
                with open(os.path.join(CONV_DIR, fn), encoding="utf-8") as f:
                    d = json.load(f)
                out.append({"id": d.get("id"), "title": d.get("title", "Untitled"),
                            "updated": d.get("updated", "")})
            except Exception:
                pass
    out.sort(key=lambda x: x["updated"], reverse=True)
    return out


def delete_conversation(cid):
    try:
        os.remove(_conv_path(cid))
    except Exception:
        pass


try:
    import rag
    _rag_available = True
except Exception:
    _rag_available = False


# Warm the embedding model once and keep it resident across reruns, so the first
# documentation query isn't slow. @st.cache_resource persists it for the session.
@st.cache_resource(show_spinner=False)
def _warm_embedder():
    if _rag_available:
        try:
            return rag._embedder()
        except Exception:
            return None
    return None

try:
    import anthropic
    _anthropic_client = (anthropic.Anthropic()
                         if os.getenv("ANTHROPIC_API_KEY") else None)
except Exception:
    _anthropic_client = None

# --------------------------------------------------------------------------- #
# Model provider selection — makes the app model-agnostic.
#   LODESTAR_PROVIDER = "claude"  → Anthropic API (best quality, per-token cost)
#   LODESTAR_PROVIDER = "ollama"  → local model via Ollama (free, private, needs a
#                              running Ollama server; e.g. IBM Granite, Llama)
# Switch by setting the env var; nothing else in the app changes.
# --------------------------------------------------------------------------- #
PROVIDER = os.getenv("LODESTAR_PROVIDER", "claude").lower()

# Text model (RAG answers, catalog, log analysis).
MODEL = os.getenv("LODESTAR_MODEL",
                  "claude-sonnet-4-5" if PROVIDER == "claude" else "granite4.1:8b")

# Vision model (reading screenshots). With Claude the same model handles vision, so
# this defaults to MODEL. With Ollama, text models can't see images — set a vision
# model like Granite Vision or llava. If left equal to a text-only model, screenshot
# reading is skipped with a clear message.
VISION_MODEL = os.getenv("LODESTAR_VISION_MODEL",
                         MODEL if PROVIDER == "claude" else "granite3.2-vision")

OLLAMA_URL = os.getenv("LODESTAR_OLLAMA_URL", "http://localhost:11434")

# _client is truthy when the selected provider is usable (drives UI warnings).
_client = _anthropic_client if PROVIDER == "claude" else True

# --------------------------------------------------------------------------- #
# Pricing (USD per million tokens). Applies to the Claude provider. For a local
# provider (Ollama) the per-token API cost is zero — you pay only for the
# hardware/electricity to run the model, which this app doesn't meter.
# Claude Sonnet 4.5 rates; UPDATE if you change models. Source:
# https://platform.claude.com/docs/en/about-claude/pricing
# --------------------------------------------------------------------------- #
PRICE_INPUT_PER_MTOK = float(os.getenv("LODESTAR_PRICE_INPUT", "3.00"))
PRICE_OUTPUT_PER_MTOK = float(os.getenv("LODESTAR_PRICE_OUTPUT", "15.00"))
PRICE_CACHE_READ_PER_MTOK = float(os.getenv("LODESTAR_PRICE_CACHE_READ", "0.30"))
PRICE_CACHE_WRITE_PER_MTOK = float(os.getenv("LODESTAR_PRICE_CACHE_WRITE", "3.75"))


def estimate_cost(usage) -> float:
    """
    Turn a usage object into an estimated USD cost for this call. Local providers
    (Ollama) have no per-token API fee, so cost is always 0 there. For Claude, uses
    the per-million-token rates above. Returns 0.0 if usage is unavailable.
    """
    if usage is None or PROVIDER != "claude":
        return 0.0
    inp = getattr(usage, "input_tokens", 0) or 0
    out = getattr(usage, "output_tokens", 0) or 0
    c_read = getattr(usage, "cache_read_input_tokens", 0) or 0
    c_write = getattr(usage, "cache_creation_input_tokens", 0) or 0
    return (inp * PRICE_INPUT_PER_MTOK
            + out * PRICE_OUTPUT_PER_MTOK
            + c_read * PRICE_CACHE_READ_PER_MTOK
            + c_write * PRICE_CACHE_WRITE_PER_MTOK) / 1_000_000.0


def usage_tokens(usage):
    """Return (input, output, cache_read, cache_write) token counts, all ints."""
    if usage is None:
        return 0, 0, 0, 0
    return (getattr(usage, "input_tokens", 0) or 0,
            getattr(usage, "output_tokens", 0) or 0,
            getattr(usage, "cache_read_input_tokens", 0) or 0,
            getattr(usage, "cache_creation_input_tokens", 0) or 0)

# The assistant's name and greeting. Change ASSISTANT_NAME to rebrand everywhere.
ASSISTANT_NAME = "Lodestar"
ASSISTANT_AVATAR = None  # no avatar; CSS hides the avatar slot


def lodestar_icon(size=20, color="var(--logo-emblem)"):
    """
    The Lodestar mark: a four-point guiding star as inline SVG, so it scales
    crisply anywhere (greeting avatar, reply label) with no image asset.
    """
    return (
        f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" '
        f'xmlns="http://www.w3.org/2000/svg" style="display:block;flex:none;" '
        f'aria-label="Lodestar">'
        # main four-point star
        f'<path d="M12 1.2 L14.3 9.7 L22.8 12 L14.3 14.3 L12 22.8 L9.7 14.3 '
        f'L1.2 12 L9.7 9.7 Z" fill="{color}"/>'
        # soft inner highlight gives it depth
        f'<path d="M12 6.8 L13.1 10.9 L17.2 12 L13.1 13.1 L12 17.2 L10.9 13.1 '
        f'L6.8 12 L10.9 10.9 Z" fill="#FFFFFF" opacity="0.38"/>'
        # small companion spark, top-right
        f'<path d="M19.2 2.6 L19.8 4.6 L21.8 5.2 L19.8 5.8 L19.2 7.8 L18.6 5.8 '
        f'L16.6 5.2 L18.6 4.6 Z" fill="{color}" opacity="0.85"/>'
        f'</svg>')
GREETING_TITLE = f"Hey, I'm {ASSISTANT_NAME} 👋"
GREETING_BODY = ("I help with product questions. Ask a question, paste an error, or "
                 "attach a screenshot or log file to get started.")


class _TextBlock:
    """Minimal Anthropic-like content block so existing code keeps working."""
    def __init__(self, text):
        self.type = "text"
        self.text = text


class _Usage:
    def __init__(self, i=0, o=0, cr=0, cw=0):
        self.input_tokens = i
        self.output_tokens = o
        self.cache_read_input_tokens = cr
        self.cache_creation_input_tokens = cw


class _LLMResponse:
    """Uniform response: .content is a list of blocks, .usage has token counts."""
    def __init__(self, text, usage=None):
        self.content = [_TextBlock(text)]
        self.usage = usage or _Usage()


def _ollama_generate(model, system, messages, max_tokens):
    """
    Call a local Ollama server. Converts Anthropic-style messages (which may include
    image blocks) into Ollama's chat format. Images are passed as base64 for
    vision-capable models; text-only models just get the text.
    """
    import requests
    ol_messages = []
    if system:
        ol_messages.append({"role": "system", "content": system})
    for m in messages:
        content = m.get("content")
        if isinstance(content, str):
            ol_messages.append({"role": m["role"], "content": content})
        else:
            # content is a list of blocks (text / image)
            text_parts, images = [], []
            for blk in content:
                if blk.get("type") == "text":
                    text_parts.append(blk["text"])
                elif blk.get("type") == "image":
                    src = blk.get("source", {})
                    if src.get("type") == "base64":
                        images.append(src.get("data", ""))
            msg = {"role": m["role"], "content": "\n".join(text_parts)}
            if images:
                msg["images"] = images
            ol_messages.append(msg)

    resp = requests.post(
        f"{OLLAMA_URL}/api/chat",
        json={"model": model, "messages": ol_messages, "stream": False,
              # Low-ish temperature keeps answers faithful to the context without
              # being so rigid that the model omits facts that ARE there. 0.3 is a
              # good balance for grounded support answers.
              "options": {"num_predict": max_tokens, "temperature": 0.3}},
        timeout=180,
    )
    resp.raise_for_status()
    data = resp.json()
    text = (data.get("message") or {}).get("content", "")
    # Ollama returns token counts under these keys (when available).
    usage = _Usage(i=data.get("prompt_eval_count", 0),
                   o=data.get("eval_count", 0))
    return _LLMResponse(text, usage)

@observe(as_type="generation", name="llm", capture_input=False, capture_output=False)
def _llm_create(model=None, system=None, messages=None, max_tokens=1024, **kwargs):
    """
    Provider-agnostic LLM call with retry on transient errors. Routes to Claude
    (Anthropic API) or Ollama (local) based on LODESTAR_PROVIDER. Returns a uniform
    response object so the rest of the app doesn't care which model answered.
    """
    model = model or MODEL
    delays = [1.0, 3.0, 6.0]
    last_err = None
    for attempt in range(len(delays) + 1):
        try:
            if PROVIDER == "ollama":
                resp = _ollama_generate(model, system, messages or [], max_tokens)
            else:
                resp = _anthropic_client.messages.create(
                    model=model, system=system or anthropic.NOT_GIVEN,
                    messages=messages or [], max_tokens=max_tokens, **kwargs)
            # Accumulate token usage for the current turn.
            try:
                acc = st.session_state.get("_turn_usage")
                if acc is not None and getattr(resp, "usage", None) is not None:
                    i, o, cr, cw = usage_tokens(resp.usage)
                    acc["input"] += i
                    acc["output"] += o
                    acc["cache_read"] += cr
                    acc["cache_write"] += cw
                    acc["cost"] += estimate_cost(resp.usage)
            except Exception:
                pass
            i, o, cr, cw = usage_tokens(getattr(resp, "usage", None))
            get_client().update_current_generation(
                model=model or MODEL,
                usage_details={"input": i, "output": o},
                output=resp.content[0].text if getattr(resp, "content", None) else None,
            )
            return resp
        except Exception as e:
            last_err = e
            name = type(e).__name__.lower()
            transient = ("overloaded" in name or "ratelimit" in name
                         or "timeout" in name or "connection" in name
                         or "apistatus" in name or "529" in str(e) or "429" in str(e)
                         or "503" in str(e))
            if transient and attempt < len(delays):
                time.sleep(delays[attempt])
                continue
            raise last_err


def extract_error_text_from_image(image_bytes, media_type):
    if _client is None:
        return ""
    b64 = base64.standard_b64encode(image_bytes).decode("utf-8")
    msg = _llm_create(
        model=VISION_MODEL, max_tokens=1500,
        messages=[{"role": "user", "content": [
            {"type": "image", "source": {"type": "base64",
                                         "media_type": media_type, "data": b64}},
            {"type": "text", "text": (
                "This is a screenshot from a user asking for technical support. "
                "Transcribe ALL readable text you can see — error messages, terminal "
                "output, commands, logs, config, UI labels, dialog boxes. Preserve the "
                "wording. If it's a terminal or code, keep it verbatim. If there's no "
                "text but the image shows something relevant (a diagram, a UI state), "
                "briefly describe what's shown. Output only the transcription/"
                "description, no preamble.")},
        ]}],
    )
    return "".join(b.text for b in msg.content if b.type == "text").strip()


def phrase_catalog_answer(error_text, entry):
    if _client is None:
        return _render_entry_plain(entry)
    grounding = {
        "meaning": entry.meaning,
        "possible_causes": entry.possible_causes,
        "diagnostics": [{"command": d.command, "purpose": d.purpose}
                        for d in entry.diagnostics],
        "resolution_steps": [{"instruction": s.instruction, "command": s.command,
                              "requires_human": s.requires_human}
                             for s in entry.resolution_steps],
        "escalate_when": entry.escalate_when,
    }
    system = (
        "You are an L1 support assistant. Answer in "
        "English. Ground your answer ONLY in the matched catalog entry JSON. Do not "
        "invent commands or causes. Present diagnostics as safe read-only checks. Any "
        "resolution step with requires_human=true must be shown with a clear warning "
        "that it changes state and needs engineer approval. If multiple causes are "
        "possible, ask the customer to run the diagnostics and report back rather than "
        "asserting one. End by noting that unresolved cases go to a human engineer. "
        "Be concise. Use plain, everyday English — avoid stiff words like 'caveats' "
        "or 'considerations'. NEVER name a specific customer, bank, or company, even "
        "if you recognize one — answer only about the technical issue.")
    msg = _llm_create(
        model=MODEL, max_tokens=1200, system=system,
        messages=[{"role": "user", "content": (
            f"Customer error text:\n{error_text}\n\n"
            f"Matched catalog entry:\n{json.dumps(grounding, indent=2)}")}],
    )
    return "".join(b.text for b in msg.content if b.type == "text").strip()


def phrase_rag_answer(question, chunks, turns=None):
    if _client is None or not chunks:
        return ""
    context = "\n\n".join(f"[Source: {c.source}]\n{c.text}" for c in chunks)
    system = (
        "You are an L1 support assistant. Answer in English using ONLY the "
        "provided documentation context.\n\n"
        "Your job is to relay what the documentation says — thoroughly and faithfully. "
        "Do NOT over-summarize or compress the answer into one or two lines. If the "
        "documentation gives steps, commands, config, or explanation, present them in "
        "full. It's better to give the complete documented answer than a short "
        "paraphrase.\n\n"
        "How to answer:\n"
        "- Start with a one-line direct answer, then give the full details from the "
        "documentation: all relevant steps, commands, settings, and explanation.\n"
        "- Quote exact values verbatim from the context: file names, variables, "
        "commands, ports, flags, paths, version numbers.\n"
        "- FORMAT COMMANDS AS PROPER MARKDOWN CODE BLOCKS. Put any shell command, "
        "code, or config inside a fenced code block with triple backticks on their "
        "OWN lines, like:\n"
        "```bash\n"
        "helm repo add example https://charts.example.com\n"
        "```\n"
        "Never put a command inline in the middle of a sentence. Never write the word "
        "'bash' inline before a command — it goes as the language tag right after the "
        "opening triple backticks. Each command block on its own lines.\n"
        "- Use numbered steps for procedures and bullet points for lists. Use short "
        "**bold** lead-ins instead of large markdown headings (no '#'/'##').\n"
        "- LINE BREAKS MATTER. Put each bullet and each numbered step on its OWN line, "
        "with a blank line between items. Never write list items inline like "
        "'- a - b - c' or '1. x 2. y' in one paragraph — that breaks the formatting. "
        "Each '- item' and each '1. step' starts a new line. Keep a bullet's label and "
        "its text on the SAME line: write '- **Root Cause:** the module fails', never "
        "put the bullet marker and its bold label on separate lines.\n"
        "- Plain, everyday English. Avoid stiff words like 'caveats', 'considerations', "
        "'aforementioned', 'utilize'.\n\n"
        "Accuracy & safety:\n"
        "- Base everything on the documentation context. Use what IS in the context "
        "fully — if it states a fix, workaround, version, or command, include it. "
        "Don't hold back information that's there.\n"
        "- Do NOT invent facts, version numbers, commands, or steps that don't appear "
        "in the context. If it's in the context, use it; if not, don't write it.\n"
        "- If the context does NOT actually answer the question, say so plainly "
        "(e.g. 'The documentation I have does not cover the default SmartDashboard "
        "admin password'). Do not guess, extrapolate, or stretch a loosely related "
        "passage into an answer. It is fine to add what the context DOES say about "
        "the topic, clearly labelled as related information, not as the answer.\n"
        "- Example and placeholder values in the documentation are NOT real values. "
        "Sample passwords, tokens, secrets, hostnames, IDs, or domains that appear in "
        "config examples (app.conf, credentials.conf, YAML/JSON samples, 'e.g.' "
        "values, <your-...> placeholders) are illustrations only. Never present them "
        "as the actual or default value. Say the real value is generated per "
        "environment or set by the customer, and point to where it is configured.\n"
        "- NEVER name a specific customer, bank, company, or organization. Even if you "
        "recognize one, do not write it. If the context refers to a customer "
        "generically (e.g. [CUSTOMER]), keep it generic. Placeholders like "
        "<your-username> and <your-password> are fine to keep as-is.\n"
        "- Only if the context is genuinely unrelated to the question, reply with "
        "EXACTLY the token INSUFFICIENT_CONTEXT and nothing else.\n"
        "- Flag any state-changing step (delete, regenerate keys, edit configs) as "
        "needing engineer approval.\n"
        "- Do NOT mention or cite the source file names in your answer.\n"
        "- If recent conversation is provided, use it to understand follow-up "
        "questions (resolve 'it', 'that', etc.), but base technical facts only on the "
        "documentation context.")
    convo_block = ""
    if turns:
        convo_block = f"Recent conversation:\n{_history_text(turns)}\n\n"
    msg = _llm_create(
        model=MODEL, max_tokens=2000, system=system,
        messages=[{"role": "user", "content": (
            f"{convo_block}"
            f"Customer question / error:\n{question}\n\n"
            f"Documentation context:\n{context}")}],
    )
    return "".join(b.text for b in msg.content if b.type == "text").strip()


def _render_entry_plain(entry):
    lines = [f"### {entry.title}", "", f"**What it means:** {entry.meaning}", "",
             "**Possible causes:**"]
    lines += [f"- {c}" for c in entry.possible_causes]
    lines += ["", "**Safe diagnostic checks (read-only):**"]
    for d in entry.diagnostics:
        lines.append(f"- {d.purpose}")
        lines.append(f"  ```bash\n  {d.command}\n  ```")
    lines += ["", "**Resolution:**"]
    for s in entry.resolution_steps:
        flag = "  (!) Changes state - needs engineer approval." if s.requires_human else ""
        lines.append(f"- {s.instruction}{flag}")
        if s.command:
            lines.append(f"  ```bash\n  {s.command}\n  ```")
    lines += ["", "_If these checks don't resolve it, this case will be escalated to a "
              "human engineer._"]
    return "\n".join(lines)


def escalation_message(text):
    return (
        "I couldn't confidently answer this from known errors or the docs, so rather "
        "than guess I'm flagging it for a human engineer. To speed that up, please "
        "share:\n"
        "- The full command you ran\n"
        "- The complete error output (not just the last line)\n"
        "- Your environment name and, if relevant, `uname -m`")


def analyze_attachment(text):
    """
    Directly analyze a message that contains attached log/screenshot content.
    Skips the catalog/RAG routing (which can mis-fire on log keywords) and asks the
    LLM to read THIS specific content and explain the actual error in it.
    """
    if _client is None:
        return [{"kind": "escalate", "title": "", "badge": "",
                 "body": "I need an API key to analyze attached logs or screenshots."}]
    # Also pull any relevant docs to ground the explanation, but the focus is the
    # attached content itself.
    doc_context = ""
    if _rag_available and rag.index_exists():
        try:
            chunks = rag.retrieve(text, k=5)
            doc_context = "\n\n".join(f"[{c.source}]\n{c.text}" for c in chunks)
        except Exception:
            pass
    system = (
        "You are a support assistant analyzing a specific log or screenshot the "
        "user just shared. Read the ATTACHED content carefully and:\n"
        "- Identify the actual error(s) present in THIS content — quote the exact "
        "error lines.\n"
        "- Explain what the error means and the most likely cause.\n"
        "- Give concrete next steps. Flag any state-changing step as needing engineer "
        "approval.\n"
        "- Base your answer on what's actually in the attached content, not on generic "
        "assumptions. If documentation context is provided, use it to explain, but the "
        "attached content is the source of truth for what went wrong.\n"
        "- Reply in the user's language. Keep it chat-friendly, no huge headings. Use "
        "plain, everyday words — avoid stiff terms like 'caveats' or 'considerations'. "
        "NEVER name a specific customer, bank, or company, even if you recognize one — "
        "describe only the technical problem and fix.")
    user = f"{text}"
    if doc_context:
        user += f"\n\n---\nRelevant documentation for reference:\n{doc_context}"
    try:
        msg = _llm_create(
            model=MODEL, max_tokens=2000, system=system,
            messages=[{"role": "user", "content": user}])
        body = "".join(b.text for b in msg.content if b.type == "text").strip()
    except Exception as e:
        body = f"I couldn't analyze the attachment: {e}"
    return [{"kind": "docs", "title": "", "badge": "", "body": body}]


def _strip_customer_mentions(text):
    """
    Chat-output filter: turn any masking tags in the assistant's reply into neutral,
    readable wording so the chat never shows technical tags like [CUSTOMER] or
    [MASKED-TERM]. (The knowledge base keeps the raw tags for auditability; this is
    only for what the user sees in chat.)

    Removing a name entirely can break sentences ("when Acme calls" →
    "when calls"), so we substitute neutral words instead, then tidy up.
    """
    import re
    try:
        import jira_sync as _js
    except Exception:
        return text
    masked = _js.mask(text)

    # Customer tags → "the customer".
    masked = re.sub(r"\[CUSTOMER\]'s", "the customer's", masked)
    masked = re.sub(r"\b(the|a|an)\s+\[CUSTOMER\]\b", "the customer", masked,
                    flags=re.IGNORECASE)
    masked = re.sub(r"\[CUSTOMER\]", "the customer", masked)

    # Other masking tags → neutral wording. A tag can be glued to a suffix
    # (e.g. "[MASKED-TERM]ApiCall"); absorb the trailing word so no fragment is left.
    # Order matters: most specific first.
    masked = re.sub(r"\[MASKED-JWT\]\w*", "a token", masked)
    masked = re.sub(r"\[MASKED-PRIVATE-KEY\]\w*", "a private key", masked)
    masked = re.sub(r"\[MASKED-HEX\]\w*", "a value", masked)
    masked = re.sub(r"\[MASKED-B64\]\w*", "a value", masked)
    # Internal term (possibly glued to a suffix like "ApiCall") → "an internal call"
    # when a suffix is attached, else "an internal identifier".
    masked = re.sub(r"\[MASKED-TERM\][A-Za-z]\w*", "an internal call", masked)
    masked = re.sub(r"\[MASKED-TERM\]", "an internal identifier", masked)
    masked = re.sub(r"\[MASKED\]\w*", "a value", masked)
    masked = re.sub(r"\[USER\]", "a user", masked)
    masked = re.sub(r"\[IP\]", "an IP address", masked)
    masked = re.sub(r"\[EMAIL\]", "an email address", masked)
    masked = re.sub(r"\[TEAMS-LINK\]", "an internal link", masked)
    masked = re.sub(r"\[TEAMS-REF\]", "an internal discussion", masked)

    # Fix artifacts: doubled article ("the the customer" / "a an internal"), spacing,
    # capitalization. Also "the an internal..." → "the internal..." when a definite
    # article already precedes a neutral phrase we inserted with a/an.
    masked = re.sub(r"\b(the|a|an)\s+the customer\b", "the customer", masked,
                    flags=re.IGNORECASE)
    masked = re.sub(r"\bthe\s+an?\s+(internal|value|token|private key|user|IP|email)",
                    r"the \1", masked, flags=re.IGNORECASE)
    masked = re.sub(r"\b(a|an)\s+(a|an)\s+", r"\1 ", masked, flags=re.IGNORECASE)
    masked = re.sub(r"\s{2,}", " ", masked)
    masked = re.sub(r"\s+([,.;:!?])", r"\1", masked)
    # Capitalize a leading "the customer" at a sentence start.
    masked = re.sub(r"(^|[.!?]\s+)the customer",
                    lambda m: m.group(0)[:-12] + "The customer", masked)
    return masked.strip()


def _mask_blocks(blocks):
    """
    Final safety net for CHAT output: remove customer names / secrets from the
    assistant's OWN reply before it reaches the user. The model can emit a known
    bank name from its own training data even when it's not in the context; this
    catches that regardless of why it appeared. Customer names are removed entirely
    (no visible tag) so the chat reads naturally.
    """
    for b in blocks:
        if b.get("body"):
            b["body"] = _strip_customer_mentions(b["body"])
        if b.get("title"):
            b["title"] = _strip_customer_mentions(b["title"])
    return blocks


def _recent_turns(history, max_turns=4):
    """
    Pull the last `max_turns` user/assistant exchanges from history as plain text,
    for giving the model short-term memory. Assistant turns are stored as blocks;
    we flatten their bodies. Returns a list of {"role", "text"} newest-last.
    """
    out = []
    for m in history or []:
        role = m.get("role")
        if role == "user":
            txt = m.get("display") or m.get("content") or ""
        elif role == "assistant":
            blocks = m.get("blocks") or []
            txt = "\n".join(b.get("body", "") for b in blocks if b.get("body"))
        else:
            continue
        if txt.strip():
            out.append({"role": role, "text": txt.strip()})
    # Keep the last max_turns*2 messages (a turn ≈ user + assistant).
    return out[-(max_turns * 2):]


def _history_text(turns):
    """Format recent turns as a compact transcript for a prompt."""
    lines = []
    for t in turns:
        who = "User" if t["role"] == "user" else "Assistant"
        lines.append(f"{who}: {t['text']}")
    return "\n".join(lines)


def _contextualize_query(question, turns):
    """
    Rewrite a follow-up question into a standalone, searchable query using the recent
    conversation. E.g. "what if it's expired?" → "what if the activation code is
    expired?". Falls back to the original question on any error. Only runs when there
    is prior context; a first question is already standalone.
    """
    if not turns:
        return question
    try:
        system = (
            "Rewrite the user's latest question into a single standalone question "
            "that can be searched on its own, resolving references like 'it', 'that', "
            "'this' using the conversation. Keep it short. Output ONLY the rewritten "
            "question, nothing else. If it's already standalone, return it unchanged.")
        convo = _history_text(turns)
        result = _llm_create(
            model=MODEL, max_tokens=120, system=system,
            messages=[{"role": "user",
                       "content": f"Conversation:\n{convo}\n\nLatest question: "
                                  f"{question}\n\nStandalone question:"}])
        rewritten = "".join(b.text for b in result.content
                            if b.type == "text").strip()
        # Guard against the model over-explaining; take the first line, cap length.
        rewritten = rewritten.split("\n")[0].strip().strip('"')
        return rewritten if rewritten else question
    except Exception:
        return question

@observe(name="lodestar-chat")
def generate_reply(text, has_attachment=False, history=None):
    """
    Run the answering pipeline and return (blocks, meta). With `history` (recent
    conversation turns), the assistant has short-term memory: follow-up questions are
    understood in context, RAG search is done on a context-resolved query, and the
    answer model sees the recent exchange.
    All answer text passes through _mask_blocks() as a final output filter so no
    customer identifier reaches the user, even one the model produced on its own.
    """
    turns = _recent_turns(history, max_turns=4)

    if has_attachment:
        blocks = analyze_attachment(text)
        return _mask_blocks(blocks), {"route": "attachment", "num_sources": 0}

    # 1) Known-error catalog (matched on the raw message; errors are self-contained)
    matches = ec.find_matches(text)
    if matches:
        blocks = []
        for entry in matches:
            blocks.append({
                "kind": "catalog",
                "title": entry.title,
                "badge": "Catalog match",
                "body": phrase_catalog_answer(text, entry),
            })
        return _mask_blocks(blocks), {"route": "catalog", "num_sources": len(matches)}

    # 2) Documentation RAG — search on a context-resolved query, answer with memory
    if _rag_available and _client is not None and rag.index_exists():
        search_query = _contextualize_query(text, turns)
        chunks = rag.retrieve(search_query, k=12)
        # Relevance filtering, but keep whole documents together: if a document has a
        # relevant chunk, keep its other chunks too (they're the rest of the same
        # ticket/page and often hold the detail).
        # Threshold calibration (reranker scores, golden.jsonl, 2026-09-04): real
        # questions score 0.75-1.0 on their best chunk; off-topic/unknown questions
        # top out around 0.1-0.45. 0.5 keeps the relevant docs and drops the weak
        # 0.15-0.45 band that used to leak unrelated chunks into the prompt.
        min_score = float(os.getenv("LODESTAR_MIN_SOURCE_SCORE", "0.5"))
        strong_docs = {os.path.basename(str(c.source))
                       for c in chunks if float(c.score) >= min_score}
        if strong_docs:
            chunks = [c for c in chunks
                      if os.path.basename(str(c.source)) in strong_docs]
        elif chunks:
            # Nothing cleared the bar; still try the single best chunk if it's
            # moderately relevant (hard cross-document questions can score ~0.3),
            # but not the near-zero noise. The prompt's "say when the context does
            # not answer" rule handles the remaining borderline cases.
            fallback = float(os.getenv("LODESTAR_FALLBACK_SOURCE_SCORE", "0.3"))
            top = max(chunks, key=lambda c: float(c.score))
            chunks = [top] if float(top.score) >= fallback else []
        if chunks:
            answer = phrase_rag_answer(text, chunks, turns=turns)
            if answer and "INSUFFICIENT_CONTEXT" not in answer:
                return (_mask_blocks([{
                    "kind": "docs", "title": "", "badge": "",
                    "body": answer,
                    "sources": [(str(c.source), float(c.score)) for c in chunks],
                }]), {"route": "docs", "num_sources": len(chunks)})

    # 3) Escalate
    return (_mask_blocks([{"kind": "escalate", "title": "Routing to a human engineer",
              "badge": "Escalated", "body": escalation_message(text)}]),
            {"route": "escalate", "num_sources": 0})


st.set_page_config(page_title="Lodestar", page_icon="🌟", layout="centered",
                   initial_sidebar_state="collapsed")

# Load the embedding model once, up front, so the first doc query is fast.
_warm_embedder()

# ---------------------------------------------------------------------------- #
# Dark brand theme
# Injected as CSS so the working Streamlit app matches the branded mockup.
# ---------------------------------------------------------------------------- #
st.markdown("""
<style>
  /* Typography: Docusaurus/Infima-style system-font stack */

  /* Typography matched to a Docusaurus / Infima documentation site.
     Such sites use the system font stack, 16px base, and a
     relaxed 1.65 line-height. No web-font download — fast and native. */

  :root {
    --primary: #3A67EF;
    --secondary: #6B8FF5;
    --text-primary: #ECEDEE;          /* light text on dark */
    --text-secondary: #B4B8C0;
    --text-tertiary: #8A8F99;
    --background: #0F0E13;             /* dark near-black */
    --surface: #1A1B21;               /* cards / assistant bubbles */
    --surface-2: #23242B;             /* hover / raised */
    --border: #2A2C34;
    --user-bubble: #3A67EF;           /* user message accent */
    --white: #FFFFFF;
    --logo-emblem: #8FA8FF;           /* Lodestar star mark, bright on dark bg */
    --success: #1DAC76;
    --radius-card: 16px;
    --radius-ctrl: 14px;
    --font-base: system-ui, -apple-system, "Segoe UI", Roboto, Ubuntu, Cantarell,
                 "Noto Sans", BlinkMacSystemFont, "Helvetica Neue", Arial, sans-serif;
  }

  /* Base — dark theme */
  .stApp { background: var(--background); }
  html, body, [class*="css"], [data-testid="stAppViewContainer"] {
    font-family: var(--font-base) !important;
    color: var(--text-primary);
    font-size: 16px;
    line-height: 1.65;
  }
  .stApp h1, .stApp h2, .stApp h3, .stApp h4 {
    font-family: var(--font-base) !important;
    font-weight: 700 !important;
    line-height: 1.25 !important;
    color: var(--text-primary) !important;
  }
  /* Body copy sizing to match the docs site */
  .stApp p, .stApp li { font-size: 16px; line-height: 1.65; }
  .block-container { padding-top: 1.5rem; max-width: 860px; }

  /* Hide Streamlit chrome for a cleaner app feel */
  /* Hide only the Streamlit hamburger menu and footer. Do NOT touch the header
     region or any sidebar controls — hiding those traps the sidebar closed. */
  #MainMenu { visibility: hidden; }
  header[data-testid="stHeader"] { background: transparent; }
  /* Force every known sidebar collapse/expand control to stay visible across
     Streamlit versions (the testid has changed between releases). */
  [data-testid="stSidebarCollapsedControl"],
  [data-testid="stSidebarCollapseButton"],
  [data-testid="collapsedControl"],
  [data-testid="stExpandSidebarButton"],
  button[kind="header"] {
    display: flex !important;
    visibility: visible !important;
    opacity: 1 !important;
    z-index: 999999 !important;
  }
  /* Keep the sidebar itself visible when expanded. */
  section[data-testid="stSidebar"] { display: block !important; }

  /* Hero title */
  h1, h2, h3 { font-family: var(--font-base) !important; }
  .stApp h1 { font-weight: 700 !important; font-size: 30px !important; letter-spacing: -0.01em; }

  /* Text areas & inputs (dark) */
  textarea, .stTextArea textarea {
    font-family: ui-monospace, "SF Mono", Menlo, monospace !important;
    border-radius: var(--radius-ctrl) !important;
    border: 1px solid var(--border) !important;
    background: var(--surface) !important;
    color: var(--text-primary) !important;
    font-size: 14px !important;
  }
  .stTextArea textarea:focus {
    border-color: var(--primary) !important;
    box-shadow: 0 0 0 2px rgba(58,103,239,0.25) !important;
  }

  /* Primary button */
  .stButton > button[kind="primary"], .stButton > button:first-child {
    background: var(--primary) !important;
    color: #fff !important;
    border: none !important;
    border-radius: var(--radius-ctrl) !important;
    font-weight: 600 !important;
    padding: 12px 24px !important;
    transition: filter 300ms ease !important;
  }
  .stButton > button:first-child:hover { filter: brightness(1.12) !important; }

  /* Expander / info cards (dark) */
  div[data-testid="stExpander"] {
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-card) !important;
    background: var(--surface) !important;
  }
  div[data-testid="stExpander"] summary { color: var(--text-secondary) !important; }

  /* Code blocks (dark) */
  .stCodeBlock, pre, code {
    background: #0B0A0E !important;
    border-radius: 10px !important;
    color: #E6E7EB !important;
  }

  /* Alerts (st.info / st.warning / st.error) dark surfaces */
  [data-testid="stAlert"] {
    background: var(--surface) !important;
    color: var(--text-secondary) !important;
    border-radius: 12px !important;
  }

  /* Badges / captions */
  .stCaption, .caption { color: var(--text-tertiary) !important; }

  /* Section divider */
  hr { border-color: var(--border) !important; }

  /* Compact feedback buttons (👍 👎) — transparent, small, side by side.
     Scoped to the main area so sidebar buttons keep their normal styling. */
  section.main div[data-testid="stHorizontalBlock"] .stButton > button,
  [data-testid="stAppViewContainer"] > section:not([data-testid="stSidebar"])
    div[data-testid="stHorizontalBlock"] .stButton > button {
    background: transparent !important;
    border: none !important;
    padding: 2px 4px !important;
    min-height: 30px !important;
    height: 30px !important;
    font-size: 15px !important;
    border-radius: 8px !important;
    opacity: 0.65;
  }
  [data-testid="stAppViewContainer"] > section:not([data-testid="stSidebar"])
    div[data-testid="stHorizontalBlock"] .stButton > button:hover {
    background: var(--surface) !important;
    opacity: 1;
  }
  [data-testid="stAppViewContainer"] > section:not([data-testid="stSidebar"])
    div[data-testid="stHorizontalBlock"] .stButton > button[kind="primary"] {
    background: var(--surface) !important;
    opacity: 1;
  }

  /* Assistant bubble: bordered container in dark surface */
  [data-testid="stVerticalBlockBorderWrapper"] {
    background: var(--surface) !important;
    border-color: var(--border) !important;
    border-radius: 4px 16px 16px 16px !important;
  }

  /* Sidebar dark */
  section[data-testid="stSidebar"] {
    background: #141319 !important;
    border-right: 1px solid var(--border) !important;
  }
  section[data-testid="stSidebar"] * { color: var(--text-secondary) !important; }
  section[data-testid="stSidebar"] h3 { color: var(--text-primary) !important; }
  section[data-testid="stSidebar"] h4 { color: var(--text-primary) !important; }
  /* Conversation buttons: single line, ellipsis instead of overlapping wrap */
  section[data-testid="stSidebar"] .stButton > button {
    white-space: nowrap !important;
    overflow: hidden !important;
    text-overflow: ellipsis !important;
    text-align: left !important;
    line-height: 1.4 !important;
    min-height: 40px !important;
    height: auto !important;
    display: block !important;
    font-size: 13px !important;
  }
  /* But keep the New Chat button centered */
  section[data-testid="stSidebar"] .stButton > button[kind="secondary"]#newchat,
  section[data-testid="stSidebar"] div:first-child .stButton > button {
    text-align: center !important;
  }

  /* Hide chat avatars (not used, but belt-and-braces) */
  [data-testid="stChatMessageAvatar"],
  [data-testid="chatAvatarIcon-user"],
  [data-testid="chatAvatarIcon-assistant"],
  [data-testid="stChatMessageAvatarUser"],
  [data-testid="stChatMessageAvatarAssistant"] {
    display: none !important; width: 0 !important; margin: 0 !important;
  }
  [data-testid="stChatMessage"] { gap: 0 !important; padding-left: 0 !important; }

  /* Chat input (dark, blended) */
  [data-testid="stChatInput"],
  [data-testid="stBottomBlockContainer"],
  [data-testid="stBottom"],
  [data-testid="stBottom"] > div {
    background: var(--background) !important;
  }
  [data-testid="stChatInput"] textarea,
  [data-testid="stChatInputTextArea"] {
    background: var(--surface) !important;
    color: var(--text-primary) !important;
    border-radius: 14px !important;
  }
  [data-testid="stChatInput"] > div {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 16px !important;
    box-shadow: none !important;
  }
</style>
""", unsafe_allow_html=True)

st.markdown(f"""
<div style="display:inline-flex;align-items:center;gap:8px;font-size:11px;
            font-weight:600;letter-spacing:.06em;text-transform:uppercase;
            color:var(--secondary);background:rgba(107,143,245,0.12);
            border:1.5px solid rgba(107,143,245,0.2);border-radius:50px;
            padding:5px 12px;margin-bottom:8px;">
  <span style="width:8px;height:8px;border-radius:50%;background:var(--success);
               display:inline-block;"></span>
  {ASSISTANT_NAME} · Online
</div>
""", unsafe_allow_html=True)

if _client is None:
    st.info("No ANTHROPIC_API_KEY set - screenshot reading and doc-based answers are "
            "disabled. The known-error catalog still works.")
if _rag_available and not rag.index_exists():
    st.warning("Documentation not indexed yet. Add docs from the sidebar to enable "
               "documentation answers.")


def _fix_markdown(text):
    """
    Repair markdown the model produced without proper line breaks. Small models often
    write list items and steps inline ("- a - b - c" or "1. x 2. y") instead of on
    their own lines, so Streamlit renders them as one run-on paragraph. This inserts
    the blank lines / newlines that markdown needs, makes fenced code blocks sit on
    their own lines, and removes empty bullets/numbers left when a list marker and
    its bold lead-in ("- **Root Cause:** ...") get split across lines.
    """
    import re
    if not text:
        return text
    t = text
    # Ensure a fenced code block opener/closer is on its own line.
    t = re.sub(r'(?<!\n)```', r'\n```', t)
    t = re.sub(r'```(\w+)?[ \t]+', r'```\1\n', t)  # ```bash cmd -> ```bash\ncmd
    # Put a newline before a bullet "- " that follows sentence text on the same line.
    t = re.sub(r'(?<=[^\n])\s+-\s+(?=[A-Z0-9`*])', r'\n\n- ', t)
    # Put a newline before numbered steps "1. " / "2) " that run inline.
    t = re.sub(r'(?<=[^\n])\s+(\d{1,2})[.)]\s+(?=[A-Z`*])', r'\n\n\1. ', t)
    # A bold lead-in like "**Note:**" mid-line -> start on a new line, but ONLY when
    # it is NOT already the content of a list marker (so we don't split "- **X:**").
    t = re.sub(r'(?<=[^\n*\-.\d\s])\s+(\*\*[A-Z][^*]{1,40}:\*\*)', r'\n\n\1', t)
    # Merge a list marker that got separated from its bold lead-in:
    #   "- \n\n**Root Cause:** x"  ->  "- **Root Cause:** x"
    t = re.sub(r'(?m)^(\s*(?:[-*]|\d{1,2}[.)]))\s*\n\s*\n(\*\*)', r'\1 \2', t)
    # Drop any bullet/number that is now empty (nothing after the marker on its line).
    t = re.sub(r'(?m)^\s*(?:[-*]|\d{1,2}[.)])\s*$\n?', '', t)
    # Collapse 3+ newlines to a clean paragraph break.
    t = re.sub(r'\n{3,}', '\n\n', t)
    return t.strip()


def render_blocks(blocks):
    """Render each answer block inside a bordered assistant bubble, with sources."""
    for b in blocks:
        with st.container(border=True):
            st.markdown(_fix_markdown(b["body"]))
            # Show which documents the answer came from, with retrieval scores.
            sources = b.get("sources")
            if sources:
                # De-duplicate by source, keeping the highest score per document.
                # (After JSON round-trip, each entry may be a list, not a tuple.)
                _disp_min = float(os.getenv("LODESTAR_MIN_SOURCE_SCORE", "0.15"))
                best = {}
                for entry in sources:
                    src, score = entry[0], entry[1]
                    name = os.path.basename(str(src))
                    score = float(score)
                    if score < _disp_min:
                        continue  # hide weakly-relevant sources from the UI too
                    if name not in best or score > best[name]:
                        best[name] = score
                ranked = sorted(best.items(), key=lambda x: x[1], reverse=True)[:5]
                chips = []
                for name, score in ranked:
                    # Colour by confidence: green (strong), amber (medium), grey (weak).
                    if score >= 0.6:
                        col = "#1DAC76"
                    elif score >= 0.4:
                        col = "#E0A93B"
                    else:
                        col = "#8A8F99"
                    chips.append(
                        f'<span style="display:inline-flex;align-items:center;gap:6px;'
                        f'background:var(--surface-2);border:1px solid var(--border);'
                        f'border-radius:8px;padding:3px 9px;margin:3px 6px 3px 0;'
                        f'font-size:12px;color:var(--text-secondary);">'
                        f'📄 {name}'
                        f'<span style="color:{col};font-weight:600;font-variant-numeric:'
                        f'tabular-nums;">{round(score * 100)}%</span></span>')
                st.markdown(
                    '<div style="font-size:11px;color:var(--text-tertiary);'
                    'text-transform:uppercase;letter-spacing:.05em;margin:10px 0 4px;">'
                    'Sources</div>'
                    '<div style="display:flex;flex-wrap:wrap;">' + "".join(chips)
                    + '</div>', unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Chat state + history
# --------------------------------------------------------------------------- #
if "current_conv_id" not in st.session_state:
    st.session_state.current_conv_id = uuid.uuid4().hex[:12]
if "history" not in st.session_state:
    st.session_state.history = []  # list of {"role","content"|"blocks"}


def _save_current():
    """Save the current conversation if it has any messages."""
    if st.session_state.history:
        # Title = first user message (trimmed), like most chat apps.
        title = "New chat"
        for m in st.session_state.history:
            if m["role"] == "user":
                t = m.get("display") or m.get("content", "")
                title = (t[:48] + "…") if len(t) > 48 else (t or "New chat")
                break
        save_conversation(st.session_state.current_conv_id, title,
                          st.session_state.history)


def start_new_chat():
    _save_current()
    st.session_state.current_conv_id = uuid.uuid4().hex[:12]
    st.session_state.history = []


def switch_to_conversation(cid):
    _save_current()
    conv = load_conversation(cid)
    if conv:
        st.session_state.current_conv_id = cid
        st.session_state.history = conv.get("messages", [])


def user_bubble(text):
    """Render a user message as a right-aligned bubble (no avatar)."""
    safe = (text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))
    st.markdown(
        f'<div style="display:flex;justify-content:flex-end;margin:10px 0;">'
        f'<div style="background:var(--primary);color:#fff;padding:10px 14px;'
        f'border-radius:16px 16px 4px 16px;max-width:80%;font-size:14px;'
        f'white-space:pre-wrap;word-break:break-word;">{safe}</div></div>',
        unsafe_allow_html=True)


def assistant_open():
    """Small assistant label with the star mark above a reply."""
    st.markdown(
        '<div style="display:flex;align-items:center;gap:6px;margin:8px 0 4px;">'
        f'{lodestar_icon(13)}'
        f'<span style="font-size:12px;font-weight:600;color:var(--secondary);">'
        f'{ASSISTANT_NAME}</span></div>', unsafe_allow_html=True)


# Greeting with the circular star-mark avatar
st.markdown(f"""
<div style="display:flex;align-items:flex-start;gap:14px;margin-bottom:8px;
            font-family:var(--font-base);">
  <div style="flex:none;width:44px;height:44px;border-radius:50%;
              background:radial-gradient(circle at 50% 45%, #1E2340 0%, var(--surface) 70%);
              border:1px solid var(--border);
              box-shadow:0 0 14px rgba(143,168,255,0.25);
              display:flex;align-items:center;justify-content:center;">
    {lodestar_icon(24)}
  </div>
  <div>
    <div style="font-size:22px;font-weight:700;color:var(--text-primary);
                margin-bottom:6px;">{GREETING_TITLE}</div>
    <div style="font-size:15px;color:var(--text-secondary);line-height:1.6;">
      {GREETING_BODY}</div>
  </div>
</div>
""", unsafe_allow_html=True)

# Replay history
for msg in st.session_state.history:
    if msg["role"] == "user":
        # Prefer the clean display text; fall back to content for old messages.
        shown = msg.get("display") or msg["content"]
        if len(shown) > 600:
            shown = shown[:600] + f"\n… (+{len(shown) - 600:,} more characters)"
        user_bubble(shown)
    else:
        assistant_open()
        if msg.get("blocks") is not None:
            render_blocks(msg["blocks"])
        else:
            st.markdown(msg["content"])
        # Performance line + feedback + one-click copy for assistant turns.
        tid = msg.get("turn_id")
        if tid:
            meta = msg.get("meta", {})
            lat = msg.get("latency", 0)
            cost = msg.get("cost", 0.0)
            cost_str = f" · ~${cost:.4f}" if cost else ""
            st.markdown(
                f'<div style="font-size:11px;color:var(--text-tertiary);margin:2px 0 6px;">'
                f'{lat:.1f}s · {meta.get("num_sources", 0)} source(s) · '
                f'via {meta.get("route", "?")}{cost_str}</div>', unsafe_allow_html=True)
            fb_key = f"fb_{tid}"
            current = st.session_state.get(fb_key)
            answer_text = "\n\n".join(b.get("body", "")
                                      for b in (msg.get("blocks") or []))
            import json as _json
            js_text = _json.dumps(answer_text).replace("</", "<\\/")
            c1, c2, c3, _sp = st.columns([1, 1, 1, 10], gap="small")
            if c1.button("👍" if current != "up" else "👍",
                         key=f"up_{tid}",
                         type="primary" if current == "up" else "secondary"):
                st.session_state[fb_key] = "up"
                update_feedback(tid, "up")
                st.rerun()
            if c2.button("👎", key=f"down_{tid}",
                         type="primary" if current == "down" else "secondary"):
                st.session_state[fb_key] = "down"
                update_feedback(tid, "down")
                st.rerun()
            with c3:
                # Copy: transparent SVG icon button, direct clipboard write.
                components.html(
                    "<!DOCTYPE html><html><head><meta charset='utf-8'></head>"
                    "<body style='margin:0;'>"
                    "<button id='cp' title='Copy' style='background:transparent;"
                    "border:none;cursor:pointer;padding:5px;border-radius:8px;'>"
                    "<svg id='cpicon' width='18' height='18' viewBox='0 0 24 24' "
                    "fill='none' stroke='#8A8F99' stroke-width='2' "
                    "stroke-linecap='round' stroke-linejoin='round'>"
                    "<rect width='14' height='14' x='8' y='8' rx='2' ry='2'/>"
                    "<path d='M4 16c-1.1 0-2-.9-2-2V4c0-1.1.9-2 2-2h10c1.1 0 2 .9 2 "
                    "2'/></svg></button>"
                    "<script>const t=" + js_text + ";"
                    "document.getElementById('cp').onclick=()=>{"
                    "navigator.clipboard.writeText(t).then(()=>{"
                    "const i=document.getElementById('cpicon');"
                    "i.setAttribute('stroke','#1DAC76');"
                    "setTimeout(()=>i.setAttribute('stroke','#8A8F99'),1200);});};"
                    "</script></body></html>",
                    height=34)

# Chat input with an attach button built in (screenshots, logs, or text files).
composer = st.chat_input(
    f"Ask a question, or attach a screenshot or log file…",
    accept_file=True,
    file_type=["png", "jpg", "jpeg", "log", "txt", "out", "err"],
)

typed = None
attachment = None
if composer is not None:
    typed = (composer.text or "").strip() or None
    if composer.files:
        attachment = composer.files[0]


def _read_log_text(file_obj) -> str:
    """
    Read a log/text attachment as UTF-8 text. Logs can be huge, so instead of a
    blind head+tail cut, find the lines that actually matter — errors, exceptions,
    failures, stack traces — and keep those plus surrounding context. This focuses
    the assistant on the real problem instead of thousands of routine INFO lines.
    """
    raw = file_obj.getvalue()
    try:
        text = raw.decode("utf-8", errors="ignore")
    except Exception:
        text = str(raw)

    max_chars = 12000
    if len(text) <= max_chars:
        return text

    import re
    lines = text.splitlines()

    # Lines that signal a real problem.
    err_re = re.compile(
        r"\b(error|fatal|exception|traceback|panic|fail(ed|ure)?|"
        r"critical|denied|refused|timeout|cannot|unable|"
        r"\bERR\b|\bWARN(ING)?\b)\b", re.IGNORECASE)

    # Collect indices of interesting lines and a window of context around each.
    context = 4
    keep = set()
    hits = 0
    for i, ln in enumerate(lines):
        if err_re.search(ln):
            hits += 1
            for j in range(max(0, i - context), min(len(lines), i + context + 1)):
                keep.add(j)

    if not keep:
        # No obvious error lines — fall back to head + tail.
        head = text[:2000]
        tail = text[-(max_chars - 2000):]
        return (f"{head}\n\n... [log truncated — no explicit error lines found; "
                f"showing head and tail of a {len(text):,}-char file] ...\n\n{tail}")

    # Build a focused excerpt from the kept lines, marking gaps.
    kept_sorted = sorted(keep)
    out_lines = []
    prev = None
    for idx in kept_sorted:
        if prev is not None and idx > prev + 1:
            out_lines.append(f"... [skipped {idx - prev - 1} lines] ...")
        out_lines.append(lines[idx])
        prev = idx
    excerpt = "\n".join(out_lines)

    # If still too big, keep the most relevant portion (errors cluster late in logs).
    if len(excerpt) > max_chars:
        excerpt = ("... [showing the most relevant tail of the error lines] ...\n"
                   + excerpt[-(max_chars - 200):])

    header = (f"[Log '{getattr(file_obj, 'name', 'file')}' — {len(lines):,} lines, "
              f"{hits} error/warning line(s) found; showing those with context]\n\n")
    return header + excerpt


# Build the user's message. The attachment is ALWAYS processed, but its extracted
# text goes only to the assistant — the chat bubble shows a clean attachment chip,
# not the raw dump.
process_parts = []   # full text sent to the assistant
display_parts = []   # what the user sees in their bubble
if composer is not None:
    if typed:
        process_parts.append(typed)
        display_parts.append(typed)
    if attachment is not None:
        name = (attachment.name or "").lower()
        is_image = name.endswith((".png", ".jpg", ".jpeg"))
        if is_image:
            if _client is None:
                st.error("Reading screenshots needs an API key. Please paste the text.")
            else:
                with st.spinner("Reading the screenshot…"):
                    media = "image/png" if name.endswith(".png") else "image/jpeg"
                    try:
                        ocr = extract_error_text_from_image(attachment.getvalue(), media)
                    except Exception as e:
                        ocr = ""
                        st.error(f"Couldn't read the screenshot: {e}")
                if ocr and ocr.strip():
                    process_parts.append(f"[From the attached screenshot]\n{ocr.strip()}")
                    display_parts.append(f"📎 {attachment.name}")
                else:
                    st.warning("I couldn't extract text from that screenshot. Try a "
                               "clearer image or paste the text.")
        else:
            with st.spinner(f"Reading {attachment.name}…"):
                log_text = _read_log_text(attachment)
            if log_text.strip():
                process_parts.append(f"[From the attached log file '{attachment.name}']\n"
                                     f"{log_text.strip()}")
                display_parts.append(f"📎 {attachment.name}")
            else:
                st.warning("That file appears to be empty or unreadable.")

user_msg = "\n\n".join(process_parts) if process_parts else None
display_msg = "\n".join(display_parts) if display_parts else None
has_attachment = attachment is not None and len(process_parts) > (1 if typed else 0)

if user_msg:
    turn_id = uuid.uuid4().hex[:12]
    # Store both: full content for the assistant, clean display for the bubble.
    st.session_state.history.append({"role": "user", "content": user_msg,
                                     "display": display_msg})
    user_bubble(display_msg)
    assistant_open()
    t0 = time.time()
    # Reset the per-turn token accumulator before generating the reply.
    st.session_state["_turn_usage"] = {"input": 0, "output": 0, "cache_read": 0,
                                       "cache_write": 0, "cost": 0.0}
    with st.spinner(f"{ASSISTANT_NAME} is looking into it…"):
        try:
            # Pass prior turns (everything before the message just added) so the
            # assistant has short-term memory.
            prior = st.session_state.history[:-1]
            blocks, meta = generate_reply(user_msg.strip(),
                                          has_attachment=has_attachment,
                                          history=prior)
        except Exception as e:
            # Turn any backend error (e.g. Anthropic 529 Overloaded after retries,
            # network issues) into a friendly message instead of a raw traceback.
            name = type(e).__name__.lower()
            if "overloaded" in name or "529" in str(e):
                friendly = ("The AI service is briefly overloaded right now. "
                            "Please try again in a moment — this is usually temporary.")
            elif "ratelimit" in name or "429" in str(e):
                friendly = ("We've hit a rate limit for the moment. "
                            "Please wait a few seconds and try again.")
            elif "authentication" in name or "401" in str(e) or "api_key" in name:
                friendly = ("The AI service rejected the request — the API key may be "
                            "missing or invalid. Please check the configuration.")
            else:
                friendly = ("Something went wrong reaching the AI service. "
                            "Please try again in a moment.")
            blocks = [{"kind": "escalate", "title": "", "badge": "", "body": friendly}]
            meta = {"route": "error", "num_sources": 0}
    latency = time.time() - t0
    render_blocks(blocks)

    answer_chars = sum(len(b.get("body", "")) for b in blocks)
    u = st.session_state.get("_turn_usage", {})

    # Log the interaction for later analysis.
    log_metric({
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "turn_id": turn_id,
        "question_preview": (user_msg[:120].replace("\n", " ")),
        "route": meta["route"],
        "num_sources": meta["num_sources"],
        "latency_seconds": f"{latency:.2f}",
        "answer_chars": answer_chars,
        "input_tokens": u.get("input", 0),
        "output_tokens": u.get("output", 0),
        "cache_read_tokens": u.get("cache_read", 0),
        "cache_write_tokens": u.get("cache_write", 0),
        "est_cost_usd": f"{u.get('cost', 0.0):.6f}",
        "feedback": "",
    })

    st.session_state.history.append({"role": "assistant", "blocks": blocks,
                                     "turn_id": turn_id, "latency": latency,
                                     "meta": meta, "cost": u.get("cost", 0.0)})
    _save_current()  # persist so it appears in the history panel
    st.rerun()

# --------------------------------------------------------------------------- #
# Sidebar: Knowledge base — add documents from inside the app.
# Drop a PDF/MD/TXT in and it's saved + indexed on the spot (no manual steps).
# --------------------------------------------------------------------------- #
with st.sidebar:
    # New chat + conversation history (like polished assistants).
    if st.button("✏️  New Chat", use_container_width=True, key="new_chat_btn"):
        start_new_chat()
        st.rerun()

    st.markdown("#### Conversations")
    convs = list_conversations()
    if convs:
        for c in convs:
            is_current = c["id"] == st.session_state.current_conv_id
            label = ("● " if is_current else "") + c["title"]
            row = st.columns([5, 1])
            if row[0].button(label, key=f"conv_{c['id']}",
                             use_container_width=True,
                             type="primary" if is_current else "secondary"):
                switch_to_conversation(c["id"])
                st.rerun()
            if row[1].button("🗑", key=f"del_{c['id']}"):
                delete_conversation(c["id"])
                if is_current:
                    start_new_chat()
                st.rerun()
    else:
        st.caption("No past conversations yet.")

    st.markdown("---")

    # --------------------------------------------------------------------- #
    # Knowledge base — add documents from inside the app.
    # --------------------------------------------------------------------- #
    st.markdown("### 📚 Knowledge base")
    st.caption("Add documentation the assistant can answer from. Files are "
               "saved locally and indexed immediately — nothing is sent anywhere "
               "except answer generation.")

    if _rag_available:
        kb_file = st.file_uploader("Add a document (PDF, MD, TXT)",
                                   type=["pdf", "md", "txt"],
                                   key="kb_uploader")
        if kb_file is not None:
            # Avoid re-processing the same file on every rerun.
            sig = f"{kb_file.name}:{kb_file.size}"
            if st.session_state.get("_last_kb_sig") != sig:
                with st.spinner(f"Indexing '{kb_file.name}'…"):
                    result = rag.add_uploaded_document(kb_file.name, kb_file.getvalue())
                st.session_state["_last_kb_sig"] = sig
                if result.get("ok"):
                    st.success(f"Added '{result['saved_as']}' — "
                               f"{result['chunks']} chunks indexed.")
                else:
                    st.error(result.get("error", "Upload failed."))

        # Add from public URLs (e.g. product documentation pages) — one per line.
        st.markdown("**Add from URLs**")
        urls_val = st.text_area("Public doc URLs (one per line)",
                                placeholder="https://docs.example.com/page-1\n"
                                            "https://docs.example.com/page-2",
                                label_visibility="collapsed",
                                height=90,
                                key="kb_urls")
        if st.button("Fetch & index", key="kb_url_btn"):
            urls = [u for u in (urls_val or "").splitlines() if u.strip()]
            if urls:
                with st.spinner(f"Fetching {len(urls)} page(s) and indexing…"):
                    result = rag.add_urls_bulk(urls)
                if result.get("saved"):
                    st.success(f"Added {len(result['saved'])} page(s) — "
                               f"{result.get('chunks', 0)} chunks indexed.")
                for u, why in result.get("failed", []):
                    st.warning(f"Skipped: {u[:50]} — {why}")
            else:
                st.warning("Enter at least one URL.")
        st.caption("Fetched once and indexed locally (not looked up live), so answers "
                   "stay fast. Re-fetch to refresh if a page changes.")

        st.markdown("**Indexed documents**")
        docs_now = rag.list_documents()
        if docs_now:
            for d in docs_now:
                st.markdown(f"- `{d}`")
        else:
            st.caption("No documents yet. Upload one above to get started.")

        st.caption("⚠️ Redact secrets, customer names, IPs, and hostnames before "
                   "adding — the assistant may repeat anything in these docs. Only add "
                   "public pages that are OK to share with customers.")

        # ------------------------------------------------------------------- #
        # Jira: pull resolved-ticket SOLUTIONS as knowledge, with masking.
        # ------------------------------------------------------------------- #
        st.markdown("---")
        with st.expander("🎫 Import from Jira (resolved tickets)"):
            st.caption("Pull how past issues were solved. Credentials, IPs, emails "
                       "and customer names are auto-masked — but you review each "
                       "one before anything is indexed.")

            import jira_sync as jira

            jira_url = st.text_input("Jira base URL",
                                     placeholder="https://jira.example.com",
                                     key="jira_url")
            jira_token = st.text_input(
                "Personal Access Token", type="password", key="jira_token",
                help="Self-hosted Jira: Profile → Personal Access Tokens. Not "
                     "stored; kept only for this session.")
            verify_tls = st.checkbox("Verify TLS certificate", value=True,
                                     key="jira_tls")

            col_p, col_l = st.columns(2)
            project = col_p.text_input("Project key", placeholder="PROJ",
                                       key="jira_proj")
            account = col_l.text_input("Account (optional)",
                                       placeholder="ACME_PROD",
                                       key="jira_account")
            account_field = st.text_input(
                "Account field name in JQL", value="Account",
                key="jira_acc_field",
                help="Account is often a custom field. If 'Account' doesn't work, "
                     "try the exact field name, or 'cf[12345]' with its custom-field "
                     "id. Find the id via the field's admin page or a working JQL.")
            resolved_only = st.checkbox(
                "Only resolved/done tickets", value=True, key="jira_resolved",
                help="On: only closed tickets (where solutions live). Off: all "
                     "tickets.")
            max_n = st.slider("Max tickets to fetch", 10, 100, 50, key="jira_max")

            st.text_input(
                "Extra keywords to mask (comma-separated)",
                placeholder="internalBackendApiName, internalToolName, projectCodeword",
                key="jira_keywords",
                help="Custom terms that should never appear in the knowledge base — "
                     "internal identifiers, backend/API names, project codewords. "
                     "Matched anywhere, case-insensitive.")
            st.checkbox(
                "Also pull text from ticket attachments (PDF + screenshots)",
                value=False, key="jira_attach",
                help="Extracts text from PDF and image attachments (OCR), then "
                     "masks it like everything else. Raw files are never indexed — "
                     "only their cleaned, reviewed text. Slower (OCR per image).")

            if st.button("🔌 Connect & search", key="jira_search_btn"):
                if not jira_url or not jira_token:
                    st.warning("Enter the Jira URL and token.")
                else:
                    try:
                        client = jira.JiraClient(jira_url, jira_token, verify_tls)
                        who = client.test_connection()
                        jql = jira.build_jql(project=project, account=account,
                                             resolved_only=resolved_only,
                                             account_field=account_field)
                        found = client.search(jql, max_results=max_n)
                        st.session_state["_jira_issues"] = found
                        st.session_state["_jira_creds"] = (jira_url, jira_token,
                                                           verify_tls)
                        st.success(f"Connected as {who['name']}. Found {len(found)} "
                                   f"ticket(s).")
                        st.caption(f"JQL used: `{jql}`")
                        if not found:
                            st.info("No tickets matched. Check the project key / "
                                    "account, or the Account field name (it may be a "
                                    "custom field — see the help above).")
                    except Exception as e:
                        st.error(f"Jira search failed: {e}")
                        st.caption("If it's a JQL error about the Account field, the "
                                   "field name is probably different — try the exact "
                                   "name or cf[id].")

            found_issues = st.session_state.get("_jira_issues", [])
            if found_issues:
                options = {f"{i['key']} — {i['summary'][:50]}": i["key"]
                           for i in found_issues}
                chosen = st.multiselect("Select tickets to import",
                                        list(options.keys()), key="jira_pick")
                st.checkbox("Second-pass audit with local model (recommended)",
                            value=True, key="jira_audit",
                            help="After regex masking, the local Granite model "
                                 "re-reads each ticket and removes any customer/"
                                 "person identifier that survived. Slower but safer.")
                if st.button("Preview with masking", key="jira_prev_btn") and chosen:
                    creds = st.session_state.get("_jira_creds")
                    client = jira.JiraClient(*creds)
                    acct = st.session_state.get("jira_account", "")
                    run_audit = st.session_state.get("jira_audit", True)
                    pull_attach = st.session_state.get("jira_attach", False)
                    # Parse comma-separated extra keywords.
                    kw_raw = st.session_state.get("jira_keywords", "")
                    extra_kw = [w.strip() for w in kw_raw.split(",") if w.strip()]
                    import redaction_audit as auditor
                    previews = []
                    with st.spinner("Fetching, masking, and auditing…"):
                        for label_str in chosen:
                            key = options[label_str]
                            try:
                                issue = client.get_issue(key)
                                # Optionally pull + extract text from attachments.
                                att_texts = []
                                if pull_attach:
                                    for att in issue.get("attachments", []):
                                        try:
                                            data = client.download_attachment(
                                                att["url"])
                                            txt = jira.attachment_to_text(
                                                att, data,
                                                ocr_fn=extract_error_text_from_image)
                                            if txt:
                                                att_texts.append(
                                                    (att["filename"], txt))
                                        except Exception:
                                            pass
                                k = jira.issue_to_knowledge(
                                    issue, account=acct, extra_keywords=extra_kw,
                                    attachment_texts=att_texts)
                                if run_audit:
                                    a = auditor.audit_masking(k["redacted"])
                                    k["audited"] = a["ok"]
                                    k["audit_changed"] = a.get("changed", False)
                                    k["audit_error"] = a.get("error", "")
                                    if a["ok"]:
                                        k["redacted"] = a["text"]
                                else:
                                    k["audited"] = False
                                    k["audit_changed"] = False
                                    k["audit_error"] = ""
                                k["n_attachments"] = len(att_texts)
                                previews.append(k)
                            except Exception as e:
                                st.warning(f"{key}: {e}")
                    st.session_state["_jira_previews"] = previews

            previews = st.session_state.get("_jira_previews", [])
            if previews:
                st.markdown("**Review before indexing**")
                for p in previews:
                    rep = p["report"]
                    # Customer code name (anonymous) for this ticket.
                    import customer_map as cmap
                    code = cmap.code_for(p.get("account", ""))
                    audit_badge = ""
                    if p.get("audited"):
                        audit_badge = ("· 🤖 audited"
                                       + (" (changed)" if p.get("audit_changed")
                                          else " (no change)"))
                    elif p.get("audit_error"):
                        audit_badge = "· ⚠️ audit failed"
                    att_badge = ""
                    if p.get("n_attachments"):
                        att_badge = f" · 📎 {p['n_attachments']} attachment(s)"
                    st.markdown(f"**{p['key']} — {p['title'][:60]}**  "
                                f"· {rep['total_masked']} masked item(s) "
                                f"· 🔒 {code} {audit_badge}{att_badge}")
                    st.text_area(f"Masked content ({p['key']})", p["redacted"],
                                 height=180, key=f"jira_red_{p['key']}")
                    st.caption(f"Masked by type: {rep['by_type']}")
                    if p.get("audit_error"):
                        st.caption(f"⚠️ Auditor didn't run: {p['audit_error']} — "
                                   "review this one extra carefully.")

                st.warning("Read each preview above. Confirm no customer credential "
                           "or name survived before indexing. The customer is stored "
                           "only as an anonymous code (🔒), never by real name.")
                confirm = st.checkbox("I reviewed these and they're safe to index",
                                      key="jira_confirm")
                if st.button("✅ Approve & index", key="jira_index_btn",
                             disabled=not confirm):
                    import customer_map as cmap
                    added, failed = 0, 0
                    for p in previews:
                        text = st.session_state.get(f"jira_red_{p['key']}",
                                                    p["redacted"])
                        # Anonymous filename: the project prefix (which reveals the
                        # customer) is replaced with the customer CODE. The real key
                        # never lands in the filename, Sources UI, or doc list.
                        acct = p.get("account", "")
                        fname = cmap.anonymous_filename(p["key"], acct)
                        try:
                            res = rag.add_uploaded_document(fname,
                                                            text.encode("utf-8"))
                            if res.get("ok"):
                                added += 1
                                cmap.record(fname, acct)
                            else:
                                failed += 1
                        except Exception:
                            failed += 1
                    st.session_state["_jira_previews"] = []
                    st.success(f"Indexed {added} ticket solution(s)." +
                               (f" {failed} failed." if failed else ""))
                    st.rerun()
    else:
        st.caption("RAG module not available. Install requirements to enable the "
                   "knowledge base.")

    # ----------------------------------------------------------------------- #
    # Admin: look up which anonymous customer CODE a solution belongs to.
    # The bot never sees this; it's an internal, name-free lookup.
    # ----------------------------------------------------------------------- #
    with st.expander("🔒 Customer code lookup (admin)"):
        st.caption("Solutions are mapped to anonymous customer codes — never real "
                   "names. Look up a code here to relate a new ticket to past ones. "
                   "Only you know which real customer a code is.")
        try:
            import customer_map as cmap
            codes = cmap.all_codes()
            if codes:
                st.markdown("**Customer codes in the knowledge base:**")
                for code, cnt in sorted(codes.items()):
                    st.markdown(f"🔒 **{code}** — {cnt} solution(s)")
                lookup = st.text_input("Look up solutions for a code",
                                       placeholder="CUST-A7F3", key="cmap_lookup")
                if lookup:
                    sols = cmap.solutions_for(lookup.strip().upper())
                    if sols:
                        st.markdown(f"Solutions for **{lookup.strip().upper()}**:")
                        for s in sols:
                            st.markdown(f"- `{s}`")
                    else:
                        st.caption("No solutions for that code.")
                st.divider()
                acc_test = st.text_input(
                    "What's the code for an account?",
                    placeholder="ACME_PROD", key="cmap_accodetest",
                    help="Type an account identifier to see its code — useful when a "
                         "new ticket arrives and you want to find past solutions.")
                if acc_test:
                    st.markdown(f"→ Code: 🔒 **{cmap.code_for(acc_test.strip())}**")
            else:
                st.caption("No customer mappings yet. Import tickets from Jira first.")
        except Exception as e:
            st.caption(f"Lookup unavailable: {e}")

    st.markdown("---")

    # Active model / provider indicator.
    _prov_label = "Claude (Anthropic API)" if PROVIDER == "claude" else f"Local · Ollama"
    _cost_note = ("per-token cost" if PROVIDER == "claude" else "free / self-hosted")
    _vision_line = ("" if VISION_MODEL == MODEL
                    else f'<b>Vision:</b> {VISION_MODEL}<br>')
    st.markdown(
        f'<div style="font-size:12px;color:var(--text-secondary);line-height:1.6;">'
        f'<b>Model:</b> {MODEL}<br>'
        f'{_vision_line}'
        f'<b>Provider:</b> {_prov_label}<br>'
        f'<span style="color:var(--text-tertiary);">{_cost_note}</span></div>',
        unsafe_allow_html=True)
    st.markdown("---")

    # Usage & cost summary (read from metrics.csv).
    st.markdown("#### 📊 Usage & cost")
    try:
        if os.path.exists(METRICS_FILE):
            with open(METRICS_FILE, newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            n = len(rows)
            total_cost = sum(float(r.get("est_cost_usd") or 0) for r in rows)
            total_in = sum(int(r.get("input_tokens") or 0) for r in rows)
            total_out = sum(int(r.get("output_tokens") or 0) for r in rows)
            ups = sum(1 for r in rows if r.get("feedback") == "up")
            downs = sum(1 for r in rows if r.get("feedback") == "down")
            avg_cost = (total_cost / n) if n else 0
            # Cost lines only make sense for a paid provider.
            cost_lines = ""
            if PROVIDER == "claude":
                cost_lines = (f'Est. total cost: <b>${total_cost:.4f}</b><br>'
                              f'Avg / question: <b>${avg_cost:.4f}</b><br>')
            st.markdown(
                f'<div style="font-size:12px;line-height:1.7;color:var(--text-secondary);">'
                f'Questions: <b>{n}</b><br>'
                f'{cost_lines}'
                f'Tokens in/out: {total_in:,} / {total_out:,}<br>'
                f'Feedback: 👍 {ups} · 👎 {downs}</div>',
                unsafe_allow_html=True)
            if PROVIDER == "claude":
                st.caption(f"Priced at ${PRICE_INPUT_PER_MTOK:.2f}/M in, "
                           f"${PRICE_OUTPUT_PER_MTOK:.2f}/M out. Estimate only.")
            else:
                st.caption("Local model — no API cost.")
        else:
            st.caption("No usage logged yet.")
    except Exception:
        st.caption("Usage summary unavailable.")

    st.markdown("---")
    st.markdown("""
    <div style="color:var(--text-tertiary);font-size:11px;line-height:1.6;">
      Known errors → catalog<br>
      Other questions → docs (RAG)<br>
      Unknown / destructive → human
    </div>
    """, unsafe_allow_html=True)

