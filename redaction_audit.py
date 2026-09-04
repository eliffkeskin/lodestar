"""
Second-pass masking: an LLM auditor.

Regex masking (in jira_sync.py) catches structured secrets and known names, but
misses context-dependent identifiers — a customer name embedded mid-sentence, an
environment detail that reveals who it is, a person's name without a greeting. This
module asks the local model (same provider as the app — Granite by default) to read
the already-masked text and remove anything customer-identifying that survived.

Defense in depth: regex first (fast, deterministic), LLM second (context-aware).
The LLM only ever sees text that regex already cleaned, and it runs locally, so no
data leaves the machine.
"""
import os

# Reuse the app's provider settings so this uses the same local Granite model.
PROVIDER = os.getenv("LODESTAR_PROVIDER", "ollama").lower()
MODEL = os.getenv("LODESTAR_MODEL", "granite4.1:8b")
OLLAMA_URL = os.getenv("LODESTAR_OLLAMA_URL", "http://localhost:11434")

_AUDIT_SYSTEM = (
    "You are a strict data-privacy masker for a bank's support knowledge base. "
    "You receive text describing a technical problem and its solution. Your ONLY job "
    "is to remove anything that could identify a specific CUSTOMER or PERSON, while "
    "keeping the technical solution fully intact.\n\n"
    "Remove / replace:\n"
    "- Company or bank names (any customer) → [CUSTOMER]\n"
    "- Person names → [USER]\n"
    "- Hostnames, environment names, project codes that identify a customer → "
    "[ENV]\n"
    "- Anything that survived earlier masking and still points to who this is\n\n"
    "KEEP intact:\n"
    "- The technical problem and its solution\n"
    "- Product names (e.g. Keycloak, Kubernetes, Helm), config keys, file "
    "names, commands, error messages, version numbers\n\n"
    "Output ONLY the cleaned text, same structure, nothing else. Do not add "
    "commentary. If nothing needs changing, return the text unchanged.")


def _ollama(text: str) -> str:
    import requests
    r = requests.post(
        f"{OLLAMA_URL}/api/chat",
        json={"model": MODEL, "stream": False,
              "messages": [{"role": "system", "content": _AUDIT_SYSTEM},
                           {"role": "user", "content": text}],
              "options": {"num_predict": 2000, "temperature": 0}},
        timeout=180)
    r.raise_for_status()
    return (r.json().get("message") or {}).get("content", "").strip()


def _claude(text: str) -> str:
    import anthropic
    client = anthropic.Anthropic()
    msg = client.messages.create(
        model=MODEL, max_tokens=2000, system=_AUDIT_SYSTEM,
        messages=[{"role": "user", "content": text}])
    return "".join(b.text for b in msg.content if b.type == "text").strip()


def audit_masking(text: str) -> dict:
    """
    Run the LLM auditor over already-regex-masked text. Returns:
      {ok, text, changed}  — cleaned text and whether the auditor changed anything.
    On any failure, returns the input unchanged with ok=False so the caller can warn
    the reviewer (and the human review step still applies).
    """
    if not text.strip():
        return {"ok": True, "text": text, "changed": False}
    try:
        cleaned = _ollama(text) if PROVIDER == "ollama" else _claude(text)
        if not cleaned:
            return {"ok": False, "text": text, "changed": False,
                    "error": "empty response"}
        return {"ok": True, "text": cleaned, "changed": cleaned.strip() != text.strip()}
    except Exception as e:
        return {"ok": False, "text": text, "changed": False, "error": str(e)}
