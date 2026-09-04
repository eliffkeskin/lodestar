"""
Jira integration — pull resolved tickets and turn their SOLUTIONS into knowledge,
while aggressively redacting anything that looks like a customer credential.

Design decisions (banking context):
  - Self-hosted Jira (Server / Data Center): auth via a Personal Access Token
    sent as a Bearer header. (Cloud uses email + API token with Basic auth; this
    module targets self-hosted per the deployment.)
  - We fetch RESOLVED tickets in a chosen project/label, take the summary +
    description + comments, and redact secrets BEFORE anything is shown or saved.
  - Nothing is indexed automatically. This module only fetches + redacts and hands
    the result back for human review. The Streamlit layer does the approve→index.

Redaction is best-effort and conservative: when in doubt, redact. It is NOT a
guarantee — the human review step is the real safety control.
"""
import os
import re
import requests


# --- Redaction -------------------------------------------------------------
# Patterns that commonly carry secrets or customer-identifying data. Each match
# is replaced with a placeholder so the *shape* of the solution survives but the
# secret does not.
_REDACTION_RULES = [
    # key: value style secrets (password: hunter2, api_key = abc...). Includes
    # Turkish credential words (şifre/parola/gizli/anahtar) since tickets are often
    # written in Turkish. The value must follow a real : or = and must NOT be an
    # angle-bracket placeholder like <your-password> (those are template examples,
    # not real secrets) — otherwise a placeholder in a curl example gets mangled.
    (re.compile(r'(?i)\b(password|passwd|pwd|secret|token|api[_-]?key|apikey|'
                r'client[_-]?secret|private[_-]?key|access[_-]?key|bearer|'
                r'şifre|sifre|şifresi|sifresi|parola|parolası|parolasi|gizli|'
                r'anahtar|kullanıcı\s*adı|kullanici\s*adi)\b'
                r'\s*[:=]\s*(?!<[^>]*>)([^\s<]\S*)'), r'\1: [MASKED]'),
    # Authorization headers
    (re.compile(r'(?i)authorization\s*:\s*\S+'), 'Authorization: [MASKED]'),
    # Bearer / Basic tokens inline
    (re.compile(r'(?i)\b(bearer|basic)\s+[A-Za-z0-9._\-+/=]{8,}'), r'\1 [MASKED]'),
    # JWT-looking strings (three base64 segments)
    (re.compile(r'\beyJ[A-Za-z0-9._\-]{10,}\b'), '[MASKED-JWT]'),
    # PEM private key blocks
    (re.compile(r'-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----',
                re.S), '[MASKED-PRIVATE-KEY]'),
    # Connection strings with embedded creds (proto://user:pass@host)
    (re.compile(r'([a-zA-Z][a-zA-Z0-9+.\-]*://)[^:\s/]+:[^@\s/]+@'), r'\1[MASKED]@'),
    # IPv4 addresses
    (re.compile(r'\b(?:\d{1,3}\.){3}\d{1,3}\b'), '[IP]'),
    # Email addresses
    (re.compile(r'\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b'), '[EMAIL]'),
    # Long hex/base64 blobs (>=24 chars) that look like keys
    (re.compile(r'\b[A-Fa-f0-9]{24,}\b'), '[MASKED-HEX]'),
    (re.compile(r'\b[A-Za-z0-9+/]{32,}={0,2}\b'), '[MASKED-B64]'),
    # Jira user mentions: [~username] or [~accountid:xxx]
    (re.compile(r'\[~[^\]]+\]'), '[USER]'),
    # Greeting lines with a name: "Hi Joel," / "Hello John Doe," / "Merhaba Ali,"
    (re.compile(r'(?im)^\s*(hi|hello|hey|dear|merhaba|selam|sayın)\b[^\n,]*,'),
     r'\1 [USER],'),
    # @mentions: @joel.cordonnier / @joel
    (re.compile(r'(?<!\w)@[A-Za-z][A-Za-z0-9._\-]{2,}'), '@[USER]'),
    # Microsoft Teams links and thread references (internal, identifying).
    (re.compile(r'https?://teams\.microsoft\.com/\S+'), '[TEAMS-LINK]'),
    (re.compile(r'(?i)\bhttps?://\S*teams\S+'), '[TEAMS-LINK]'),
    # Teams thread/meeting/channel mentions in prose ("Teams thread", "Teams
    # channel", "in the Teams call") — reference removed, not the whole sentence.
    (re.compile(r'(?i)\b(teams)\s+(thread|channel|meeting|call|chat|conversation|'
                r'message|link|post)\b'), '[TEAMS-REF]'),
]

# Customer names to always scrub. Case-insensitive, whole word. Configure them in
# the environment (comma-separated) so no real customer name is hard-coded here:
#   LODESTAR_CUSTOMER_NAMES="Acme Bank,Acme,ACME_PROD"
# Include account codes / project identifiers that reveal the customer.
_CUSTOMER_NAMES = [n.strip() for n in
                   os.getenv("LODESTAR_CUSTOMER_NAMES", "").split(",") if n.strip()]


def mask(text: str, extra_names=None, extra_keywords=None) -> str:
    """
    Redact secrets, PII, and known customer names from free text.

    extra_names     : additional customer/person names → [CUSTOMER]
    extra_keywords  : user-supplied terms to always strip (e.g. internal identifiers
                      like "internalBackendApiName") → [MASKED-TERM]. Matched
                      case-insensitively; substring match (not whole-word) so
                      compound identifiers are caught wherever they appear.
    """
    if not text:
        return ""
    out = text
    for pattern, repl in _REDACTION_RULES:
        out = pattern.sub(repl, out)
    names = list(_CUSTOMER_NAMES) + list(extra_names or [])
    for name in names:
        if name.strip():
            out = re.sub(rf'(?i)\b{re.escape(name.strip())}\b', '[CUSTOMER]', out)
    for kw in (extra_keywords or []):
        if kw.strip():
            out = re.sub(re.escape(kw.strip()), '[MASKED-TERM]', out,
                         flags=re.IGNORECASE)
    return out


def mask_report(original: str, redacted: str) -> dict:
    """Count how much was redacted, so the reviewer can sanity-check."""
    markers = {
        "SECRET": ["[MASKED]", "[MASKED-JWT]", "[MASKED-PRIVATE-KEY]",
                   "[MASKED-HEX]", "[MASKED-B64]"],
        "IP": ["[IP]"], "EMAIL": ["[EMAIL]"], "CUSTOMER": ["[CUSTOMER]"],
        "USER": ["[USER]"], "TEAMS": ["[TEAMS-LINK]", "[TEAMS-REF]"],
        "TERM": ["[MASKED-TERM]"],
    }
    counts = {label: sum(redacted.count(m) for m in ms)
              for label, ms in markers.items()}
    return {"total_masked": sum(counts.values()), "by_type": counts}


# --- Jira client (self-hosted) --------------------------------------------
class JiraClient:
    def __init__(self, base_url: str, token: str, verify_tls: bool = True):
        self.base = base_url.rstrip("/")
        self.session = requests.Session()
        # Self-hosted Jira uses Bearer PAT auth.
        self.session.headers.update({
            "Authorization": f"Bearer {token}",
            "Accept": "application/json",
        })
        self.verify = verify_tls

    def test_connection(self) -> dict:
        """Verify credentials by fetching the current user."""
        r = self.session.get(f"{self.base}/rest/api/2/myself",
                             verify=self.verify, timeout=20)
        r.raise_for_status()
        me = r.json()
        return {"ok": True, "name": me.get("displayName") or me.get("name")}

    def search(self, jql: str, max_results: int = 25) -> list:
        """Run a JQL search, return a list of issue keys + summaries."""
        r = self.session.get(
            f"{self.base}/rest/api/2/search",
            params={"jql": jql, "maxResults": max_results,
                    "fields": "summary,status,resolution"},
            verify=self.verify, timeout=30)
        r.raise_for_status()
        issues = r.json().get("issues", [])
        return [{"key": i["key"],
                 "summary": i["fields"].get("summary", ""),
                 "status": (i["fields"].get("status") or {}).get("name", ""),
                 "resolution": (i["fields"].get("resolution") or {}).get("name", "")}
                for i in issues]

    def get_issue(self, key: str) -> dict:
        """Fetch one issue with description, comments, and attachment metadata."""
        r = self.session.get(
            f"{self.base}/rest/api/2/issue/{key}",
            params={"fields": "summary,description,status,resolution,comment,labels,"
                              "attachment"},
            verify=self.verify, timeout=30)
        r.raise_for_status()
        f = r.json().get("fields", {})
        comments = [(c.get("author", {}).get("displayName", "?"), c.get("body", ""))
                    for c in (f.get("comment", {}) or {}).get("comments", [])]
        attachments = [{"filename": a.get("filename", ""),
                        "mime": a.get("mimeType", ""),
                        "url": a.get("content", ""),
                        "size": a.get("size", 0)}
                       for a in (f.get("attachment", []) or [])]
        return {
            "key": key,
            "summary": f.get("summary", ""),
            "description": f.get("description", "") or "",
            "status": (f.get("status") or {}).get("name", ""),
            "resolution": (f.get("resolution") or {}).get("name", ""),
            "labels": f.get("labels", []),
            "comments": comments,
            "attachments": attachments,
        }

    def download_attachment(self, url: str) -> bytes:
        """Download an attachment's raw bytes by its content URL."""
        r = self.session.get(url, verify=self.verify, timeout=60)
        r.raise_for_status()
        return r.content


def extract_pdf_text(data: bytes) -> str:
    """Extract text from a PDF's bytes. Returns '' if it can't be read."""
    try:
        import io
        from pypdf import PdfReader
        reader = PdfReader(io.BytesIO(data))
        parts = []
        for page in reader.pages:
            t = page.extract_text() or ""
            if t.strip():
                parts.append(t)
        return "\n".join(parts).strip()
    except Exception:
        return ""


def attachment_to_text(att: dict, data: bytes, ocr_fn=None) -> str:
    """
    Turn an attachment into text so it can be redacted like everything else.
      - PDF   → extract embedded text (pypdf)
      - image → OCR via ocr_fn(image_bytes, media_type) if provided
      - other → skipped (returns '')
    ocr_fn is injected by the caller (app.py passes its vision-model OCR function),
    so this module doesn't depend on the LLM layer.
    """
    name = (att.get("filename") or "").lower()
    mime = (att.get("mime") or "").lower()
    if name.endswith(".pdf") or "pdf" in mime:
        return extract_pdf_text(data)
    if (name.endswith((".png", ".jpg", ".jpeg", ".gif", ".webp"))
            or mime.startswith("image/")):
        if ocr_fn is None:
            return ""
        media = "image/png" if name.endswith(".png") else "image/jpeg"
        try:
            return (ocr_fn(data, media) or "").strip()
        except Exception:
            return ""
    return ""


def build_jql(project: str = "", account: str = "",
              resolved_only: bool = True, account_field: str = "Account") -> str:
    """
    Compose a JQL query. `account` filters by the (custom) Account field — the
    field name may differ per Jira, so account_field is configurable. If your Jira
    exposes Account only by custom-field id, pass account_field='cf[XXXXX]'.
    """
    conds = []
    if project:
        conds.append(f'project = "{project}"')
    if account:
        # Quote the field name unless it's a cf[...] reference.
        field = account_field if account_field.startswith("cf[") else f'"{account_field}"'
        conds.append(f'{field} = "{account}"')
    if resolved_only:
        conds.append('statusCategory = Done')
    where = " AND ".join(conds)
    order = "ORDER BY updated DESC"
    return f"{where} {order}".strip() if where else order


def issue_to_knowledge(issue: dict, extra_names=None, account: str = "",
                       extra_keywords=None, attachment_texts=None) -> dict:
    """
    Turn a fetched issue into a redacted, knowledge-shaped markdown document plus
    a redaction report. Focuses on the PROBLEM and SOLUTION, not on who reported it.

    `account`          : customer/account id (e.g. "ACME_PROD"). NOT written
                         into content — returned separately for anonymous code
                         mapping. The real id never enters the knowledge base.
    `extra_keywords`   : user-supplied terms to strip (e.g. "internalBackendApiName").
    `attachment_texts` : list of (filename, extracted_text) from PDF/image
                         attachments. Text is included then redacted like the rest —
                         raw files are never indexed, only their cleaned text.
    """
    raw_parts = [f"# {issue['summary']}", ""]
    if issue["description"]:
        raw_parts += ["## Problem", issue["description"], ""]
    if issue["comments"]:
        raw_parts += ["## Resolution discussion"]
        for author, body in issue["comments"]:
            if body.strip():
                raw_parts.append(body.strip())
                raw_parts.append("")
    if attachment_texts:
        raw_parts += ["## From attachments"]
        for fname, text in attachment_texts:
            if text and text.strip():
                raw_parts.append(f"[Attachment: {fname}]")
                raw_parts.append(text.strip())
                raw_parts.append("")
    raw = "\n".join(raw_parts)

    redacted = mask(raw, extra_names=extra_names, extra_keywords=extra_keywords)
    report = mask_report(raw, redacted)
    return {
        "key": issue["key"],
        "title": issue["summary"],
        "raw": raw,
        "redacted": redacted,
        "report": report,
        "account": account,   # kept out of content; used only for code-name mapping
    }
