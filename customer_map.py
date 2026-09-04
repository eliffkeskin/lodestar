"""
Customer pseudonymization + solution↔customer mapping.

The knowledge base is fully anonymous — no real customer names anywhere. But we
still want to relate a new ticket to a customer's past solutions. We do that with
a CODE NAME, never the real name:

    real account id (e.g. "ACME_PROD")  --hash-->  "CUST-A7F3"

The mapping stored on disk is only:  solution_id -> CUST-A7F3
The REAL name is never written anywhere by this system. Only you know which real
customer a code maps to (or you don't — the system can't tell you).

Design choices:
  - Simple (fast) hash, but with a fixed local salt so someone can't just hash a
    guessed account id and match it. Change SALT below to rotate all code names.
  - The map DB is a plain JSON file, kept SEPARATE from the vector index. The
    answering bot never reads it — only the admin "lookup" tool does.
"""
import hashlib
import json
import os

# Change this to rotate every code name. Keep it out of shared docs/screenshots.
SALT = os.getenv("LODESTAR_CUSTOMER_SALT", "lodestar-local-salt-change-me")

MAP_FILE = os.getenv("LODESTAR_CUSTOMER_MAP", "customer_map.json")


def code_for(account_identifier: str) -> str:
    """
    Deterministic, one-way code name for a customer/account identifier.
    Same input → same code; the code can't be reversed to the identifier.
    """
    if not account_identifier:
        return "CUST-UNKNOWN"
    h = hashlib.sha256((SALT + "::" + account_identifier.strip().lower())
                       .encode("utf-8")).hexdigest()
    # Short, readable code: CUST- + 4 hex chars (65k space; fine for a support KB).
    return f"CUST-{h[:4].upper()}"


def _load_map() -> dict:
    if not os.path.exists(MAP_FILE):
        return {"solutions": {}}
    try:
        with open(MAP_FILE, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"solutions": {}}


def _save_map(data: dict):
    with open(MAP_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def record(solution_id: str, account_identifier: str) -> str:
    """
    Map a solution (indexed md) to a customer CODE name. Stores only the code,
    never the real identifier. Returns the code name.
    """
    code = code_for(account_identifier)
    data = _load_map()
    data["solutions"][solution_id] = code
    _save_map(data)
    return code


def customer_of(solution_id: str) -> str:
    """Admin lookup: which customer CODE a solution belongs to (never a real name)."""
    return _load_map()["solutions"].get(solution_id, "CUST-UNKNOWN")


def solutions_for(code: str) -> list:
    """Admin lookup: all solution ids belonging to a customer CODE."""
    data = _load_map()
    return [sid for sid, c in data["solutions"].items() if c == code]


def all_codes() -> dict:
    """Return {code: count} — how many solutions each customer code has."""
    data = _load_map()
    counts = {}
    for c in data["solutions"].values():
        counts[c] = counts.get(c, 0) + 1
    return counts


def anonymous_filename(ticket_key: str, account_identifier: str) -> str:
    """
    Build an anonymous filename for an indexed ticket solution.

    The ticket key (e.g. "PROJ-16") leaks the customer via its project prefix
    ("PROJ"). Replace that prefix with the customer CODE, keeping the ticket number
    for traceability:

        "PROJ-16", "ACME_PROD"  ->  "jira-CUST-A3F9-16.md"

    The number alone doesn't identify the customer; the code is one-way. The real
    project key never appears in the filename, the Sources UI, or the document list.
    """
    code = code_for(account_identifier)
    # Split "PROJ-16" -> number "16"; if no number, fall back to a short hash.
    num = ""
    if "-" in ticket_key:
        num = ticket_key.rsplit("-", 1)[-1]
    if not num.isdigit():
        num = hashlib.sha256(ticket_key.encode("utf-8")).hexdigest()[:4]
    return f"jira-{code}-{num}.md"
