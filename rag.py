"""
Documentation RAG layer
===================

Indexes Markdown, PDF and text docs into a local Chroma vector store and retrieves the most
relevant chunks for a customer question. This is what makes the bot "know" the product
without fine-tuning: the knowledge lives in the vector store (easy to update by
re-indexing), not baked into model weights.

Two responsibilities:
  1. build_index()  — read docs/*.md, split into chunks, embed, store in Chroma.
  2. retrieve()     — given a question, return the top-k most relevant chunks
                      (with their source file) to ground the LLM's answer.

Embeddings use Anthropic's Voyage-style approach? No — Anthropic has no public
embeddings endpoint, so we use a local sentence-transformers model by default.
This keeps everything on your machine: docs never leave, no embedding API calls.

Run indexing once (or whenever docs change):
    python rag.py --index

Then the Streamlit app queries it automatically.
"""

import argparse
import glob
import os
from dataclasses import dataclass
from typing import List
from langfuse import observe

# Local, on-machine embeddings — no external API, docs stay local.
# Install: pip install chromadb sentence-transformers
import chromadb
from chromadb.utils import embedding_functions

DOCS_DIR = os.getenv("LODESTAR_DOCS_DIR", "docs")
CHROMA_DIR = os.getenv("LODESTAR_CHROMA_DIR", "chroma_store")
COLLECTION = "lodestar_docs"
# Stronger, multilingual embedding model. Better for technical docs and for the
# English+German mix in technical documentation than the tiny all-MiniLM-L6-v2.
# ~470MB on first download, then fully local/offline.
EMBED_MODEL = os.getenv("LODESTAR_EMBED_MODEL",
                        "paraphrase-multilingual-mpnet-base-v2")


@dataclass
class RetrievedChunk:
    text: str
    source: str
    score: float


def _client():
    return chromadb.PersistentClient(path=CHROMA_DIR)


# Cache the embedding function so the model is loaded ONCE per process, not on
# every retrieve/index call. Reloading a ~470MB model repeatedly is the main
# source of slowness; this keeps it resident in memory.
_EMBEDDER_CACHE = None


def _embedder():
    global _EMBEDDER_CACHE
    if _EMBEDDER_CACHE is None:
        _EMBEDDER_CACHE = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=EMBED_MODEL)
    return _EMBEDDER_CACHE


def _chunk_markdown(text: str, max_chars: int = 900, overlap: int = 150) -> List[str]:
    """
    Split text into smaller, semantically tighter chunks for sharper retrieval.
    Strategy: first split on Markdown headings; then, for any section still larger
    than max_chars, split on paragraph (blank-line) boundaries; only fall back to a
    character window for very long paragraphs. Smaller chunks keep a single topic
    (e.g. "reverse proxy") from being diluted by unrelated text, which raises
    similarity scores.
    """
    import re
    out: List[str] = []

    def add_by_paragraphs(block: str):
        block = block.strip()
        if not block:
            return
        if len(block) <= max_chars:
            out.append(block)
            return
        # Split on blank lines (paragraphs), accumulate up to max_chars.
        paras = re.split(r"\n\s*\n", block)
        buf = ""
        for p in paras:
            p = p.strip()
            if not p:
                continue
            if len(buf) + len(p) + 2 <= max_chars:
                buf = (buf + "\n\n" + p) if buf else p
            else:
                if buf:
                    out.append(buf)
                if len(p) <= max_chars:
                    buf = p
                else:
                    # Very long paragraph: sliding character window.
                    start = 0
                    while start < len(p):
                        out.append(p[start:start + max_chars])
                        start += max_chars - overlap
                    buf = ""
        if buf:
            out.append(buf)

    sections = re.split(r"(?=^#{1,6}\s)", text, flags=re.MULTILINE)
    for sec in sections:
        add_by_paragraphs(sec)
    return out


def _read_pdf(path: str) -> str:
    """
    Extract text from a PDF. Uses pypdf (pure-Python, no system deps).
    Note: scanned/image PDFs won't yield text without OCR — for those, convert
    to text first. Most docs are text PDFs, so this is fine.
    """
    try:
        from pypdf import PdfReader
    except Exception:
        raise SystemExit(
            "PDF support needs pypdf. Install it:  pip install pypdf")
    reader = PdfReader(path)
    parts = []
    for page in reader.pages:
        txt = page.extract_text() or ""
        if txt.strip():
            parts.append(txt)
    return "\n\n".join(parts)


def _read_document(path: str) -> str:
    """Read a document to plain text based on its extension."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        return _read_pdf(path)
    # .md / .txt and anything else: read as UTF-8 text
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def build_index() -> int:
    """Read docs/*.{md,pdf,txt}, chunk, embed, and (re)store in Chroma."""
    patterns = ["*.md", "*.pdf", "*.txt"]
    paths = []
    for pat in patterns:
        paths += glob.glob(os.path.join(DOCS_DIR, "**", pat), recursive=True)
    paths = sorted(set(paths))
    if not paths:
        raise SystemExit(
            f"No documents found under '{DOCS_DIR}/'. "
            f"Put your redacted docs there (.md, .pdf, or .txt) and retry.")

    client = _client()
    # Fresh collection each time so re-indexing reflects doc changes cleanly.
    try:
        client.delete_collection(COLLECTION)
    except Exception:
        pass
    coll = client.create_collection(COLLECTION, embedding_function=_embedder())

    ids, docs, metas = [], [], []
    skipped = []
    for path in paths:
        try:
            content = _read_document(path)
        except SystemExit:
            raise
        except Exception as e:
            skipped.append((os.path.relpath(path, DOCS_DIR), str(e)))
            continue
        if not content.strip():
            skipped.append((os.path.relpath(path, DOCS_DIR),
                            "no extractable text (scanned PDF?)"))
            continue
        for i, chunk in enumerate(_chunk_markdown(content)):
            ids.append(f"{os.path.relpath(path, DOCS_DIR)}::{i}")
            docs.append(chunk)
            metas.append({"source": os.path.relpath(path, DOCS_DIR)})

    if skipped:
        print("Skipped (no text / unreadable):")
        for name, why in skipped:
            print(f"  - {name}: {why}")

    # Chroma embeds via the collection's embedding function on add.
    coll.add(ids=ids, documents=docs, metadatas=metas)
    return len(ids)


import re as _re


def _keyword_hits(question: str, k: int) -> List["RetrievedChunk"]:
    """
    Literal keyword search over all chunks. Catches exact terms (e.g. 'app.conf',
    a variable name, a flag) that semantic search can miss when the phrasing
    differs. Returns chunks scored by how many distinct query terms they contain.
    """
    client = _client()
    try:
        coll = client.get_collection(COLLECTION, embedding_function=_embedder())
    except Exception:
        return []
    # Pull all chunks (fine for the doc sizes here).
    try:
        data = coll.get(include=["documents", "metadatas"])
    except Exception:
        return []
    docs = data.get("documents", []) or []
    metas = data.get("metadatas", []) or []

    # Distinctive terms from the question: keep words with letters/dots/underscores,
    # length >= 3, so 'app.conf', 'PUBLIC_DOMAIN', 'hostname' survive; 'the' drops.
    raw = _re.findall(r"[A-Za-z0-9_.]+", question.lower())
    terms = [t for t in raw if len(t) >= 3 and t not in _STOP]
    if not terms:
        return []

    scored = []
    for text, meta in zip(docs, metas):
        low = text.lower()
        hits = sum(1 for t in set(terms) if t in low)
        if hits:
            scored.append((hits, text, meta))
    scored.sort(key=lambda x: x[0], reverse=True)
    out = []
    for hits, text, meta in scored[:k]:
        # Normalize a keyword score into a comparable 0..1-ish range.
        out.append(RetrievedChunk(text=text,
                                   source=(meta or {}).get("source", "?"),
                                   score=min(1.0, 0.4 + 0.15 * hits)))
    return out


_STOP = {"the", "and", "for", "are", "you", "how", "what", "where", "should",
         "write", "does", "can", "with", "this", "that", "from", "have", "was",
         "will", "into", "your", "there", "which", "when", "why", "who", "information"}

@observe(name="retrieve")
def retrieve(question: str, k: int = 4) -> List["RetrievedChunk"]:
    """
    Hybrid retrieval: combine semantic (embedding) search with literal keyword
    search, then merge and de-duplicate. Semantic finds paraphrased matches;
    keyword guarantees chunks containing exact terms (like 'app.conf') are not
    missed. This fixes the case where the right file/variable is in the docs but
    phrased differently from the question.
    """
    client = _client()
    try:
        coll = client.get_collection(COLLECTION, embedding_function=_embedder())
    except Exception:
        return []  # index not built yet

    # Pull MORE candidates than needed (k*4), so the reranker has a rich pool to
    # pick from. Retrieval casts a wide net; reranking picks the sharpest matches.
    pool = max(k * 4, 20)

    # 1) Semantic results
    sem: List[RetrievedChunk] = []
    try:
        res = coll.query(query_texts=[question], n_results=pool)
        docs = res.get("documents", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
        dists = res.get("distances", [[]])[0]
        for text, meta, dist in zip(docs, metas, dists):
            sem.append(RetrievedChunk(text=text,
                                      source=(meta or {}).get("source", "?"),
                                      score=1.0 - float(dist)))
    except Exception:
        pass

    # 2) Keyword results
    kw = _keyword_hits(question, pool)

    # 3) Merge unique candidates
    best = {}
    for c in sem + kw:
        key = c.text
        if key not in best or c.score > best[key].score:
            best[key] = c
    candidates = list(best.values())

    # 4) Rerank with a cross-encoder if available (much better relevance than the
    #    bi-encoder retrieval scores). Falls back gracefully to the merged order.
    reranked = _rerank(question, candidates, k)
    if reranked is not None:
        return reranked

    # Fallback: no reranker available — use merged retrieval order.
    candidates.sort(key=lambda c: c.score, reverse=True)
    return candidates[:k]


# Cache the reranker model once per process (like the embedder).
_RERANKER_CACHE = None
_RERANKER_TRIED = False


def _reranker():
    global _RERANKER_CACHE, _RERANKER_TRIED
    if _RERANKER_TRIED:
        return _RERANKER_CACHE
    _RERANKER_TRIED = True
    try:
        from sentence_transformers import CrossEncoder
        # Small, fast, runs locally/offline after first download (~80MB).
        _RERANKER_CACHE = CrossEncoder(
            os.getenv("LODESTAR_RERANK_MODEL",
                      "cross-encoder/ms-marco-MiniLM-L-6-v2"))
    except Exception:
        _RERANKER_CACHE = None
    return _RERANKER_CACHE

@observe(name="rerank")
def _rerank(question: str, candidates: List["RetrievedChunk"], k: int):
    """
    Reorder candidates by cross-encoder relevance to the question. Returns the top-k,
    or None if the reranker isn't available (so the caller can fall back).
    """
    if not candidates:
        return []
    model = _reranker()
    if model is None:
        return None
    try:
        pairs = [(question, c.text) for c in candidates]
        scores = model.predict(pairs)
        # Cross-encoder returns unbounded logits (e.g. +4.3, -3.0). Map them to a
        # 0..1 relevance with a sigmoid so scores are comparable and displayable.
        import math
        for c, s in zip(candidates, scores):
            c.score = float(1.0 / (1.0 + math.exp(-float(s))))
        candidates.sort(key=lambda c: c.score, reverse=True)
        return candidates[:k]
    except Exception:
        return None


def index_exists() -> bool:
    try:
        _client().get_collection(COLLECTION, embedding_function=_embedder())
        return True
    except Exception:
        return False


def add_uploaded_document(filename: str, data: bytes) -> dict:
    """
    Save an uploaded document into docs/ and rebuild the index so it's immediately
    searchable. Returns a small status dict for the UI.

    This is what lets a user add knowledge from inside the app — drop a file in,
    and it's indexed on the spot. No manual 'Build Index' step, no touching the
    docs/ folder by hand. Under the hood it's still RAG indexing (not training),
    so everything stays current, grounded, and local.
    """
    os.makedirs(DOCS_DIR, exist_ok=True)

    # Basic safety on the filename (no path traversal); keep the extension.
    base = os.path.basename(filename).replace("/", "_").replace("\\", "_")
    ext = os.path.splitext(base)[1].lower()
    if ext not in (".pdf", ".md", ".txt"):
        return {"ok": False, "error": f"Unsupported type '{ext}'. Use PDF, MD, or TXT."}

    dest = os.path.join(DOCS_DIR, base)
    # If a file with the same name exists, don't silently overwrite — suffix it.
    if os.path.exists(dest):
        stem, e = os.path.splitext(base)
        i = 2
        while os.path.exists(os.path.join(DOCS_DIR, f"{stem}-{i}{e}")):
            i += 1
        dest = os.path.join(DOCS_DIR, f"{stem}-{i}{e}")

    with open(dest, "wb") as f:
        f.write(data)

    # Verify the new file yields text (catch scanned PDFs early).
    try:
        text = _read_document(dest)
    except Exception as e:
        os.remove(dest)
        return {"ok": False, "error": f"Could not read the file: {e}"}
    if not text.strip():
        os.remove(dest)
        return {"ok": False,
                "error": "No extractable text (a scanned/image PDF?). Convert it to "
                         "text first, then upload again."}

    # Rebuild the whole index so the new doc is live immediately.
    try:
        n = build_index()
    except Exception as e:
        return {"ok": False, "error": f"Saved the file but indexing failed: {e}"}

    return {"ok": True, "saved_as": os.path.basename(dest), "chunks": n}


def _slugify_url(url: str) -> str:
    """Turn a URL into a safe .md filename."""
    from urllib.parse import urlparse
    p = urlparse(url)
    slug = (p.netloc + p.path).strip("/")
    slug = _re.sub(r"[^A-Za-z0-9._-]+", "-", slug).strip("-") or "page"
    return slug[:120] + ".md"

import json
import re
import requests
from urllib.parse import urljoin

def _find_openapi_spec_url(html: str, page_url: str):
    """Redoc/Swagger/Redocusaurus sayfalarindaki OpenAPI spec linkini bulur."""
    pats = [
        r'href=["\']([^"\']*redocusaurus[^"\']*\.(?:ya?ml|json))["\']',
        r'href=["\']([^"\']*(?:openapi|swagger)[^"\']*\.(?:ya?ml|json))["\']',
        r'spec-url=["\']([^"\']+)["\']',
        r'"specUrl"\s*:\s*"([^"]+)"',
        r'["\']([^"\']*(?:openapi|swagger)[^"\']*\.(?:ya?ml|json))["\']',
    ]
    for p in pats:
        m = re.search(p, html, re.IGNORECASE)
        if m:
            return urljoin(page_url, m.group(1))
    return None

def _fetch_openapi_spec(spec_url: str):
    r = requests.get(spec_url, headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
    r.raise_for_status()
    # Decode bytes as UTF-8 ourselves: servers that omit charset make requests
    # fall back to Latin-1, which injects control chars that break YAML parsing.
    body = r.content.decode("utf-8", errors="replace")
    try:
        return json.loads(body)
    except Exception:
        import yaml                      # pip install pyyaml
        return yaml.safe_load(body)

def _resolve_ref(spec: dict, node):
    """Follow #/components/... $ref pointers (with a loop guard)."""
    hops = 0
    while isinstance(node, dict) and "$ref" in node and hops < 10:
        ref = node["$ref"]
        if not isinstance(ref, str) or not ref.startswith("#/"):
            break
        cur = spec
        for part in ref[2:].split("/"):
            if not isinstance(cur, dict) or part not in cur:
                return {}
            cur = cur[part]
        node = cur
        hops += 1
    return node if isinstance(node, dict) else {}


def _schema_example(spec: dict, schema, depth: int = 0):
    """Build a skeleton example value from an OpenAPI schema, like Redoc's
    'Request samples' box: honor example/default/enum, recurse into objects
    and arrays, cap depth against reference cycles."""
    if depth > 4 or not isinstance(schema, dict):
        return None
    schema = _resolve_ref(spec, schema)
    if "example" in schema:
        return schema["example"]
    if "default" in schema:
        return schema["default"]
    if schema.get("enum"):
        return schema["enum"][0]
    if isinstance(schema.get("allOf"), list):
        merged = {}
        for part in schema["allOf"]:
            ex = _schema_example(spec, part, depth + 1)
            if isinstance(ex, dict):
                merged.update(ex)
        return merged or None
    for key in ("oneOf", "anyOf"):
        if isinstance(schema.get(key), list) and schema[key]:
            return _schema_example(spec, schema[key][0], depth + 1)
    t = schema.get("type")
    if t == "object" or "properties" in schema:
        return {name: _schema_example(spec, sub, depth + 1)
                for name, sub in (schema.get("properties") or {}).items()}
    if t == "array":
        item = _schema_example(spec, schema.get("items") or {}, depth + 1)
        return [item] if item is not None else []
    if t == "integer" or t == "number":
        return 0
    if t == "boolean":
        return False
    if t == "string":
        return "2019-08-24T14:15:22Z" if schema.get("format") == "date-time" else "string"
    return None


def _spec_to_markdown(spec: dict, source_url: str = "") -> str:
    """OpenAPI spec'ini chunker-dostu markdown'a cevirir: her endpoint bir ## basligi."""
    info = spec.get("info") or {}
    lines = [f"# {info.get('title', 'API Reference')} ({info.get('version', '')})", ""]
    if info.get("description"):
        lines += [str(info["description"]), ""]
    servers = [s.get("url", "") for s in (spec.get("servers") or []) if isinstance(s, dict)]
    if not servers and spec.get("host"):          # Swagger 2.0 fallback
        scheme = (spec.get("schemes") or ["https"])[0]
        servers = [f"{scheme}://{spec.get('host')}{spec.get('basePath', '')}"]
    # Resolve relative/empty server URLs against where the spec is served from --
    # this is exactly what Redoc does to build the full request URL.
    base = (servers[0].strip() if servers else "")
    if source_url:
        base = urljoin(source_url, base or "/")
    base = base.rstrip("/")
    if base:
        lines += [f"Base URL: {base}", ""]
    for path, ops in (spec.get("paths") or {}).items():
        if not isinstance(ops, dict):
            continue
        for method, op in ops.items():
            if method.lower() not in ("get", "post", "put", "patch", "delete") or not isinstance(op, dict):
                continue
            lines.append(f"## {method.upper()} {path}")
            if base:
                lines.append(f"Full URL: {method.upper()} {base}{path}")
            if op.get("summary"):
                lines.append(f"**{op['summary']}**")
            if op.get("description"):
                lines.append(str(op["description"]))
            prm = [f"- {p.get('name')} ({p.get('in')}{', required' if p.get('required') else ''}): {p.get('description', '')}"
                   for p in (op.get("parameters") or []) if isinstance(p, dict)]
            if prm:
                lines.append("Parameters:\n" + "\n".join(prm))
            rb = _resolve_ref(spec, op.get("requestBody") or {})
            content = rb.get("content") or {}
            media = content.get("application/json") or (list(content.values())[0] if content else {})
            schema = media.get("schema") if isinstance(media, dict) else None
            if schema:
                example = _schema_example(spec, schema)
                if example is not None:
                    dumped = json.dumps(example, indent=2, ensure_ascii=False)
                    if len(dumped) > 1500:
                        dumped = json.dumps(example, ensure_ascii=False)[:1500] + " ..."
                    lines.append("Request body example:\n```json\n" + dumped + "\n```")
                else:
                    props = (_resolve_ref(spec, schema).get("properties") or {})
                    if props:
                        lines.append("Request body fields: " + ", ".join(props.keys()))
            resps = op.get("responses") or {}
            if isinstance(resps, dict) and resps:
                parts = []
                for code, r in list(resps.items())[:8]:
                    desc = ""
                    if isinstance(r, dict):
                        desc = str(_resolve_ref(spec, r).get("description", "")).strip().splitlines()
                        desc = desc[0][:70] if desc else ""
                    parts.append(f"{code} ({desc})" if desc else str(code))
                lines.append("Responses: " + "; ".join(parts))
            lines.append("")
    return "\n".join(lines)



def _looks_fragmented(text: str) -> bool:
    """
    True if extracted text looks mangled -- the signature of a JS-rendered page
    (Swagger/Redoc API refs) flattened to noise: mostly ultra-short 1-2 word
    lines plus stray single-char lines (exploded JSON/punctuation). Skips very
    short docs where the ratios aren't meaningful.
    """
    # Ignore fenced code blocks (```...```): JSON/code examples legitimately
    # contain many tiny lines and must not trip the fragmentation check.
    text = re.sub(r"```.*?```", "", text, flags=re.DOTALL)
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if len(lines) < 25:
        return False
    tiny = sum(1 for ln in lines if len(ln) <= 2)
    shortish = sum(1 for ln in lines if len(ln.split()) < 4)
    return (tiny / len(lines)) > 0.15 or (shortish / len(lines)) > 0.75


def _html_to_text(html: str) -> str:
    """
    Extract readable text from an HTML page. Prefers BeautifulSoup if available
    (better quality), falls back to a simple tag-stripper otherwise. Drops
    script/style/nav/footer noise so the indexed text is mostly real content.
    """
    try:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "noscript", "nav", "footer", "header",
                         "svg", "form", "aside"]):
            tag.decompose()
        # Preserve heading levels as markdown (#/##/...) so _chunk_markdown can
        # split on sections and every chunk keeps its section title.
        for level in range(1, 7):
            for h in soup.find_all(f"h{level}"):
                htext = h.get_text(" ", strip=True)
                h.replace_with(soup.new_string("\n" + "#" * level + " " + htext + "\n"))
        # Drop Docusaurus "On this page" table-of-contents columns.
        for tag in soup.select('[class*="tableOfContents"]'):
            tag.decompose()
        # Wrap <pre> code blocks in markdown fences: correct representation for
        # the LLM, and the fragmentation gate already ignores fenced blocks --
        # code-heavy SDK pages must not be mistaken for JS-soup.
        for pre in soup.find_all("pre"):
            code = pre.get_text("\n").strip("\n")
            pre.replace_with(soup.new_string("\n```\n" + code + "\n```\n"))
        # Tables: flatten each row to ONE "cell | cell | ..." line. Otherwise
        # every cell lands on its own line, which reads like JS-soup and gets
        # (correctly) rejected by the fragmentation gate -- but tabular pages
        # like error-code references are legitimate, valuable content.
        for table in soup.find_all("table"):
            trs = table.find_all("tr")
            headers = []
            if trs and trs[0].find("th"):
                headers = [c.get_text(" ", strip=True) for c in trs[0].find_all(["th", "td"])]
            rows = []
            for tr in (trs[1:] if headers else trs):
                cells = [c.get_text(" ", strip=True) for c in tr.find_all(["th", "td"])]
                if not any(cells):
                    continue
                if headers and len(cells) == len(headers):
                    # Self-describing rows survive chunking: every line carries
                    # its column names, so retrieval never loses the semantics.
                    rows.append(" | ".join(f"{h}: {v}" for h, v in zip(headers, cells) if v))
                else:
                    rows.append(" | ".join(v for v in cells if v))
            table.replace_with(soup.new_string("\n" + "\n".join(rows) + "\n"))
        # Prefer a <main>/<article> region if present.
        main = soup.find("main") or soup.find("article") or soup.body or soup
        text = main.get_text("\n")
    except Exception:
        # Minimal fallback: strip tags crudely.
        no_scripts = _re.sub(r"(?is)<(script|style).*?>.*?</\1>", " ", html)
        text = _re.sub(r"(?s)<[^>]+>", " ", no_scripts)
    for zw in ("\u200b", "\ufeff", "\u200e", "\u200f"):
        text = text.replace(zw, "")
    # Collapse excessive whitespace/blank lines.
    lines = [ln.strip() for ln in text.splitlines()]
    lines = [ln for ln in lines if ln]
    return "\n".join(lines)


def add_url_document(url: str) -> dict:
    """
    Fetch a public web page (e.g. a documentation site), convert it to text,
    save it into docs/, and rebuild the index so it's immediately searchable.

    This is a one-time fetch (not a live lookup on every question), which keeps
    answers fast and offline afterwards. Re-add the URL to refresh it if the page
    changes. Only use this for pages that are OK to repeat back to customers.
    """
    url = url.split("#", 1)[0].strip()

    if not url.lower().startswith(("http://", "https://")):
        return {"ok": False, "error": "Enter a full URL starting with http:// or https://"}

    try:
        import requests
        resp = requests.get(url, timeout=20,
                            headers={"User-Agent": "Lodestar/1.0"})
        resp.raise_for_status()
    except Exception as e:
        return {"ok": False, "error": f"Could not fetch the page: {e}"}

    ctype = resp.headers.get("Content-Type", "").lower()
    if "pdf" in ctype or url.lower().endswith(".pdf"):
        # Save as PDF and let the normal PDF path handle it.
        return add_uploaded_document(_slugify_url(url).replace(".md", ".pdf"),
                                     resp.content)
    # API reference page? Prefer the raw OpenAPI spec over mangled HTML.
    page_html = resp.content.decode("utf-8", errors="replace")
    spec_url = _find_openapi_spec_url(page_html, url)
    if spec_url:
        try:
            text = _spec_to_markdown(_fetch_openapi_spec(spec_url), spec_url)
        except Exception as e:
            return {"ok": False,
                    "error": f"OpenAPI spec found ({spec_url}) but it could not be "
                             f"downloaded/parsed: {e}"}
        text = f"# Source: {url} (OpenAPI spec: {spec_url})\n\n{text}"
    else:
        text = _html_to_text(page_html)
        if len(text.strip()) < 100:
            return {"ok": False,
                    "error": "The page had very little extractable text (JavaScript-only "
                             "site?). Try a more specific documentation page."}
        if _looks_fragmented(text):
            return {"ok": False,
                    "error": "The page extracted to fragmented text (JavaScript-rendered "
                             "site, e.g. Swagger/Redoc) and no OpenAPI spec link was found "
                             "on it. Not indexed. Use the raw OpenAPI spec URL "
                             "(often .../openapi.json or .yaml) or a static docs page."}
        # Prepend a small source header so retrieval/citations know where it came from.
        text = f"# Source: {url}\n\n{text}"

    os.makedirs(DOCS_DIR, exist_ok=True)
    dest = os.path.join(DOCS_DIR, _slugify_url(url))
    with open(dest, "w", encoding="utf-8") as f:
        f.write(text)

    try:
        n = build_index()
    except Exception as e:
        return {"ok": False, "error": f"Saved the page but indexing failed: {e}"}

    return {"ok": True, "saved_as": os.path.basename(dest), "chunks": n,
            "chars": len(text)}


def add_urls_bulk(urls: list) -> dict:
    """
    Fetch and index several URLs in one go. Each page is fetched once, converted
    to text, and saved into docs/; the index is rebuilt ONCE at the end (not per
    page) so bulk adds stay fast. Returns per-URL results plus a final chunk count.
    """
    saved, failed = [], []
    for url in urls:
        url = url.split("#", 1)[0].strip()   # fragments never reach the server
        if not url:
            continue
        if not url.lower().startswith(("http://", "https://")):
            failed.append((url, "not a valid URL"))
            continue
        try:
            import requests
            resp = requests.get(url, timeout=20,
                                headers={"User-Agent": "Lodestar/1.0"})
            resp.raise_for_status()
        except Exception as e:
            failed.append((url, f"fetch error: {e}"))
            continue

        ctype = resp.headers.get("Content-Type", "").lower()
        os.makedirs(DOCS_DIR, exist_ok=True)
        try:
            if "pdf" in ctype or url.lower().endswith(".pdf"):
                dest = os.path.join(DOCS_DIR,
                                    _slugify_url(url).replace(".md", ".pdf"))
                with open(dest, "wb") as f:
                    f.write(resp.content)
                # Validate it yields text.
                if not _read_document(dest).strip():
                    os.remove(dest)
                    failed.append((url, "no text in PDF"))
                    continue
            else:
                page_html = resp.content.decode("utf-8", errors="replace")
                spec_url = _find_openapi_spec_url(page_html, url)
                if spec_url:
                    try:
                        text = _spec_to_markdown(_fetch_openapi_spec(spec_url), spec_url)
                        text = f"# Source: {url} (OpenAPI spec: {spec_url})\n\n{text}"
                    except Exception as e:
                        failed.append((url, f"OpenAPI spec found but failed: {e}"))
                        continue
                else:
                    text = _html_to_text(page_html)
                    if len(text.strip()) < 100:
                        failed.append((url, "little/no extractable text"))
                        continue
                    if _looks_fragmented(text):
                        failed.append((url, "fragmented extraction (JS-rendered page?)"))
                        continue
                    text = f"# Source: {url}\n\n{text}"
                dest = os.path.join(DOCS_DIR, _slugify_url(url))
                with open(dest, "w", encoding="utf-8") as f:
                    f.write(text)
            saved.append(os.path.basename(dest))
        except Exception as e:
            failed.append((url, f"save error: {e}"))

    # Rebuild the index once for the whole batch.
    chunks = 0
    if saved:
        try:
            chunks = build_index()
        except Exception as e:
            return {"ok": False, "saved": saved, "failed": failed,
                    "error": f"Saved pages but indexing failed: {e}"}

    return {"ok": True, "saved": saved, "failed": failed, "chunks": chunks}


def list_documents() -> list:
    """Return the list of document filenames currently in docs/."""
    if not os.path.isdir(DOCS_DIR):
        return []
    names = []
    for pat in ("*.md", "*.pdf", "*.txt"):
        names += [os.path.basename(p)
                  for p in glob.glob(os.path.join(DOCS_DIR, "**", pat), recursive=True)]
    return sorted(set(names))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", action="store_true", help="(Re)build the vector index")
    ap.add_argument("--ask", type=str, help="Test a retrieval query")
    args = ap.parse_args()

    if args.index:
        n = build_index()
        print(f"Indexed {n} chunks from '{DOCS_DIR}/' into '{CHROMA_DIR}/'.")
    if args.ask:
        for c in retrieve(args.ask):
            print(f"\n--- {c.source} (sim={c.score:.2f}) ---\n{c.text[:400]}")
    if not args.index and not args.ask:
        ap.print_help()