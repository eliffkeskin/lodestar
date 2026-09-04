"""
run_eval.py — Lodestar golden-set evaluation.

Usage (inside venv, .env loaded):
    python run_eval.py                # all 20 questions
    python run_eval.py --ids q07 q20  # subset
    python run_eval.py --no-langfuse  # print only, no score upload

Produces: eval_results.jsonl + a per-category summary on stdout.
"""
import argparse
import json
import logging
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

# --no-langfuse must take effect BEFORE langfuse/app are imported (the langfuse
# client reads LANGFUSE_TRACING_ENABLED at construction time), so peek at argv here.
if "--no-langfuse" in sys.argv:
    os.environ["LANGFUSE_TRACING_ENABLED"] = "false"

# app.py is a Streamlit script; importing it outside `streamlit run` works but logs
# a "missing ScriptRunContext" warning per st.* call. Silence those.
os.environ.setdefault("STREAMLIT_LOGGER_LEVEL", "error")
logging.getLogger("streamlit").setLevel(logging.ERROR)

from dotenv import load_dotenv
from langfuse import get_client, observe, propagate_attributes
from openai import OpenAI

load_dotenv()
langfuse = get_client()

# ---------------------------------------------------------------------------
# 1. ADAPTER — the only part that depends on your code.
#    Must return (answer_text, context_text).
#    context_text = the reranked chunks the LLM actually saw, joined by "\n\n---\n\n".
# ---------------------------------------------------------------------------
from app import generate_reply  # noqa: E402
import rag  # noqa: E402


def ask(question: str) -> tuple[str, str]:
    """
    app.generate_reply(text, has_attachment=False, history=None) returns
    (blocks, meta): blocks is a list of dicts {kind, title, badge, body, sources?}
    where sources is [(source_path, score), ...]; meta has {"route", "num_sources"}.
    It does not hand back the chunk texts, so we re-run the same retrieval
    (rag.retrieve, k=12, as generate_reply does) and keep the chunks of the documents
    that were actually cited — that is the context the LLM saw.
    """
    blocks, meta = generate_reply(question, has_attachment=False, history=None)
    answer = "\n\n".join(b.get("body", "") for b in blocks if b.get("body"))

    cited = {os.path.basename(str(src))
             for b in blocks for src, _score in (b.get("sources") or [])}
    texts = []
    if meta.get("route") == "docs" and cited:
        try:
            for c in rag.retrieve(question, k=12):
                if os.path.basename(str(c.source)) in cited:
                    texts.append(f"[Source: {os.path.basename(str(c.source))}]\n{c.text}")
        except Exception:
            pass
    return answer, "\n\n---\n\n".join(texts)


# ---------------------------------------------------------------------------
# 2. JUDGE (self-contained; same Ollama OpenAI-compatible endpoint as judge.py)
# ---------------------------------------------------------------------------
JUDGE_MODEL = os.getenv("LODESTAR_JUDGE_MODEL", "llama3.1:8b")  # use a different model than the chatbot generator if you can
ollama = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")

JUDGE_SYSTEM = Path("judge_prompt.md").read_text().split("```")[1].strip()  # system block from judge_prompt.md

URL_RE = re.compile(r"https?://[^\s`'\"<>]+")


def url_grounding(answer: str, context: str):
    urls = [u.rstrip(".,;:!?)") for u in URL_RE.findall(answer)]
    if not urls:
        return None, []
    missing = [u for u in urls if u not in context]
    return 1 - len(missing) / len(urls), missing


def judge(question, context, answer, expected, expect_refusal) -> dict:
    user = (
        f"QUESTION:\n{question}\n\nEXPECT_REFUSAL: {expect_refusal}\n\n"
        f"CONTEXT:\n{context}\n\nANSWER:\n{answer}\n\nEXPECTED_ANSWER:\n{expected}"
    )
    resp = ollama.chat.completions.create(
        model=JUDGE_MODEL,
        temperature=0,
        messages=[{"role": "system", "content": JUDGE_SYSTEM}, {"role": "user", "content": user}],
        response_format={"type": "json_object"},
    )
    raw = resp.choices[0].message.content.strip()
    raw = re.sub(r"^```(json)?|```$", "", raw, flags=re.M).strip()
    try:
        out = json.loads(raw)
        out["faithfulness"] = float(out["faithfulness"])
        out["correctness"] = float(out["correctness"])
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        # Unparseable verdict: no score rather than a fake 0.0. The runner skips
        # None rows in averages and flags them with "??".
        out = {"faithfulness": None, "correctness": None, "unsupported_claims": [],
               "reason": "judge_parse_error"}
    out.setdefault("unsupported_claims", [])
    out.setdefault("reason", "")
    return out


# ---------------------------------------------------------------------------
# 3. RUNNER
# ---------------------------------------------------------------------------
@observe(name="lodestar-eval")
def run_one(item: dict, use_langfuse: bool) -> dict:
    # langfuse 4.x has no update_current_trace(); trace-level name/metadata/tags are
    # set with propagate_attributes() (applies to this span and all children).
    with propagate_attributes(
        trace_name=f"eval-{item['id']}",
        metadata={"golden_id": item["id"], "category": item["category"],
                  "expect_refusal": str(item["expect_refusal"])},
        tags=["golden-eval"],
    ):
        trace_id = langfuse.get_current_trace_id()
        answer, context = ask(item["question"])
    verdict = judge(item["question"], context, answer, item["expected_answer"], item["expect_refusal"])
    u_score, missing = url_grounding(answer, context)

    if use_langfuse and verdict["faithfulness"] is not None:
        langfuse.create_score(trace_id=trace_id, name="faithfulness", value=verdict["faithfulness"], data_type="NUMERIC", comment=verdict["reason"][:400])
        langfuse.create_score(trace_id=trace_id, name="correctness", value=verdict["correctness"], data_type="NUMERIC", comment=verdict["reason"][:400])
        langfuse.create_score(trace_id=trace_id, name="hallucination", value=verdict["faithfulness"] < 0.5, data_type="BOOLEAN", comment="; ".join(verdict["unsupported_claims"])[:400])
        if u_score is not None:
            langfuse.create_score(trace_id=trace_id, name="url_grounding", value=float(u_score), data_type="NUMERIC", comment="; ".join(missing)[:400] or "all URLs present in context")

    return {
        "id": item["id"], "category": item["category"], "expect_refusal": item["expect_refusal"],
        "trace_id": trace_id, "answer": answer,
        "faithfulness": verdict["faithfulness"], "correctness": verdict["correctness"],
        "url_grounding": u_score, "unsupported_claims": verdict["unsupported_claims"], "reason": verdict["reason"],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--golden", default="golden.jsonl")
    ap.add_argument("--ids", nargs="*")
    ap.add_argument("--no-langfuse", action="store_true")
    args = ap.parse_args()

    items = [json.loads(l) for l in open(args.golden) if l.strip()]
    if args.ids:
        items = [i for i in items if i["id"] in args.ids]

    results = []
    with open("eval_results.jsonl", "w") as out:
        for item in items:
            r = run_one(item, use_langfuse=not args.no_langfuse)
            results.append(r)
            out.write(json.dumps(r, ensure_ascii=False) + "\n")
            if r["faithfulness"] is None:
                flag = "??"   # judge_parse_error: no score, excluded from averages
            elif r["faithfulness"] < 0.5:
                flag = "!!"
            else:
                flag = "  "
            print(f"{flag} {r['id']:<4} {r['category']:<12} F={fmt(r['faithfulness'])} C={fmt(r['correctness'])} URL={r['url_grounding']}  {r['reason'][:70]}")
            sys.stdout.flush()

    langfuse.flush()

    by_cat = defaultdict(list)
    for r in results:
        by_cat[r["category"]].append(r)
    print("\n== summary ==")
    print(f"{'category':<12} {'n':>2} {'faith':>6} {'corr':>6} {'halluc':>7} {'??':>3}")
    for cat, rs in sorted(by_cat.items()):
        f, c, h, u = summarize(rs)
        print(f"{cat:<12} {len(rs):>2} {fmt(f):>6} {fmt(c):>6} {h:>7} {u:>3}")
    f_all, c_all, h_all, u_all = summarize(results)
    print(f"{'ALL':<12} {len(results):>2} {fmt(f_all):>6} {fmt(c_all):>6} {h_all:>7} {u_all:>3}")


def fmt(v):
    """Format a score; None (judge_parse_error) prints as '??'."""
    return "??" if v is None else f"{v:.2f}"


def summarize(rs):
    """
    Averages over scored rows only. Rows whose judge verdict could not be parsed
    (faithfulness/correctness None) are excluded from the averages and the
    hallucination count, and reported separately as `unscored`.
    Returns (faith_avg, corr_avg, hallucinations, unscored); averages are None
    when no row was scored.
    """
    scored = [r for r in rs if r["faithfulness"] is not None]
    unscored = len(rs) - len(scored)
    if not scored:
        return None, None, 0, unscored
    f = sum(r["faithfulness"] for r in scored) / len(scored)
    c = sum(r["correctness"] for r in scored) / len(scored)
    h = sum(r["faithfulness"] < 0.5 for r in scored)
    return f, c, h, unscored


if __name__ == "__main__":
    main()