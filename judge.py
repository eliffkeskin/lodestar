
import json, sys
from dotenv import load_dotenv
load_dotenv()

from langfuse import get_client
from openai import OpenAI


import re
URL_RE = re.compile(r"https?://[^\s`'\"<>]+")

def url_grounding(answer: str, context: str):
    urls = [u.rstrip(".,;:!?)") for u in URL_RE.findall(answer)]
    if not urls:
        return None, []
    missing = [u for u in urls if u not in context]
    return 1 - len(missing) / len(urls), missing

langfuse = get_client()
ollama = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")

JUDGE_PROMPT = """You are a strict evaluator. Below is CONTEXT (documentation chunks
retrieved by a RAG system) and an ANSWER given to a user.

Check every factual claim in ANSWER - especially URLs, API endpoints, commands,
parameter names. A claim counts as supported ONLY if it appears in CONTEXT or
follows directly from it. Invented URLs/endpoints are unsupported.

CONTEXT:
{context}

ANSWER:
{answer}

Return ONLY JSON, nothing else:
{{"score": <float 0-1, fraction of claims supported>, "unsupported": ["claim", ...]}}"""

def judge_trace(trace_id: str):
    res = langfuse.api.observations.get_many(
        trace_id=trace_id, limit=100, fields="core,basic,io,usage",
    )
    obs = {o.name: o for o in res.data}

    answer  = obs["llm"].output 
    if not isinstance(answer, str):
        answer = json.dumps(answer, ensure_ascii=False)

    chunks  = obs["retrieve"].output                 
    if isinstance(chunks, str):
        chunks = json.loads(chunks)              

    texts = []
    for c in chunks:
        if isinstance(c, dict):
            texts.append(c.get("text") or c.get("page_content") or str(c))
        elif isinstance(c, (list, tuple)):       
            texts.append(str(c[0]))
        else:
            texts.append(str(c))
    context = "\n\n---\n\n".join(texts)
    u_score, missing = url_grounding(answer, context)
    if u_score is not None:
        langfuse.create_score(
            trace_id=trace_id,
            name="url_grounding",
            value=float(u_score),
            data_type="NUMERIC",
            comment="; ".join(missing)[:400] if missing else "all URLs present in context",
        )
    resp = ollama.chat.completions.create(
        model="granite4.1:8b",                            
        messages=[{"role": "user",
                   "content": JUDGE_PROMPT.format(context=context, answer=answer)}],
    )
    raw = resp.choices[0].message.content
    verdict = json.loads(raw[raw.find("{"): raw.rfind("}") + 1]) 
    langfuse.create_score(
        trace_id=trace_id,
        name="faithfulness",
        value=float(verdict["score"]),
        data_type="NUMERIC",
        comment="; ".join(verdict.get("unsupported", []))[:500],
    )
    langfuse.flush()
    print(f"{trace_id} -> {verdict['score']}  unsupported: {verdict.get('unsupported')}")

if __name__ == "__main__":
    judge_trace(sys.argv[1])