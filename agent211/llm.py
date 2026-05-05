from __future__ import annotations

import json
import os
from pathlib import Path

from .models import SearchResult


def load_dotenv(path: Path = Path(".env")) -> None:
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def make_openai_client(provider: str = "openrouter"):
    from openai import OpenAI

    load_dotenv()
    provider = provider.lower()
    if provider == "openai":
        return OpenAI(
            api_key=os.environ["OPENAI_API_KEY"],
            base_url=os.getenv("OPENAI_BASE_URL") or None,
        )
    return OpenAI(
        api_key=os.environ["OPENROUTER_API_KEY"],
        base_url=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
        default_headers=_openrouter_headers(),
    )


def rerank_with_llm(
    query: str,
    results: list[SearchResult],
    client,
    model: str,
    limit: int = 10,
) -> list[SearchResult]:
    if not results:
        return []
    candidates = []
    for result in results:
        r = result.resource
        candidates.append(
            {
                "resource_id": r.resource_id,
                "service_name": r.service_name,
                "agency_name": r.agency_name,
                "categories": r.benchmark_categories,
                "subcategories": r.curated_subcategories,
                "service_area": r.service_area,
                "city": r.city,
                "eligibility": r.eligibility,
                "application_process": r.application_process,
                "documents_required": r.documents_required,
                "retrieval_score": result.score,
            }
        )
    prompt = (
        "Rerank these 211 resource candidates for the user query. "
        "Return only JSON: {\"resource_ids\": [..]} with the best resource_ids first. "
        "Use only resource_ids from the candidates.\n\n"
        f"User query: {query}\n\n"
        f"Candidates:\n{json.dumps(candidates, ensure_ascii=False)}"
    )
    response = client.chat.completions.create(
        model=model,
        temperature=0,
        messages=[
            {"role": "system", "content": "You rerank candidates and return strict JSON only."},
            {"role": "user", "content": prompt},
        ],
        response_format={"type": "json_object"},
    )
    data = _json_object(response.choices[0].message.content or "{}")
    requested_order = [rid for rid in data.get("resource_ids", []) if isinstance(rid, str)]
    by_id = {result.resource.resource_id: result for result in results}
    reranked = [by_id[rid] for rid in requested_order if rid in by_id]
    seen = {result.resource.resource_id for result in reranked}
    reranked.extend(result for result in results if result.resource.resource_id not in seen)
    return reranked[:limit]


def _json_object(value: str) -> dict:
    try:
        data = json.loads(value)
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def _openrouter_headers() -> dict[str, str]:
    headers = {}
    if os.getenv("OPENROUTER_HTTP_REFERER"):
        headers["HTTP-Referer"] = os.environ["OPENROUTER_HTTP_REFERER"]
    if os.getenv("OPENROUTER_APP_TITLE"):
        headers["X-Title"] = os.environ["OPENROUTER_APP_TITLE"]
    return headers
