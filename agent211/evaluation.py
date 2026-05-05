from __future__ import annotations

import json
from pathlib import Path

from .agent import Agent211
from .models import BenchmarkCase, EvaluationRecord, EvaluationSummary, SearchRequest


def load_benchmark_cases(path: Path | str) -> list[BenchmarkCase]:
    cases = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            cases.append(
                BenchmarkCase(
                    query_id=str(row.get("query_id") or row.get("id") or len(cases)),
                    user_query=str(row["user_query"]),
                    primary_gt_resource_ids=tuple(row.get("primary_gt_resource_ids") or ()),
                    acceptable_gt_resource_ids=tuple(row.get("acceptable_gt_resource_ids") or ()),
                    hard_negative_resource_ids=tuple(row.get("hard_negative_resource_ids") or ()),
                    constraints=row.get("constraints") or {},
                    difficulty=str(row.get("difficulty", "")),
                )
            )
    return cases


def evaluate_cases(
    agent: Agent211,
    cases: list[BenchmarkCase],
    limit: int = 10,
    use_constraints: bool = False,
) -> EvaluationSummary:
    records = []
    for case in cases:
        request = _request_from_constraints(case, limit) if use_constraints else None
        response = agent.ask(case.user_query, request=request, limit=limit)
        retrieved_ids = tuple(result.resource.resource_id for result in response.results)
        gt_ids = tuple(sorted(case.all_gt_resource_ids))
        records.append(
            EvaluationRecord(
                query_id=case.query_id,
                user_query=case.user_query,
                retrieved_resource_ids=retrieved_ids,
                gt_resource_ids=gt_ids,
                hit_at_1=_hit_at_k(retrieved_ids, gt_ids, 1),
                hit_at_3=_hit_at_k(retrieved_ids, gt_ids, 3),
                hit_at_5=_hit_at_k(retrieved_ids, gt_ids, 5),
                reciprocal_rank=_reciprocal_rank(retrieved_ids, gt_ids),
                tool_calls=response.tool_calls,
            )
        )

    count = len(records)
    return EvaluationSummary(
        case_count=count,
        recall_at_1=_mean(record.hit_at_1 for record in records),
        recall_at_3=_mean(record.hit_at_3 for record in records),
        recall_at_5=_mean(record.hit_at_5 for record in records),
        mrr=_mean(record.reciprocal_rank for record in records),
        records=tuple(records),
    )


def _request_from_constraints(case: BenchmarkCase, limit: int) -> SearchRequest:
    constraints = case.constraints
    return SearchRequest(
        text_query=case.user_query,
        counties=_as_tuple(constraints.get("counties") or constraints.get("county")),
        cities=_as_tuple(constraints.get("cities") or constraints.get("city")),
        states=_as_tuple(constraints.get("states") or constraints.get("state")),
        zipcodes=_as_tuple(constraints.get("zipcodes") or constraints.get("zipcode")),
        benchmark_categories=_as_tuple(constraints.get("benchmark_categories")),
        taxonomy_categories=_as_tuple(constraints.get("taxonomy_categories")),
        subcategories=_as_tuple(constraints.get("subcategories")),
        curated_subcategories=_as_tuple(constraints.get("curated_subcategories")),
        limit=limit,
    )


def _hit_at_k(retrieved_ids: tuple[str, ...], gt_ids: tuple[str, ...], k: int) -> bool:
    return bool(set(retrieved_ids[:k]) & set(gt_ids))


def _reciprocal_rank(retrieved_ids: tuple[str, ...], gt_ids: tuple[str, ...]) -> float:
    gt = set(gt_ids)
    for idx, resource_id in enumerate(retrieved_ids, start=1):
        if resource_id in gt:
            return 1.0 / idx
    return 0.0


def _mean(values) -> float:
    values = list(values)
    if not values:
        return 0.0
    return sum(float(value) for value in values) / len(values)


def _as_tuple(value: object) -> tuple[str, ...]:
    if isinstance(value, str):
        return (value,)
    if isinstance(value, list):
        return tuple(item for item in value if isinstance(item, str))
    return ()
