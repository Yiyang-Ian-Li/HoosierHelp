from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Resource:
    resource_id: str
    service_name: str
    agency_name: str
    site_name: str
    benchmark_categories: tuple[str, ...]
    source_subcategories: tuple[str, ...]
    curated_subcategories: tuple[str, ...]
    service_area: tuple[str, ...]
    city: str
    state: str
    zipcode: str
    address_1: str
    phone: str
    website: str
    eligibility: str
    application_process: str
    fees: str
    documents_required: str
    search_text: str


@dataclass(frozen=True)
class SearchRequest:
    text_query: str = ""
    resource_ids: tuple[str, ...] = ()
    agency_ids: tuple[str, ...] = ()
    site_ids: tuple[str, ...] = ()
    service_names: tuple[str, ...] = ()
    agency_names: tuple[str, ...] = ()
    site_names: tuple[str, ...] = ()
    counties: tuple[str, ...] = ()
    cities: tuple[str, ...] = ()
    states: tuple[str, ...] = ()
    zipcodes: tuple[str, ...] = ()
    benchmark_categories: tuple[str, ...] = ()
    taxonomy_categories: tuple[str, ...] = ()
    subcategories: tuple[str, ...] = ()
    curated_subcategories: tuple[str, ...] = ()
    eligibility_keywords: tuple[str, ...] = ()
    application_keywords: tuple[str, ...] = ()
    document_keywords: tuple[str, ...] = ()
    fee_keywords: tuple[str, ...] = ()
    contact_required: bool = False
    address_required: bool = False
    limit: int = 10


@dataclass(frozen=True)
class SearchResult:
    resource: Resource
    score: float
    matched_filters: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class AgentResponse:
    query: str
    request: SearchRequest
    results: tuple[SearchResult, ...]
    answer: str
    tool_calls: tuple[dict, ...]


@dataclass(frozen=True)
class BenchmarkCase:
    query_id: str
    user_query: str
    primary_gt_resource_ids: tuple[str, ...] = ()
    acceptable_gt_resource_ids: tuple[str, ...] = ()
    hard_negative_resource_ids: tuple[str, ...] = ()
    constraints: dict = field(default_factory=dict)
    difficulty: str = ""

    @property
    def all_gt_resource_ids(self) -> set[str]:
        return set(self.primary_gt_resource_ids) | set(self.acceptable_gt_resource_ids)


@dataclass(frozen=True)
class EvaluationRecord:
    query_id: str
    user_query: str
    retrieved_resource_ids: tuple[str, ...]
    gt_resource_ids: tuple[str, ...]
    hit_at_1: bool
    hit_at_3: bool
    hit_at_5: bool
    reciprocal_rank: float
    tool_calls: tuple[dict, ...]


@dataclass(frozen=True)
class EvaluationSummary:
    case_count: int
    recall_at_1: float
    recall_at_3: float
    recall_at_5: float
    mrr: float
    records: tuple[EvaluationRecord, ...]
