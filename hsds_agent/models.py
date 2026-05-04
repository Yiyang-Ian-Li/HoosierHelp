from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class SearchRequest:
    query: str
    location: str | None = None
    radius_miles: float = 10.0
    categories: tuple[str, ...] = ()
    languages: tuple[str, ...] = ()
    eligibility: tuple[str, ...] = ()
    open_now: bool = False
    limit: int = 5


@dataclass(frozen=True)
class ResourceCandidate:
    service_id: str
    service_name: str
    organization_name: str
    description: str
    categories: tuple[str, ...]
    address: str
    city: str
    region: str
    postal_code: str
    distance_miles: float | None
    phone: str | None
    website: str | None
    languages: tuple[str, ...]
    eligibility: str | None
    schedule: str | None
    score: float
    source_fields: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class AgentAnswer:
    answer: str
    needs_follow_up: bool
    tool_calls: tuple[dict, ...]
    candidates: tuple[ResourceCandidate, ...]
