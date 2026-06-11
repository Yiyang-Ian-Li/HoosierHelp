from __future__ import annotations

from dataclasses import dataclass, field

from .indiana211_schedule import ScheduleWindow


@dataclass(frozen=True)
class Resource:
    resource_id: str
    service_name: str
    service_categories: tuple[str, ...]
    counties: tuple[str, ...]
    city: str
    zipcode: str
    schedule_windows: tuple[ScheduleWindow, ...]
    intake_methods: tuple[str, ...] = ()
    document_requirements: tuple[str, ...] = ()
    eligibility_tags: tuple[str, ...] = ()

    @property
    def resource_name(self) -> str:
        return self.service_name


@dataclass(frozen=True)
class SearchRequest:
    service_categories: tuple[str, ...] = ()
    schedule: dict = field(default_factory=dict)
    counties: tuple[str, ...] = ()
    cities: tuple[str, ...] = ()
    zipcodes: tuple[str, ...] = ()
    intake_methods: tuple[str, ...] = ()
    available_documents: tuple[str, ...] = ()
    eligibility: tuple[str, ...] = ()


@dataclass(frozen=True)
class SearchResult:
    resource: Resource
    score: float
    matched_filters: tuple[str, ...] = field(default_factory=tuple)


class ResourceIndex:
    def __init__(self, resources: list[Resource]):
        self.resources = resources
        self.by_id = {resource.resource_id: resource for resource in resources}
        self.counties = sorted({county for r in resources for county in r.counties})
        self.cities = sorted({r.city for r in resources if r.city})
        self.service_categories = sorted({category for r in resources for category in r.service_categories})
        self.intake_methods = sorted({method for r in resources for method in r.intake_methods})
        self.document_requirements = sorted({doc for r in resources for doc in r.document_requirements})
        self.eligibility_tags = sorted({tag for r in resources for tag in r.eligibility_tags})
