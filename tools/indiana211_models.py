from __future__ import annotations

from dataclasses import dataclass, field

from .indiana211_schedule import ScheduleWindow


@dataclass(frozen=True)
class Resource:
    resource_id: str
    service_name: str
    agency_name: str
    site_name: str
    service_categories: tuple[str, ...]
    service_area: tuple[str, ...]
    city: str
    state: str
    zipcode: str
    address_1: str
    phone: str
    website: str
    schedule_status: str
    schedule_windows: tuple[ScheduleWindow, ...]
    intake_methods: tuple[str, ...] = ()
    document_requirements: tuple[str, ...] = ()


@dataclass(frozen=True)
class SearchRequest:
    counties: tuple[str, ...] = ()
    cities: tuple[str, ...] = ()
    zipcodes: tuple[str, ...] = ()
    service_categories: tuple[str, ...] = ()
    available_days: tuple[str, ...] = ()
    available_time_windows: tuple[dict, ...] = ()
    requires_24_hours: bool = False
    intake_methods: tuple[str, ...] = ()
    documents_available: tuple[str, ...] = ()
    limit: int = 10


@dataclass(frozen=True)
class SearchResult:
    resource: Resource
    score: float
    matched_filters: tuple[str, ...] = field(default_factory=tuple)


class ResourceIndex:
    def __init__(self, resources: list[Resource]):
        self.resources = resources
        self.by_id = {resource.resource_id: resource for resource in resources}
        self.counties = sorted({county for r in resources for county in r.service_area})
        self.cities = sorted({r.city for r in resources if r.city})
        self.service_categories = sorted({category for r in resources for category in r.service_categories})
