from __future__ import annotations

from dataclasses import dataclass, field

from .curated_categories import validate_curated_category_coverage
from .indiana211_schedule import ScheduleWindow


@dataclass(frozen=True)
class Resource:
    resource_id: str
    service_name: str
    agency_name: str
    site_name: str
    taxonomy_categories: tuple[str, ...]
    subcategories: tuple[str, ...]
    service_categories: tuple[str, ...]
    service_area: tuple[str, ...]
    city: str
    state: str
    zipcode: str
    address_1: str
    phone: str
    website: str
    eligibility: str
    site_schedule: str
    schedule_status: str
    schedule_windows: tuple[ScheduleWindow, ...]
    site_details: str
    fee_structure: str
    documents_required: str
    eligibility_tags: tuple[str, ...] = ()
    intake_methods: tuple[str, ...] = ()
    document_requirements: tuple[str, ...] = ()
    fee_options: tuple[str, ...] = ()


@dataclass(frozen=True)
class SearchRequest:
    counties: tuple[str, ...] = ()
    cities: tuple[str, ...] = ()
    zipcodes: tuple[str, ...] = ()
    service_categories: tuple[str, ...] = ()
    eligibility_tags: tuple[str, ...] = ()
    available_days: tuple[str, ...] = ()
    available_at_or_after: str = ""
    requires_weekend: bool = False
    requires_24_hours: bool = False
    allow_appointment_only: bool = False
    intake_methods: tuple[str, ...] = ()
    document_requirements: tuple[str, ...] = ()
    fee_options: tuple[str, ...] = ()
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
        self.taxonomy_categories = sorted(
            {category for r in resources for category in r.taxonomy_categories}
        )
        self.subcategories = sorted({subcategory for r in resources for subcategory in r.subcategories})
        validate_curated_category_coverage(set(self.subcategories))
        self.service_categories = sorted({category for r in resources for category in r.service_categories})
