from __future__ import annotations

import json
from pathlib import Path

from .models import Resource

DEFAULT_RESOURCE_INDEX = Path("data/indiana211/benchmark_curated/resource_index_curated.jsonl")
DEFAULT_FULL_INDIANA_CSV = Path("data/indiana211/indiana211_resources_deduped.csv")


class ResourceIndex:
    def __init__(self, resources: list[Resource]):
        self.resources = resources
        self.by_id = {resource.resource_id: resource for resource in resources}
        self.counties = sorted({county for r in resources for county in r.service_area})
        self.cities = sorted({r.city for r in resources if r.city})
        self.benchmark_categories = sorted(
            {category for r in resources for category in r.benchmark_categories}
        )
        self.subcategories = sorted(
            {subcategory for r in resources for subcategory in r.curated_subcategories}
        )


def load_resource_index(path: Path | str = DEFAULT_RESOURCE_INDEX) -> ResourceIndex:
    resources = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                resources.append(_resource_from_json(json.loads(line)))
    return ResourceIndex(resources)


def load_indiana_csv(path: Path | str = DEFAULT_FULL_INDIANA_CSV) -> ResourceIndex:
    import csv
    import re

    resources = []
    with Path(path).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            service_name = _clean(row.get("service_name", ""))
            slug = re.sub(r"[^a-z0-9]+", "-", service_name.lower()).strip("-")
            resources.append(
                Resource(
                    resource_id=f"in211-{row['agency_id']}-{row['site_id']}-{slug}",
                    service_name=service_name,
                    agency_name=_clean(row.get("agency_name", "")),
                    site_name=_clean(row.get("site_name", "")),
                    benchmark_categories=tuple(_split(row.get("taxonomy_categories", ""))),
                    source_subcategories=tuple(_split(row.get("subcategories", ""))),
                    curated_subcategories=tuple(_split(row.get("subcategories", ""))),
                    service_area=tuple(_split(row.get("counties_served", ""))),
                    city=_clean(row.get("city", "")),
                    state=_clean(row.get("state_province", "")),
                    zipcode=_clean(row.get("zipcode", "")),
                    address_1=_clean(row.get("address_1", "")),
                    phone=_clean(row.get("site_number", "")),
                    website=_clean(row.get("service_website", "")),
                    eligibility=_clean(row.get("site_eligibility", "")),
                    application_process=_clean(row.get("site_details", "")),
                    fees=_clean(row.get("fee_structure", "")),
                    documents_required=_clean(row.get("documents_required", "")),
                    search_text=_clean(
                        " ".join(
                            row.get(field, "")
                            for field in (
                                "service_name",
                                "agency_name",
                                "site_name",
                                "taxonomy_categories",
                                "subcategories",
                                "site_eligibility",
                                "agency_desc",
                                "site_details",
                                "documents_required",
                                "fee_structure",
                                "city",
                                "zipcode",
                                "counties_served",
                            )
                        )
                    ),
                )
            )
    return ResourceIndex(resources)


def _resource_from_json(row: dict) -> Resource:
    location = row.get("location") or {}
    contact = row.get("contact") or {}
    return Resource(
        resource_id=str(row["resource_id"]),
        service_name=str(row.get("service_name", "")),
        agency_name=str(row.get("agency_name", "")),
        site_name=str(row.get("site_name", "")),
        benchmark_categories=tuple(row.get("benchmark_categories") or ()),
        source_subcategories=tuple(row.get("source_subcategories") or ()),
        curated_subcategories=tuple(row.get("curated_subcategories") or ()),
        service_area=tuple(row.get("service_area") or ()),
        city=str(location.get("city", "")),
        state=str(location.get("state", "")),
        zipcode=str(location.get("zipcode", "")),
        address_1=str(location.get("address_1", "")),
        phone=str(contact.get("phone", "")),
        website=str(contact.get("website", "")),
        eligibility=str(row.get("eligibility", "")),
        application_process=str(row.get("application_process", "")),
        fees=str(row.get("fees", "")),
        documents_required=str(row.get("documents_required", "")),
        search_text=str(row.get("search_text", "")),
    )


def _split(value: str) -> list[str]:
    return [part.strip() for part in (value or "").split(";") if part.strip()]


def _clean(value: object) -> str:
    if value is None:
        return ""
    return " ".join(str(value).split())
