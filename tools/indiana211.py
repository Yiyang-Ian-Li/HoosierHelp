from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass, field
from pathlib import Path


DEFAULT_RESOURCE_INDEX = Path("data/indiana211/indiana211_resources_raw_all_counties.json")
DEFAULT_FULL_INDIANA_CSV = Path("data/indiana211/indiana211_resources_deduped.csv")

TOOL_DESCRIPTION = """
Search Indiana 211 resource records using structured filters.

Each record is one service/resource offered by an agency at a site. Requested
fields are filters: a record must match every non-empty field to be returned.
Multiple values inside one field are alternatives.

Use only facts stated by the user. Do not infer nearby cities, counties, ZIP
codes, agencies, services, eligibility rules, intake steps, documents, fees, or
hours that the user did not state.
"""


@dataclass(frozen=True)
class Resource:
    resource_id: str
    service_name: str
    agency_name: str
    site_name: str
    taxonomy_categories: tuple[str, ...]
    subcategories: tuple[str, ...]
    service_area: tuple[str, ...]
    city: str
    state: str
    zipcode: str
    address_1: str
    phone: str
    website: str
    eligibility: str
    site_schedule: str
    site_details: str
    fee_structure: str
    documents_required: str
    eligibility_tags: tuple[str, ...] = ()
    schedule_tags: tuple[str, ...] = ()
    intake_methods: tuple[str, ...] = ()
    document_requirements: tuple[str, ...] = ()
    fee_options: tuple[str, ...] = ()


@dataclass(frozen=True)
class SearchRequest:
    counties: tuple[str, ...] = ()
    cities: tuple[str, ...] = ()
    zipcodes: tuple[str, ...] = ()
    taxonomy_categories: tuple[str, ...] = ()
    subcategories: tuple[str, ...] = ()
    eligibility_tags: tuple[str, ...] = ()
    schedule_tags: tuple[str, ...] = ()
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


def load_resource_index(path: Path | str = DEFAULT_RESOURCE_INDEX) -> ResourceIndex:
    resources = []
    text = Path(path).read_text(encoding="utf-8").strip()
    if not text:
        return ResourceIndex(resources)
    if text.startswith("["):
        return ResourceIndex(_resources_from_raw_rows(json.loads(text)))
    for line in text.splitlines():
        if line.strip():
            resources.append(_resource_from_json(json.loads(line)))
    return ResourceIndex(resources)


def load_indiana_csv(path: Path | str = DEFAULT_FULL_INDIANA_CSV) -> ResourceIndex:
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
                    taxonomy_categories=tuple(_split(row.get("taxonomy_categories", ""))),
                    subcategories=tuple(_split(row.get("subcategories", ""))),
                    service_area=tuple(_split(row.get("counties_served", ""))),
                    city=_clean(row.get("city", "")),
                    state=_clean(row.get("state_province", "")),
                    zipcode=_clean(row.get("zipcode", "")),
                    address_1=_clean(row.get("address_1", "")),
                    phone=_clean(row.get("site_number", "")),
                    website=_clean(row.get("service_website", "")),
                    eligibility=_clean(row.get("site_eligibility", "")),
                    site_schedule=_clean(row.get("site_schedule", "")),
                    site_details=_clean(row.get("site_details", "")),
                    fee_structure=_clean(row.get("fee_structure", "")),
                    documents_required=_clean(row.get("documents_required", "")),
                    eligibility_tags=_eligibility_tags(row.get("site_eligibility", "")),
                    schedule_tags=_schedule_tags(row.get("site_schedule", "")),
                    intake_methods=_intake_methods(row.get("site_details", "")),
                    document_requirements=_document_requirements(row.get("documents_required", "")),
                    fee_options=_fee_options(row.get("fee_structure", "")),
                )
            )
    return ResourceIndex(resources)


def search_resources_tool_schema(index: ResourceIndex) -> dict:
    return {
        "type": "function",
        "name": "search_resources",
        "description": TOOL_DESCRIPTION,
        "parameters": {
            "type": "object",
            "properties": {
                "counties": _array_schema(
                    "Service-area counties in uppercase, such as MARION or ALLEN. Use county names here, not city names."
                ),
                "cities": _array_schema(
                    "Physical site city names, such as South Bend or Indianapolis. Only include cities explicitly named by the user."
                ),
                "zipcodes": _array_schema(
                    "Physical site ZIP codes as strings. Only include ZIP codes explicitly named by the user."
                ),
                "taxonomy_categories": _enum_array_schema(
                    index.taxonomy_categories,
                    "Broad Indiana 211 taxonomy categories.",
                ),
                "subcategories": _enum_array_schema(
                    index.subcategories,
                    "Specific Indiana 211 service subcategories. Prefer this when the user's need is specific.",
                ),
                "eligibility_tags": _enum_array_schema(
                    [
                        "open",
                        "resident",
                        "income",
                        "senior",
                        "children",
                        "disability",
                        "veteran",
                        "pregnant",
                        "homeless",
                    ],
                    "Structured eligibility facts explicitly stated by the user.",
                ),
                "schedule_tags": _enum_array_schema(
                    ["weekdays", "weekends", "evening", "24_hours", "varies"],
                    "Structured schedule facts explicitly needed by the user.",
                ),
                "intake_methods": _enum_array_schema(
                    ["call", "walk_in", "online", "appointment", "email", "text", "mail"],
                    "How the user can or needs to start the application/intake process.",
                ),
                "document_requirements": _enum_array_schema(
                    [
                        "none",
                        "varies",
                        "photo_id",
                        "proof_of_income",
                        "proof_of_address",
                        "lease",
                        "insurance_card",
                        "social_security",
                        "birth_certificate",
                        "utility_bill",
                    ],
                    "Documents the user says they have, need, or cannot provide.",
                ),
                "fee_options": _enum_array_schema(
                    ["free", "sliding_scale", "varies", "insurance", "payment_required"],
                    "Fee/payment constraints explicitly stated by the user.",
                ),
            },
            "required": [],
            "additionalProperties": False,
        },
    }


def execute_search_resources(index: ResourceIndex, args: dict, limit: int = 10) -> dict:
    request = request_from_tool_args(args, limit=limit)
    return search_resources_tool_result(search_resources(index, request))


def search_resources(index: ResourceIndex, request: SearchRequest) -> list[SearchResult]:
    results = []
    for resource in index.resources:
        matched = _matched_filters(resource, request)
        if matched is None:
            continue
        results.append(
            SearchResult(
                resource=resource,
                score=float(len(matched)),
                matched_filters=tuple(matched),
            )
        )

    results.sort(
        key=lambda result: (
            result.resource.service_name.lower(),
            result.resource.agency_name.lower(),
            result.resource.resource_id,
        )
    )
    return results[: request.limit]


def request_from_tool_args(args: dict, limit: int = 10) -> SearchRequest:
    return SearchRequest(
        counties=_string_tuple(args.get("counties")),
        cities=_string_tuple(args.get("cities")),
        zipcodes=_string_tuple(args.get("zipcodes")),
        taxonomy_categories=_string_tuple(args.get("taxonomy_categories")),
        subcategories=_string_tuple(args.get("subcategories")),
        eligibility_tags=_string_tuple(args.get("eligibility_tags")),
        schedule_tags=_string_tuple(args.get("schedule_tags")),
        intake_methods=_string_tuple(args.get("intake_methods")),
        document_requirements=_string_tuple(args.get("document_requirements")),
        fee_options=_string_tuple(args.get("fee_options")),
        limit=limit,
    )


def search_resources_tool_result(results: list[SearchResult]) -> dict:
    return {
        "resources": [
            {
                "resource_id": result.resource.resource_id,
                "service_name": result.resource.service_name,
                "agency_name": result.resource.agency_name,
                "site_name": result.resource.site_name,
                "taxonomy_categories": result.resource.taxonomy_categories,
                "subcategories": result.resource.subcategories,
                "service_area": result.resource.service_area,
                "city": result.resource.city,
                "state": result.resource.state,
                "zipcode": result.resource.zipcode,
                "address": result.resource.address_1,
                "phone": result.resource.phone,
                "website": result.resource.website,
                "eligibility": result.resource.eligibility,
                "site_schedule": result.resource.site_schedule,
                "site_details": result.resource.site_details,
                "fee_structure": result.resource.fee_structure,
                "documents_required": result.resource.documents_required,
                "eligibility_tags": result.resource.eligibility_tags,
                "schedule_tags": result.resource.schedule_tags,
                "intake_methods": result.resource.intake_methods,
                "document_requirements": result.resource.document_requirements,
                "fee_options": result.resource.fee_options,
                "matched_filters": result.matched_filters,
            }
            for result in results
        ]
    }


def _matched_filters(resource: Resource, request: SearchRequest) -> list[str] | None:
    checks = [
        ("counties", request.counties, resource.service_area),
        ("cities", request.cities, (resource.city,)),
        ("zipcodes", request.zipcodes, (resource.zipcode,)),
        ("taxonomy_categories", request.taxonomy_categories, resource.taxonomy_categories),
        ("subcategories", request.subcategories, resource.subcategories),
        ("eligibility_tags", request.eligibility_tags, resource.eligibility_tags),
        ("schedule_tags", request.schedule_tags, resource.schedule_tags),
        ("intake_methods", request.intake_methods, resource.intake_methods),
        ("document_requirements", request.document_requirements, resource.document_requirements),
        ("fee_options", request.fee_options, resource.fee_options),
    ]
    matched = []
    for name, requested, available in checks:
        if requested:
            if not _exact_any(requested, available):
                return None
            matched.append(name)
    return matched


def _resource_from_json(row: dict) -> Resource:
    location = row.get("location") or {}
    contact = row.get("contact") or {}
    return Resource(
        resource_id=str(row["resource_id"]),
        service_name=str(row.get("service_name", "")),
        agency_name=str(row.get("agency_name", "")),
        site_name=str(row.get("site_name", "")),
        taxonomy_categories=tuple(row.get("taxonomy_categories") or ()),
        subcategories=tuple(row.get("subcategories") or ()),
        service_area=tuple(row.get("service_area") or ()),
        city=str(location.get("city", "")),
        state=str(location.get("state", "")),
        zipcode=str(location.get("zipcode", "")),
        address_1=str(location.get("address_1", "")),
        phone=str(contact.get("phone", "")),
        website=str(contact.get("website", "")),
        eligibility=str(row.get("eligibility", "")),
        site_schedule=str(row.get("site_schedule", "")),
        site_details=str(row.get("site_details", "")),
        fee_structure=str(row.get("fee_structure", "")),
        documents_required=str(row.get("documents_required", "")),
        eligibility_tags=tuple(row.get("eligibility_tags") or _eligibility_tags(row.get("eligibility", ""))),
        schedule_tags=tuple(row.get("schedule_tags") or _schedule_tags(row.get("site_schedule", ""))),
        intake_methods=tuple(row.get("intake_methods") or _intake_methods(row.get("site_details", ""))),
        document_requirements=tuple(row.get("document_requirements") or _document_requirements(row.get("documents_required", ""))),
        fee_options=tuple(row.get("fee_options") or _fee_options(row.get("fee_structure", ""))),
    )


def _resources_from_raw_rows(rows: list[dict]) -> list[Resource]:
    grouped: dict[tuple[str, str, str], dict] = {}
    for row in rows:
        key = (
            str(row.get("agency_id") or ""),
            str(row.get("site_id") or ""),
            _clean(row.get("service_name", "")),
        )
        if key not in grouped:
            grouped[key] = {**row, "_counties": [], "_categories": [], "_subcategories": []}
        group = grouped[key]
        _append_unique(group["_counties"], _clean(row.get("county", "")))
        _append_unique(group["_categories"], _clean(row.get("taxonomy_category", "")))
        _append_unique(group["_subcategories"], _clean(row.get("subcategory", "")))
    return [_resource_from_raw_group(row) for row in grouped.values()]


def _resource_from_raw_group(row: dict) -> Resource:
    service_name = _clean(row.get("service_name", ""))
    slug = "-".join(part for part in service_name.lower().split() if part)
    return Resource(
        resource_id=f"in211-{row.get('agency_id', '')}-{row.get('site_id', '')}-{slug}",
        service_name=service_name,
        agency_name=_clean(row.get("agency_name", "")),
        site_name=_clean(row.get("site_name", "")),
        taxonomy_categories=tuple(row.get("_categories") or ()),
        subcategories=tuple(row.get("_subcategories") or ()),
        service_area=tuple(row.get("_counties") or ()),
        city=_clean(row.get("city", "")),
        state=_clean(row.get("state_province", "")),
        zipcode=_clean(row.get("zipcode", "")),
        address_1=_clean(row.get("address_1", "")),
        phone=_clean(row.get("site_number", "")),
        website=_clean(row.get("service_website", "")),
        eligibility=_clean(row.get("site_eligibility", "")),
        site_schedule=_clean(row.get("site_schedule", "")),
        site_details=_clean(row.get("site_details", "")),
        fee_structure=_clean(row.get("fee_structure", "")),
        documents_required=_clean(row.get("documents_required", "")),
        eligibility_tags=_eligibility_tags(row.get("site_eligibility", "")),
        schedule_tags=_schedule_tags(row.get("site_schedule", "")),
        intake_methods=_intake_methods(row.get("site_details", "")),
        document_requirements=_document_requirements(row.get("documents_required", "")),
        fee_options=_fee_options(row.get("fee_structure", "")),
    )


def _array_schema(description: str) -> dict:
    return {"type": "array", "items": {"type": "string"}, "description": description}


def _enum_array_schema(values: list[str], description: str) -> dict:
    return {
        "type": "array",
        "items": {"type": "string", "enum": values},
        "description": description,
    }


def _exact_any(requested: tuple[str, ...], available: tuple[str, ...]) -> bool:
    req = {_norm(item) for item in requested}
    av = {_norm(item) for item in available}
    return bool(req & av)


def _norm(value: str) -> str:
    return value.lower().strip().replace(".", "").replace("-", "_").replace(" ", "_")


def _string_tuple(value: object) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    return tuple(item.strip() for item in value if isinstance(item, str) and item.strip())


def _append_unique(values: list[str], value: str) -> None:
    if value and value not in values:
        values.append(value)


def _split(value: str) -> list[str]:
    return [part.strip() for part in (value or "").split(";") if part.strip()]


def _clean(value: object) -> str:
    if value is None:
        return ""
    return " ".join(str(value).split())


def _eligibility_tags(text: object) -> tuple[str, ...]:
    text = _clean(text).lower()
    tags = []
    _tag(tags, "open", "open" in text and len(text) < 80)
    _tag(tags, "resident", "living in" in text or "resident" in text or "county" in text)
    _tag(tags, "income", "income" in text or "poverty" in text)
    _tag(tags, "senior", "senior" in text or "older" in text or "age 60" in text or "age 65" in text)
    _tag(tags, "children", "child" in text or "children" in text or "youth" in text or "age 0-18" in text)
    _tag(tags, "disability", "disab" in text)
    _tag(tags, "veteran", "veteran" in text or "military" in text)
    _tag(tags, "pregnant", "pregnan" in text)
    _tag(tags, "homeless", "homeless" in text)
    return tuple(tags)


def _schedule_tags(text: object) -> tuple[str, ...]:
    text = _clean(text).lower()
    tags = []
    _tag(tags, "weekdays", any(day in text for day in ("mon", "tues", "wed", "thur", "fri")))
    _tag(tags, "weekends", "sat" in text or "sun" in text or "weekend" in text)
    _tag(tags, "evening", "pm" in text and any(hour in text for hour in ("5", "6", "7", "8", "9", "10", "11")))
    _tag(tags, "24_hours", "24 hour" in text or "24/7" in text or "daily 24" in text)
    _tag(tags, "varies", "vary" in text or "varies" in text)
    return tuple(tags)


def _intake_methods(text: object) -> tuple[str, ...]:
    text = _clean(text).lower()
    tags = []
    _tag(tags, "call", "call" in text or "phone" in text)
    _tag(tags, "walk_in", "walk in" in text or "walk-in" in text)
    _tag(tags, "online", "online" in text or "website" in text or "visit www" in text)
    _tag(tags, "appointment", "appointment" in text or "schedule" in text)
    _tag(tags, "email", "email" in text or "e-mail" in text)
    _tag(tags, "text", "text" in text)
    _tag(tags, "mail", "mail" in text)
    return tuple(tags)


def _document_requirements(text: object) -> tuple[str, ...]:
    text = _clean(text).lower()
    tags = []
    _tag(tags, "none", "nothing needed" in text or "nothing required" in text or text in {"none", "n/a"})
    _tag(tags, "varies", "varies" in text or "call for" in text)
    _tag(tags, "photo_id", "photo id" in text or "identification" in text)
    _tag(tags, "proof_of_income", "proof of income" in text or "pay stub" in text or "income documentation" in text)
    _tag(tags, "proof_of_address", "proof of address" in text or "current address" in text or "residency" in text)
    _tag(tags, "lease", "lease" in text)
    _tag(tags, "insurance_card", "insurance card" in text)
    _tag(tags, "social_security", "social security" in text)
    _tag(tags, "birth_certificate", "birth certificate" in text)
    _tag(tags, "utility_bill", "utility bill" in text)
    return tuple(tags)


def _fee_options(text: object) -> tuple[str, ...]:
    text = _clean(text).lower()
    tags = []
    _tag(tags, "free", "free" in text or "no fee" in text or text == "none")
    _tag(tags, "sliding_scale", "sliding" in text)
    _tag(tags, "varies", "varies" in text or "vary" in text)
    _tag(tags, "insurance", "insurance" in text or "medicaid" in text or "medicare" in text)
    _tag(tags, "payment_required", "$" in text or "copay" in text or "fee" in text or "cost" in text)
    return tuple(tags)


def _tag(tags: list[str], tag: str, condition: bool) -> None:
    if condition and tag not in tags:
        tags.append(tag)
