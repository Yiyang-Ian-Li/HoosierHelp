from __future__ import annotations

from .index import ResourceIndex
from .models import Resource, SearchRequest, SearchResult

TOOL_DESCRIPTION = """
Search Indiana 211 resource records using structured filters.

Each record is one service/resource offered by an agency at a site. Use filters
the way a person would use a resource-directory search page:
- counties: service areas, e.g. MARION, ALLEN, LAKE, STATEWIDE.
- cities/states/zipcodes: physical site address fields, e.g. Indianapolis, IN, 46204.
- service_names/agency_names/site_names: text contained in those names.
- benchmark_categories/taxonomy_categories: broad need categories.
- subcategories/curated_subcategories: specific need types, usually the best category filter.
- eligibility/application/document/fee keywords: text that must appear in those fields.
- contact_required/address_required: require contact or address fields to exist.
- text_query: accepted for future retriever experiments, but ignored by this filter-only tool.

Non-empty filters are applied conjunctively across fields. Multiple values in
one field are alternatives. Use only filters that are supported by the user's
request; leave uncertain filters empty and put the uncertain need words in
text_query.
"""


def search_resources(index: ResourceIndex, request: SearchRequest) -> list[SearchResult]:
    results = []
    for resource in index.resources:
        matched_filters = _matched_filters(resource, request)
        if matched_filters is None:
            continue
        results.append(
            SearchResult(
                resource=resource,
                score=float(len(matched_filters)),
                matched_filters=tuple(matched_filters),
            )
        )

    results.sort(
        key=lambda result: (
            -result.score,
            result.resource.service_name.lower(),
            result.resource.agency_name.lower(),
            result.resource.resource_id,
        )
    )
    return results[: request.limit]


def search_resources_tool_schema(index: ResourceIndex, limit: int) -> dict:
    return {
        "type": "function",
        "function": {
            "name": "search_resources",
            "description": TOOL_DESCRIPTION,
            "parameters": {
                "type": "object",
                "properties": {
                    "text_query": {
                        "type": "string",
                        "description": (
                            "Accepted for future fuzzy retriever experiments. "
                            "Currently ignored by this filter-only tool."
                        ),
                    },
                    "resource_ids": _array_schema(
                        "Exact resource IDs, shaped like in211-{agency_id}-{site_id}-{service-slug}."
                    ),
                    "agency_ids": _array_schema("Exact numeric agency IDs as strings."),
                    "site_ids": _array_schema("Exact numeric site IDs as strings."),
                    "service_names": _array_schema(
                        "Service-name contains filters, e.g. Food Pantry, Legal Aid, Township Assistance."
                    ),
                    "agency_names": _array_schema(
                        "Agency-name contains filters, e.g. Gleaners, Catholic Charities."
                    ),
                    "site_names": _array_schema(
                        "Site-name contains filters, usually a location/program name."
                    ),
                    "counties": _enum_array_schema(
                        index.counties,
                        "Service-area counties in uppercase, such as MARION or ALLEN. Use county names here, not city names.",
                    ),
                    "cities": _enum_array_schema(
                        index.cities[:500],
                        "Physical site cities, such as Indianapolis, Fort Wayne, Bloomington.",
                    ),
                    "states": _enum_array_schema(
                        sorted({r.state for r in index.resources if r.state}),
                        "Physical site state abbreviations, usually IN but can include out-of-state/national offices.",
                    ),
                    "zipcodes": _enum_array_schema(
                        sorted({r.zipcode for r in index.resources if r.zipcode}),
                        "Physical site ZIP codes as strings.",
                    ),
                    "benchmark_categories": _enum_array_schema(
                        index.benchmark_categories,
                        "Broad need categories. Use when the need is broad or subcategory is uncertain.",
                    ),
                    "taxonomy_categories": _enum_array_schema(
                        index.benchmark_categories,
                        "Original broad Indiana 211 categories. Values look like Basic Needs, Health Care, Consumer Services.",
                    ),
                    "subcategories": _enum_array_schema(
                        index.subcategories,
                        "Specific Indiana 211 need types, e.g. Food, Utilities, Housing/Shelter, Legal Services.",
                    ),
                    "curated_subcategories": _enum_array_schema(
                        index.subcategories,
                        "Specific benchmark need types. In full data mode these match subcategories.",
                    ),
                    "eligibility_keywords": _array_schema(
                        "Words or phrases that must appear in eligibility text, e.g. senior, veteran, children, resident."
                    ),
                    "application_keywords": _array_schema(
                        "Words or phrases that must appear in application/intake text, e.g. call, walk in, online, appointment."
                    ),
                    "document_keywords": _array_schema(
                        "Words or phrases that must appear in required documents text, e.g. photo ID, proof of address, lease."
                    ),
                    "fee_keywords": _array_schema(
                        "Words or phrases that must appear in fee text, e.g. free, sliding scale, varies."
                    ),
                    "contact_required": {
                        "type": "boolean",
                        "description": "Require phone or website.",
                    },
                    "address_required": {
                        "type": "boolean",
                        "description": "Require address_1, city, state, and zipcode.",
                    },
                    "limit": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": max(10, limit),
                    },
                },
                "required": [],
                "additionalProperties": False,
            },
        },
    }


def request_from_tool_args(args: dict, fallback_query: str, limit: int) -> SearchRequest:
    return SearchRequest(
        text_query=_string(args.get("text_query"), fallback_query),
        resource_ids=_string_tuple(args.get("resource_ids")),
        agency_ids=_string_tuple(args.get("agency_ids")),
        site_ids=_string_tuple(args.get("site_ids")),
        service_names=_string_tuple(args.get("service_names")),
        agency_names=_string_tuple(args.get("agency_names")),
        site_names=_string_tuple(args.get("site_names")),
        counties=_string_tuple(args.get("counties")),
        cities=_string_tuple(args.get("cities")),
        states=_string_tuple(args.get("states")),
        zipcodes=_string_tuple(args.get("zipcodes")),
        benchmark_categories=_string_tuple(args.get("benchmark_categories")),
        taxonomy_categories=_string_tuple(args.get("taxonomy_categories")),
        subcategories=_string_tuple(args.get("subcategories")),
        curated_subcategories=_string_tuple(args.get("curated_subcategories")),
        eligibility_keywords=_string_tuple(args.get("eligibility_keywords")),
        application_keywords=_string_tuple(args.get("application_keywords")),
        document_keywords=_string_tuple(args.get("document_keywords")),
        fee_keywords=_string_tuple(args.get("fee_keywords")),
        contact_required=bool(args.get("contact_required", False)),
        address_required=bool(args.get("address_required", False)),
        limit=_int(args.get("limit"), limit),
    )


def request_to_tool_call(request: SearchRequest) -> dict:
    return {
        "tool": "search_resources",
        "arguments": {
            "text_query": request.text_query,
            "resource_ids": list(request.resource_ids),
            "agency_ids": list(request.agency_ids),
            "site_ids": list(request.site_ids),
            "service_names": list(request.service_names),
            "agency_names": list(request.agency_names),
            "site_names": list(request.site_names),
            "counties": list(request.counties),
            "cities": list(request.cities),
            "states": list(request.states),
            "zipcodes": list(request.zipcodes),
            "benchmark_categories": list(request.benchmark_categories),
            "taxonomy_categories": list(request.taxonomy_categories),
            "subcategories": list(request.subcategories),
            "curated_subcategories": list(request.curated_subcategories),
            "eligibility_keywords": list(request.eligibility_keywords),
            "application_keywords": list(request.application_keywords),
            "document_keywords": list(request.document_keywords),
            "fee_keywords": list(request.fee_keywords),
            "contact_required": request.contact_required,
            "address_required": request.address_required,
            "limit": request.limit,
        },
    }


def tool_result(results: list[SearchResult]) -> dict:
    return {
        "resources": [
            {
                "resource_id": result.resource.resource_id,
                "service_name": result.resource.service_name,
                "agency_name": result.resource.agency_name,
                "site_name": result.resource.site_name,
                "benchmark_categories": result.resource.benchmark_categories,
                "taxonomy_categories": result.resource.benchmark_categories,
                "subcategories": result.resource.source_subcategories,
                "curated_subcategories": result.resource.curated_subcategories,
                "service_area": result.resource.service_area,
                "city": result.resource.city,
                "state": result.resource.state,
                "zipcode": result.resource.zipcode,
                "address": result.resource.address_1,
                "phone": result.resource.phone,
                "website": result.resource.website,
                "eligibility": result.resource.eligibility,
                "application_process": result.resource.application_process,
                "fees": result.resource.fees,
                "documents_required": result.resource.documents_required,
                "score": result.score,
                "matched_filters": result.matched_filters,
            }
            for result in results
        ]
    }


def with_limit(request: SearchRequest, limit: int) -> SearchRequest:
    return SearchRequest(**{**request.__dict__, "limit": limit})


def _matched_filters(resource: Resource, request: SearchRequest) -> list[str] | None:
    matched = []
    checks = [
        ("resource_ids", request.resource_ids, (resource.resource_id,), _exact_any),
        ("agency_ids", request.agency_ids, (_agency_id(resource),), _exact_any),
        ("site_ids", request.site_ids, (_site_id(resource),), _exact_any),
        ("service_names", request.service_names, (resource.service_name,), _contains_any),
        ("agency_names", request.agency_names, (resource.agency_name,), _contains_any),
        ("site_names", request.site_names, (resource.site_name,), _contains_any),
        ("counties", request.counties, resource.service_area, _exact_any),
        ("cities", request.cities, (resource.city,), _exact_any),
        ("states", request.states, (resource.state,), _exact_any),
        ("zipcodes", request.zipcodes, (resource.zipcode,), _exact_any),
        ("benchmark_categories", request.benchmark_categories, resource.benchmark_categories, _exact_any),
        ("taxonomy_categories", request.taxonomy_categories, resource.benchmark_categories, _exact_any),
        ("subcategories", request.subcategories, resource.source_subcategories, _exact_any),
        ("curated_subcategories", request.curated_subcategories, resource.curated_subcategories, _exact_any),
        ("eligibility_keywords", request.eligibility_keywords, (resource.eligibility,), _contains_all),
        ("application_keywords", request.application_keywords, (resource.application_process,), _contains_all),
        ("document_keywords", request.document_keywords, (resource.documents_required,), _contains_all),
        ("fee_keywords", request.fee_keywords, (resource.fees,), _contains_all),
    ]
    for name, requested, available, matcher in checks:
        if requested:
            if not matcher(requested, available):
                return None
            matched.append(name)
    if request.contact_required and not (resource.phone or resource.website):
        return None
    if request.contact_required:
        matched.append("contact_required")
    if request.address_required and not (
        resource.address_1 and resource.city and resource.state and resource.zipcode
    ):
        return None
    if request.address_required:
        matched.append("address_required")
    return matched


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


def _contains_any(requested: tuple[str, ...], available: tuple[str, ...]) -> bool:
    haystack = " ".join(available).lower()
    return any(item.lower() in haystack for item in requested)


def _contains_all(requested: tuple[str, ...], available: tuple[str, ...]) -> bool:
    haystack = " ".join(available).lower()
    return all(item.lower() in haystack for item in requested)


def _agency_id(resource: Resource) -> str:
    return resource.resource_id.split("-")[1] if "-" in resource.resource_id else ""


def _site_id(resource: Resource) -> str:
    parts = resource.resource_id.split("-")
    return parts[2] if len(parts) > 2 else ""


def _norm(value: str) -> str:
    return value.lower().strip().replace(".", "")


def _string(value: object, default: str) -> str:
    return value.strip() if isinstance(value, str) and value.strip() else default


def _string_tuple(value: object) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    return tuple(item.strip() for item in value if isinstance(item, str) and item.strip())


def _int(value: object, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return max(1, min(50, parsed))
