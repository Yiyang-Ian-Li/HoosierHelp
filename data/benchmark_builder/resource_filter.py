from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.curated_categories import service_categories_for_raw_subcategories
from tools.indiana211_schedule import format_minutes, schedule_status, schedule_windows


DOCUMENT_AMBIGUOUS_PATTERNS = (
    r"\bvar(?:y|ies)\b",
    r"\bcall for\b",
    r"\bcall .*information\b",
    r"\bcall .*details\b",
    r"\bmay be requested\b",
    r"\bif applicable\b",
    r"\bif available\b",
    r"\bencouraged\b",
    r"\badditional documents?\b",
)

DOCUMENT_NONE_PATTERNS = (
    r"\bnothing needed\b",
    r"\bnothing required\b",
    r"\bno documentation is required\b",
    r"\bno documents? (?:needed|required)\b",
    r"^none$",
    r"^n/a$",
)

INTAKE_METHOD_PATTERNS = {
    "call": (r"\bcall\b", r"\bphone\b"),
    "walk_in": (r"\bwalk[ -]?in\b",),
    "online": (r"\bonline\b", r"\bwebsite\b", r"\bweb site\b", r"\bwww\.", r"\bweb portal\b"),
    "appointment": (r"\bappointment\b", r"\bschedule\b"),
    "email": (r"\be-?mail\b",),
    "text": (r"\btext(?:ing)?\b", r"\bsms\b"),
    "mail": (r"(?<!e-)\bmail\b", r"\bpostal\b", r"\bfax\b"),
}

DEFAULT_INPUT = Path("data/indiana211/indiana211_resources_deduped.csv")
DEFAULT_ORIGINAL_OUT = Path("data/benchmark/filtered_resources_raw.csv")
DEFAULT_TAGGED_OUT = Path("data/benchmark/filtered_resources_tagged.csv")


@dataclass(frozen=True)
class RawResource:
    resource_id: str
    agency_name: str
    site_name: str
    service_name: str
    eligibility: str
    address_1: str
    city: str
    zipcode: str
    state: str
    phone: str
    site_schedule: str
    site_details: str
    fee_structure: str
    documents_required: str
    website: str
    taxonomy_categories: tuple[str, ...]
    subcategories: tuple[str, ...]
    service_categories: tuple[str, ...]
    service_area: tuple[str, ...]
    schedule_status: str
    schedule_windows: tuple
    intake_methods: tuple[str, ...]
    document_requirements: tuple[str, ...]

RAW_FIELDS = [
    "resource_id",
    "agency_name",
    "site_name",
    "service_name",
    "site_eligibility",
    "agency_desc",
    "address_1",
    "city",
    "zipcode",
    "state_province",
    "site_number",
    "site_schedule",
    "site_details",
    "fee_structure",
    "documents_required",
    "service_website",
    "taxonomy_categories",
    "subcategories",
    "counties_served",
]

TAGGED_FIELDS = [
    "resource_id",
    "agency_name",
    "site_name",
    "service_name",
    "service_categories",
    "service_area",
    "city",
    "state",
    "zipcode",
    "address_1",
    "phone",
    "website",
    "schedule_status",
    "schedule_windows",
    "intake_methods",
    "document_requirements",
]


def load_raw_indiana_csv(path: Path | str) -> list[RawResource]:
    resources = []
    with Path(path).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            service_name = clean(row.get("service_name", ""))
            resource_id = f"in211-{row['agency_id']}-{row['site_id']}-{slugify(service_name)}"
            site_schedule = clean(row.get("site_schedule", ""))
            site_details = clean(row.get("site_details", ""))
            documents_required = clean(row.get("documents_required", ""))
            subcategories = tuple(split_semicolon(row.get("subcategories", "")))
            resources.append(
                RawResource(
                    resource_id=resource_id,
                    agency_name=clean(row.get("agency_name", "")),
                    site_name=clean(row.get("site_name", "")),
                    service_name=service_name,
                    eligibility=clean(row.get("site_eligibility", "")),
                    address_1=clean(row.get("address_1", "")),
                    city=clean(row.get("city", "")),
                    zipcode=clean(row.get("zipcode", "")),
                    state=clean(row.get("state_province", "")),
                    phone=clean(row.get("site_number", "")),
                    site_schedule=site_schedule,
                    site_details=site_details,
                    fee_structure=clean(row.get("fee_structure", "")),
                    documents_required=documents_required,
                    website=clean(row.get("service_website", "")),
                    taxonomy_categories=tuple(split_semicolon(row.get("taxonomy_categories", ""))),
                    subcategories=subcategories,
                    service_categories=service_categories_for_raw_subcategories(subcategories),
                    service_area=tuple(split_semicolon(row.get("counties_served", ""))),
                    schedule_status=schedule_status(site_schedule),
                    schedule_windows=schedule_windows(site_schedule),
                    intake_methods=parse_intake_methods(site_details),
                    document_requirements=parse_document_requirements(documents_required),
                )
            )
    return resources


def parse_intake_methods(text: object) -> tuple[str, ...]:
    raw = clean(text).lower()
    methods = [
        method
        for method, patterns in INTAKE_METHOD_PATTERNS.items()
        if any(re.search(pattern, raw, re.IGNORECASE) for pattern in patterns)
    ]
    return tuple(methods) if methods else ("empty",)


def parse_document_requirements(text: object) -> tuple[str, ...]:
    raw = clean(text).lower()
    tags = []
    append_if(tags, "none", bool(raw) and any(re.search(pattern, raw, re.IGNORECASE) for pattern in DOCUMENT_NONE_PATTERNS))
    append_if(tags, "varies", any(re.search(pattern, raw, re.IGNORECASE) for pattern in DOCUMENT_AMBIGUOUS_PATTERNS))
    append_if(tags, "photo_id", "photo id" in raw or "identification" in raw)
    append_if(tags, "proof_of_income", "proof of income" in raw or "pay stub" in raw or "income documentation" in raw)
    append_if(tags, "proof_of_address", "proof of address" in raw or "current address" in raw or "residency" in raw)
    append_if(tags, "lease", "lease" in raw)
    append_if(tags, "insurance_card", "insurance card" in raw)
    append_if(tags, "social_security", "social security" in raw)
    append_if(tags, "birth_certificate", "birth certificate" in raw)
    append_if(tags, "utility_bill", "utility bill" in raw)
    return tuple(tags) if tags else ("none",)


def append_if(values: list[str], value: str, condition: bool) -> None:
    if condition and value not in values:
        values.append(value)


def slugify(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")


def split_semicolon(value: str) -> list[str]:
    return [part.strip() for part in (value or "").split(";") if part.strip()]


def main() -> None:
    args = parse_args()
    resources = load_raw_indiana_csv(args.input)
    filtered = filter_benchmark_resources(resources)
    write_original_resources_csv(filtered, args.original_out)
    write_tagged_resources_csv(filtered, args.tagged_out)
    print(f"[resource-filter] wrote_raw_csv={args.original_out}")
    print(f"[resource-filter] wrote_tagged_csv={args.tagged_out}")


def filter_benchmark_resources(resources: list[RawResource], verbose: bool = True) -> list[RawResource]:
    current = list(resources)
    if verbose:
        print(f"[resource-filter] start: {len(current)}")
    current = filter_step(
        current,
        "in_state_local_service_area",
        in_state_local_service_area,
        verbose,
    )
    current = filter_step(
        current,
        "parseable_schedule",
        has_parseable_schedule,
        verbose,
    )
    current = filter_step(
        current,
        "parseable_or_empty_documents",
        has_parseable_or_empty_documents,
        verbose,
    )
    current = filter_step(
        current,
        "parseable_intake_methods",
        has_parseable_intake_methods,
        verbose,
    )
    current = filter_step(
        current,
        "benchmark_metadata",
        has_benchmark_metadata,
        verbose,
    )
    if verbose:
        print("[resource-filter] fee_and_eligibility: ignored")
    return current


def filter_step(
    resources: list[RawResource],
    name: str,
    predicate: Callable[[RawResource], bool],
    verbose: bool,
) -> list[RawResource]:
    kept = [resource for resource in resources if predicate(resource)]
    if verbose:
        removed = len(resources) - len(kept)
        print(f"[resource-filter] {name}: kept={len(kept)} removed={removed}")
    return kept


def in_state_local_service_area(resource: RawResource) -> bool:
    if (resource.state or "").upper() != "IN":
        return False
    service_area = {county.upper() for county in resource.service_area}
    return bool(service_area) and not (service_area & {"STATEWIDE", "ALL"})


def has_parseable_schedule(resource: RawResource) -> bool:
    return resource.schedule_status == "structured" and bool(resource.schedule_windows)


def has_parseable_or_empty_documents(resource: RawResource) -> bool:
    raw = clean(resource.documents_required)
    if not raw:
        return True
    if any(re.search(pattern, raw, re.IGNORECASE) for pattern in DOCUMENT_NONE_PATTERNS):
        return True
    if any(re.search(pattern, raw, re.IGNORECASE) for pattern in DOCUMENT_AMBIGUOUS_PATTERNS):
        return False
    parsed = set(resource.document_requirements) - {"empty", "none", "varies"}
    return bool(parsed)


def has_parseable_intake_methods(resource: RawResource) -> bool:
    raw = clean(resource.site_details).lower()
    methods = tuple(method for method in resource.intake_methods if method != "empty")
    if not raw or not methods:
        return False
    return all(intake_method_has_evidence(raw, method) for method in methods)


def intake_method_has_evidence(raw: str, method: str) -> bool:
    patterns = INTAKE_METHOD_PATTERNS.get(method)
    if not patterns:
        return False
    return any(re.search(pattern, raw, re.IGNORECASE) for pattern in patterns)


def has_benchmark_metadata(resource: RawResource) -> bool:
    return (
        bool(resource.service_categories)
        and bool(resource.zipcode)
        and bool(resource.phone or resource.website)
        and len(resource.service_categories) <= 3
    )


def clean(value: object) -> str:
    if value is None:
        return ""
    return " ".join(str(value).split())


def write_original_resources_csv(resources: list[RawResource], path: Path) -> None:
    write_csv(path, RAW_FIELDS, [raw_resource_row(resource) for resource in resources])


def write_tagged_resources_csv(resources: list[RawResource], path: Path) -> None:
    write_csv(path, TAGGED_FIELDS, [tagged_resource_row(resource) for resource in resources])


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def raw_resource_row(resource: RawResource) -> dict:
    return {
        "resource_id": resource.resource_id,
        "agency_name": resource.agency_name,
        "site_name": resource.site_name,
        "service_name": resource.service_name,
        "site_eligibility": resource.eligibility,
        "agency_desc": "",
        "address_1": resource.address_1,
        "city": resource.city,
        "zipcode": resource.zipcode,
        "state_province": resource.state,
        "site_number": resource.phone,
        "site_schedule": resource.site_schedule,
        "site_details": resource.site_details,
        "fee_structure": resource.fee_structure,
        "documents_required": resource.documents_required,
        "service_website": resource.website,
        "taxonomy_categories": pipe_join(resource.taxonomy_categories),
        "subcategories": pipe_join(resource.subcategories),
        "counties_served": pipe_join(resource.service_area),
    }


def tagged_resource_row(resource: RawResource) -> dict:
    return {
        "resource_id": resource.resource_id,
        "agency_name": resource.agency_name,
        "site_name": resource.site_name,
        "service_name": resource.service_name,
        "service_categories": pipe_join(resource.service_categories),
        "service_area": pipe_join(resource.service_area),
        "city": resource.city,
        "state": resource.state,
        "zipcode": resource.zipcode,
        "address_1": resource.address_1,
        "phone": resource.phone,
        "website": resource.website,
        "schedule_status": resource.schedule_status,
        "schedule_windows": json.dumps(
            [
                {
                    "day": window.day,
                    "start": format_minutes(window.start_minute),
                    "end": format_minutes(window.end_minute),
                }
                for window in resource.schedule_windows
            ],
            ensure_ascii=False,
        ),
        "intake_methods": pipe_join(resource.intake_methods),
        "document_requirements": pipe_join(resource.document_requirements),
    }


def pipe_join(values: tuple[str, ...] | list[str]) -> str:
    return "|".join(str(value) for value in values if str(value))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Filter benchmark-friendly Indiana 211 resources.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--original-out", type=Path, default=DEFAULT_ORIGINAL_OUT)
    parser.add_argument("--tagged-out", type=Path, default=DEFAULT_TAGGED_OUT)
    return parser.parse_args()


if __name__ == "__main__":
    main()
