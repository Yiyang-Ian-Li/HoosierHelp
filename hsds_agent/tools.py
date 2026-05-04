from __future__ import annotations

import json
import sqlite3
from datetime import datetime

from . import database
from .models import ResourceCandidate, SearchRequest


def search_services(
    conn: sqlite3.Connection, request: SearchRequest
) -> list[ResourceCandidate]:
    origin = database.resolve_coordinates(request.location)
    rows = database.fetch_service_rows(conn)
    candidates = []

    for row in rows:
        categories = tuple(json.loads(row["categories"]))
        languages = tuple(json.loads(row["languages"]))
        distance = None
        if origin:
            distance = database.distance_miles(
                origin[0], origin[1], row["latitude"], row["longitude"]
            )
            if distance > request.radius_miles:
                continue

        if request.languages and not _has_overlap(request.languages, languages):
            continue

        if request.categories and not _has_overlap(request.categories, categories):
            continue

        if request.open_now and not is_service_open_now(conn, row["service_id"]):
            continue

        score = _score_row(row, request, categories, languages, distance)
        if score <= 0:
            continue

        schedule = format_schedule(database.fetch_schedule(conn, row["service_id"]))
        candidates.append(
            ResourceCandidate(
                service_id=row["service_id"],
                service_name=row["service_name"],
                organization_name=row["organization_name"],
                description=row["description"],
                categories=categories,
                address=row["address_1"],
                city=row["city"],
                region=row["region"],
                postal_code=row["postal_code"],
                distance_miles=round(distance, 1) if distance is not None else None,
                phone=row["phone"],
                website=row["service_url"] or row["organization_website"],
                languages=languages,
                eligibility=row["eligibility"],
                schedule=schedule,
                score=round(score, 3),
                source_fields=(
                    "service.name",
                    "service.description",
                    "service.taxonomy",
                    "organization.name",
                    "location.address",
                    "phone.number",
                    "schedule",
                    "service.eligibility",
                ),
            )
        )

    candidates.sort(
        key=lambda item: (
            -item.score,
            item.distance_miles if item.distance_miles is not None else 999,
        )
    )
    return candidates[: request.limit]


def get_service_details(
    conn: sqlite3.Connection, service_id: str
) -> ResourceCandidate | None:
    request = SearchRequest(query="", limit=100)
    for candidate in search_services(conn, request):
        if candidate.service_id == service_id:
            return candidate
    return None


def format_schedule(rows: list[sqlite3.Row]) -> str | None:
    if not rows:
        return None
    by_range: dict[tuple[str, str], list[str]] = {}
    for row in rows:
        by_range.setdefault((row["opens_at"], row["closes_at"]), []).append(row["weekday"])
    parts = []
    for (opens_at, closes_at), weekdays in by_range.items():
        parts.append(f"{_format_weekdays(weekdays)} {opens_at}-{closes_at}")
    return "; ".join(parts)


def is_service_open_now(conn: sqlite3.Connection, service_id: str) -> bool:
    now = datetime.now()
    weekday = now.strftime("%A").lower()
    current = now.strftime("%H:%M")
    rows = database.fetch_schedule(conn, service_id)
    return any(
        row["weekday"] == weekday and row["opens_at"] <= current <= row["closes_at"]
        for row in rows
    )


def _score_row(
    row: sqlite3.Row,
    request: SearchRequest,
    categories: tuple[str, ...],
    languages: tuple[str, ...],
    distance: float | None,
) -> float:
    query_terms = _terms(request.query)
    searchable = " ".join(
        [
            row["service_name"],
            row["description"],
            " ".join(categories),
            row["organization_name"],
            row["eligibility"] or "",
            row["fees"] or "",
            " ".join(languages),
        ]
    ).lower()

    score = 0.0
    for term in query_terms:
        if term in searchable:
            score += 2.0

    for category in request.categories:
        if category.lower() in [item.lower() for item in categories]:
            score += 4.0

    available_languages = {_normalize_token(item) for item in languages}
    for language in request.languages:
        if _normalize_token(language) in available_languages:
            score += 2.0

    for eligibility in request.eligibility:
        if eligibility.lower() in (row["eligibility"] or "").lower():
            score += 1.5

    if "free" in query_terms and "free" in (row["fees"] or "").lower():
        score += 2.0

    if distance is not None:
        score += max(0.0, 3.0 - distance / 5.0)

    if not query_terms and not request.categories:
        score += 1.0

    return score


def _terms(value: str) -> set[str]:
    ignored = {"i", "me", "my", "need", "needs", "a", "an", "the", "near", "for"}
    return {
        token.strip(".,?!:;").lower()
        for token in value.split()
        if len(token.strip(".,?!:;")) > 2 and token.lower() not in ignored
    }


def _has_overlap(requested: tuple[str, ...], available: tuple[str, ...]) -> bool:
    available_normalized = {_normalize_token(item) for item in available}
    return any(_normalize_token(item) in available_normalized for item in requested)


def _normalize_token(value: str) -> str:
    normalized = value.lower().strip().replace("-", " ")
    aliases = {
        "en": "english",
        "eng": "english",
        "es": "spanish",
        "esp": "spanish",
        "espanol": "spanish",
        "español": "spanish",
    }
    return aliases.get(normalized, normalized)


def _format_weekdays(weekdays: list[str]) -> str:
    if weekdays == ["monday", "tuesday", "wednesday", "thursday", "friday"]:
        return "Mon-Fri"
    return ", ".join(day[:3].title() for day in weekdays)
