from __future__ import annotations

import re

from .index import ResourceIndex
from .models import SearchRequest


KEYWORD_SUBCATEGORY_RULES = {
    "Food": ("food", "pantry", "meal", "grocer", "hungry", "eat"),
    "Housing/Shelter": ("rent", "shelter", "housing", "homeless", "eviction", "sleep"),
    "Utilities": ("utility", "utilities", "electric", "water", "gas", "bill", "shut off", "lights"),
    "Temporary Financial Assistance": ("financial", "money", "cash", "emergency assistance"),
    "Transportation": ("ride", "transport", "bus", "medical ride", "driver"),
    "Public Assistance Programs": ("snap", "medicaid", "tanf", "benefits", "wic"),
    "Mental Health Assessment and Treatment": ("mental", "therapy", "counseling", "depression", "anxiety"),
    "Substance Use Disorder Services": ("addiction", "substance", "detox", "recovery", "opioid"),
    "Legal Services": ("legal", "lawyer", "attorney", "court", "tenant rights"),
    "Employment": ("job", "employment", "work", "career", "resume"),
    "Material Goods": ("diaper", "clothing", "furniture", "baby item", "household"),
    "Health Supportive Services": ("health", "medical", "clinic", "doctor"),
    "Consumer Assistance and Protection": ("consumer", "scam", "fraud", "complaint"),
    "Educational Programs": ("class", "school", "education", "ged", "english"),
}


def plan_search(query: str, index: ResourceIndex, limit: int = 10) -> SearchRequest:
    normalized = query.lower()
    counties = tuple(
        county
        for county in index.counties
        if county != "STATEWIDE" and _phrase_in_text(county, normalized)
    )
    cities = tuple(
        city
        for city in index.cities
        if _phrase_in_text(city, normalized)
        and not re.search(rf"\b{re.escape(city.lower())}\s+county\b", normalized)
    )
    subcategories = tuple(
        subcategory
        for subcategory, keywords in KEYWORD_SUBCATEGORY_RULES.items()
        if any(keyword in normalized for keyword in keywords)
    )
    categories = () if subcategories else tuple(_categories_for_subcategories(subcategories, index))
    return SearchRequest(
        text_query=query,
        counties=counties,
        cities=cities,
        benchmark_categories=categories,
        curated_subcategories=subcategories,
        limit=limit,
    )


def _categories_for_subcategories(
    subcategories: tuple[str, ...], index: ResourceIndex
) -> tuple[str, ...]:
    categories = []
    for resource in index.resources:
        if set(subcategories) & set(resource.curated_subcategories):
            categories.extend(resource.benchmark_categories)
    return tuple(sorted(set(categories)))


def _phrase_in_text(phrase: str, normalized_text: str) -> bool:
    phrase = phrase.lower().replace(".", "")
    normalized_text = normalized_text.replace(".", "")
    if len(phrase) <= 3:
        return bool(re.search(rf"\b{re.escape(phrase)}\b", normalized_text))
    return phrase in normalized_text
