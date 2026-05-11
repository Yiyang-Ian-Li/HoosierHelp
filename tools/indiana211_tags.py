from __future__ import annotations


ELIGIBILITY_TAG_MEANINGS = {
    "open": "resource appears open to the general public or has broad eligibility.",
    "resident": "requires or emphasizes residence in a place or service area.",
    "income": "has income, poverty, or low-income eligibility.",
    "senior": "for older adults or people above a stated age.",
    "children": "for children, youth, or households with children.",
    "disability": "for people with disabilities or disability-related needs.",
    "veteran": "for veterans, military members, or military families.",
    "pregnant": "for pregnant people or pregnancy-related needs.",
    "homeless": "for people experiencing homelessness or housing instability.",
}

INTAKE_METHOD_MEANINGS = {
    "call": "start by phone.",
    "walk_in": "walk in without first applying online.",
    "online": "start on a website, online form, or web portal.",
    "appointment": "appointment or scheduled intake is needed.",
    "email": "start or contact by email.",
    "text": "start or contact by text message.",
    "mail": "start or submit materials by postal mail.",
}

DOCUMENT_REQUIREMENT_MEANINGS = {
    "none": "records indicate nothing is needed or no documents required.",
    "varies": "documents vary or user must call/check.",
    "photo_id": "photo ID or identification is needed.",
    "proof_of_income": "income documentation, pay stubs, or similar proof is needed.",
    "proof_of_address": "proof of address, residence, or current address is needed.",
    "lease": "lease or rental agreement is needed.",
    "insurance_card": "insurance card is needed.",
    "social_security": "Social Security card/number/document is needed.",
    "birth_certificate": "birth certificate is needed.",
    "utility_bill": "utility bill is needed.",
}

FEE_OPTION_MEANINGS = {
    "unknown": "source data has no usable fee/payment tag.",
    "free": "no fee or free service is indicated.",
    "sliding_scale": "fees vary by income or sliding scale.",
    "varies": "cost varies or user should call/check.",
    "insurance": "insurance, Medicaid, or Medicare may be accepted/required.",
    "payment_required": "some fee, copay, cost, or payment is indicated.",
}


def eligibility_tags(text: object) -> tuple[str, ...]:
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
    return _empty_tag_if_missing(tags)


def intake_methods(text: object) -> tuple[str, ...]:
    text = _clean(text).lower()
    tags = []
    _tag(tags, "call", "call" in text or "phone" in text)
    _tag(tags, "walk_in", "walk in" in text or "walk-in" in text)
    _tag(tags, "online", "online" in text or "website" in text or "visit www" in text)
    _tag(tags, "appointment", "appointment" in text or "schedule" in text)
    _tag(tags, "email", "email" in text or "e-mail" in text)
    _tag(tags, "text", "text" in text)
    _tag(tags, "mail", "mail" in text)
    return _empty_tag_if_missing(tags)


def document_requirements(text: object) -> tuple[str, ...]:
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
    return tuple(tags) if tags else ("none",)


def fee_options(text: object) -> tuple[str, ...]:
    text = _clean(text).lower()
    tags = []
    _tag(tags, "free", "free" in text or "no fee" in text or text == "none")
    _tag(tags, "sliding_scale", "sliding" in text)
    _tag(tags, "varies", "varies" in text or "vary" in text)
    _tag(tags, "insurance", "insurance" in text or "medicaid" in text or "medicare" in text)
    _tag(tags, "payment_required", "$" in text or "copay" in text or "fee" in text or "cost" in text)
    return tuple(tags) if tags else ("unknown",)


def _empty_tag_if_missing(tags: list[str]) -> tuple[str, ...]:
    return tuple(tags) if tags else ("empty",)


def _tag(tags: list[str], tag: str, condition: bool) -> None:
    if condition and tag not in tags:
        tags.append(tag)


def _clean(value: object) -> str:
    if value is None:
        return ""
    return " ".join(str(value).split())
