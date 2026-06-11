from __future__ import annotations


SERVICE_CATEGORY_DESCRIPTIONS = {
    "Food Assistance": "food pantries, groceries, prepared meals, and nutrition support.",
    "Material Goods": "clothing, diapers, furniture, household goods, and basic supplies.",
    "Housing and Shelter": "shelters, housing search, rent help, and eviction-related housing support.",
    "Utility Assistance": "electric, gas, water, heating, cooling, and shutoff prevention help.",
    "Financial Assistance and Benefits": "cash assistance, benefits, Social Security, debt, budgeting, and help paying urgent bills.",
    "Transportation": "rides, transit help, medical transportation, and transportation assistance.",
    "Legal and Court Help": "civil legal aid, court services, filings, tenant legal help, immigration legal help, and legal clinics.",
    "Public Safety": "police, sheriff, emergency reporting, safety planning, and public safety services.",
    "Health Care": "clinics, hospitals, emergency medical care, screenings, public health, specialty medicine, and general medical care.",
    "Medical Support Services": "medical navigation, medical equipment, prescriptions, health supplies, and prevention support.",
    "Mental Health Care": "counseling, therapy, psychiatric care, crisis assessment, peer support, support groups, and mental health facilities.",
    "Substance Use Services": "detox, recovery, substance use treatment, harm reduction, and recovery support.",
    "Pregnancy and Reproductive Health": "pregnancy, prenatal care, family planning, and reproductive health.",
    "Disability and Rehabilitation": "rehabilitation, disability support, independent living, and accessibility services.",
    "Family and Caregiver Services": "case management, caregiver help, child and family services, and family support.",
    "Education and Youth Programs": "schools, GED, ESL, tutoring, school supplies, youth development, after-school, and student support.",
    "Employment and Job Training": "job search, job training, resumes, work clothing, and employment support.",
    "Disaster and Environmental Services": "disaster recovery, emergency relief, environmental hazards, and cleanup.",
    "Pet and Animal Services": "pet food, animal shelters, animal control, and veterinary-related animal services.",
    "Tax Help": "tax filing, tax issues, tax relief, and tax offices.",
    "Community and Recreation": "community centers, recreation, arts, culture, parks, and activities.",
}

BENCHMARK_SERVICE_CATEGORIES = tuple(SERVICE_CATEGORY_DESCRIPTIONS)

RAW_SUBCATEGORY_TO_SERVICE_CATEGORIES = {
    "Arts and Culture": ("Community and Recreation",),
    "Community Facilities/Centers": ("Community and Recreation",),
    "Courts": ("Legal and Court Help",),
    "Disaster Services": ("Disaster and Environmental Services",),
    "Domestic Animal Services": ("Pet and Animal Services",),
    "Educational Institutions/Schools": ("Education and Youth Programs",),
    "Educational Programs": ("Education and Youth Programs",),
    "Educational Support Services": ("Education and Youth Programs",),
    "Emergency Medical Care": ("Health Care",),
    "Employment": ("Employment and Job Training",),
    "Environmental Protection and Improvement": ("Disaster and Environmental Services",),
    "Food": ("Food Assistance",),
    "Health Screening/Diagnostic Services": ("Health Care",),
    "Health Supportive Services": ("Medical Support Services",),
    "Housing/Shelter": ("Housing and Shelter",),
    "Human Reproduction": ("Pregnancy and Reproductive Health",),
    "Individual and Family Support Services": ("Family and Caregiver Services",),
    "Inpatient Health Facilities": ("Health Care",),
    "Judicial Services": ("Legal and Court Help",),
    "Law Enforcement Agencies": ("Public Safety",),
    "Law Enforcement Services": ("Public Safety",),
    "Legal Services": ("Legal and Court Help",),
    "Leisure Activities/Recreation": ("Community and Recreation",),
    "Material Goods": ("Material Goods",),
    "Mental Health Assessment and Treatment": ("Mental Health Care",),
    "Mental Health Care Facilities": ("Mental Health Care",),
    "Mental Health Support Services": ("Mental Health Care",),
    "Money Management": ("Financial Assistance and Benefits",),
    "Mutual Support": ("Mental Health Care",),
    "Outpatient Health Facilities": ("Health Care",),
    "Public Assistance Programs": ("Financial Assistance and Benefits",),
    "Public Health": ("Health Care",),
    "Public Safety": ("Public Safety",),
    "Rehabilitation/Habilitation Services": ("Disability and Rehabilitation",),
    "Social Development and Enrichment": ("Education and Youth Programs",),
    "Social Insurance Programs": ("Financial Assistance and Benefits",),
    "Specialized Treatment and Prevention": ("Medical Support Services",),
    "Specialty Medicine": ("Health Care",),
    "Substance Use Disorder Services": ("Substance Use Services",),
    "Tax Organizations and Services": ("Tax Help",),
    "Temporary Financial Assistance": ("Financial Assistance and Benefits",),
    "Transportation": ("Transportation",),
    "Utilities": ("Utility Assistance",),
}

DROPPED_RAW_SUBCATEGORIES = {
    "Community Economic Development and Finance",
    "Community Groups and Government/Administrative Offices",
    "Community Planning and Public Works",
    "Consumer Assistance and Protection",
    "Consumer Regulation",
    "Criminal Correctional System",
    "Death Certification/Burial Arrangements",
    "Information Services",
    "Military Service",
    "Political Organization and Participation",
}


def service_categories_for_raw_subcategories(raw_subcategories) -> tuple[str, ...]:
    categories = []
    for raw_subcategory in raw_subcategories or ():
        for category in RAW_SUBCATEGORY_TO_SERVICE_CATEGORIES.get(raw_subcategory, ()):
            if category not in categories:
                categories.append(category)
    return tuple(categories)


def validate_curated_category_coverage(raw_subcategories: set[str]) -> None:
    known = set(RAW_SUBCATEGORY_TO_SERVICE_CATEGORIES) | DROPPED_RAW_SUBCATEGORIES
    missing = sorted(raw_subcategories - known)
    if missing:
        raise RuntimeError(f"Benchmark category mapping is missing raw subcategories: {missing}")
