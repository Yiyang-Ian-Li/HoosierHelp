from __future__ import annotations


SERVICE_CATEGORY_DESCRIPTIONS = {
    "Food": "food pantries, groceries, prepared meals, nutrition support.",
    "Material Goods": "clothing, diapers, furniture, household goods, and basic supplies.",
    "Housing and Shelter": "shelters, housing help, eviction-related housing support.",
    "Utility Bill Help": "electric, gas, water, heating, cooling, and shutoff prevention help.",
    "Emergency Financial Assistance": "short-term financial help for urgent bills or basic needs.",
    "Benefits and Public Assistance": "SNAP, Medicaid, TANF, WIC, township assistance, and benefit applications.",
    "Transportation": "rides, transit help, medical transportation, and transportation assistance.",
    "Legal Help": "civil legal aid, tenant legal help, immigration legal help, and legal clinics.",
    "Courts and Court Services": "court locations, filings, public defenders, judicial services, and court navigation.",
    "Corrections and Reentry": "probation, correctional facilities, reentry, and post-release support.",
    "Police and Public Safety": "police, sheriff, emergency reporting, safety planning, and public safety services.",
    "Consumer Complaints and Regulation": "consumer complaints, scams, regulated services, licensing, and permits.",
    "Medical Care and Clinics": "clinics, outpatient care, specialty medicine, and general medical care.",
    "Emergency and Hospital Care": "emergency medical care, hospitals, and inpatient care.",
    "Medical Tests and Screenings": "health screenings, diagnostic tests, and disease testing.",
    "Health Navigation and Support": "medical navigation, medical equipment, prescriptions, and health support services.",
    "Pregnancy and Reproductive Health": "pregnancy, prenatal care, family planning, and reproductive health.",
    "Public Health": "public health departments, vaccines, disease prevention, and health information.",
    "Mental Health Treatment": "counseling, therapy, psychiatric care, crisis assessment, and mental health facilities.",
    "Mental Health and Peer Support": "peer support, emotional support, support groups, and non-clinical mental health help.",
    "Substance Use Help": "detox, recovery, substance use treatment, harm reduction, and recovery support.",
    "Disability and Rehabilitation": "rehabilitation, disability support, independent living, and accessibility services.",
    "Family and Caregiver Support": "case management, caregiver help, child/family services, and family support.",
    "Youth and School Support": "school supplies, tutoring, youth development, after-school, and student support.",
    "Education and Classes": "GED, ESL, adult education, schools, classes, and education programs.",
    "Employment and Job Training": "job search, job training, resumes, work clothing, and employment support.",
    "Financial Counseling and Debt": "budgeting, debt help, credit counseling, and financial literacy.",
    "Social Security and Medicare": "Social Security, Medicare, disability benefits, and social insurance help.",
    "Tax Help": "tax filing, tax issues, tax relief, and tax offices.",
    "Veteran and Military Support": "veteran, military, and service-member programs.",
    "Disaster and Environmental Help": "disaster recovery, emergency relief, environmental hazards, and cleanup.",
    "Government Offices and Public Services": "local government offices, public works, administrative offices, and public services.",
    "Information and Referral": "information lines, referral services, locators, directories, and navigation help.",
    "Community Centers and Recreation": "community centers, recreation, arts, culture, parks, and activities.",
    "Voting and Civic Participation": "voting, voter ID, voter registration, and civic participation.",
    "Pet and Animal Services": "pet food, animal shelters, animal control, and veterinary-related animal services.",
    "Business and Community Development": "small business help, community finance, and economic development programs.",
    "Burial and Funeral Help": "death certificates, burial, cremation, and funeral-related help.",
}


RAW_SUBCATEGORY_TO_SERVICE_CATEGORIES = {
    "Arts and Culture": ("Community Centers and Recreation",),
    "Community Economic Development and Finance": ("Business and Community Development",),
    "Community Facilities/Centers": ("Community Centers and Recreation",),
    "Community Groups and Government/Administrative Offices": ("Government Offices and Public Services",),
    "Community Planning and Public Works": ("Government Offices and Public Services",),
    "Consumer Assistance and Protection": ("Consumer Complaints and Regulation",),
    "Consumer Regulation": ("Consumer Complaints and Regulation",),
    "Courts": ("Courts and Court Services",),
    "Criminal Correctional System": ("Corrections and Reentry",),
    "Death Certification/Burial Arrangements": ("Burial and Funeral Help",),
    "Disaster Services": ("Disaster and Environmental Help",),
    "Domestic Animal Services": ("Pet and Animal Services",),
    "Educational Institutions/Schools": ("Education and Classes",),
    "Educational Programs": ("Education and Classes",),
    "Educational Support Services": ("Youth and School Support",),
    "Emergency Medical Care": ("Emergency and Hospital Care",),
    "Employment": ("Employment and Job Training",),
    "Environmental Protection and Improvement": ("Disaster and Environmental Help",),
    "Food": ("Food",),
    "Health Screening/Diagnostic Services": ("Medical Tests and Screenings",),
    "Health Supportive Services": ("Health Navigation and Support",),
    "Housing/Shelter": ("Housing and Shelter",),
    "Human Reproduction": ("Pregnancy and Reproductive Health",),
    "Individual and Family Support Services": ("Family and Caregiver Support",),
    "Information Services": ("Information and Referral",),
    "Inpatient Health Facilities": ("Emergency and Hospital Care",),
    "Judicial Services": ("Courts and Court Services",),
    "Law Enforcement Agencies": ("Police and Public Safety",),
    "Law Enforcement Services": ("Police and Public Safety",),
    "Legal Services": ("Legal Help",),
    "Leisure Activities/Recreation": ("Community Centers and Recreation",),
    "Material Goods": ("Material Goods",),
    "Mental Health Assessment and Treatment": ("Mental Health Treatment",),
    "Mental Health Care Facilities": ("Mental Health Treatment",),
    "Mental Health Support Services": ("Mental Health and Peer Support",),
    "Military Service": ("Veteran and Military Support",),
    "Money Management": ("Financial Counseling and Debt",),
    "Mutual Support": ("Mental Health and Peer Support",),
    "Outpatient Health Facilities": ("Medical Care and Clinics",),
    "Political Organization and Participation": ("Voting and Civic Participation",),
    "Public Assistance Programs": ("Benefits and Public Assistance",),
    "Public Health": ("Public Health",),
    "Public Safety": ("Police and Public Safety",),
    "Rehabilitation/Habilitation Services": ("Disability and Rehabilitation",),
    "Social Development and Enrichment": ("Youth and School Support",),
    "Social Insurance Programs": ("Social Security and Medicare",),
    "Specialized Treatment and Prevention": ("Health Navigation and Support",),
    "Specialty Medicine": ("Medical Care and Clinics",),
    "Substance Use Disorder Services": ("Substance Use Help",),
    "Tax Organizations and Services": ("Tax Help",),
    "Temporary Financial Assistance": ("Emergency Financial Assistance",),
    "Transportation": ("Transportation",),
    "Utilities": ("Utility Bill Help",),
}


def service_categories_for_raw_subcategories(raw_subcategories) -> tuple[str, ...]:
    categories = []
    for raw_subcategory in raw_subcategories or ():
        for category in RAW_SUBCATEGORY_TO_SERVICE_CATEGORIES.get(raw_subcategory, ()):
            if category not in categories:
                categories.append(category)
    return tuple(categories)


def validate_curated_category_coverage(raw_subcategories: set[str]) -> None:
    missing = sorted(raw_subcategories - set(RAW_SUBCATEGORY_TO_SERVICE_CATEGORIES))
    if missing:
        raise RuntimeError(f"Curated category mapping is missing raw subcategories: {missing}")
