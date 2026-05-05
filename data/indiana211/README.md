# Indiana 211 Resource Data

This folder contains a snapshot of resource data from the Indiana 211 resource search site.

- Source page: https://indiana211-resource.fssa.in.gov/
- Retrieved date: 2026-05-05
- Source API used by the front-end: `https://in211-api.azurewebsites.net/api/HttpTrigger_in211api_dev`
- API mode: `blobresources`

Indiana 211 states that resource information is updated continuously. Treat this dataset as a point-in-time snapshot, not a permanent source of truth.

## Files

| File | Rows | Meaning |
| --- | ---: | --- |
| `indiana211_resources_deduped.csv` | 9,987 | Main resource-level table. One row is one service/resource offered by an agency at a site. |
| `indiana211_resource_county_rows.csv` | 31,358 | Lossless row-level export from the API. Rows can repeat the same resource across taxonomy/subcategory/county records. |
| `indiana211_resources_raw_all_counties.json` | 31,358 JSON objects | Raw API response for all counties requested from the site. |
| `indiana211_counties.csv` | 92 | County list hard-coded in the Indiana 211 front-end. |
| `README.json` | 1 | Machine-readable summary of source, dates, row counts, and dedupe rule. |

## Recommended Table

Use `indiana211_resources_deduped.csv` for most analysis.

The dedupe key is:

```text
agency_id + site_id + service_name
```

This is resource-level deduplication. It keeps separate services offered at the same site as separate rows. Using only `agency_id + site_id` would collapse different services at the same location and lose 3,012 service-level records.

Use `indiana211_resource_county_rows.csv` when you need the uncollapsed API rows, especially for auditing, taxonomy detail, or reproducing the exact source export.

## Field Dictionary

Fields in `indiana211_resources_deduped.csv`:

| Field | Meaning | Notes |
| --- | --- | --- |
| `agency_id` | Indiana 211 agency identifier. | 4,297 unique values. |
| `site_id` | Indiana 211 site/location identifier. | 6,975 unique values. |
| `site_name` | Name of the site where the service is offered. | Often the same as `agency_name`, but not always. |
| `service_name` | Name of the specific resource/service. | 3,210 unique values. |
| `site_eligibility` | Eligibility or target group for the service. | Free text. |
| `agency_desc` | Description of the agency. | Free text. |
| `address_1` | Street address line 1. | 77 blank rows. |
| `address_2` | Street address line 2. | Often blank. |
| `city` | City. | 678 unique nonblank values. |
| `zipcode` | ZIP code. | 869 unique nonblank values. |
| `state_province` | State/province for the site. | Mostly `IN`, but national or remote resources may be outside Indiana. |
| `site_number` | Contact phone number. | Free text; may include multiple formats. |
| `site_schedule` | Operating hours or service schedule. | Free text. |
| `site_details` | Application process, intake instructions, or usage details. | Free text. |
| `fee_structure` | Fee information. | Free text; blank in 6,740 rows. |
| `documents_required` | Documents required to access the service. | Free text. |
| `service_website` | Service or agency website. | Normalized to include `https://` when possible. |
| `service_email` | Contact email. | Free text/email. |
| `agency_name` | Name of the agency. | 4,260 unique values. |
| `latitude` | Latitude. | Blank in 5 rows. |
| `longitude` | Longitude. | Blank in 5 rows. |
| `taxonomy_categories` | One or more high-level resource categories. | Semicolon-separated; fixed atomic list shown below. |
| `subcategories` | One or more resource subcategories. | Semicolon-separated; fixed atomic list shown below. |
| `counties_served` | County or service area label from the source data. | Usually one county or `STATEWIDE`; one row has two county labels after aggregation. |
| `county_count` | Count of county labels in `counties_served`. | Mostly `1`; one resource has `2`. |
| `source_row_count` | Number of raw API rows collapsed into this resource row. | Useful for spotting resources with multiple taxonomy/subcategory records. |

## Fixed or Semi-Fixed Values

### Taxonomy Categories

`taxonomy_categories` is aggregated from these 10 source categories:

| Category | Resource rows containing category |
| --- | ---: |
| Basic Needs | 3,897 |
| Individual and Family Life | 2,365 |
| Health Care | 2,357 |
| Organizational/Community/International Services | 1,917 |
| Income Support and Employment | 1,351 |
| Mental Health and Substance Use Disorder Services | 1,120 |
| Criminal Justice and Legal Services | 746 |
| Consumer Services | 579 |
| Environment and Public Health/Safety | 213 |
| Education | 127 |

In the deduped table this field can contain multiple categories separated by `; `.

### Subcategories

There are 53 atomic subcategories in this snapshot. Top subcategories by resource count:

| Subcategory | Resource rows containing subcategory |
| --- | ---: |
| Food | 2,442 |
| Material Goods | 1,704 |
| Housing/Shelter | 1,626 |
| Health Supportive Services | 1,429 |
| Utilities | 1,268 |
| Transportation | 1,240 |
| Public Assistance Programs | 1,124 |
| Temporary Financial Assistance | 1,023 |
| Death Certification/Burial Arrangements | 990 |
| Community Groups and Government/Administrative Offices | 827 |
| Information Services | 824 |
| Individual and Family Support Services | 635 |
| Substance Use Disorder Services | 607 |
| Mental Health Assessment and Treatment | 525 |
| Mutual Support | 451 |

Full subcategory list:

```text
Arts and Culture
Community Economic Development and Finance
Community Facilities/Centers
Community Groups and Government/Administrative Offices
Community Planning and Public Works
Consumer Assistance and Protection
Consumer Regulation
Courts
Criminal Correctional System
Death Certification/Burial Arrangements
Disaster Services
Domestic Animal Services
Educational Institutions/Schools
Educational Programs
Educational Support Services
Emergency Medical Care
Employment
Environmental Protection and Improvement
Food
Health Screening/Diagnostic Services
Health Supportive Services
Housing/Shelter
Human Reproduction
Individual and Family Support Services
Information Services
Inpatient Health Facilities
Judicial Services
Law Enforcement Agencies
Law Enforcement Services
Legal Services
Leisure Activities/Recreation
Material Goods
Mental Health Assessment and Treatment
Mental Health Care Facilities
Mental Health Support Services
Military Service
Money Management
Mutual Support
Outpatient Health Facilities
Political Organization and Participation
Public Assistance Programs
Public Health
Public Safety
Rehabilitation/Habilitation Services
Social Development and Enrichment
Social Insurance Programs
Specialized Treatment and Prevention
Specialty Medicine
Substance Use Disorder Services
Tax Organizations and Services
Temporary Financial Assistance
Transportation
Utilities
```

### Counties

The source front-end lists 92 Indiana counties. The resource data also uses the service area label `STATEWIDE`.

Top county/service-area labels by resource count:

| County/service area | Resource rows |
| --- | ---: |
| STATEWIDE | 1,135 |
| MARION | 1,113 |
| ALLEN | 422 |
| LAKE | 389 |
| ST. JOSEPH | 288 |
| VANDERBURGH | 249 |
| ELKHART | 213 |
| MONROE | 210 |
| DELAWARE | 207 |
| TIPPECANOE | 192 |
| VIGO | 183 |
| MADISON | 177 |
| HAMILTON | 168 |
| LAPORTE | 158 |
| BARTHOLOMEW | 150 |

The complete county list is in `indiana211_counties.csv`.

### State/Province

`state_province` is not a strict Indiana-only field. Most records are in Indiana, but some statewide, national, remote, or federal resources have blank or out-of-state locations.

Top values:

| State/province | Resource rows |
| --- | ---: |
| IN | 9,479 |
| blank | 183 |
| DC | 50 |
| VA | 33 |
| KY | 29 |
| IL | 25 |
| MD | 23 |
| CA | 23 |
| FL | 19 |
| OH | 17 |

## Summary Statistics

### Overall Counts

| Metric | Count |
| --- | ---: |
| Deduped resources | 9,987 |
| Raw API rows | 31,358 |
| Agencies | 4,297 |
| Sites | 6,975 |
| Unique service names | 3,210 |
| Cities | 678 |
| ZIP codes | 869 |
| Atomic taxonomy categories | 10 |
| Atomic subcategories | 53 |

### Site and Agency Structure

| Metric | Count |
| --- | ---: |
| Sites with 2 or more services | 1,678 |
| Maximum services at one site | 11 |
| Agencies with 2 or more resources | 1,574 |
| Maximum resources under one agency | 71 |

Top sites by number of distinct services:

| Services | Agency | Site | City |
| ---: | --- | --- | --- |
| 11 | Community Action of Southern Indiana | Community Action of Southern Indiana | Jeffersonville |
| 11 | Vanderburgh County Health Department | Vanderburgh County Health Department | Evansville |
| 10 | Area Four Agency on Aging and Community Action Programs | Area Four Agency on Aging and Community Action Programs | Lafayette |
| 10 | Catholic Charities of Fort Wayne - South Bend | Catholic Charities of Fort Wayne - South Bend - Fort Wayne | Fort Wayne |
| 10 | Christamore House Family and Community Center | Christamore House Family and Community Center | Indianapolis |
| 10 | CoAction | Housing Counseling | Crown Point |
| 10 | Indiana Family and Social Services Administration - Division of Family Resources | Temporary Assistance for Needy Families (TANF) | Indianapolis |
| 10 | Mary Rigg Neighborhood Center | Mary Rigg Neighborhood Center | Indianapolis |
| 10 | St. Luke's United Methodist Church | St. Luke's United Methodist Church | Indianapolis |

Top agencies by number of resource rows:

| Resources | Agency |
| ---: | --- |
| 71 | Indianapolis Public Library |
| 60 | Bowen Health |
| 59 | Women's Care Center |
| 43 | Indiana Family and Social Services Administration - Disability, Aging, and Rehabilitative Services |
| 40 | Meridian Health Services |
| 36 | LifeSpring Health Systems |
| 36 | Strengthening Indiana Families |
| 33 | Eskenazi Health |
| 32 | Indiana Department of Health |
| 31 | Aspire Indiana Health |

### Top Service Names

| Service name | Resource rows |
| --- | ---: |
| Township Assistance | 984 |
| Food Pantry | 699 |
| Public Library | 276 |
| Police Department | 153 |
| Financial Assistance | 128 |
| Support Group | 110 |
| Community Meal | 106 |
| 911 Services | 94 |
| Assessor | 94 |
| Sandbags | 92 |
| Hospital | 90 |
| Transitional Housing | 82 |
| Soup Kitchen | 82 |
| Highway Department | 80 |
| Clothing Pantry | 80 |

### Geographic Distribution

Top cities:

| City | Resource rows |
| --- | ---: |
| Indianapolis | 1,463 |
| Fort Wayne | 419 |
| Evansville | 261 |
| South Bend | 229 |
| State-wide | 210 |
| Bloomington | 204 |
| Muncie | 184 |
| Lafayette | 178 |
| Terre Haute | 172 |
| Columbus | 142 |

Top ZIP codes:

| ZIP | Resource rows |
| --- | ---: |
| 46204 | 277 |
| blank | 247 |
| 46202 | 146 |
| 47201 | 114 |
| 46205 | 113 |
| 47374 | 110 |
| 46802 | 110 |
| 46208 | 102 |
| 47404 | 91 |
| 46016 | 90 |

### Data Completeness

Blank counts in the deduped resource table:

| Field | Blank rows |
| --- | ---: |
| `fee_structure` | 6,740 |
| `address_2` | 8,341 |
| `service_email` | 2,829 |
| `documents_required` | 1,025 |
| `service_website` | 983 |
| `zipcode` | 247 |
| `state_province` | 183 |
| `site_number` | 131 |
| `address_1` | 77 |
| `city` | 42 |
| `latitude` | 5 |
| `longitude` | 5 |

Common `documents_required` values:

| Value | Resource rows |
| --- | ---: |
| Nothing needed | 2,838 |
| blank | 1,025 |
| Photo ID -- Proof of address (lease usually required for rent assistance) -- Income and expense documentation (call to confirm details) -- Other items may be requested | 981 |
| Varies by service | 584 |
| Photo ID | 374 |

Common `fee_structure` values:

| Value | Resource rows |
| --- | ---: |
| blank | 6,740 |
| Varies by service | 384 |
| Varies by item | 129 |
| Varies | 66 |
| Varies by program and insurance coverage | 50 |

## Example Rows

### Example 1

```text
agency_name: 1 Voice
site_name: 1 Voice
service_name: Harm Reduction Services
city: Aurora
zipcode: 47001
site_number: 812-954-5154
site_schedule: Mon-Fri 11am-7pm; Sat noon-8pm; Sun noon-7pm Eastern Time
site_eligibility: Open
documents_required: Nothing needed
taxonomy_categories: Mental Health and Substance Use Disorder Services; Organizational/Community/International Services
subcategories: Community Planning and Public Works; Substance Use Disorder Services
counties_served: DEARBORN
```

### Example 2

At one site, multiple resources are intentionally separate rows:

```text
agency_name: Community Action of Southern Indiana
site_name: Community Action of Southern Indiana
city: Jeffersonville
```

Separate `service_name` values at this site include:

```text
Computer Access
Financial Literacy
Food Cabinet
Food Pantry
Food Pantry - Produce Wednesdays
Section 8 Housing Choice Vouchers
Weatherization
```

This is why the recommended dedupe key includes `service_name`.

## Practical Usage Notes

- For a service/resource directory, use `indiana211_resources_deduped.csv`.
- For a site/location directory, group `indiana211_resources_deduped.csv` by `agency_id + site_id` and aggregate `service_name`.
- For county-level or taxonomy-level audits, use `indiana211_resource_county_rows.csv`.
- `taxonomy_categories`, `subcategories`, and `counties_served` in the deduped table are semicolon-separated multi-value fields.
- Some fields are free text and should not be treated as fixed enumerations, especially `site_eligibility`, `site_schedule`, `site_details`, `fee_structure`, and `documents_required`.
