# Deterministic Case Spec Benchmark Data

- Resource rows: 9987
- Benchmark-eligible resources: 6919
- Candidate probes: 8793
- Attempted probes: 595
- Valid case specs selected: 200
- Max attempts: 24000
- Category-cap skips: 108
- Categories covered: 29
- Difficulty targets: {"easy": 50, "hard": 50, "medium": 100}

## Matching Semantics

- `user_requirements` are user-stated needs.
- Every case has an explicit `intake_methods` requirement.
- Difficulty is the number of non-location/category user requirements: easy=1, medium=2, hard=3.
- `user_qualification` describes whether the user qualifies for resource-side eligibility, fee, and document requirements.
- A case is kept only when exactly one resource satisfies both `user_requirements` and `user_qualification`.
- Ground truth is embedded in each case spec as the singleton `ground_truth_resource_ids` field.
- Schedule requirements without schedule-relevant intake method: 0

## Difficulty

- `medium`: 100
- `easy`: 50
- `hard`: 50

## User Requirement Fields

- `intake_methods`: 200
- `zipcodes`: 96
- `cities`: 63
- `available_days`: 15
- `requires_weekend`: 12
- `available_at_or_after`: 12
- `requires_24_hours`: 2

## User Requirement Values

### `available_at_or_after`

- `18:00`: 12

### `available_days`

- `sat`: 14
- `sun`: 1

### `cities`

- `Evansville`: 3
- `Fort Wayne`: 3
- `Muncie`: 3
- `New Albany`: 2
- `Indianapolis`: 2
- `Akron`: 2
- `Gary`: 2
- `South Bend`: 2
- `North Vernon`: 1
- `Goshen`: 1
- `Vincennes`: 1
- `Paoli`: 1
- `Aurora`: 1
- `Crown Point`: 1
- `Corydon`: 1
- `Washington`: 1
- `Bremen`: 1
- `Winchester`: 1
- `Zionsville`: 1
- `Rushville`: 1
- `Delphi`: 1
- `Griffith`: 1
- `Bluffton`: 1
- `Elkhart`: 1
- `Anderson`: 1
- `Westport`: 1
- `Portland`: 1
- `Bristow`: 1
- `Greenwood`: 1
- `Crawfordsville`: 1
- `Shelbyville`: 1
- `Rockville`: 1
- `Kokomo`: 1
- `Marion`: 1
- `Honey Creek`: 1
- `Lafayette`: 1
- `Southport`: 1
- `Richmond`: 1
- `Kirklin`: 1
- `Columbus`: 1
- `Seymour`: 1
- `North Judson`: 1
- `Spencer`: 1
- `Wolcott`: 1
- `Valparaiso`: 1
- `Madison`: 1
- `Brazil`: 1
- `Fulton`: 1
- `Clarksville`: 1
- `Peru`: 1
- `Tell City`: 1
- `Lawrenceburg`: 1

### `intake_methods`

- `call`: 67
- `walk_in`: 54
- `appointment`: 53
- `online`: 13
- `mail`: 5
- `email`: 4
- `text`: 4

### `requires_24_hours`

- `true`: 2

### `requires_weekend`

- `true`: 12

### `zipcodes`

- `46235`: 3
- `47304`: 3
- `46615`: 2
- `46312`: 2
- `47150`: 2
- `46202`: 2
- `47303`: 2
- `46403`: 2
- `46910`: 2
- `46204`: 2
- `46410`: 2
- `46254`: 2
- `46221`: 2
- `46953`: 2
- `46915`: 2
- `46970`: 2
- `46803`: 2
- `47807`: 2
- `46237`: 1
- `46815`: 1
- `46526`: 1
- `47591`: 1
- `47001`: 1
- `47042`: 1
- `46205`: 1
- `47130`: 1
- `47501`: 1
- `46544`: 1
- `47167`: 1
- `46614`: 1
- `46806`: 1
- `46173`: 1
- `46923`: 1
- `46220`: 1
- `46319`: 1
- `46550`: 1
- `46140`: 1
- `46567`: 1
- `46714`: 1
- `46013`: 1
- `46703`: 1
- `46808`: 1
- `47012`: 1
- `46320`: 1
- `46962`: 1
- `46402`: 1
- `47515`: 1
- `47441`: 1
- `46143`: 1
- `46516`: 1
- `47904`: 1
- `47374`: 1
- `47711`: 1
- `46151`: 1
- `47201`: 1
- `47274`: 1
- `46203`: 1
- `47847`: 1
- `47951`: 1
- `46074`: 1
- `46324`: 1
- `47995`: 1
- `46222`: 1
- `47933`: 1
- `47619`: 1
- `46787`: 1
- `46256`: 1
- `46804`: 1
- `47834`: 1
- `46761`: 1
- `46260`: 1
- `47713`: 1
- `46805`: 1
- `46224`: 1
- `46125`: 1
- `47922`: 1

## Location Counties

- MARION: 23
- LAKE: 15
- ALLEN: 8
- DELAWARE: 7
- ST. JOSEPH: 6
- VIGO: 5
- VANDERBURGH: 5
- ELKHART: 5
- TIPPECANOE: 5
- BARTHOLOMEW: 5
- RANDOLPH: 4
- WASHINGTON: 4
- CLARK: 4
- MARSHALL: 4
- MORGAN: 4
- PORTER: 4
- JOHNSON: 4
- GRANT: 4
- FLOYD: 3
- FULTON: 3
- CARROLL: 3
- WELLS: 3
- JACKSON: 2
- KNOX: 2
- ORANGE: 2
- DEARBORN: 2
- RIPLEY: 2
- HANCOCK: 2
- SHELBY: 2
- BOONE: 2
- KOSCIUSKO: 2
- HOWARD: 2
- MADISON: 2
- HENDRICKS: 2
- LAWRENCE: 2
- OWEN: 2
- WABASH: 2
- PERRY: 2
- MONTGOMERY: 2
- JEFFERSON: 2
- MIAMI: 2
- NOBLE: 2
- NEWTON: 2
- WHITE: 2
- POSEY: 2
- CLAY: 2
- JENNINGS: 1
- PIKE: 1
- HARRISON: 1
- DAVIESS: 1
- RUSH: 1
- CASS: 1
- SULLIVAN: 1
- DECATUR: 1
- STEUBEN: 1
- FRANKLIN: 1
- JAY: 1
- GREENE: 1
- DEKALB: 1
- PARKE: 1
- WAYNE: 1
- CLINTON: 1
- VERMILLION: 1
- STARKE: 1
- HAMILTON: 1
- WARRICK: 1
- WHITLEY: 1
- LAPORTE: 1
- LAGRANGE: 1
- HENRY: 1

## Location Cities

- Indianapolis: 23
- Fort Wayne: 8
- Muncie: 7
- South Bend: 5
- Evansville: 5
- Columbus: 5
- Terre Haute: 4
- East Chicago: 4
- Winchester: 4
- Salem: 4
- Gary: 4
- Lafayette: 4
- Marion: 4
- Hammond: 3
- New Albany: 3
- Jeffersonville: 3
- Bluffton: 3
- Martinsville: 3
- Elkhart: 3
- Plymouth: 3
- Seymour: 2
- Vincennes: 2
- Paoli: 2
- Shelbyville: 2
- Akron: 2
- Zionsville: 2
- Valparaiso: 2
- Greenwood: 2
- Merrillville: 2
- Franklin: 2
- Kokomo: 2
- Anderson: 2
- Bedford: 2
- Spencer: 2
- Crawfordsville: 2
- Madison: 2
- Burlington: 2
- Peru: 2
- Mount Vernon: 2
- North Vernon: 1
- Goshen: 1
- Aurora: 1
- Versailles: 1
- Velpen: 1
- Crown Point: 1
- Corydon: 1
- Washington: 1
- Bremen: 1
- Mishawaka: 1
- Rushville: 1
- Delphi: 1
- Griffith: 1
- Nappanee: 1
- Greenfield: 1
- Logansport: 1
- Syracuse: 1
- Plainfield: 1
- Brownsburg: 1
- Sullivan: 1
- Westport: 1
- Angola: 1
- Brookville: 1
- Chesterton: 1
- North Manchester: 1
- Otterbein: 1
- Portland: 1
- Bristow: 1
- Linton: 1
- Auburn: 1
- Rockville: 1
- Osgood: 1
- Honey Creek: 1
- Warsaw: 1
- Southport: 1
- Richmond: 1
- Kirklin: 1
- Albion: 1
- Dana: 1
- North Judson: 1
- Kentland: 1
- Westfield: 1
- Wolcott: 1
- Lynnville: 1
- South Whitley: 1
- Monticello: 1
- Portage: 1
- Brazil: 1
- Kendallville: 1
- Fulton: 1
- Michigan City: 1
- LaGrange: 1
- Clarksville: 1
- Clay City: 1
- Tell City: 1
- Wabash: 1
- Eminence: 1
- Lawrenceburg: 1
- New Castle: 1
- Brook: 1

## Fee Capacity

- `can_pay`: 192
- `must_be_free`: 4
- `has_insurance`: 4

## Eligibility Facts

- `resident`: 148
- `senior`: 22
- `income`: 15
- `pregnant`: 11
- `children`: 7
- `disability`: 7
- `veteran`: 6
- `homeless`: 5

## Document Facts

- `photo_id`: 139
- `proof_of_address`: 117
- `insurance_card`: 109
- `proof_of_income`: 109
- `utility_bill`: 11
- `social_security`: 8
- `lease`: 6
- `birth_certificate`: 5

## Primary Resource Schedule Status

- `structured`: 200

## Primary Resource Intake Methods

- `call`: 175
- `appointment`: 105
- `walk_in`: 93
- `online`: 48
- `mail`: 15
- `email`: 12
- `text`: 6

## Primary Resource Fee Options

- `unknown`: 112
- `varies`: 64
- `payment_required`: 54
- `sliding_scale`: 32
- `free`: 6
- `insurance`: 6

## Primary Resource Document Requirements

- `none`: 90
- `photo_id`: 87
- `proof_of_address`: 40
- `insurance_card`: 24
- `varies`: 22
- `proof_of_income`: 21
- `utility_bill`: 11
- `social_security`: 8
- `lease`: 6
- `birth_certificate`: 5

## Primary Resource Eligibility Tags

- `resident`: 103
- `open`: 53
- `empty`: 27
- `senior`: 22
- `children`: 17
- `income`: 15
- `pregnant`: 11
- `veteran`: 7
- `disability`: 7
- `homeless`: 5

## Categories

- Information and Referral: 21
- Material Goods: 15
- Food: 15
- Housing and Shelter: 14
- Mental Health Treatment: 12
- Medical Care and Clinics: 11
- Substance Use Help: 10
- Pregnancy and Reproductive Health: 9
- Transportation: 9
- Family and Caregiver Support: 8
- Pet and Animal Services: 7
- Health Navigation and Support: 7
- Mental Health and Peer Support: 7
- Emergency and Hospital Care: 7
- Medical Tests and Screenings: 6
- Police and Public Safety: 6
- Legal Help: 5
- Government Offices and Public Services: 5
- Utility Bill Help: 4
- Employment and Job Training: 4
- Disaster and Environmental Help: 3
- Youth and School Support: 3
- Financial Counseling and Debt: 2
- Education and Classes: 2
- Business and Community Development: 2
- Tax Help: 2
- Community Centers and Recreation: 2
- Consumer Complaints and Regulation: 1
- Corrections and Reentry: 1
