# ClinicalTrials.gov API v2 — JSON Schema Reference

Derived from live API responses. Endpoint: `https://clinicaltrials.gov/api/v2/studies`

Each study object has two top-level keys: `protocolSection` and `derivedSection`.

---

## protocolSection

### identificationModule
| Field | Type | Notes |
|---|---|---|
| `.nctId` | str | Primary key — e.g. `"NCT05732857"` |
| `.briefTitle` | str | Short human-readable trial name |
| `.officialTitle` | str | Full formal title |
| `.orgStudyIdInfo.id` | str | Sponsor's internal ID |
| `.secondaryIdInfos[].id` | str | Optional additional IDs |
| `.secondaryIdInfos[].type` | str | e.g. `"OTHER"` |
| `.secondaryIdInfos[].domain` | str | e.g. `"JHM IRB"` |
| `.organization.fullName` | str | Sponsoring institution |
| `.organization.class` | str | e.g. `"OTHER"`, `"INDUSTRY"` |

---

### statusModule
| Field | Type | Notes |
|---|---|---|
| `.overallStatus` | str | `RECRUITING`, `COMPLETED`, `TERMINATED`, `UNKNOWN`, etc. |
| `.whyStopped` | str | Nullable — only present if terminated |
| `.statusVerifiedDate` | str | `"YYYY-MM"` format |
| `.startDateStruct.date` | str | `"YYYY-MM-DD"` |
| `.startDateStruct.type` | str | `"ACTUAL"` or `"ESTIMATED"` |
| `.primaryCompletionDateStruct.date` | str | |
| `.completionDateStruct.date` | str | |
| `.studyFirstSubmitDate` | str | |
| `.lastUpdateSubmitDate` | str | |

---

### sponsorCollaboratorsModule
| Field | Type | Notes |
|---|---|---|
| `.leadSponsor.name` | str | Primary sponsor name |
| `.leadSponsor.class` | str | `"INDUSTRY"`, `"OTHER"`, `"NIH"`, etc. |
| `.collaborators[].name` | str | List of collaborating orgs |
| `.collaborators[].class` | str | |
| `.responsibleParty.type` | str | `"SPONSOR"` or `"PRINCIPAL_INVESTIGATOR"` |
| `.responsibleParty.investigatorFullName` | str | Nullable — only if PI type |

---

### descriptionModule
| Field | Type | Notes |
|---|---|---|
| `.briefSummary` | str | **NER source** — narrative description; often names conditions, drugs, biomarkers |
| `.detailedDescription` | str | Nullable — longer scientific background; rich clinical terminology |

---

### conditionsModule
| Field | Type | Notes |
|---|---|---|
| `.conditions` | list[str] | **NER source** — cancer types as submitted by sponsor (e.g. `"Glioblastoma"`, `"Cancer of Kidney"`) |
| `.keywords` | list[str] | Nullable — free-text keywords |

---

### designModule
| Field | Type | Notes |
|---|---|---|
| `.studyType` | str | `"INTERVENTIONAL"` or `"OBSERVATIONAL"` |
| `.phases` | list[str] | `"PHASE1"`, `"PHASE2"`, `"PHASE3"`, `"PHASE4"`, `"NA"` |
| `.designInfo.allocation` | str | `"RANDOMIZED"`, `"NON_RANDOMIZED"` |
| `.designInfo.interventionModel` | str | `"PARALLEL"`, `"CROSSOVER"`, etc. |
| `.designInfo.primaryPurpose` | str | `"TREATMENT"`, `"PREVENTION"`, etc. |
| `.designInfo.maskingInfo.masking` | str | `"NONE"`, `"SINGLE"`, `"DOUBLE"`, etc. |
| `.enrollmentInfo.count` | int | Target or actual enrollment |
| `.enrollmentInfo.type` | str | `"ESTIMATED"` or `"ACTUAL"` |

---

### armsInterventionsModule

#### armGroups (each arm in the trial)
| Field | Type | Notes |
|---|---|---|
| `.armGroups[].label` | str | Arm name |
| `.armGroups[].type` | str | `"EXPERIMENTAL"`, `"ACTIVE_COMPARATOR"`, `"PLACEBO_COMPARATOR"`, etc. |
| `.armGroups[].description` | str | **NER source** — describes drugs, doses, schedule |
| `.armGroups[].interventionNames` | list[str] | Cross-reference to interventions |

#### interventions (the actual drugs/devices)
| Field | Type | Notes |
|---|---|---|
| `.interventions[].type` | str | `"DRUG"`, `"DEVICE"`, `"BIOLOGICAL"`, `"PROCEDURE"`, etc. |
| `.interventions[].name` | str | **NER source (DRUG)** — drug/device name e.g. `"Nivolumab"` |
| `.interventions[].description` | str | **NER source** — dosing, route, schedule details |
| `.interventions[].otherNames` | list[str] | Nullable — aliases e.g. `["anti-PD1"]` |
| `.interventions[].armGroupLabels` | list[str] | Which arms use this intervention |

---

### outcomesModule
| Field | Type | Notes |
|---|---|---|
| `.primaryOutcomes[].measure` | str | **NER source** — endpoint descriptions; often contain lab values, scales, thresholds |
| `.primaryOutcomes[].description` | str | Nullable — additional detail |
| `.primaryOutcomes[].timeFrame` | str | **NER source (TIMEFRAME)** — e.g. `"Up to 9 weeks"` |
| `.secondaryOutcomes[].measure` | str | Nullable — secondary endpoints |
| `.secondaryOutcomes[].timeFrame` | str | |

---

### eligibilityModule
| Field | Type | Notes |
|---|---|---|
| `.eligibilityCriteria` | str | **Primary NER + classifier source** — full inclusion/exclusion free text blob |
| `.minimumAge` | str | Nullable — e.g. `"18 Years"` (note: string not int) |
| `.maximumAge` | str | **Often absent** — nullable |
| `.sex` | str | `"ALL"`, `"MALE"`, `"FEMALE"` |
| `.healthyVolunteers` | bool | Whether healthy volunteers accepted |
| `.stdAges` | list[str] | `"CHILD"`, `"ADULT"`, `"OLDER_ADULT"` |

---

### contactsLocationsModule
| Field | Type | Notes |
|---|---|---|
| `.locations[].facility` | str | Institution name |
| `.locations[].city` | str | |
| `.locations[].state` | str | Nullable |
| `.locations[].country` | str | |
| `.locations[].geoPoint.lat` | float | |
| `.locations[].geoPoint.lon` | float | |
| `.overallOfficials[].name` | str | Nullable |
| `.overallOfficials[].role` | str | e.g. `"PRINCIPAL_INVESTIGATOR"` |

---

### ipdSharingStatementModule
| Field | Type | Notes |
|---|---|---|
| `.ipdSharing` | str | `"YES"` or `"NO"` — whether individual patient data is shared |

---

## derivedSection

Pre-computed by ClinicalTrials.gov. Standardized MeSH vocabulary — highly valuable for NER training labels.

### conditionBrowseModule
| Field | Type | Notes |
|---|---|---|
| `.meshes[].id` | str | MeSH concept ID |
| `.meshes[].term` | str | **NER training gold label (CONDITION)** — normalized condition name e.g. `"Glioblastoma"` |
| `.ancestors[].id` | str | MeSH hierarchy parent IDs |
| `.ancestors[].term` | str | Broader condition categories |

### interventionBrowseModule
| Field | Type | Notes |
|---|---|---|
| `.meshes[].id` | str | MeSH concept ID |
| `.meshes[].term` | str | **NER training gold label (DRUG)** — normalized drug name e.g. `"Nivolumab"`, `"Ipilimumab"` |
| `.ancestors[].term` | str | Drug class hierarchy e.g. `"Antibodies, Monoclonal"` |

---

## Fields relevant to each NER entity type

| NER Entity | Primary source fields |
|---|---|
| `CONDITION` | `conditionsModule.conditions`, `derivedSection.conditionBrowseModule.meshes[].term`, `descriptionModule.briefSummary` |
| `DRUG` | `armsInterventionsModule.interventions[].name`, `interventions[].otherNames`, `derivedSection.interventionBrowseModule.meshes[].term` |
| `LAB_VALUE` | `eligibilityModule.eligibilityCriteria`, `outcomesModule.primaryOutcomes[].measure` |
| `THRESHOLD` | `eligibilityModule.eligibilityCriteria`, `outcomesModule.primaryOutcomes[].measure` |
| `DEMOGRAPHIC` | `eligibilityModule.eligibilityCriteria`, `eligibilityModule.minimumAge`, `eligibilityModule.sex` |
| `SCALE` | `eligibilityModule.eligibilityCriteria`, `descriptionModule.briefSummary` |
| `TIMEFRAME` | `eligibilityModule.eligibilityCriteria`, `outcomesModule.primaryOutcomes[].timeFrame` |

---

## Fields NOT captured (and why)

| Field | Reason skipped |
|---|---|
| `oversightModule` | FDA regulation flags — not relevant to eligibility matching |
| `ipdSharingStatementModule` | Administrative — not relevant to NER or retrieval |
| `contactsLocationsModule` | Location/contact info — not needed for NLP pipeline |
| `derivedSection.conditionBrowseModule.ancestors` | Too broad (e.g. "Neoplasms") — adds noise to CONDITION NER |
| `secondaryIdInfos` | Administrative IDs — no analytical value |
| `sponsorCollaboratorsModule` | Sponsor info — not relevant to eligibility |
