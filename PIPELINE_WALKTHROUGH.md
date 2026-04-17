# Pipeline Walkthrough — One Trial, End to End

This document traces a single real clinical trial through each stage of the
pipeline, explaining what happens at each step and why the design decisions
were made the way they were.

The trial used throughout is **NCT00127920** — a Phase 2 study of Taxol,
Carboplatin, and Bevacizumab in advanced ovarian carcinoma. It was chosen
because its eligibility criteria are short enough to read in full, diverse
enough to illustrate every label type, and realistic enough to expose the
genuine ambiguities the system must handle.

New sections will be appended as subsequent pipeline steps are completed.

---

## Step 2 — Ingestion

### What happens

The ingestion pipeline calls the ClinicalTrials.gov REST API v2 and stores
the result in a local DuckDB database. Two tables are populated:

- **`trials`** — one row per trial, containing metadata and the raw
  eligibility text as a single free-text blob
- **`criteria`** — one row per individual criterion sentence, populated by
  the NLP layer in Step 3

### The raw API response (abbreviated)

The API returns a deeply nested JSON object. The ingestion parser
(`ingestion/parser.py`) walks the relevant paths and flattens them into a
single dict matching the `trials` table schema.

For NCT00127920, the key fields extracted are:

| Field | Value |
|---|---|
| `nct_id` | `NCT00127920` |
| `brief_title` | Pilot Study of Taxol, Carboplatin, and Bevacizumab in Advanced Stage Ovarian Carcinoma Patients |
| `conditions` | `['Ovarian Neoplasms']` |
| `interventions` | `['Avastin']` |
| `phases` | `['PHASE2']` |
| `status` | `COMPLETED` |
| `min_age` | `18 Years` |
| `max_age` | `None` (not specified) |
| `sex` | `FEMALE` |
| `mesh_conditions` | `['Ovarian Neoplasms']` |
| `mesh_interventions` | `['Bevacizumab']` |

Two things worth noting here:

**MeSH terms vs raw text.** The `conditions` field contains whatever the
trial sponsor wrote — it could be "Ovarian Cancer", "OC", "Ovarian
Neoplasms", etc. The `mesh_conditions` field contains the standardised
Medical Subject Heading term assigned by NLM, which is always
"Ovarian Neoplasms" regardless of how the sponsor phrased it. Later
pipeline steps use MeSH terms as pre-normalised labels for entity
extraction, so we store both.

**`max_age` is nullable.** Many trials enrol adults of any age above a
minimum. The parser defaults this to `None` rather than inventing a value.

### The raw eligibility text

This is stored verbatim in `trials.eligibility_text`. It is a single
free-text blob, mixing both inclusion and exclusion criteria under prose
headers, with bullet points in varying formats:

```
Inclusion Criteria:

* Subjects with a histologic or cytologic diagnosis of stage III/IV ovarian
  cancer, fallopian tube epithelial cancer, or peritoneal cancer who have not
  received prior chemotherapy or radiotherapy.
* Subjects must have the appropriate surgery for their gynecologic cancer.
  However, subjects may be treated in a neoadjuvant manner, with surgery being
  performed after chemotherapy cycles 1, 2, or 3.
* If neoadjuvant therapy is not administered, subjects must receive their first
  dose no more than six weeks postoperatively.
* Subjects must have adequate bone marrow, renal and hepatic function as
  defined by WBC > 3,000 cells/cu ml., platelets > 100,000/cu.ml., calculated
  creatinine clearance > 50 ccs/min., bilirubin < 1.5 mg/dl, and SGOT < three
  times normal.
* Karnofsky performance status > 50%.
* Subjects who have signed an institutional review board (IRB) approved
  informed consent form.

Exclusion Criteria:

* Subjects with epithelial ovarian cancer of low malignancy potential.
* Subjects with septicemia, severe infection, or acute hepatitis.
* Subjects with severe gastrointestinal bleeding.
* Subjects with a history of congestive heart failure, angina, or a history of
  myocardial infarction within the past six months.
```

This blob is not yet usable for machine learning — it is one long string with
no structure. Step 3 turns it into individual classifiable sentences.

---

## Step 3 — Criterion Splitter and Weak Labeler

### What happens

Two modules process the eligibility text in sequence:

1. **`nlp/criterion_splitter.py`** — splits the blob into individual
   criterion sentences and assigns each to its section
2. **`nlp/weak_labeler.py`** — assigns noisy labels to each sentence without
   any manual annotation

The result is written to the `criteria` table: one row per criterion, with
labels that will serve as training signal for the SciBERT classifier in Step 4.

---

### Part A — Criterion Splitter

#### The problem

The eligibility text blob has no machine-readable structure. Headers appear
in a dozen real-world formats ("INCLUSION CRITERIA:", "Inclusion criteria",
"* INCLUSION CRITERIA:", no colon, all-caps, etc.). Bullet points use
asterisks, numbers, or letters. Sub-points can be indented under a parent
criterion. Some trials use entirely non-standard headers like
"DISEASE CHARACTERISTICS:" that don't map to inclusion or exclusion at all.

#### What the splitter does

It scans the text line by line, maintaining a small amount of state:

- **Current section** (`inclusion`, `exclusion`, or `unknown`) — set whenever
  a header line is detected
- **Current bullet buffer** — accumulates lines belonging to the same
  criterion (a top-level bullet line plus any sub-points indented beneath it)

When a new top-level bullet is encountered, the buffer is flushed: combined
into a single string, cleaned (escape sequences like `\>` → `>`), and — if
at least 10 characters long — appended to the results as a criterion dict
with `{text, section, position}`.

Sub-points (indented 3+ spaces) are folded into the parent criterion rather
than emitted as separate rows. This is correct because sub-points are
qualifications of the parent, not independent criteria.

#### Output for NCT00127920

The splitter produces 10 criterion dicts. Sections are assigned from the
header context:

| # (Position) | Section | Text (abbreviated) |
|---|---|---|
| 0 | inclusion | Subjects with a histologic or cytologic diagnosis of stage III/IV... |
| 1 | inclusion | Subjects must have the appropriate surgery... |
| 2 | inclusion | If neoadjuvant therapy is not administered... |
| 3 | inclusion | Subjects must have adequate bone marrow, renal and hepatic function... |
| 4 | inclusion | Karnofsky performance status > 50%. |
| 5 | inclusion | Subjects who have signed an IRB approved informed consent form. |
| 6 | exclusion | Subjects with epithelial ovarian cancer of low malignancy potential. |
| 7 | exclusion | Subjects with septicemia, severe infection, or acute hepatitis. |
| 8 | exclusion | Subjects with severe gastrointestinal bleeding. |
| 9 | exclusion | Subjects with a history of congestive heart failure... |

**Position** is a 0-based index across the entire trial, not within each
section. This preserves ordering for the downstream model and allows
reconstruction of the original criteria sequence.

---

### Part B — Weak Labeler

#### Why weak labels at all?

The SciBERT model in Step 4 needs labeled training data. Manual annotation
of 256,000 criteria is not feasible before training. Weak supervision is the
standard approach: generate noisy labels programmatically, use them to get a
first model, then refine with a small set of manually verified labels.

The labeler assigns three independent labels to each criterion:

#### B1 — Inclusion vs Exclusion

**Signal:** the `section` field from the splitter.

This is the most reliable label (~90% accuracy). If the criterion was found
under an "Inclusion Criteria" header, `b1_label = 1`. Under "Exclusion
Criteria", `b1_label = 0`. Under a non-standard header, `b1_label = None`.

There is no confidence score for B1 — it is rule-derived. The training code
assigns a fixed confidence of 0.90 to all labeled B1 rows.

For this trial, criteria 0–5 get `b1_label = 1` and criteria 6–9 get
`b1_label = 0`. No unknowns, because both headers are standard.

#### B2 — Objective vs Subjective

**Signal:** regex pattern matching against the criterion text.

**Objective (b2_label = 1):** a criterion is objective if it has a numeric
threshold, a standardised clinical scale with a number, or a reference to
normal range. These can be verified unambiguously — either the patient's
creatinine is above 1.5 mg/dL or it isn't.

**Subjective (b2_label = 0):** language that requires clinical judgment —
"adequate", "willing", "at the discretion of the investigator", "significant".
No number is given; the assessor decides.

**Confidence:** the ratio of pattern-type hits to total hits. If 3 objective
patterns fire and 1 subjective pattern fires, confidence = 0.75 objective.
If nothing fires, `b2_label = None` with confidence 0.0.

Criterion [3] illustrates this well:

> *"Subjects must have adequate bone marrow, renal and hepatic function as
> defined by WBC > 3,000 cells/cu ml., platelets > 100,000/cu.ml., calculated
> creatinine clearance > 50 ccs/min., bilirubin < 1.5 mg/dl, and SGOT < three
> times normal."*

The word **"adequate"** fires a subjective pattern (it signals judgment-
dependent language). But the explicit numeric thresholds (`> 3,000`,
`> 100,000`, `> 50`, `< 1.5`) fire multiple objective patterns, as does
"three times normal" (a ULN reference). Objective hits outnumber subjective,
so `b2_label = 1` with `b2_confidence = 0.80` — the system correctly
identifies this as objective despite the "adequate" phrasing.

Criterion [5] — the informed consent criterion — fires only subjective
patterns ("signed", "consent"), so `b2_label = 0` with full confidence.

Criteria [0], [6], [7], [8], [9] fire no patterns at all: `b2_label = None`.
These criteria describe diagnoses, disease states, and conditions that are
clinically meaningful but carry no objectivity signal the regex can detect.
The SciBERT model will learn to classify them from context.

#### B3 — Observable vs Unobservable

**Signal:** a combination of EHR field lookup (known lab names, demographics,
diagnosis terminology) and unobservability keyword patterns.

**Observable (b3_label = 1):** assessable from standard electronic health
record data — lab values, diagnoses, performance status scores, treatment
history, documented allergies. The key question is: *would a clinician
reviewing a standard patient chart be able to answer this criterion?*

**Unobservable (b3_label = 0):** requires data that typically does not
exist in an EHR — consent status, life expectancy estimates, internet access,
geographic constraints, intent to conceive.

The results for this trial:

| Criterion | B3 | Reasoning |
|---|---|---|
| [0] Histologic diagnosis, no prior chemo | Observable | "diagnosis", "histolog", "prior", "chemotherapy" all in EHR |
| [1] Appropriate surgery, neoadjuvant OK | Observable | "chemotherapy" fires observable; "appropriate" fires nothing unobservable |
| [2] First dose ≤ 6 weeks postoperatively | **None** | No EHR field keyword fires; no unobservable signal either |
| [3] Lab value thresholds | Observable | WBC, platelets, creatinine, bilirubin are all standard lab fields |
| [4] Karnofsky > 50% | Observable | "Karnofsky" is a named clinical performance scale |
| [5] Signed informed consent | **Unobservable** | "signed", "consent" fire unobservable patterns |
| [6] Low malignancy potential OC | Observable | "cancer", "malignancy" fire observable |
| [7] Septicemia, severe infection | **None** | These are clinical conditions, but no specific EHR keyword fires |
| [8] Severe GI bleeding | **None** | Same — the concept is real but no pattern matches |
| [9] History of cardiac conditions | Observable | "history of" fires the known-history observable pattern |

Criterion [2] is genuinely ambiguous — "first dose no more than six weeks
postoperatively" is a timing constraint that *could* be derived from surgical
records, but no pattern fires either way, so None is the correct outcome.
The SciBERT model will be trained to make a decision where the weak labeler
cannot.

---

### The full criteria table for NCT00127920

This is what gets written to the `criteria` table after both modules run:

| pos | section | B1 | B2 | B2 conf | B3 | B3 conf | text |
|---|---|---|---|---|---|---|---|
| 0 | inclusion | 1 | None | 0.0 | 1 | 1.0 | Histologic diagnosis of stage III/IV... |
| 1 | inclusion | 1 | 0 | 1.0 | 1 | 1.0 | Appropriate surgery... neoadjuvant... |
| 2 | inclusion | 1 | None | 0.0 | None | 0.0 | First dose ≤ 6 weeks postoperatively |
| 3 | inclusion | 1 | 1 | 0.8 | 1 | 1.0 | Lab value thresholds (WBC, platelets...) |
| 4 | inclusion | 1 | 1 | 1.0 | 1 | 1.0 | Karnofsky performance status > 50% |
| 5 | inclusion | 1 | 0 | 1.0 | 0 | 1.0 | Signed informed consent |
| 6 | exclusion | 0 | None | 0.0 | 1 | 1.0 | Low malignancy potential OC |
| 7 | exclusion | 0 | None | 0.0 | None | 0.0 | Septicemia, severe infection |
| 8 | exclusion | 0 | None | 0.0 | None | 0.0 | Severe GI bleeding |
| 9 | exclusion | 0 | None | 0.0 | 1 | 1.0 | History of cardiac conditions |

**What the None labels mean in practice.** Of the 10 criteria, 6 have no B2
label and 3 have no B3 label. These are not errors — they are criteria where
the regex-based weak labeler genuinely has no signal. The B1 label is always
present because it comes from section context, not text content.

The None rows will contribute zero loss during SciBERT training (their
confidence weight is 0.0), so the model trains only on what the weak labeler
is confident about, and learns to generalise from context to fill in the rest.

---

*Next section will be added after Step 4 (SciBERT training) is complete.*
