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

---

## Step 4 — Multi-task SciBERT Classifier

### What happens

The SciBERT model learns to fill in the labels the weak labeler could not assign.
It is a fine-tuned version of `allenai/scibert_scivocab_uncased` — a BERT-scale
model (110M parameters) pretrained on scientific text — with three independent
classification heads added on top, one per label type.

The model takes a criterion sentence as input and simultaneously predicts all
three labels in a single forward pass. This is the multi-task design: the shared
encoder builds a single representation of the text that is then passed to each
head independently. The heads do not share parameters with each other, only
with the encoder.

### Architecture

```
criterion text
      ↓
SciBERT encoder (shared, 110M params)
      ↓
  CLS token [768-dim vector]
      ↓
  ┌──────────┬──────────┬──────────┐
  │ head_b1  │ head_b2  │ head_b3  │
  │ Linear   │ Linear   │ Linear   │
  │ (768→2)  │ (768→2)  │ (768→2)  │
  └──────────┴──────────┴──────────┘
      ↓           ↓           ↓
  B1 logits   B2 logits   B3 logits
  (inc/exc)  (obj/subj)  (obs/unobs)
```

The CLS token — the first output token of the encoder — serves as a
sentence-level representation. Each head is a single linear layer projecting
it to a 2-class logit vector.

### Training signal: confidence-weighted loss

The weak labeler assigns a confidence score alongside each label. When
confidence is 0.0 (a None label), that example contributes zero gradient
for that head — it is effectively masked. When confidence is 1.0, the example
contributes full signal. This is the mechanism that allows the model to train
on all 256,663 criteria simultaneously without being corrupted by the
uncertain cases.

The loss for each head is:

```
loss_head = sum(cross_entropy(logit, label) * confidence) / sum(confidence)
```

The three head losses are combined with task weights: B1=0.30, B2=0.30, B3=0.40.
B3 is weighted highest because unobservable classification errors are the most
consequential downstream — a misclassified unobservable criterion bypasses the
Bayesian uncertainty quantification entirely.

### Class imbalance correction

Inspecting the output probability distributions after the first training probe
revealed that B3 had collapsed to predicting "observable" on nearly all criteria
(85% of predictions had P(observable) > 0.90). This is a direct consequence of
the 8:1 observable-to-unobservable ratio in the labeled corpus.

The fix: class weights in the loss function. Each unobservable example is
weighted 8× relative to observable, and each subjective example is weighted 3×
relative to objective. This restores balanced gradient signal without changing
the architecture or the data.

### Training results

Training ran for 8 epochs on 205,330 criteria (80/20 train/val split, seed 48697),
on Apple MPS hardware. Best checkpoint saved at epoch 6 by validation macro F1.

| Epoch | Loss | Macro F1 | B1 excl/incl | B2 subj/obj | B3 unobs/obs |
|---|---|---|---|---|---|
| 1 | 0.178 | 0.976 | 0.937/0.934 | 0.988/0.996 | 0.990/0.999 |
| 2 | 0.065 | 0.978 | 0.943/0.939 | 0.991/0.997 | 0.993/0.999 |
| 3 | 0.045 | 0.979 | 0.945/0.941 | 0.994/0.998 | 0.995/0.999 |
| 4 | 0.031 | 0.981 | 0.947/0.945 | 0.994/0.998 | 0.995/0.999 |
| **5** | **0.023** | **0.980** | 0.947/0.944 | 0.994/0.998 | 0.994/0.999 |
| **6 ✓** | **0.016** | **0.981** | **0.948/0.945** | **0.994/0.998** | **0.995/0.999** |
| 7 | 0.012 | 0.980 | 0.947/0.944 | 0.995/0.998 | 0.995/0.999 |
| 8 | 0.009 | 0.981 | 0.947/0.944 | 0.994/0.998 | 0.996/0.999 |

Loss continued falling through epoch 8, but validation macro F1 plateaued from
epoch 4 onward — the model converged cleanly with no overfitting signal. The
best checkpoint is epoch 6.

### What the val F1 actually measures

An important caveat: validation F1 is computed only on criteria where the weak
labeler assigned a confident label (confidence > 0). This means val F1 measures
how well SciBERT reproduces the weak labeler's own patterns — not how well it
generalises to genuinely ambiguous cases.

The ~138K B2 None rows and ~88K B3 None rows are excluded from val F1 entirely.
These are the criteria the model must generalise to from context, and their
quality can only be assessed through manual annotation (Step 5).

---

## Step 5 — Manual Annotation and Generalization Evaluation

### Why manual annotation is necessary

The val F1 of 0.997/0.999 is misleading in isolation. It reflects the model's
ability to reproduce regex patterns, not genuine understanding. The only honest
generalization test is to evaluate the model on criteria it has never seen labeled
— specifically the None rows, which are the cases the weak labeler could not classify.

### Sampling strategy

500 criteria were sampled from the None-row pool (criteria where b2_label IS NULL
OR b3_label IS NULL), stratified by section to mirror the corpus distribution:

| Section | Count |
|---|---|
| exclusion | 258 |
| inclusion | 185 |
| unknown | 57 |
| **Total** | **500** |

The checkpoint from Step 4 was used to generate SciBERT predictions
(predicted label + softmax probability) for all 500 sampled criteria before
any human or LLM labels were produced, so there is no contamination between
the model output and the ground truth.

### Hybrid annotation workflow

Labels were produced using a hybrid approach:

1. **LLM annotation:** Claude Sonnet was given an explicit written rubric for B2
   and B3 and asked to label each criterion independently, with a self-reported
   confidence score (0.0–1.0).
2. **Conflict flagging:** rows where LLM and SciBERT disagreed, or where LLM
   confidence fell below 0.80, were flagged for human review.
3. **Human adjudication:** a human reviewer examined all flagged rows and confirmed
   or overrode the LLM label.

Of 493 successfully annotated rows (7 API failures), 256 were flagged for human
review — a 51% flag rate, which reflects the genuine difficulty of the None-row
population. These are the criteria the weak labeler abstained on precisely because
they are ambiguous.

The human reviewer confirmed agreement with all LLM calls reviewed, making the
final ground truth equivalent to the LLM labels for this sample.

### Generalization results

| Head | Accuracy | Minority recall | Notes |
|---|---|---|---|
| B2 (obj/subj) | 73% | subj: **0.52** | misses ~half of subjective criteria |
| B3 (obs/unobs) | 78% | unobs: **0.30** | majority-class collapse persists |

**B2:** The model correctly identifies most objective criteria (recall 0.84) but
misclassifies nearly half of subjective criteria as objective. This is consistent
with the training distribution — the weak labeler's definition of "objective" was
anchored to numeric thresholds and named scales. Criteria that are subjective but
don't use the canonical hedging language ("adequate", "willing") are invisible to
that definition.

**B3:** The observable-class collapse is the more significant finding. Despite the
8× class weight correction, only 30% of genuinely unobservable criteria are
correctly identified on the None-row population. The model learned to output
"observable" as a near-default on out-of-distribution text. The calibration
analysis confirms this: 437 of 493 criteria received P(observable) > 0.90,
yet only 78% of those were actually observable.

### Design decision: propagate uncertainty into the Bayesian model

Rather than retraining with corrected labels (which would improve B3 unobservable
recall at the cost of another training cycle), the decision was made to accept
these results and encode the residual uncertainty into the Bayesian model's prior
structure. Specifically:

- B2 SciBERT probability feeds the Beta prior strength for subjective criteria
- B3 SciBERT probability modulates marginalization width for unobservable criteria
- High-confidence predictions (P > 0.9) are treated as strong priors
- Middle-band predictions (P 0.4–0.8) are treated as weak priors, resulting in
  wider credible intervals on the final P(eligible)

This is the principled use of Bayesian modeling: SciBERT outputs are noisy
measurements, not ground truth, and PyMC naturally represents that.

---

## Step 6 — Named Entity Recognition

### What happens

With criteria classified, the next requirement is to make objective/observable
criteria **directly comparable to patient data**. The raw text
"platelet count must be ≥ 100,000/µL" needs to be decomposed into a lab name
(`platelet`) and a numeric threshold (`≥ 100,000`) before it can be evaluated
against a patient's most recent platelet measurement.

The NER extractor populates seven columns in the criteria table that were
previously NULL:

| Column | Entity type | Example |
|---|---|---|
| `extracted_conditions` | Cancer types and comorbidities | `['breast neoplasms']` |
| `extracted_drugs` | Drug and intervention names | `['carboplatin', 'bevacizumab']` |
| `extracted_lab_values` | Lab test names | `['platelet', 'creatinine']` |
| `extracted_thresholds` | Numeric thresholds with comparators | `['≥ 100,000', '< 1.5 mg/dL']` |
| `extracted_scales` | Named clinical scales | `['Karnofsky > 50']` |
| `extracted_demographics` | Age, sex, BMI patterns | `['18 years of age', 'female']` |
| `extracted_timeframes` | Time window constraints | `['within 6 months']` |

### Why not a general NER model?

The standard approach would be to fine-tune a biomedical NER model such as
scispaCy or a BERT NER head on a labeled corpus. Three factors motivated a
different approach here:

1. **Installation incompatibility:** scispaCy's dependency chain (Cython, blis)
   does not build against Python 3.13 on Apple Silicon, ruling out that option.
2. **Domain mismatch:** general biomedical NER models (e.g. trained on BC5CDR)
   cover chemical and disease names but not THRESHOLD, SCALE, TIMEFRAME, or
   DEMOGRAPHIC entities, which are unique to the eligibility criteria domain.
3. **A better data source already exists:** every trial in the database already
   has MeSH-normalized condition and drug terms (`mesh_conditions`,
   `mesh_interventions`) plus drug aliases (`intervention_other_names`). Matching
   these trial-specific terms against the criterion text is higher precision
   than any general model, because it is anchored to exactly the entities that
   trial is about.

### Strategy by entity type

| Entity | Method | Rationale |
|---|---|---|
| CONDITION | Trial MeSH dictionary match | MeSH-normalized, trial-specific, zero false positives from other disease areas |
| DRUG | MeSH + alias dictionary match | Same; aliases cover brand names and investigational compound codes |
| LAB_VALUE | Regex (curated lab name list) | Pattern space is bounded; same list used in weak labeler |
| THRESHOLD | Regex (comparator + number + unit) | Highly structured syntax, regex is the right tool |
| SCALE | Regex (named scale + score) | Small closed set of scale names |
| DEMOGRAPHIC | Regex (age/sex/BMI patterns) | Structured, bounded pattern space |
| TIMEFRAME | Regex ("within N days/weeks/months") | Closed syntactic pattern |

### Results

All 256,663 criteria processed. Fill rates:

| Entity | Criteria with ≥1 match | % of corpus |
|---|---|---|
| THRESHOLD | 41,270 | 16.1% |
| TIMEFRAME | 30,653 | 11.9% |
| SCALE | 28,023 | 10.9% |
| DEMOGRAPHIC | 26,751 | 10.4% |
| LAB_VALUE | 22,997 | 9.0% |
| DRUG | 13,981 | 5.4% |
| CONDITION | 13,440 | 5.2% |

Condition and drug fill rates are lower (5%) by design. The dictionary matching
only captures entities from the trial's own MeSH vocabulary — criteria that
reference comorbidities outside the trial's primary focus return empty lists.
This is a deliberate precision/recall trade-off: no false positives at the cost
of lower recall on out-of-vocabulary conditions.

### How NER connects to the Bayesian model

NER is the bridge between the classifier output and the patient matching layer.
The Bayesian model's deterministic branch — criteria classified as B2=Objective
and B3=Observable — requires a structured comparand to evaluate against a
patient's EHR values. Without NER, the model would need to parse thresholds
from raw text at query time. With NER, the threshold is pre-extracted and stored,
and the evaluator performs a simple numeric comparison:

```
patient.creatinine ≤ extracted_threshold   →   criterion met (1)
patient.creatinine > extracted_threshold   →   criterion not met (0)
```

Criteria classified as B2=Subjective or B3=Unobservable bypass this comparison
entirely and enter the probabilistic branches of the Bayesian model instead.

---

*Next section will be added after Step 7 (embedding pipeline and ChromaDB) is complete.*
