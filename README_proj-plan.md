# Clinical Trial Eligibility Intelligence System

## Overview

An end-to-end clinical decision support system combining retrieval-augmented generation (RAG), transformer-based NLP, and Bayesian uncertainty quantification to match patients to oncology clinical trials. Given a patient profile, the system retrieves relevant trials, classifies eligibility criteria along three dimensions, and computes a posterior probability of eligibility with credible intervals reflecting uncertainty from subjective and unobservable criteria.

**Type:** Independent production ML/AI project
**Status:** Implementation in progress
**Total cost to run:** $0 (fully open-source stack)

---

## Architecture

```
ClinicalTrials.gov REST API (oncology subset ~15,000 trials)
        ↓
Ingestion Pipeline (Python + requests)
        ↓
DuckDB (local analytical store → raw trial metadata + criteria text)
        ↓
┌─────────────────────────────────────────────────────┐
│  NLP Layer (HuggingFace + PyTorch)                  │
│  Multi-task SciBERT with three classification heads │
│  B1: Inclusion vs Exclusion                         │
│  B2: Objective vs Subjective                        │
│  B3: Observable vs Unobservable                     │
└─────────────────────────────────────────────────────┘
        ↓
DuckDB (structured criteria store → labeled criterion objects)
        ↓
┌──────────────────────────┐   ┌────────────────────────────┐
│  Embedding Layer         │   │  Named Entity              │
│  sentence-               │   │  Recognition               │
│  transformers            │   │  (SciBERT NER head)        │
│  all-MiniLM-L6-v2        │   │  Conditions, drugs,        │
└──────────────────────────┘   │  lab values, thresholds,   │
        ↓                      │  demographics              │
ChromaDB                       └────────────────────────────┘
(local persistent                            ↓
 vector store)                  DuckDB structured
        ↓                       entity store
        └──────────────┬────────────────────┘
                       ↓
┌─────────────────────────────────────────────────────┐
│  RAG Layer (LlamaIndex)                             │
│  - Query understanding                              │
│  - Semantic retrieval from ChromaDB                 │
│  - Cross-encoder reranking                          │
│  - Response generation (Mistral-7B via Ollama)      │
│  - RAGAS evaluation                                 │
└─────────────────────────────────────────────────────┘
        ↓
┌─────────────────────────────────────────────────────┐
│  Bayesian Eligibility Scorer (PyMC)                 │
│  - Deterministic: B2=Objective + B3=Observable      │
│  - Beta prior: B2=Subjective                        │
│  - Marginalization: B3=Unobservable                 │
│  - Posterior P(eligible) with credible intervals    │
└─────────────────────────────────────────────────────┘
        ↓
┌─────────────────────────────────────────────────────┐
│  Streamlit Interface                                │
│  Panel 1: Trial search (natural language query)     │
│  Panel 2: Patient profile input                     │
│  Panel 3: Eligibility assessment + uncertainty      │
│  Panel 4: Criterion-by-criterion explainability     │
└─────────────────────────────────────────────────────┘
```

---

## Dataset

| Property | Detail |
|---|---|
| Source | ClinicalTrials.gov REST API v2 |
| Endpoint | `https://beta.clinicaltrials.gov/api/v2/studies` |
| Scope | Oncology trials (~15,000 for development) |
| Access | No credentials required |
| Format | JSON, paginated at 1,000 records per page |
| Rate limit | 10 requests/second |

**Key fields extracted:**
- `protocolSection.identificationModule.nctId` → trial identifier
- `protocolSection.identificationModule.briefTitle` → human-readable trial name
- `protocolSection.eligibilityModule.eligibilityCriteria` → full criteria free text
- `protocolSection.eligibilityModule.minimumAge` / `maximumAge` (maximumAge may be absent — nullable)
- `protocolSection.eligibilityModule.sex` → sex eligibility
- `protocolSection.eligibilityModule.stdAges` → age category list
- `protocolSection.conditionsModule.conditions` → cancer types (list of str)
- `protocolSection.armsInterventionsModule.interventions[].name` → drug/intervention names
- `protocolSection.armsInterventionsModule.interventions[].type` → intervention type (DRUG, DEVICE, etc.)
- `protocolSection.designModule.phases` → trial phase (list of str)
- `protocolSection.statusModule.overallStatus` → recruiting status
- `protocolSection.outcomesModule.primaryOutcomes[].measure` → primary endpoints
- `protocolSection.descriptionModule.briefSummary` → narrative trial summary (rich NER source)
- `derivedSection.conditionBrowseModule.meshes[].term` → MeSH-normalized condition terms
- `derivedSection.interventionBrowseModule.meshes[].term` → MeSH-normalized drug/intervention terms

**Scalability note:**

| Tier | Pattern | When to use |
|---|---|---|
| Development | API → local DuckDB | <10GB, portfolio demonstration |
| Staging | API → GCS → DuckDB reads from GCS | 10GB–1TB, single-node |
| Production | Streaming API → GCS → Spark/BigQuery | >1TB, distributed |

DuckDB supports GCS and S3 reads natively with identical query syntax. The transition from local to cloud storage requires only connection configuration changes, no query rewrites.

---

## Project Structure

```
clinical_trial_eligibility/
├── data/
│   ├── raw/                    # gitignored → API JSON responses
│   ├── processed/              # gitignored → DuckDB files, parquet
│   └── labeled/                # manually labeled validation sets
├── ingestion/
│   ├── api_client.py           # ClinicalTrials.gov API wrapper with pagination
│   ├── parser.py               # JSON → structured DuckDB records
│   └── database.py             # DuckDB connection and schema
├── nlp/
│   ├── criterion_splitter.py   # split criteria blob into sentences
│   ├── weak_labeler.py         # regex/heuristic weak supervision
│   ├── multitask_classifier.py # SciBERT multi-task model (B1/B2/B3)
│   ├── ner_extractor.py        # clinical entity extraction
│   ├── trainer.py              # training loop with W&B logging
│   └── evaluate.py             # F1 evaluation per subtask
├── rag/
│   ├── embedder.py             # sentence-transformers embedding pipeline
│   ├── vector_store.py         # ChromaDB operations
│   ├── retriever.py            # retrieval + cross-encoder reranking
│   ├── generator.py            # Mistral-7B via Ollama
│   ├── pipeline.py             # end-to-end RAG orchestration
│   └── evaluate_ragas.py       # RAGAS evaluation suite
├── bayesian/
│   ├── criterion_evaluator.py  # patient vs criterion matching
│   ├── eligibility_model.py    # PyMC Bayesian model
│   └── uncertainty.py          # credible interval computation
├── app/
│   └── streamlit_app.py        # Streamlit interface (4 panels)
├── tests/
│   ├── test_api_client.py
│   ├── test_classifier.py
│   ├── test_embed.py
│   └── test_bayesian.py
├── notebooks/
│   └── exploration.ipynb       # EDA and prototyping
├── embed.py                    # Embedding pipeline CLI
├── .env                        # gitignored
├── .env.example
├── .gitignore
├── requirements.txt
├── docker-compose.yml          # Ollama service
└── README.md
```

---

## Stack

| Component | Tool | Cost |
|---|---|---|
| Data ingestion | requests, DuckDB | Free |
| Criterion splitting | regex | Free |
| Weak supervision | regex, lookup tables | Free |
| Multi-task NLP | HuggingFace Transformers, PyTorch | Free |
| NER | regex + MeSH dictionary matching | Free |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 | Free |
| Vector store | ChromaDB (local persistent) | Free |
| LLM | Mistral-7B via Ollama (local) | Free |
| RAG orchestration | LlamaIndex | Free |
| Reranking | cross-encoder/ms-marco-MiniLM-L-6-v2 | Free |
| RAG evaluation | RAGAS | Free |
| Bayesian modeling | PyMC | Free |
| Frontend | Streamlit | Free |
| Containerization | Docker + docker-compose (Ollama service) | Free |

---

## Implementation Steps

### Step 1 — Repository Scaffold and Environment Setup ✓

**Files:** `.gitignore`, `.env`, `.env.example`, `requirements.txt`, full folder structure

**Tasks:**
- Initialize git repo and `.gitignore`
- Create `.venv` Python virtual environment
- Install core dependencies
- Create `.env.example` with placeholder keys
- Create folder structure as defined in Project Structure above

**Key packages:**
```bash
pip install duckdb requests python-dotenv
pip install torch transformers sentence-transformers
pip install chromadb llama-index ragas
pip install pymc streamlit
pip install pytest pandas anthropic python-dotenv
```

**Install Ollama and pull Mistral:**
```bash
brew install ollama
ollama pull mistral
```

---

### Step 2 — DuckDB Schema and API Ingestion Pipeline ✓

**Files:** `ingestion/database.py`, `ingestion/api_client.py`, `ingestion/parser.py`, `ingest.py`

**Two DuckDB tables:**

- **`trials`** — one row per trial. Key fields: `nct_id`, `brief_title`, `conditions`, `interventions`, `intervention_types`, `intervention_other_names`, `phases`, `status`, `min_age`, `max_age`, `sex`, `std_ages`, `primary_outcomes`, `secondary_outcomes`, `intervention_descriptions`, `brief_summary`, `detailed_description`, `eligibility_text`, `mesh_conditions`, `mesh_interventions`, `ingested_at`

- **`criteria`** — one row per individual criterion sentence. Key fields: `criterion_id`, `nct_id`, `text`, `section`, `position`, `b1_label`, `b2_label`, `b3_label`, `b2_confidence`, `b3_confidence`, `extracted_conditions`, `extracted_drugs`, `extracted_lab_values`, `extracted_thresholds`, `extracted_demographics`, `extracted_scales`, `extracted_timeframes`, `processed_at`

**API client:** Paginates against the ClinicalTrials.gov REST API v2 using `nextPageToken`. Fetches oncology trials filtered by condition keywords and status. Rate-limited at 10 req/s.

**Acceptance criteria:**
- [x] DuckDB file created at `data/processed/trials.duckdb`
- [x] Both tables exist with correct schema
- [x] Trials written to `trials` table with no null `nct_id` values
- [x] `eligibility_text` populated for >90% of records

---

### Step 3 — Criterion Splitter and Weak Labeler ✓

**Files:** `nlp/criterion_splitter.py`, `nlp/weak_labeler.py`, `label.py`

**Criterion splitter:** Scans eligibility text line by line, detecting section headers (inclusion/exclusion) via regex — including non-standard formats such as `* INCLUSION CRITERIA:`. Accumulates bullet lines into individual criterion dicts with `{text, section, position}`. Sub-points indented 3+ spaces are folded into the parent criterion rather than emitted as separate rows. Criteria shorter than 10 characters are discarded as noise.

**Weak labeler:** Assigns three noisy labels per criterion using regex pattern matching:

- **B1 (inclusion/exclusion):** derived from section header context (~90% accuracy). Confidence fixed at 0.90 for labeled rows; None rows get confidence 0.0.
- **B2 (objective/subjective):** objective patterns (numeric thresholds, named scales, binary clinical states) vs subjective patterns (judgment language, consent, willingness). Label assigned by whichever pattern type fires more. Confidence = hit ratio.
- **B3 (observable/unobservable):** EHR field patterns (lab names, diagnoses, performance scales, treatment history) vs unobservable patterns (consent, life expectancy, geographic constraints, judgment-adjective + organ/function). Same confidence mechanism.

Criteria where neither pattern type fires for a given head receive `label = None` with `confidence = 0.0`. These are not errors — they represent cases where the weak labeler has no signal. SciBERT learns to classify them from context.

**Acceptance criteria:**
- [x] Splitter correctly separates inclusion and exclusion sections
- [x] Individual criterion sentences written to `criteria` table
- [x] Weak labels computed for B1, B2, B3 on all criteria
- [x] `label.py --reprocess` rebuilds all criteria from scratch

---

### Step 4 — Multi-task SciBERT Classifier ✓

**Files:** `nlp/multitask_classifier.py`, `nlp/trainer.py`

**Model architecture:** `allenai/scibert_scivocab_uncased` (110M parameters) with three independent linear classification heads on the CLS token — one per label type (B1/B2/B3), each projecting 768 → 2 logits.

**Loss function:** Confidence-weighted multi-task cross-entropy. Per-example loss is scaled by weak label confidence, so None rows (confidence=0.0) contribute zero gradient. Task weights: B1=0.30, B2=0.30, B3=0.40. B3 weighted highest because unobservable errors are most consequential downstream.

**Class imbalance correction:** Class weights in the loss to counteract majority-class suppression. B2: subjective upweighted 3× (3:1 imbalance). B3: unobservable upweighted 8× (8:1 imbalance). B1 requires no correction.

**Training configuration:**
- Optimizer: AdamW, lr=2e-5, weight decay=0.01
- Scheduler: linear warmup over 10% of total steps
- Batch size: 32, epochs: 8, seed: 48697
- Train/val split: 80/20
- Device: Apple MPS (Metal GPU)
- Best checkpoint saved by validation macro F1 to `nlp/checkpoints/best_model.pt`

**Note on validation F1:** Computed only on conf>0 rows per head. This measures reproduction of weak-labeler patterns, not generalization to None rows. Generalization is evaluated in Step 5.

**Acceptance criteria:**
- [x] Model trains without errors on weak labels
- [x] Loss decreases monotonically across epochs with no instability
- [x] Per-class F1 reported for minority and majority class per head at each epoch
- [x] Best checkpoint saved to `nlp/checkpoints/`

---

### Step 5 — Generalization Evaluation via Hybrid Annotation ✓

**Files:** `scripts/sample_annotation.py`, `scripts/llm_annotate.py`, `scripts/review_annotations.py`, `data/annotation/sample.csv`

**Goal:** Evaluate SciBERT generalization on None rows — the only population where true generalization can be measured.

**Workflow:**
1. Sample criteria from None rows (b2_label IS NULL OR b3_label IS NULL), stratified by section. Run SciBERT inference to pre-populate predicted labels and probabilities.
2. LLM annotation: Claude labels each criterion for B2 and B3 using an explicit written rubric, with self-reported confidence (0.0–1.0).
3. Conflict flagging: rows where LLM and SciBERT disagree, or where LLM confidence < 0.80, are flagged for human review.
4. Human adjudication: reviewer examines flagged rows and confirms or overrides LLM label.
5. Metrics: per-class F1 and calibration table (SciBERT accuracy by probability decile) computed against ground truth.

**B2 rubric — Objective (1) vs Subjective (0):** Could two independent clinicians evaluate this criterion and always reach the same conclusion? Objective = specific threshold or verifiable clinical state. Subjective = requires clinical judgment, no numeric threshold.

**B3 rubric — Observable (1) vs Unobservable (0):** Can a clinician answer this criterion from a standard EHR without asking the patient or predicting the future? Observable = lab values, diagnoses, medications, documented history. Unobservable = consent, willingness, life expectancy, geographic constraints.

**Design decision:** Residual model uncertainty is propagated into the Bayesian model's prior structure rather than triggering a retrain. SciBERT probability scores modulate prior strength and marginalization width in PyMC.

**Acceptance criteria:**
- [x] Minimum 300 criteria annotated with ground truth labels for B2 and B3
- [x] Per-class F1 and calibration computed and reviewed
- [x] Design decision documented for how residual uncertainty is handled downstream

---

### Step 6 — NER Extractor ✓

**File:** `nlp/ner_extractor.py`

**Entity types extracted and method per type:**

| Entity | Column | Method |
|---|---|---|
| CONDITION | `extracted_conditions` | Trial MeSH dictionary match (`mesh_conditions` + `conditions`) |
| DRUG | `extracted_drugs` | Trial MeSH + alias dictionary match (`mesh_interventions` + `intervention_other_names`) |
| LAB_VALUE | `extracted_lab_values` | Regex over curated lab name list |
| THRESHOLD | `extracted_thresholds` | Regex: comparator + number + optional unit |
| SCALE | `extracted_scales` | Regex: named clinical scale + optional score |
| DEMOGRAPHIC | `extracted_demographics` | Regex: age/sex/BMI patterns |
| TIMEFRAME | `extracted_timeframes` | Regex: "within N days/weeks/months/years" |

**Design rationale:** MeSH-based dictionary matching for CONDITION and DRUG is higher precision than a general biomedical NER model because it is anchored to the trial's own normalized vocabulary. Regex is appropriate for the remaining types because their pattern space is syntactically bounded.

**Connection to Bayesian model:** For criteria classified as B2=Objective and B3=Observable, the Bayesian deterministic branch compares the extracted threshold against the patient's EHR value. NER performs this extraction offline so the evaluator executes a simple numeric comparison at query time.

**Processing:** Keyset pagination on `criterion_id` for incremental runs. `--reprocess` flag forces full rebuild. `--spot-check` prints a sample of extractions for manual inspection.

**Acceptance criteria:**
- [x] All criteria in the criteria table processed
- [x] Seven NER columns populated
- [x] `--spot-check` output reviewed for extraction quality

---

### Step 7 — Embedding Pipeline and ChromaDB Vector Store ✓

**Files:** `rag/embedder.py`, `rag/vector_store.py`, `embed.py`, `tests/test_embed.py`

**Model:** `sentence-transformers/all-MiniLM-L6-v2` — 384-dim embeddings, 256-token max, ~22MB, designed for cosine similarity. Unit-normalised outputs enable dot-product cosine similarity without explicit normalisation at query time.

**Embedding strategy — mean pooling of overlapping chunks:**

A composite document per trial (brief_title + brief_summary + eligibility_text) typically runs 500–700 tokens, well above the model's 256-token limit. Single-pass truncation discards the eligibility criteria entirely, which are the primary patient-matching signal. The solution is token-boundary chunking with mean pooling:

| Parameter | Value |
|---|---|
| Chunk size | 256 tokens (model max) |
| Overlap | 32 tokens (preserves cross-boundary context) |
| Pooling | Mean of per-chunk unit vectors |
| Normalisation | Explicit L2 after pooling (mean of unit vectors ≠ unit vector) |

78% of 15,010 trials (11,690) exceeded 256 tokens and required chunking. Most produced 2–3 chunks.

**Composite document ordering:** `brief_title → brief_summary → eligibility_text`

Order matters under mean pooling because the summary contains trial-specific content (drug names, biomarkers, study design) that drives self-retrieval precision, while eligibility text is often generic and shared across many similar trials. Putting the summary before the eligibility text ensures its signal is captured even when document length pushes it past the first chunk boundary.

**Implementation architecture (`rag/embedder.py`):**

- `_chunk_text(text, tokenizer)` — tokenises with the model's own tokenizer so chunk boundaries align with what the encoder sees; decodes chunks back to strings for encoding
- `_embed_and_pool(chunks, model)` — encodes all chunks with `normalize_embeddings=True`, mean-pools, applies explicit L2 normalisation
- `build_corpus(rows)` — constructs composite texts and metadata dicts from DuckDB rows
- `embed_corpus(texts, batch_size, show_progress)` — partitions documents into a fast batch path (≤256 tokens, encoded together) and a chunked path (>256 tokens, encoded one at a time with tqdm progress). Runtime: ~4 minutes for 15,010 trials on CPU.

**ChromaDB (`rag/vector_store.py`):**

- Collection: `oncology_trials`, `hnsw:space=cosine`
- `upsert` (not `add`) — safe to rerun; overwrites rather than duplicates
- Metadata stored: `nct_id`, `conditions`, `phases`, `status` — enables filtered retrieval
- Writes in 500-doc chunks to avoid memory spikes
- Score returned as `1.0 - cosine_distance` (cosine similarity)

**CLI (`embed.py`):**
```bash
python embed.py                  # embed all trials not yet in ChromaDB
python embed.py --reprocess      # re-embed all (upsert overwrites)
python embed.py --spot-check     # query with sample patient profile and exit
```

**Test suite (`tests/test_embed.py`) — 12 tests, all passing:**

*Class A — Embedding geometry (no ChromaDB required):*
- A1: Unit norm (catches missing L2 normalisation after pooling)
- A2: 384-dim output
- A3: Semantic ordering (synonym closer than unrelated term)
- A4: Determinism (identical input → identical output)
- A5: Out-of-domain floor (clinical text not close to unrelated domains)

*Class B — Retrieval correctness (requires populated ChromaDB):*
- B1: Disease area precision — all top-10 results for an ovarian cancer query are gynecologic oncology trials
- B2: Self-retrieval — NCT00127920's brief title returns it within the top 10 of 15,010 trials
- B3: Disease type separation — prostate cancer query returns no gynecologic trials in top 5
- B4: Recurrent vs naive differentiation — Jaccard < 0.5 between result sets for different treatment lines
- B5: Metadata filter — status filter returns only RECRUITING trials
- B6: Score floor — top result for any reasonable oncology query scores > 0.5
- B7: Biomarker specificity — HER2+ breast cancer query returns no off-target cancer types in top 5

**Acceptance criteria for Step 7:**
- [x] ChromaDB collection created at `data/processed/chroma`
- [x] All ~15k trials embedded and stored 
- [x] Embedding pipeline completes on CPU
- [x] 12/12 unit tests pass (geometry + retrieval correctness)

---

### Step 8 — Ollama Setup and RAG Pipeline

**Files:** `rag/retriever.py`, `rag/generator.py`, `rag/pipeline.py`

**Local LLM setup:**
```bash
# Verify Ollama is running
ollama list
ollama run mistral  # test interactively first
```

**RAG pipeline:**

```python
# rag/pipeline.py
from llama_index.core import VectorStoreIndex, Settings
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from sentence_transformers import CrossEncoder

# Configure local models — no API costs
Settings.llm = Ollama(model="mistral", request_timeout=120.0)
Settings.embed_model = HuggingFaceEmbedding(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Cross-encoder for reranking
reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

def query_trials(
    query: str,
    top_k: int = 20,
    rerank_top_k: int = 5
) -> list[dict]:
    results = index.as_retriever(similarity_top_k=top_k).retrieve(query)
    pairs = [(query, r.text) for r in results]
    scores = reranker.predict(pairs)
    reranked = sorted(zip(results, scores), key=lambda x: x[1], reverse=True)
    return reranked[:rerank_top_k]
```

**Acceptance criteria for Step 8:**
- [ ] Ollama running locally with Mistral responding to prompts
- [ ] RAG pipeline returns relevant trials for sample clinical queries
- [ ] Reranking improves result order vs baseline retrieval
- [ ] End-to-end query latency <30 seconds on local hardware

---

### Step 9 — RAGAS Evaluation

**File:** `rag/evaluate_ragas.py`

**Metrics to compute:**

| Metric | What it measures |
|---|---|
| Faithfulness | Generated answers grounded in retrieved context |
| Answer relevancy | Response addresses the query |
| Context precision | Retrieved trials are relevant |
| Context recall | Relevant trials are retrieved |

**Evaluation dataset:** 50–100 manually constructed clinical query / expected answer pairs covering common query patterns such as "What phase 3 trials are recruiting for NSCLC patients with prior platinum therapy?"

```python
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall
)
```

**Acceptance criteria for Step 9:**
- [ ] 50+ evaluation Q&A pairs created and committed to `data/labeled/ragas_eval.json`
- [ ] RAGAS scores computed and logged
- [ ] Faithfulness > 0.70
- [ ] Context precision > 0.60
- [ ] Results documented in `reports/rag_evaluation.md`

---

### Step 10 — PyMC Bayesian Eligibility Model

**Files:** `bayesian/criterion_evaluator.py`, `bayesian/eligibility_model.py`, `bayesian/uncertainty.py`

**Estimand:**
```
P(eligible) = ∏ P(meets criterion_i)
```

Where each factor is:
- **0 or 1** (deterministic) if B2=Objective and B3=Observable
- **Sampled Beta** (uncertain) if B2=Subjective
- **Marginalized** (unobservable) if B3=Unobservable

```python
# bayesian/eligibility_model.py
import pymc as pm
import numpy as np
from dataclasses import dataclass

@dataclass
class Criterion:
    text: str
    label_inclusion: int      # B1
    label_objective: int      # B2
    label_observable: int     # B3
    threshold_value: float    # from NER
    patient_value: float      # from patient profile

def compute_eligibility_posterior(
    criteria: list[Criterion],
    patient_profile: dict,
    n_samples: int = 2000
) -> dict:

    with pm.Model() as model:
        criterion_probs = []

        for i, criterion in enumerate(criteria):
            if criterion.label_objective == 1 and criterion.label_observable == 1:
                # Deterministic
                meets = evaluate_objective_criterion(criterion, patient_profile)
                p = pm.math.constant(float(meets))

            elif criterion.label_objective == 0:
                # Subjective — uncertain, model as Beta
                hedging_strength = estimate_hedging(criterion.text)
                alpha = 2.0 * (1 - hedging_strength)
                beta_param = 2.0 * hedging_strength
                p = pm.Beta(f'p_subj_{i}', alpha=alpha, beta=beta_param)

            else:
                # Unobservable — marginalize with uninformative prior
                p = pm.Beta(f'p_unobs_{i}', alpha=1, beta=1)

            criterion_probs.append(p)

        if criterion_probs:
            p_eligible = pm.math.prod(pm.math.stack(criterion_probs))
        else:
            p_eligible = pm.math.constant(1.0)

        trace = pm.sample(n_samples, tune=1000,
                         progressbar=False, return_inferencedata=True)

    posterior_samples = trace.posterior['p_eligible'].values.flatten()

    return {
        'mean': float(np.mean(posterior_samples)),
        'ci_lower': float(np.percentile(posterior_samples, 2.5)),
        'ci_upper': float(np.percentile(posterior_samples, 97.5)),
        'samples': posterior_samples,
        'n_deterministic': sum(1 for c in criteria
                               if c.label_objective == 1
                               and c.label_observable == 1),
        'n_subjective': sum(1 for c in criteria if c.label_objective == 0),
        'n_unobservable': sum(1 for c in criteria if c.label_observable == 0)
    }
```

**Acceptance criteria for Step 10:**
- [ ] Model runs without errors on a sample patient and trial
- [ ] Posterior mean and 95% CI returned correctly
- [ ] Deterministic criteria produce point-mass posteriors at 0 or 1
- [ ] Subjective criteria produce diffuse posteriors
- [ ] Runtime acceptable (<60 seconds per trial assessment)

---

### Step 11 — Streamlit Interface

**File:** `app/streamlit_app.py`

**Four panels:**

**Panel 1 — Trial Search**
- Natural language query input
- Filter controls: cancer type, phase, status, intervention type
- Results table: trial ID, title, conditions, relevance score
- Click trial to load into Panel 3

**Panel 2 — Patient Profile**
Structured input for:
- Demographics: age, sex, ECOG score
- Diagnosis: cancer type, stage, histology
- Lab values: HbA1c, eGFR, CBC components, LFTs
- Treatment history: prior chemo, immunotherapy, radiation
- Current medications

**Panel 3 — Eligibility Assessment**
- Selected trial display: title, NCT ID, phase, status
- Bayesian eligibility score displayed as probability gauge (0–100%)
- 95% credible interval displayed as range
- Uncertainty decomposition:
  - N criteria deterministic / N subjective / N unobservable
- Criterion-by-criterion breakdown table

**Panel 4 — Explainability**
- Primary sources of uncertainty
- Sensitivity analysis: "If subjective criterion X is assumed met, eligibility rises from 42% to 68%"
- Which unobservable criteria would most change the score if known

**Acceptance criteria for Step 11:**
- [ ] All four panels render without errors
- [ ] Trial search returns results from ChromaDB
- [ ] Patient profile input persists in session state
- [ ] Bayesian scorer runs on selected trial and displays results
- [ ] Criterion table correctly displays B1/B2/B3 labels and patient status

---

### Step 12 — Docker Containerization (Ollama Service)

**File:** `docker-compose.yml`

```yaml
version: '3.8'

services:
  ollama:
    image: ollama/ollama:latest
    ports:
      - "11434:11434"
    volumes:
      - ollama_data:/root/.ollama
    restart: unless-stopped

  app:
    build: .
    ports:
      - "8501:8501"
    environment:
      - OLLAMA_HOST=http://ollama:11434
    depends_on:
      - ollama
    volumes:
      - ./data:/app/data

volumes:
  ollama_data:
```

**Acceptance criteria for Step 12:**
- [ ] `docker-compose up` starts both services without errors
- [ ] Streamlit app accessible at `localhost:8501`
- [ ] RAG pipeline connects to Ollama container successfully
- [ ] End-to-end query works inside Docker environment

---

### Step 13 — README and Portfolio Documentation

**Contents:**
- Project overview and motivation
- Architecture diagram
- Setup instructions (local and Docker)
- Scalability note (three-tier architecture)
- Example queries and screenshots
- Evaluation results (RAGAS scores, NLP F1 per subtask)
- Known limitations and future work

---

## Evaluation Framework

### NLP Classifier
- Per-subtask Precision, Recall, F1 on held-out labeled set
- Confusion matrix analysis
- Error analysis on misclassified criteria qualitatively

### RAG (RAGAS)
- Faithfulness, Answer Relevancy, Context Precision, Context Recall
- Evaluated on 50–100 manually constructed Q&A pairs

### Bayesian Model
- Calibration: do 95% CIs contain true eligibility 95% of the time?
- Compare posterior mean vs deterministic eligibility on fully observable criteria
- Sensitivity analysis: effect of subjective criterion assumptions on posterior

### End-to-End
- 20–30 manually constructed patient-trial pairs with known ground truth eligibility
- Precision and recall of system recommendations

---

## Key Design Decisions

1. **DuckDB over SQLite** — native JSON/Parquet support, columnar analytics performance, GCS/S3 compatibility for scaling
2. **Local Ollama over OpenAI API** — zero cost, no rate limits, no data leaving local environment
3. **Multi-task SciBERT** — shared encoder reduces parameters and encourages consistent representations across three related tasks
4. **Weighted loss (0.30/0.30/0.40)** — B3 weighted highest because unobservable errors are most costly to the Bayesian model
5. **Bayesian scorer over deterministic** — eligibility is inherently uncertain; credible intervals are clinically meaningful
6. **Cross-encoder reranking** — improves retrieval precision over bi-encoder alone at acceptable latency cost
7. **Enriched patient profile baseline** — mean of training features used as integrated gradients baseline rather than zero vector
8. **Generator and Bayesian scorer have non-overlapping roles** — the LLM verdict and the posterior probability answer adjacent but architecturally distinct questions about the same trial-patient pair. The generator (Mistral-7B) reads the full trial document and patient description together and produces a prose explanation in clinical language — its output is consumed by the clinician at the Streamlit interface. It cannot produce calibrated probabilities; a language model has no mechanism for computing a posterior over a product of independent criterion probabilities. The Bayesian scorer (PyMC) operates on structured inputs — SciBERT B2/B3 labels and NER-extracted thresholds — and computes a posterior P(eligible) with a 95% credible interval. Its credible intervals communicate *why* uncertainty exists structurally: how many criteria are subjective, how many are unobservable. It has no language understanding and no access to the trial text. Each component does something the other is architecturally incapable of. In the Streamlit interface these map to Panel 3 (probability gauge + credible interval from PyMC) and Panel 4 (criterion-level explainability + LLM narrative). The LLM verdict currently has zero formal weight in the Bayesian posterior — it is a display artifact for the clinician, not an input to the model. Connecting them (e.g. using the LLM's per-criterion uncertainty to modulate Beta prior strength) is a meaningful future extension but outside the scope of this project.

---

## Connection to UCI Diabetes Portfolio Project

This project directly extends several analytical threads from the UCI Diabetes readmission prediction project:

- **Bayesian eligibility scorer** extends the Bayesian inference and experimental design work from Part 1
- **NLP classification** extends the feature engineering and interpretability methodology
- **RAG pipeline** introduces a genuinely new LLM/GenAI capability not present in the UCI project
- **Clinical domain** — oncology trial matching is adjacent to the clinical data science work at XYZ Biosciences, grounding design decisions in real domain expertise
