# Project A: Clinical Trial Eligibility Intelligence System

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
- `nctId` → trial identifier
- `eligibilityModule.eligibilityCriteria` → full criteria free text
- `eligibilityModule.minimumAge` / `maximumAge`
- `conditionsModule.conditions` → cancer types
- `interventionsModule.interventions` → drugs and treatments
- `designModule.phases` → trial phase
- `statusModule.overallStatus` → recruiting status
- `outcomesModule.primaryOutcomes` → primary endpoints

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
│   └── test_bayesian.py
├── notebooks/
│   └── exploration.ipynb       # EDA and prototyping
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
| Criterion splitting | spaCy, regex | Free |
| Weak supervision | regex, lookup tables | Free |
| Multi-task NLP | HuggingFace Transformers, PyTorch | Free |
| NER | SciBERT fine-tuned | Free |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 | Free |
| Vector store | ChromaDB (local persistent) | Free |
| LLM | Mistral-7B via Ollama (local) | Free |
| RAG orchestration | LlamaIndex | Free |
| Reranking | cross-encoder/ms-marco-MiniLM-L-6-v2 | Free |
| RAG evaluation | RAGAS | Free |
| Bayesian modeling | PyMC | Free |
| Experiment tracking | Weights & Biases (free tier) | Free |
| Frontend | Streamlit | Free |
| Containerization | Docker + docker-compose (Ollama service) | Free |

---

## Implementation Steps

### Step 1 — Repository Scaffold and Environment Setup

**Tasks:**
- Initialize git repo and `.gitignore`
- Create conda or venv environment
- Install core dependencies
- Create `.env.example` with placeholder keys
- Create folder structure as above

**Key packages to install first:**
```bash
pip install duckdb requests python-dotenv spacy
pip install torch transformers sentence-transformers
pip install chromadb llama-index ragas
pip install pymc streamlit wandb
pip install pytest
```

**Install Ollama and pull Mistral:**
```bash
brew install ollama
ollama pull mistral
```

---

### Step 2 — DuckDB Schema and API Ingestion Pipeline

**Files:** `ingestion/database.py`, `ingestion/api_client.py`, `ingestion/parser.py`

**DuckDB schema — two tables:**

```python
# ingestion/database.py
import duckdb

def initialize_database(db_path: str = "data/processed/trials.duckdb"):
    con = duckdb.connect(db_path)

    con.execute("""
        CREATE TABLE IF NOT EXISTS trials (
            nct_id VARCHAR PRIMARY KEY,
            conditions VARCHAR[],
            interventions VARCHAR[],
            phases VARCHAR[],
            status VARCHAR,
            min_age VARCHAR,
            max_age VARCHAR,
            primary_outcomes VARCHAR[],
            eligibility_text TEXT,
            ingested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    con.execute("""
        CREATE TABLE IF NOT EXISTS criteria (
            criterion_id VARCHAR PRIMARY KEY,  -- nct_id + index
            nct_id VARCHAR REFERENCES trials(nct_id),
            criterion_text TEXT,
            -- B1/B2/B3 classification outputs (populated by NLP layer)
            label_inclusion INTEGER,
            label_objective INTEGER,
            label_observable INTEGER,
            confidence_inclusion FLOAT,
            confidence_objective FLOAT,
            confidence_observable FLOAT,
            -- NER outputs
            extracted_conditions VARCHAR[],
            extracted_drugs VARCHAR[],
            extracted_lab_values VARCHAR[],
            extracted_thresholds VARCHAR[],
            extracted_demographics VARCHAR[],
            processed_at TIMESTAMP
        )
    """)

    return con
```

**API client with pagination:**

```python
# ingestion/api_client.py
import requests
import time
from typing import Generator

BASE_URL = "https://beta.clinicaltrials.gov/api/v2/studies"

def fetch_oncology_trials(
    page_size: int = 1000,
    max_trials: int = 15000
) -> Generator[dict, None, None]:

    params = {
        "query.cond": "cancer OR oncology OR neoplasm OR tumor OR carcinoma",
        "filter.overallStatus": "RECRUITING,ACTIVE_NOT_RECRUITING,COMPLETED",
        "pageSize": page_size,
        "format": "json",
        "fields": "NCTId,EligibilityCriteria,Condition,InterventionName,"
                  "Phase,OverallStatus,MinimumAge,MaximumAge,PrimaryOutcome"
    }

    fetched = 0
    next_page_token = None

    while fetched < max_trials:
        if next_page_token:
            params["pageToken"] = next_page_token

        response = requests.get(BASE_URL, params=params)
        response.raise_for_status()
        data = response.json()

        studies = data.get("studies", [])
        for study in studies:
            yield study
            fetched += 1

        next_page_token = data.get("nextPageToken")
        if not next_page_token:
            break

        time.sleep(0.1)  # respect rate limit
```

**Acceptance criteria for Step 2:**
- [ ] DuckDB file created at `data/processed/trials.duckdb`
- [ ] Both tables exist with correct schema
- [ ] API client fetches at least 100 trials without error
- [ ] Trials written to `trials` table with no null `nct_id` values
- [ ] `eligibility_text` populated for >90% of records

---

### Step 3 — Criterion Splitter and Weak Labeler

**Files:** `nlp/criterion_splitter.py`, `nlp/weak_labeler.py`

**Criterion splitter — split eligibility text blob into individual sentences:**

```python
# nlp/criterion_splitter.py
import re
from dataclasses import dataclass

@dataclass
class RawCriterion:
    nct_id: str
    criterion_id: str
    text: str
    section_context: str  # "inclusion" or "exclusion" from header
    position: int

def split_criteria(nct_id: str, eligibility_text: str) -> list[RawCriterion]:
    inclusion_pattern = re.compile(r'inclusion criteria[:\s]*', re.IGNORECASE)
    exclusion_pattern = re.compile(r'exclusion criteria[:\s]*', re.IGNORECASE)
    # Split into sections, then split sections into individual sentences
    # Filter out sentences shorter than 10 characters (noise)
    ...
```

**Weak labeler — generate noisy training labels without manual annotation:**

```python
# nlp/weak_labeler.py

# B1: Inclusion/Exclusion from section header context (~90% accuracy)
def weak_label_inclusion(criterion: RawCriterion) -> int:
    return 1 if criterion.section_context == "inclusion" else 0

# B2: Objective/Subjective from regex patterns (~75% accuracy)
OBJECTIVE_PATTERNS = [
    r'\d+\.?\d*\s*(%|mg|ml|mmol|years|days|months)',
    r'(ECOG|NYHA|CTCAE|WHO)\s*(performance|status|grade|class)',
    r'(age|bmi|weight|height)\s*[<>≤≥]=?\s*\d+'
]
SUBJECTIVE_PATTERNS = [
    r'(adequate|significant|clinically|appropriate|reasonable)',
    r'(life expectancy|prognosis|functional status)',
    r'(willing|able|capable|consent)'
]

# B3: Observable/Unobservable from EHR field lookup (~70% accuracy)
STANDARD_EHR_FIELDS = {
    'lab_values': ['hba1c', 'egfr', 'creatinine', 'hemoglobin',
                   'platelet', 'wbc', 'alt', 'ast', 'bilirubin'],
    'demographics': ['age', 'sex', 'gender', 'bmi', 'weight'],
    'diagnoses': ['diagnosis', 'cancer', 'tumor', 'malignancy'],
    'medications': ['treatment', 'therapy', 'prior', 'medication']
}
UNOBSERVABLE_SIGNALS = [
    'willing', 'consent', 'geographic', 'life expectancy',
    'investigator', 'enrollment', 'participation'
]
```

**Acceptance criteria for Step 3:**
- [ ] `split_criteria()` correctly separates inclusion and exclusion sections
- [ ] Individual criterion sentences written to `criteria` table
- [ ] Weak labels computed for B1, B2, B3 on all criteria
- [ ] Manual inspection of 20 random criteria shows reasonable label quality
- [ ] No criteria with empty `criterion_text`

---

### Step 4 — Multi-task SciBERT Classifier

**File:** `nlp/multitask_classifier.py`, `nlp/trainer.py`

**Model architecture — shared encoder with three classification heads:**

```python
# nlp/multitask_classifier.py
from transformers import AutoModel
import torch
import torch.nn as nn

class CriterionClassifier(nn.Module):
    def __init__(
        self,
        base_model: str = 'allenai/scibert_scivocab_uncased',
        dropout: float = 0.1
    ):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(base_model)
        hidden_size = self.encoder.config.hidden_size  # 768

        self.dropout = nn.Dropout(dropout)

        # Three independent classification heads
        self.head_inclusion = nn.Linear(hidden_size, 2)   # B1
        self.head_objective = nn.Linear(hidden_size, 2)   # B2
        self.head_observable = nn.Linear(hidden_size, 2)  # B3

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids, attention_mask=attention_mask)
        cls = self.dropout(outputs.last_hidden_state[:, 0, :])

        return {
            'inclusion': self.head_inclusion(cls),
            'objective': self.head_objective(cls),
            'observable': self.head_observable(cls)
        }

# Weighted multi-task loss
# B3 weighted highest — unobservable errors are most consequential
def compute_loss(logits: dict, labels: dict) -> torch.Tensor:
    ce = nn.CrossEntropyLoss()
    return (
        0.30 * ce(logits['inclusion'], labels['inclusion']) +
        0.30 * ce(logits['objective'], labels['objective']) +
        0.40 * ce(logits['observable'], labels['observable'])
    )
```

**Training strategy:**
- Phase 1: Train on weak labels (~80K criteria from 15K trials)
- Phase 2: Fine-tune on 300–500 manually labeled criteria per subtask
- Optimizer: AdamW with linear warmup, weight decay 0.01
- Epochs: 5–10 with early stopping on validation F1
- Experiment tracking: Weights & Biases free tier

**Acceptance criteria for Step 4:**
- [ ] Model trains without errors on weak labels
- [ ] Validation F1 > 0.75 on B1 (inclusion/exclusion) — strongest signal
- [ ] Validation F1 > 0.65 on B2 (objective/subjective)
- [ ] Validation F1 > 0.60 on B3 (observable/unobservable)
- [ ] Model checkpoint saved to `nlp/checkpoints/`
- [ ] W&B run logged with training curves

---

### Step 5 — Manual Labeling of Validation Set

**Files:** `data/labeled/criteria_b1.csv`, `criteria_b2.csv`, `criteria_b3.csv`

**Goal:** Label 300–500 criteria per subtask for reliable evaluation.

**Label schema:**

| Column | Description |
|---|---|
| `criterion_id` | Foreign key to `criteria` table |
| `criterion_text` | The raw criterion sentence |
| `label` | 0 or 1 per subtask |
| `notes` | Optional — flag ambiguous cases |

**B1 labeling guide:**
- 1 = Inclusion (patient MUST have this)
- 0 = Exclusion (patient must NOT have this)

**B2 labeling guide:**
- 1 = Objective (has a numeric threshold or standardized scale)
- 0 = Subjective (requires clinical judgment)

**B3 labeling guide:**
- 1 = Observable (can be assessed from standard EHR data)
- 0 = Unobservable (requires data not in standard EHR)

**Acceptance criteria for Step 5:**
- [ ] 300+ criteria labeled for each of B1, B2, B3
- [ ] CSV files committed to `data/labeled/`
- [ ] Inter-rater reliability check on 50 criteria (if feasible)
- [ ] Fine-tuned model re-evaluated on manual labels — F1 per subtask documented

---

### Step 6 — NER Extractor

**File:** `nlp/ner_extractor.py`

**Entity types to extract:**

| Entity | Examples |
|---|---|
| CONDITION | "type 2 diabetes", "NSCLC", "metastatic melanoma" |
| DRUG | "pembrolizumab", "metformin", "carboplatin" |
| LAB_VALUE | "HbA1c", "eGFR", "ALT", "platelet count" |
| THRESHOLD | "> 7.5%", "≥ 30 mL/min", "< 40 kg/m²" |
| DEMOGRAPHIC | "age 18–75", "female", "BMI < 35" |
| SCALE | "ECOG 0–2", "NYHA Class II", "CTCAE Grade 3" |
| TIMEFRAME | "within 6 months", "prior 3 years" |

**Model:** Fine-tune `d4data/biomedical-ner-all` or `allenai/scibert_scivocab_uncased` with NER head. Training data: BC5CDR corpus (freely available) for pretraining, then adapt to eligibility criteria domain.

**Acceptance criteria for Step 6:**
- [ ] NER model extracts entities from sample criteria
- [ ] Extracted entities written to `criteria` table columns
- [ ] Manual inspection of 20 criteria shows reasonable extraction quality
- [ ] Entity types present in >80% of criteria with relevant content

---

### Step 7 — Embedding Pipeline and ChromaDB Vector Store

**Files:** `rag/embedder.py`, `rag/vector_store.py`

```python
# rag/embedder.py
from sentence_transformers import SentenceTransformer
import chromadb

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

def build_vector_store(
    trials_df,
    collection_name: str = "oncology_trials"
):
    model = SentenceTransformer(MODEL_NAME)
    client = chromadb.PersistentClient(path="data/processed/chroma")
    collection = client.get_or_create_collection(collection_name)

    texts = trials_df['eligibility_text'].tolist()
    embeddings = model.encode(texts, batch_size=64, show_progress_bar=True)

    collection.add(
        embeddings=embeddings.tolist(),
        documents=texts,
        metadatas=[{
            'nct_id': row['nct_id'],
            'conditions': str(row['conditions']),
            'phase': str(row['phases']),
            'status': row['status']
        } for _, row in trials_df.iterrows()],
        ids=trials_df['nct_id'].tolist()
    )

    return collection
```

**Acceptance criteria for Step 7:**
- [ ] ChromaDB collection created at `data/processed/chroma`
- [ ] All 15,000 trials embedded and stored
- [ ] Sample query returns semantically relevant trials
- [ ] Embedding pipeline completes in <30 minutes on CPU

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

---

## Connection to UCI Diabetes Portfolio Project

This project directly extends several analytical threads from the UCI Diabetes readmission prediction project:

- **Bayesian eligibility scorer** extends the Bayesian inference and experimental design work from Part 1
- **NLP classification** extends the feature engineering and interpretability methodology
- **RAG pipeline** introduces a genuinely new LLM/GenAI capability not present in the UCI project
- **Clinical domain** — oncology trial matching is adjacent to the clinical data science work at XYZ Biosciences, grounding design decisions in real domain expertise
