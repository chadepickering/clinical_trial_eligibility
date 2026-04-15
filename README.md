# Clinical Trial Eligibility Intelligence System

A clinical decision support system that combines RAG, transformer-based NLP, and Bayesian uncertainty quantification to match patients to oncology clinical trials. Given a patient profile, the system retrieves relevant trials, classifies eligibility criteria along three dimensions, and computes a posterior probability of eligibility with credible intervals reflecting uncertainty from subjective and unobservable criteria.

---

## Architecture

```
ClinicalTrials.gov REST API (oncology subset ~15,000 trials)
        ↓
Ingestion Pipeline (Python + requests)
        ↓
DuckDB (local analytical store — raw trial metadata + criteria text)
        ↓
┌─────────────────────────────────────────────────────┐
│  NLP Layer (HuggingFace + PyTorch)                  │
│  Multi-task SciBERT with three classification heads │
│  B1: Inclusion vs Exclusion                         │
│  B2: Objective vs Subjective                        │
│  B3: Observable vs Unobservable                     │
└─────────────────────────────────────────────────────┘
        ↓
DuckDB (structured criteria store — labeled criterion objects)
        ↓
┌──────────────────┐      ┌──────────────────────────┐
│  Embedding Layer │      │  Named Entity            │
│  sentence-       │      │  Recognition             │
│  transformers    │      │  (SciBERT NER head)      │
│  all-MiniLM-L6   │      │  Extract: conditions,    │
└──────────────────┘      │  drugs, lab values,      │
        ↓                 │  thresholds, demographics│
ChromaDB                  └──────────────────────────┘
(vector store)                      ↓
        ↓                    DuckDB structured
        └──────────┬──────── entity store
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
│  - Deterministic evaluation (B2=Objective,          │
│    B3=Observable)                                   │
│  - Beta prior (B2=Subjective)                       │
│  - Marginalization (B3=Unobservable)                │
│  - Posterior P(eligible) with credible intervals    │
└─────────────────────────────────────────────────────┘
        ↓
┌─────────────────────────────────────────────────────┐
│  Streamlit Interface                                │
│  - Trial search (natural language query)            │
│  - Patient profile input                            │
│  - Eligibility assessment with uncertainty          │
│  - Criterion-by-criterion breakdown                 │
│  - Explainability outputs                           │
└─────────────────────────────────────────────────────┘
```

---

## Dataset

- **Source:** ClinicalTrials.gov REST API v2
- **Endpoint:** `https://beta.clinicaltrials.gov/api/v2/studies`
- **Scope:** Oncology trials (~15,000 for development)
- **Access:** No credentials required
- **Format:** JSON, paginated at 1,000 records per page

**Key fields extracted:**

| Field | Description |
|---|---|
| `nctId` | Trial identifier |
| `eligibilityModule.eligibilityCriteria` | Full criteria free text |
| `eligibilityModule.minimumAge` / `maximumAge` | Age constraints |
| `conditionsModule.conditions` | Cancer types |
| `interventionsModule.interventions` | Drugs and treatments |
| `designModule.phases` | Trial phase |
| `statusModule.overallStatus` | Recruiting status |
| `outcomesModule.primaryOutcomes` | Primary endpoints |

---

## Scalability Path

| Stage | Storage |
|---|---|
| Development | API → local DuckDB (~15K trials) |
| Staging | API → GCS → DuckDB reading from GCS |
| Production | Streaming API → GCS → Spark/BigQuery |

DuckDB supports GCS/S3 reads natively — the swap from local to cloud requires no changes to the query layer, only connection configuration.

---

## Project Structure

```
clinical_trial_eligibility/
├── data/
│   ├── raw/                    # gitignored — API JSON responses
│   ├── processed/              # gitignored — DuckDB files, parquet
│   └── labeled/                # manually labeled validation sets
├── ingestion/
│   ├── api_client.py           # ClinicalTrials.gov API wrapper
│   ├── parser.py               # JSON → structured DuckDB records
│   └── database.py             # DuckDB connection and schema
├── nlp/
│   ├── criterion_splitter.py   # split criteria blob into sentences
│   ├── weak_labeler.py         # regex/heuristic weak supervision
│   ├── multitask_classifier.py # SciBERT multi-task model
│   ├── ner_extractor.py        # clinical entity extraction
│   ├── trainer.py              # training loop with W&B logging
│   └── evaluate.py             # F1 evaluation per subtask
├── rag/
│   ├── embedder.py             # sentence-transformers embedding
│   ├── vector_store.py         # ChromaDB operations
│   ├── retriever.py            # retrieval + reranking pipeline
│   ├── generator.py            # Mistral-7B via Ollama
│   ├── pipeline.py             # end-to-end RAG orchestration
│   └── evaluate_ragas.py       # RAGAS evaluation suite
├── bayesian/
│   ├── criterion_evaluator.py  # patient vs criterion matching
│   ├── eligibility_model.py    # PyMC Bayesian model
│   └── uncertainty.py          # credible interval computation
├── app/
│   └── streamlit_app.py        # Streamlit interface
├── deploy/
│   └── README.md               # deployment notes
├── notebooks/
│   └── exploration.ipynb       # EDA and prototyping
├── tests/
│   ├── test_api_client.py
│   ├── test_classifier.py
│   └── test_bayesian.py
├── .env.example
├── .gitignore
├── requirements.txt
├── README.md
└── docker-compose.yml
```

---

## Setup

```bash
git clone <repo>
cd clinical_trial_eligibility
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Start Ollama (for Mistral-7B generation)
docker compose up -d
ollama pull mistral
```
