"""
Batch evaluation harness — Stage 1 (Bayesian) + Stage 2 (Mistral).

Runs a synthetic patient cohort against a set of top-N trials per patient,
saves results to Parquet, then optionally runs Mistral on a filtered subset
for disagreement analysis.

Two-stage design
----------------
Stage 1 (Bayesian sweep):
    - For each patient × trial pair: embed query, retrieve top-N from ChromaDB,
      run evaluate_all_criteria + compute_eligibility_posterior.
    - Fast: skips PyMC sampling for short-circuited trials.
    - Output: data/eval/stage1_bayesian.parquet

Stage 2 (Mistral spot-check):
    - Reads stage1 output.
    - Filters to non-trivial pairs (coverage >= threshold, not hard-disqualified
      at coverage gate, OR a random sample of disqualified pairs for calibration).
    - Runs assess_trial on each selected pair.
    - Output: data/eval/stage2_mistral.parquet

Stage 3 (Aggregate analysis):
    - Reads both Parquet files and joins them.
    - Computes disagreement rates, dominant fail patterns, coverage distribution.
    - Prints a summary report to stdout and saves to data/eval/stage3_report.txt

Usage
-----
    # Stage 1 only (fast, can run overnight):
    python scripts/batch_eval_harness.py --stage 1

    # Stage 2 (requires stage 1 complete, requires Ollama running):
    python scripts/batch_eval_harness.py --stage 2

    # Stage 3 analysis only:
    python scripts/batch_eval_harness.py --stage 3

    # All stages sequentially:
    python scripts/batch_eval_harness.py --stage all

    # Override defaults:
    python scripts/batch_eval_harness.py --stage 1 --trials-per-patient 100 --n-draws 500

Key parameters (see CONFIG below or pass as CLI flags):
    --trials-per-patient  N  : top-N trials retrieved per patient (default 200)
    --n-draws             N  : PyMC prior predictive draws (default 500)
    --mistral-sample-rate F  : fraction of non-trivial pairs sent to Mistral (default 0.10)
    --coverage-threshold  F  : min fraction evaluable to pass coverage gate (default 0.30)
    --min-criteria        N  : min total criteria for coverage gate (default 5)
    --seed                N  : random seed for sampling (default 42)
"""

import argparse
import os
import sys
import time
import json
import random
import logging
from collections import Counter, defaultdict
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pandas as pd
import duckdb

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Output paths
# ---------------------------------------------------------------------------

EVAL_DIR        = os.path.join(PROJECT_ROOT, "data", "eval")
STAGE1_PATH     = os.path.join(EVAL_DIR, "stage1_bayesian.parquet")
STAGE2_PATH     = os.path.join(EVAL_DIR, "stage2_mistral.parquet")
STAGE3_PATH     = os.path.join(EVAL_DIR, "stage3_report.txt")
PATIENTS_PATH   = os.path.join(EVAL_DIR, "patients.json")

# ---------------------------------------------------------------------------
# Synthetic patient cohort
#
# 30 profiles spanning the key axes that drive criterion routing:
#   - Sex (female-dominant; some male for andrology/prostate trials)
#   - Age (young adult through elderly)
#   - ECOG 0-3
#   - Prior chemo (yes/no), prior RT (yes/no)
#   - Brain mets (yes/no)
#   - Pregnant (yes/no for reproductive-age females)
#   - Lab values: broad range including below-threshold values that should
#     trigger deterministic fails in well-labeled criteria
#   - Cancer type: mix of common oncology types to vary semantic retrieval
# ---------------------------------------------------------------------------

PATIENTS: list[dict] = [
    # ---- Ovarian cancer cohort (8 patients) --------------------------------
    {
        "patient_id": "OV_01",
        "description": "Female, 52yo. Stage III ovarian serous carcinoma. No prior chemotherapy or radiotherapy. ECOG 1. Karnofsky 80%. Brain metastases: none. Pregnant: no. Labs: Platelets 180,000 /mm³, Hgb 12.5 g/dL, ANC 2,800 /mm³, Creatinine 0.9 mg/dL, Bilirubin 0.7 mg/dL, ALT 28 U/L, AST 22 U/L, LVEF 62%.",
        "profile": {"age": 52, "sex": "female", "ecog": 1, "karnofsky": 80, "cancer_type": "stage III ovarian serous carcinoma", "prior_chemo": False, "prior_rt": False, "brain_mets": False, "pregnant": False, "lab_values": {"platelet_count": 180000, "hemoglobin": 12.5, "neutrophil_count": 2800, "creatinine": 0.9, "bilirubin": 0.7, "alt": 28, "ast": 22, "lvef": 62}},
    },
    {
        "patient_id": "OV_02",
        "description": "Female, 65yo. Recurrent platinum-sensitive ovarian cancer. Prior carboplatin/paclitaxel (1 line). ECOG 0. Brain metastases: none. Pregnant: no. Labs: Platelets 220,000 /mm³, Hgb 11.8 g/dL, ANC 3,500 /mm³, Creatinine 1.1 mg/dL, Bilirubin 0.5 mg/dL, ALT 32 U/L, AST 28 U/L.",
        "profile": {"age": 65, "sex": "female", "ecog": 0, "cancer_type": "recurrent platinum-sensitive ovarian cancer", "prior_chemo": True, "prior_rt": False, "brain_mets": False, "pregnant": False, "lab_values": {"platelet_count": 220000, "hemoglobin": 11.8, "neutrophil_count": 3500, "creatinine": 1.1, "bilirubin": 0.5, "alt": 32, "ast": 28}},
    },
    {
        "patient_id": "OV_03",
        "description": "Female, 58yo. Stage IV ovarian carcinoma, heavily pretreated (3 prior lines). ECOG 2. Brain metastases: none. Pregnant: no. Labs: Platelets 95,000 /mm³, Hgb 9.2 g/dL, ANC 1,200 /mm³, Creatinine 1.4 mg/dL, Bilirubin 1.8 mg/dL, ALT 55 U/L, AST 48 U/L.",
        "profile": {"age": 58, "sex": "female", "ecog": 2, "cancer_type": "stage IV ovarian carcinoma heavily pretreated", "prior_chemo": True, "prior_rt": False, "brain_mets": False, "pregnant": False, "lab_values": {"platelet_count": 95000, "hemoglobin": 9.2, "neutrophil_count": 1200, "creatinine": 1.4, "bilirubin": 1.8, "alt": 55, "ast": 48}},
    },
    {
        "patient_id": "OV_04",
        "description": "Female, 44yo. Newly diagnosed stage IIB ovarian clear cell carcinoma. No prior therapy. ECOG 0. Pregnant: no. Brain metastases: none. Labs: Platelets 310,000 /mm³, Hgb 13.5 g/dL, ANC 4,200 /mm³, Creatinine 0.7 mg/dL, Bilirubin 0.4 mg/dL, ALT 18 U/L, AST 16 U/L.",
        "profile": {"age": 44, "sex": "female", "ecog": 0, "cancer_type": "stage IIB ovarian clear cell carcinoma", "prior_chemo": False, "prior_rt": False, "brain_mets": False, "pregnant": False, "lab_values": {"platelet_count": 310000, "hemoglobin": 13.5, "neutrophil_count": 4200, "creatinine": 0.7, "bilirubin": 0.4, "alt": 18, "ast": 16}},
    },
    {
        "patient_id": "OV_05",
        "description": "Female, 72yo. Recurrent platinum-resistant ovarian cancer (progressed 3 months after last platinum). ECOG 2. Karnofsky 60%. Prior chemo: yes. Prior RT: no. Brain metastases: none. Pregnant: no. Labs: Platelets 140,000 /mm³, Hgb 10.1 g/dL, ANC 2,100 /mm³, Creatinine 1.3 mg/dL, Bilirubin 1.0 mg/dL.",
        "profile": {"age": 72, "sex": "female", "ecog": 2, "karnofsky": 60, "cancer_type": "recurrent platinum-resistant ovarian cancer", "prior_chemo": True, "prior_rt": False, "brain_mets": False, "pregnant": False, "lab_values": {"platelet_count": 140000, "hemoglobin": 10.1, "neutrophil_count": 2100, "creatinine": 1.3, "bilirubin": 1.0}},
    },
    {
        "patient_id": "OV_06",
        "description": "Female, 38yo. Stage III ovarian carcinoma. No prior chemotherapy. ECOG 1. Pregnant: yes (8 weeks gestation). Brain metastases: none. Labs: Platelets 200,000 /mm³, Hgb 11.0 g/dL, ANC 3,800 /mm³, Creatinine 0.8 mg/dL.",
        "profile": {"age": 38, "sex": "female", "ecog": 1, "cancer_type": "stage III ovarian carcinoma", "prior_chemo": False, "prior_rt": False, "brain_mets": False, "pregnant": True, "lab_values": {"platelet_count": 200000, "hemoglobin": 11.0, "neutrophil_count": 3800, "creatinine": 0.8}},
    },
    {
        "patient_id": "OV_07",
        "description": "Female, 61yo. Primary peritoneal carcinoma, FIGO Stage III. No prior chemotherapy or radiation. ECOG 1. Brain metastases: none. Pregnant: no. Labs: Platelets 165,000 /mm³, Hgb 12.0 g/dL, ANC 2,600 /mm³, Creatinine 1.0 mg/dL, Bilirubin 0.6 mg/dL, ALT 22 U/L, AST 19 U/L.",
        "profile": {"age": 61, "sex": "female", "ecog": 1, "cancer_type": "primary peritoneal carcinoma stage III", "prior_chemo": False, "prior_rt": False, "brain_mets": False, "pregnant": False, "lab_values": {"platelet_count": 165000, "hemoglobin": 12.0, "neutrophil_count": 2600, "creatinine": 1.0, "bilirubin": 0.6, "alt": 22, "ast": 19}},
    },
    {
        "patient_id": "OV_08",
        "description": "Female, 55yo. Recurrent ovarian cancer with brain metastases. Prior carboplatin/paclitaxel (2 lines) and prior radiation to pelvis. ECOG 3. Pregnant: no. Labs: Platelets 88,000 /mm³, Hgb 8.5 g/dL, ANC 900 /mm³, Creatinine 1.6 mg/dL, Bilirubin 2.1 mg/dL.",
        "profile": {"age": 55, "sex": "female", "ecog": 3, "cancer_type": "recurrent ovarian cancer with brain metastases", "prior_chemo": True, "prior_rt": True, "brain_mets": True, "pregnant": False, "lab_values": {"platelet_count": 88000, "hemoglobin": 8.5, "neutrophil_count": 900, "creatinine": 1.6, "bilirubin": 2.1}},
    },

    # ---- Breast cancer cohort (5 patients) ---------------------------------
    {
        "patient_id": "BR_01",
        "description": "Female, 48yo. HER2-positive metastatic breast cancer, first line. No prior chemotherapy. ECOG 0. Pregnant: no. Brain metastases: none. Labs: Platelets 245,000 /mm³, Hgb 13.2 g/dL, ANC 4,500 /mm³, Creatinine 0.8 mg/dL, Bilirubin 0.5 mg/dL, ALT 24 U/L, AST 20 U/L, LVEF 65%.",
        "profile": {"age": 48, "sex": "female", "ecog": 0, "cancer_type": "HER2-positive metastatic breast cancer", "prior_chemo": False, "prior_rt": False, "brain_mets": False, "pregnant": False, "lab_values": {"platelet_count": 245000, "hemoglobin": 13.2, "neutrophil_count": 4500, "creatinine": 0.8, "bilirubin": 0.5, "alt": 24, "ast": 20, "lvef": 65}},
    },
    {
        "patient_id": "BR_02",
        "description": "Female, 62yo. Triple-negative breast cancer, stage IV. Prior anthracycline + taxane (2 lines). ECOG 1. Pregnant: no. Brain metastases: none. Labs: Platelets 172,000 /mm³, Hgb 11.4 g/dL, ANC 2,200 /mm³, Creatinine 0.9 mg/dL, Bilirubin 0.8 mg/dL, ALT 35 U/L, AST 30 U/L.",
        "profile": {"age": 62, "sex": "female", "ecog": 1, "cancer_type": "triple-negative breast cancer stage IV", "prior_chemo": True, "prior_rt": False, "brain_mets": False, "pregnant": False, "lab_values": {"platelet_count": 172000, "hemoglobin": 11.4, "neutrophil_count": 2200, "creatinine": 0.9, "bilirubin": 0.8, "alt": 35, "ast": 30}},
    },
    {
        "patient_id": "BR_03",
        "description": "Female, 41yo. BRCA1-mutated breast cancer, stage II. No prior chemotherapy. ECOG 0. Pregnant: no. Brain metastases: none. Labs: Platelets 290,000 /mm³, Hgb 14.0 g/dL, ANC 5,200 /mm³, Creatinine 0.7 mg/dL, Bilirubin 0.3 mg/dL.",
        "profile": {"age": 41, "sex": "female", "ecog": 0, "cancer_type": "BRCA1-mutated breast cancer stage II", "prior_chemo": False, "prior_rt": False, "brain_mets": False, "pregnant": False, "lab_values": {"platelet_count": 290000, "hemoglobin": 14.0, "neutrophil_count": 5200, "creatinine": 0.7, "bilirubin": 0.3}},
    },
    {
        "patient_id": "BR_04",
        "description": "Female, 78yo. Hormone receptor-positive breast cancer, bone metastases. Prior tamoxifen and aromatase inhibitor. ECOG 2. Pregnant: no. Brain metastases: none. Labs: Platelets 130,000 /mm³, Hgb 10.5 g/dL, ANC 1,800 /mm³, Creatinine 1.5 mg/dL.",
        "profile": {"age": 78, "sex": "female", "ecog": 2, "cancer_type": "hormone receptor-positive breast cancer bone metastases", "prior_chemo": True, "prior_rt": False, "brain_mets": False, "pregnant": False, "lab_values": {"platelet_count": 130000, "hemoglobin": 10.5, "neutrophil_count": 1800, "creatinine": 1.5}},
    },
    {
        "patient_id": "BR_05",
        "description": "Female, 35yo. Inflammatory breast cancer, stage IIIB. No prior therapy. ECOG 1. Pregnant: no. Brain metastases: none. Labs: Platelets 260,000 /mm³, Hgb 12.8 g/dL, ANC 3,900 /mm³, Creatinine 0.8 mg/dL, Bilirubin 0.6 mg/dL, ALT 20 U/L, AST 18 U/L.",
        "profile": {"age": 35, "sex": "female", "ecog": 1, "cancer_type": "inflammatory breast cancer stage IIIB", "prior_chemo": False, "prior_rt": False, "brain_mets": False, "pregnant": False, "lab_values": {"platelet_count": 260000, "hemoglobin": 12.8, "neutrophil_count": 3900, "creatinine": 0.8, "bilirubin": 0.6, "alt": 20, "ast": 18}},
    },

    # ---- Lung cancer cohort (4 patients) -----------------------------------
    {
        "patient_id": "LU_01",
        "description": "Male, 67yo. Stage IIIB non-small cell lung cancer (NSCLC), adenocarcinoma. No prior chemotherapy. ECOG 1. Brain metastases: none. Labs: Platelets 195,000 /mm³, Hgb 12.0 g/dL, ANC 3,100 /mm³, Creatinine 1.0 mg/dL, Bilirubin 0.7 mg/dL.",
        "profile": {"age": 67, "sex": "male", "ecog": 1, "cancer_type": "stage IIIB non-small cell lung cancer adenocarcinoma", "prior_chemo": False, "prior_rt": False, "brain_mets": False, "lab_values": {"platelet_count": 195000, "hemoglobin": 12.0, "neutrophil_count": 3100, "creatinine": 1.0, "bilirubin": 0.7}},
    },
    {
        "patient_id": "LU_02",
        "description": "Female, 56yo. EGFR-mutant stage IV NSCLC. Prior erlotinib (1 line). ECOG 1. Brain metastases: yes. Labs: Platelets 155,000 /mm³, Hgb 11.5 g/dL, ANC 2,400 /mm³, Creatinine 0.9 mg/dL.",
        "profile": {"age": 56, "sex": "female", "ecog": 1, "cancer_type": "EGFR-mutant stage IV non-small cell lung cancer", "prior_chemo": True, "prior_rt": False, "brain_mets": True, "pregnant": False, "lab_values": {"platelet_count": 155000, "hemoglobin": 11.5, "neutrophil_count": 2400, "creatinine": 0.9}},
    },
    {
        "patient_id": "LU_03",
        "description": "Male, 72yo. Small cell lung cancer, extensive stage. No prior chemotherapy. ECOG 2. Brain metastases: none. Labs: Platelets 110,000 /mm³, Hgb 9.8 g/dL, ANC 1,500 /mm³, Creatinine 1.2 mg/dL.",
        "profile": {"age": 72, "sex": "male", "ecog": 2, "cancer_type": "small cell lung cancer extensive stage", "prior_chemo": False, "prior_rt": False, "brain_mets": False, "lab_values": {"platelet_count": 110000, "hemoglobin": 9.8, "neutrophil_count": 1500, "creatinine": 1.2}},
    },
    {
        "patient_id": "LU_04",
        "description": "Male, 60yo. Stage IV NSCLC, squamous histology. Prior platinum doublet + immunotherapy (2 lines). ECOG 2. Prior RT: yes. Brain metastases: none. Labs: Platelets 142,000 /mm³, Hgb 10.3 g/dL, ANC 1,900 /mm³, Creatinine 1.1 mg/dL, Bilirubin 1.2 mg/dL.",
        "profile": {"age": 60, "sex": "male", "ecog": 2, "cancer_type": "stage IV NSCLC squamous histology", "prior_chemo": True, "prior_rt": True, "brain_mets": False, "lab_values": {"platelet_count": 142000, "hemoglobin": 10.3, "neutrophil_count": 1900, "creatinine": 1.1, "bilirubin": 1.2}},
    },

    # ---- Haematologic malignancy cohort (4 patients) -----------------------
    {
        "patient_id": "HM_01",
        "description": "Female, 53yo. Diffuse large B-cell lymphoma, stage III. No prior chemotherapy. ECOG 1. Brain metastases: none. Pregnant: no. Labs: Platelets 198,000 /mm³, Hgb 11.0 g/dL, ANC 2,800 /mm³, Creatinine 0.8 mg/dL, Bilirubin 0.9 mg/dL, LDH 420 U/L.",
        "profile": {"age": 53, "sex": "female", "ecog": 1, "cancer_type": "diffuse large B-cell lymphoma stage III", "prior_chemo": False, "prior_rt": False, "brain_mets": False, "pregnant": False, "lab_values": {"platelet_count": 198000, "hemoglobin": 11.0, "neutrophil_count": 2800, "creatinine": 0.8, "bilirubin": 0.9, "ldh": 420}},
    },
    {
        "patient_id": "HM_02",
        "description": "Male, 45yo. Relapsed/refractory multiple myeloma (3 prior lines). ECOG 2. Brain metastases: none. Labs: Platelets 78,000 /mm³, Hgb 8.2 g/dL, ANC 1,100 /mm³, Creatinine 2.1 mg/dL, Bilirubin 1.0 mg/dL, Calcium 11.8 mg/dL.",
        "profile": {"age": 45, "sex": "male", "ecog": 2, "cancer_type": "relapsed refractory multiple myeloma", "prior_chemo": True, "prior_rt": False, "brain_mets": False, "lab_values": {"platelet_count": 78000, "hemoglobin": 8.2, "neutrophil_count": 1100, "creatinine": 2.1, "bilirubin": 1.0, "calcium": 11.8}},
    },
    {
        "patient_id": "HM_03",
        "description": "Female, 30yo. Acute myeloid leukemia (AML), newly diagnosed. No prior chemotherapy. ECOG 1. Pregnant: no. Brain metastases: none. Labs: Platelets 42,000 /mm³, Hgb 7.5 g/dL, WBC 45,000 /mm³, ANC 400 /mm³, Creatinine 0.9 mg/dL, Bilirubin 1.1 mg/dL.",
        "profile": {"age": 30, "sex": "female", "ecog": 1, "cancer_type": "acute myeloid leukemia newly diagnosed", "prior_chemo": False, "prior_rt": False, "brain_mets": False, "pregnant": False, "lab_values": {"platelet_count": 42000, "hemoglobin": 7.5, "wbc": 45000, "neutrophil_count": 400, "creatinine": 0.9, "bilirubin": 1.1}},
    },
    {
        "patient_id": "HM_04",
        "description": "Male, 58yo. Chronic lymphocytic leukemia (CLL), Rai stage III. Prior ibrutinib (1 line). ECOG 1. Brain metastases: none. Labs: Platelets 105,000 /mm³, Hgb 10.2 g/dL, ANC 1,600 /mm³, Creatinine 1.0 mg/dL.",
        "profile": {"age": 58, "sex": "male", "ecog": 1, "cancer_type": "chronic lymphocytic leukemia Rai stage III", "prior_chemo": True, "prior_rt": False, "brain_mets": False, "lab_values": {"platelet_count": 105000, "hemoglobin": 10.2, "neutrophil_count": 1600, "creatinine": 1.0}},
    },

    # ---- Prostate cancer cohort (3 patients) --------------------------------
    {
        "patient_id": "PR_01",
        "description": "Male, 68yo. Metastatic castration-resistant prostate cancer (mCRPC). Prior docetaxel (1 line). ECOG 1. Brain metastases: none. Labs: Platelets 185,000 /mm³, Hgb 11.8 g/dL, ANC 3,200 /mm³, Creatinine 1.1 mg/dL, PSA 45 ng/mL, Testosterone 22 ng/dL.",
        "profile": {"age": 68, "sex": "male", "ecog": 1, "cancer_type": "metastatic castration-resistant prostate cancer", "prior_chemo": True, "prior_rt": False, "brain_mets": False, "lab_values": {"platelet_count": 185000, "hemoglobin": 11.8, "neutrophil_count": 3200, "creatinine": 1.1, "psa": 45, "testosterone": 22}},
    },
    {
        "patient_id": "PR_02",
        "description": "Male, 74yo. Hormone-sensitive prostate cancer, newly diagnosed with bone metastases. No prior chemotherapy. ECOG 0. Brain metastases: none. Labs: Platelets 220,000 /mm³, Hgb 12.5 g/dL, ANC 4,000 /mm³, Creatinine 1.0 mg/dL, PSA 120 ng/mL, Testosterone 380 ng/dL.",
        "profile": {"age": 74, "sex": "male", "ecog": 0, "cancer_type": "hormone-sensitive prostate cancer bone metastases", "prior_chemo": False, "prior_rt": False, "brain_mets": False, "lab_values": {"platelet_count": 220000, "hemoglobin": 12.5, "neutrophil_count": 4000, "creatinine": 1.0, "psa": 120, "testosterone": 380}},
    },
    {
        "patient_id": "PR_03",
        "description": "Male, 62yo. mCRPC, heavily pretreated (docetaxel + cabazitaxel + enzalutamide). ECOG 2. Prior RT to pelvis: yes. Brain metastases: none. Labs: Platelets 95,000 /mm³, Hgb 9.0 g/dL, ANC 1,400 /mm³, Creatinine 1.4 mg/dL, PSA 280 ng/mL, Testosterone 18 ng/dL.",
        "profile": {"age": 62, "sex": "male", "ecog": 2, "cancer_type": "metastatic castration-resistant prostate cancer heavily pretreated", "prior_chemo": True, "prior_rt": True, "brain_mets": False, "lab_values": {"platelet_count": 95000, "hemoglobin": 9.0, "neutrophil_count": 1400, "creatinine": 1.4, "psa": 280, "testosterone": 18}},
    },

    # ---- Other solid tumour cohort (6 patients) ----------------------------
    {
        "patient_id": "GI_01",
        "description": "Male, 55yo. Metastatic colorectal cancer, MSS, prior FOLFOX (1 line). ECOG 1. Brain metastases: none. Labs: Platelets 210,000 /mm³, Hgb 11.2 g/dL, ANC 2,900 /mm³, Creatinine 0.9 mg/dL, Bilirubin 0.8 mg/dL, ALT 40 U/L, AST 35 U/L.",
        "profile": {"age": 55, "sex": "male", "ecog": 1, "cancer_type": "metastatic colorectal cancer MSS", "prior_chemo": True, "prior_rt": False, "brain_mets": False, "lab_values": {"platelet_count": 210000, "hemoglobin": 11.2, "neutrophil_count": 2900, "creatinine": 0.9, "bilirubin": 0.8, "alt": 40, "ast": 35}},
    },
    {
        "patient_id": "GI_02",
        "description": "Female, 49yo. Pancreatic adenocarcinoma, stage IV. No prior chemotherapy. ECOG 1. Pregnant: no. Brain metastases: none. Labs: Platelets 165,000 /mm³, Hgb 10.8 g/dL, ANC 2,500 /mm³, Creatinine 0.8 mg/dL, Bilirubin 2.5 mg/dL, ALT 72 U/L, AST 65 U/L, Albumin 3.0 g/dL.",
        "profile": {"age": 49, "sex": "female", "ecog": 1, "cancer_type": "pancreatic adenocarcinoma stage IV", "prior_chemo": False, "prior_rt": False, "brain_mets": False, "pregnant": False, "lab_values": {"platelet_count": 165000, "hemoglobin": 10.8, "neutrophil_count": 2500, "creatinine": 0.8, "bilirubin": 2.5, "alt": 72, "ast": 65, "albumin": 3.0}},
    },
    {
        "patient_id": "GI_03",
        "description": "Male, 63yo. Hepatocellular carcinoma (HCC), Child-Pugh A, no prior systemic therapy. ECOG 1. Brain metastases: none. Labs: Platelets 88,000 /mm³, Hgb 11.5 g/dL, ANC 1,800 /mm³, Creatinine 0.9 mg/dL, Bilirubin 1.4 mg/dL, ALT 85 U/L, AST 90 U/L, Albumin 3.5 g/dL, INR 1.3.",
        "profile": {"age": 63, "sex": "male", "ecog": 1, "cancer_type": "hepatocellular carcinoma Child-Pugh A", "prior_chemo": False, "prior_rt": False, "brain_mets": False, "child_pugh": "A", "lab_values": {"platelet_count": 88000, "hemoglobin": 11.5, "neutrophil_count": 1800, "creatinine": 0.9, "bilirubin": 1.4, "alt": 85, "ast": 90, "albumin": 3.5, "inr": 1.3}},
    },
    {
        "patient_id": "ME_01",
        "description": "Male, 50yo. Metastatic melanoma, BRAF-wild type. No prior systemic therapy. ECOG 0. Brain metastases: none. Labs: Platelets 240,000 /mm³, Hgb 14.2 g/dL, ANC 4,800 /mm³, Creatinine 0.8 mg/dL, Bilirubin 0.5 mg/dL, LDH 280 U/L.",
        "profile": {"age": 50, "sex": "male", "ecog": 0, "cancer_type": "metastatic melanoma BRAF-wild type", "prior_chemo": False, "prior_rt": False, "brain_mets": False, "lab_values": {"platelet_count": 240000, "hemoglobin": 14.2, "neutrophil_count": 4800, "creatinine": 0.8, "bilirubin": 0.5, "ldh": 280}},
    },
    {
        "patient_id": "ME_02",
        "description": "Female, 42yo. Metastatic melanoma with brain metastases, BRAF V600E mutant. Prior ipilimumab + nivolumab (1 line). ECOG 1. Pregnant: no. Brain metastases: yes. Labs: Platelets 175,000 /mm³, Hgb 11.8 g/dL, ANC 2,600 /mm³, Creatinine 0.9 mg/dL.",
        "profile": {"age": 42, "sex": "female", "ecog": 1, "cancer_type": "metastatic melanoma BRAF V600E mutant with brain metastases", "prior_chemo": False, "prior_rt": False, "brain_mets": True, "pregnant": False, "lab_values": {"platelet_count": 175000, "hemoglobin": 11.8, "neutrophil_count": 2600, "creatinine": 0.9}},
    },
    {
        "patient_id": "GU_01",
        "description": "Male, 58yo. Urothelial carcinoma, stage IV. Prior cisplatin-based chemotherapy (1 line). ECOG 1. Brain metastases: none. Labs: Platelets 188,000 /mm³, Hgb 10.9 g/dL, ANC 2,700 /mm³, Creatinine 1.3 mg/dL, Bilirubin 0.7 mg/dL.",
        "profile": {"age": 58, "sex": "male", "ecog": 1, "cancer_type": "urothelial carcinoma stage IV", "prior_chemo": True, "prior_rt": False, "brain_mets": False, "lab_values": {"platelet_count": 188000, "hemoglobin": 10.9, "neutrophil_count": 2700, "creatinine": 1.3, "bilirubin": 0.7}},
    },
]

assert len(PATIENTS) == 30, f"Expected 30 patients, got {len(PATIENTS)}"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _agreement_label(
    short_circuited: bool,
    coverage_gated: bool,
    p_mean: float,
    mistral_verdict: str,
) -> str:
    """Classify Bayesian / Mistral agreement into a categorical label."""
    if coverage_gated:
        return "COVERAGE_GATED"
    if short_circuited:
        if mistral_verdict == "NOT ELIGIBLE":
            return "AGREE_INELIGIBLE"
        if mistral_verdict == "ELIGIBLE":
            return "DISAGREE_BAYES_FAIL_MISTRAL_ELIG"
        return "PARTIAL_BAYES_FAIL_MISTRAL_UNCERTAIN"
    if p_mean >= 0.60:
        if mistral_verdict == "ELIGIBLE":
            return "AGREE_ELIGIBLE"
        if mistral_verdict == "NOT ELIGIBLE":
            return "DISAGREE_BAYES_ELIG_MISTRAL_FAIL"
        return "PARTIAL_BAYES_ELIG_MISTRAL_UNCERTAIN"
    if p_mean < 0.30:
        if mistral_verdict == "NOT ELIGIBLE":
            return "AGREE_LOW_PROB"
        if mistral_verdict == "ELIGIBLE":
            return "DISAGREE_BAYES_LOW_MISTRAL_ELIG"
        return "NEUTRAL"
    return "NEUTRAL"


def _build_from_text(patient: dict) -> str:
    """Build a one-liner description from the patient profile dict (fallback)."""
    parts = []
    sex = patient.get("sex", "")
    age = patient.get("age")
    if sex and age:
        parts.append(f"{sex.capitalize()}, {age}yo.")
    ct = patient.get("cancer_type", "")
    if ct:
        parts.append(ct + ".")
    if patient.get("prior_chemo") is not None:
        parts.append("Prior chemo: " + ("yes" if patient["prior_chemo"] else "none") + ".")
    if patient.get("prior_rt") is not None:
        parts.append("Prior RT: " + ("yes" if patient["prior_rt"] else "none") + ".")
    if patient.get("brain_mets") is not None:
        parts.append("Brain mets: " + ("yes" if patient["brain_mets"] else "none") + ".")
    if patient.get("pregnant") is not None:
        parts.append("Pregnant: " + ("yes" if patient["pregnant"] else "no") + ".")
    ecog = patient.get("ecog")
    if ecog is not None:
        parts.append(f"ECOG {ecog}.")
    labs = patient.get("lab_values") or {}
    lab_strs = []
    for k, label in [("platelet_count","Plt"), ("hemoglobin","Hgb"),
                     ("neutrophil_count","ANC"), ("creatinine","Cr"),
                     ("bilirubin","Bili"), ("alt","ALT"), ("ast","AST")]:
        if k in labs:
            lab_strs.append(f"{label} {labs[k]}")
    if lab_strs:
        parts.append("Labs: " + ", ".join(lab_strs) + ".")
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Stage 1 — Bayesian sweep
# ---------------------------------------------------------------------------

def run_stage1(
    trials_per_patient: int,
    n_draws: int,
    coverage_threshold: float,
    min_criteria: int,
) -> pd.DataFrame:
    from sentence_transformers import SentenceTransformer
    from rag.vector_store import get_client, get_collection, query_trials
    from bayesian.criterion_evaluator import load_criteria_for_trial
    from bayesian.eligibility_model import (
        compute_eligibility_posterior,
        evaluate_all_criteria,
        DETERMINISTIC_FAIL,
    )
    from bayesian.uncertainty import summarize_posterior

    os.makedirs(EVAL_DIR, exist_ok=True)

    log.info("Loading sentence transformer...")
    embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    log.info("Connecting to ChromaDB and DuckDB...")
    chroma_client = get_client(os.path.join(PROJECT_ROOT, "data", "processed", "chroma"))
    collection = get_collection(chroma_client)
    con = duckdb.connect(
        os.path.join(PROJECT_ROOT, "data", "processed", "trials.duckdb"),
        read_only=True,
    )

    records = []
    total_pairs = len(PATIENTS) * trials_per_patient
    n_done = 0
    t_start = time.time()

    for pt in PATIENTS:
        patient_id = pt["patient_id"]
        description = pt.get("description") or _build_from_text(pt["profile"])
        profile = pt["profile"]

        log.info(f"Patient {patient_id}: embedding + retrieving top {trials_per_patient} trials...")
        vec = embedder.encode(description, normalize_embeddings=True).tolist()
        top_trials = query_trials(
            collection, vec,
            n_results=trials_per_patient,
            doc_max_len=200,  # metadata only for stage1; full doc not needed
        )

        # Cache criteria by nct_id within this patient's trial set
        criteria_cache: dict = {}

        for trial_hit in top_trials:
            nct_id = trial_hit["nct_id"]
            sem_score = trial_hit["score"]

            # Load criteria (cached)
            if nct_id not in criteria_cache:
                criteria_cache[nct_id] = load_criteria_for_trial(nct_id, con)
            criteria = criteria_cache[nct_id]

            # Classify criteria (fast — no PyMC yet)
            evaluations = evaluate_all_criteria(criteria, profile)
            n_total = len(evaluations)
            n_pass   = sum(1 for e in evaluations if e["kind"] == "deterministic_pass")
            n_fail   = sum(1 for e in evaluations if e["kind"] == "deterministic_fail")
            n_subj   = sum(1 for e in evaluations if e["kind"] == "subjective")
            n_unobs  = sum(1 for e in evaluations if e["kind"] == "unobservable")
            n_uneval = sum(1 for e in evaluations if e["kind"] == "unevaluable")
            n_evaluable = n_pass + n_fail + n_subj
            coverage = n_evaluable / n_total if n_total > 0 else 0.0

            # Determine fail reason(s) for short-circuited pairs
            failing_criteria = [
                e["criterion_id"] for e in evaluations if e["kind"] == DETERMINISTIC_FAIL
            ]
            first_fail_text = next(
                (e["text"][:120] for e in evaluations if e["kind"] == DETERMINISTIC_FAIL),
                None,
            )
            short_circuited = bool(failing_criteria)

            # Coverage gate
            coverage_gated = (
                not short_circuited
                and (coverage < coverage_threshold or n_total < min_criteria)
            )

            # Run PyMC only for non-trivial pairs
            p_mean = p_lo = p_hi = None
            tier = None
            if not short_circuited and not coverage_gated:
                try:
                    result = compute_eligibility_posterior(
                        criteria, profile,
                        n_samples=n_draws,
                        random_seed=42,
                    )
                    summary = summarize_posterior(result)
                    p_mean = float(summary["mean"])
                    p_lo   = float(summary["hdi_lower"])
                    p_hi   = float(summary["hdi_upper"])
                    tier   = summary["tier"]
                except Exception as exc:
                    log.warning(f"  PyMC failed for {nct_id}: {exc}")
                    p_mean = p_lo = p_hi = 0.0
                    tier = "error"

            records.append({
                "patient_id":         patient_id,
                "nct_id":             nct_id,
                "sem_score":          round(sem_score, 5),
                "n_total":            n_total,
                "n_pass":             n_pass,
                "n_fail":             n_fail,
                "n_subj":             n_subj,
                "n_unobs":            n_unobs,
                "n_uneval":           n_uneval,
                "n_evaluable":        n_evaluable,
                "coverage":           round(coverage, 4),
                "short_circuited":    short_circuited,
                "coverage_gated":     coverage_gated,
                "first_fail_text":    first_fail_text,
                "failing_criteria":   json.dumps(failing_criteria),
                "p_mean":             p_mean,
                "p_lo":               p_lo,
                "p_hi":               p_hi,
                "tier":               tier,
            })

            n_done += 1
            if n_done % 100 == 0:
                elapsed = time.time() - t_start
                rate = n_done / elapsed
                eta_s = (total_pairs - n_done) / rate if rate > 0 else 0
                log.info(
                    f"  Progress: {n_done}/{total_pairs} pairs "
                    f"({n_done/total_pairs:.0%}) | "
                    f"elapsed {elapsed/60:.1f}m | ETA {eta_s/60:.1f}m"
                )

    con.close()

    df = pd.DataFrame(records)
    df.to_parquet(STAGE1_PATH, index=False)
    elapsed = time.time() - t_start
    log.info(f"Stage 1 complete. {len(df)} pairs in {elapsed/60:.1f} min → {STAGE1_PATH}")
    return df


# ---------------------------------------------------------------------------
# Stage 2 — Mistral spot-check
# ---------------------------------------------------------------------------

def run_stage2(
    sample_rate: float,
    seed: int,
) -> pd.DataFrame:
    import requests as _req
    from rag.vector_store import get_client, get_collection
    from rag.generator import assess_trial

    if not os.path.exists(STAGE1_PATH):
        raise FileNotFoundError(f"Stage 1 output not found: {STAGE1_PATH}. Run --stage 1 first.")

    # Check Ollama
    try:
        r = _req.get("http://127.0.0.1:11434/api/tags", timeout=3)
        models = [m["name"] for m in r.json().get("models", [])]
        if not any("mistral" in m for m in models):
            raise RuntimeError("mistral model not found in Ollama. Run: ollama pull mistral")
    except Exception as exc:
        raise RuntimeError(f"Ollama not available: {exc}")

    df1 = pd.read_parquet(STAGE1_PATH)

    # Select pairs to send to Mistral:
    # - All short-circuited pairs (sampled at sample_rate for calibration)
    # - All non-trivial, non-gated pairs (these are the interesting ones)
    rng = random.Random(seed)

    sc_pairs    = df1[df1["short_circuited"]].copy()
    sc_sample   = sc_pairs.sample(frac=sample_rate, random_state=seed)

    nontrivial  = df1[~df1["short_circuited"] & ~df1["coverage_gated"]].copy()

    gated       = df1[df1["coverage_gated"]].copy()
    gated_sample = gated.sample(frac=sample_rate, random_state=seed)

    to_run = pd.concat([sc_sample, nontrivial, gated_sample], ignore_index=True)
    to_run = to_run.drop_duplicates(subset=["patient_id", "nct_id"])

    log.info(
        f"Stage 2: {len(to_run)} pairs selected "
        f"({len(sc_sample)} SC-sampled, {len(nontrivial)} non-trivial, "
        f"{len(gated_sample)} gated-sampled)"
    )

    # Build patient lookup
    pt_lookup = {p["patient_id"]: p for p in PATIENTS}

    # Get ChromaDB collection for document retrieval
    chroma_client = get_client(os.path.join(PROJECT_ROOT, "data", "processed", "chroma"))
    collection = get_collection(chroma_client)

    records = []
    t_start = time.time()

    for i, row in enumerate(to_run.itertuples(), 1):
        patient_id = row.patient_id
        nct_id = row.nct_id
        pt = pt_lookup[patient_id]
        description = pt.get("description") or _build_from_text(pt["profile"])

        # Retrieve full document for LLM
        try:
            doc_result = collection.get(ids=[nct_id], include=["documents"])
            trial_doc = doc_result["documents"][0] if doc_result["documents"] else ""
        except Exception as exc:
            log.warning(f"  Could not fetch doc for {nct_id}: {exc}")
            trial_doc = ""

        # Run Mistral
        try:
            llm = assess_trial(
                nct_id=nct_id,
                trial_document=trial_doc,
                patient_query=description,
                temperature=0.0,
            )
            mistral_verdict = llm["verdict"]
            mistral_explanation = llm["explanation"]
        except Exception as exc:
            log.warning(f"  Mistral failed for {patient_id}/{nct_id}: {exc}")
            mistral_verdict = "ERROR"
            mistral_explanation = str(exc)

        agreement = _agreement_label(
            short_circuited=bool(row.short_circuited),
            coverage_gated=bool(row.coverage_gated),
            p_mean=row.p_mean if row.p_mean is not None else 0.0,
            mistral_verdict=mistral_verdict,
        )

        records.append({
            "patient_id":          patient_id,
            "nct_id":              nct_id,
            "mistral_verdict":     mistral_verdict,
            "mistral_explanation": mistral_explanation,
            "agreement":           agreement,
        })

        if i % 10 == 0:
            elapsed = time.time() - t_start
            rate = i / elapsed
            eta_s = (len(to_run) - i) / rate if rate > 0 else 0
            log.info(
                f"  Mistral: {i}/{len(to_run)} "
                f"| elapsed {elapsed/60:.1f}m | ETA {eta_s/60:.1f}m"
            )

    df2 = pd.DataFrame(records)
    df2.to_parquet(STAGE2_PATH, index=False)
    elapsed = time.time() - t_start
    log.info(f"Stage 2 complete. {len(df2)} pairs in {elapsed/60:.1f} min → {STAGE2_PATH}")
    return df2


# ---------------------------------------------------------------------------
# Stage 3 — Aggregate analysis
# ---------------------------------------------------------------------------

def run_stage3() -> str:
    if not os.path.exists(STAGE1_PATH):
        raise FileNotFoundError(f"Stage 1 output not found: {STAGE1_PATH}")

    df1 = pd.read_parquet(STAGE1_PATH)
    has_stage2 = os.path.exists(STAGE2_PATH)
    df2 = pd.read_parquet(STAGE2_PATH) if has_stage2 else None

    lines = []
    sep  = "=" * 72
    thin = "-" * 72

    def h(title: str):
        lines.append("")
        lines.append(sep)
        lines.append(title)
        lines.append(sep)

    def sub(title: str):
        lines.append("")
        lines.append(thin)
        lines.append(title)

    lines.append(sep)
    lines.append("BATCH EVALUATION — STAGE 3 ANALYSIS REPORT")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Stage 1 pairs: {len(df1)}")
    lines.append(f"Stage 2 pairs: {len(df2) if df2 is not None else 'not run'}")
    lines.append(sep)

    # ---- Stage 1 summary ---------------------------------------------------
    h("STAGE 1 — BAYESIAN SWEEP SUMMARY")

    total = len(df1)
    n_sc    = df1["short_circuited"].sum()
    n_gated = df1["coverage_gated"].sum()
    n_scored = total - n_sc - n_gated
    lines.append(f"Total pairs evaluated : {total}")
    lines.append(f"  Short-circuited     : {n_sc} ({n_sc/total:.1%})")
    lines.append(f"  Coverage-gated      : {n_gated} ({n_gated/total:.1%})")
    lines.append(f"  Bayesian-scored     : {n_scored} ({n_scored/total:.1%})")

    sub("Coverage distribution (non-short-circuited pairs)")
    cov_df = df1[~df1["short_circuited"]]
    for bucket, lo, hi in [
        ("0–20%", 0, 0.20), ("20–40%", 0.20, 0.40),
        ("40–60%", 0.40, 0.60), ("60–80%", 0.60, 0.80), ("80–100%", 0.80, 1.01),
    ]:
        n = ((cov_df["coverage"] >= lo) & (cov_df["coverage"] < hi)).sum()
        lines.append(f"  {bucket:<12}: {n:>6} pairs ({n/max(len(cov_df),1):.1%})")

    sub("Tier distribution (Bayesian-scored pairs only)")
    scored_df = df1[~df1["short_circuited"] & ~df1["coverage_gated"]]
    if len(scored_df) > 0:
        for tier, cnt in scored_df["tier"].value_counts().items():
            lines.append(f"  {tier:<25}: {cnt:>6} ({cnt/len(scored_df):.1%})")
    else:
        lines.append("  (no scored pairs)")

    sub("Top fail pattern analysis (short-circuited pairs)")
    fail_texts = df1[df1["short_circuited"]]["first_fail_text"].dropna()
    # Bucket by keyword pattern
    pattern_counts: Counter = Counter()
    for t in fail_texts:
        tl = t.lower()
        if any(w in tl for w in ["pregnant", "pregnan", "lactati", "breastfeed", "nursing", "childbearing"]):
            pattern_counts["pregnancy/lactation"] += 1
        elif any(w in tl for w in ["prior chemo", "chemotherapy", "prior platinum", "adjuvant chemo", "previous chemo"]):
            pattern_counts["prior chemotherapy (required or excluded)"] += 1
        elif any(w in tl for w in ["brain", "cns metasta"]):
            pattern_counts["brain metastases"] += 1
        elif any(w in tl for w in ["prior radiation", "radiotherapy", "prior rt"]):
            pattern_counts["prior radiation therapy"] += 1
        elif any(w in tl for w in ["male", "men only", "female", "women only"]):
            pattern_counts["sex restriction"] += 1
        elif any(w in tl for w in ["recur", "relaps", "refractory"]):
            pattern_counts["recurrence/relapse required"] += 1
        else:
            pattern_counts["other"] += 1
    for pat, cnt in pattern_counts.most_common():
        lines.append(f"  {pat:<45}: {cnt:>5} ({cnt/max(n_sc,1):.1%})")

    sub("Per-patient short-circuit rate")
    sc_by_pt = df1.groupby("patient_id")["short_circuited"].mean().sort_values(ascending=False)
    for pid, rate in sc_by_pt.items():
        n = df1[df1["patient_id"] == pid]["short_circuited"].sum()
        lines.append(f"  {pid:<12}: {rate:.0%} ({n} of {trials_per_patient_from_df(df1, pid)} trials)")

    # ---- Stage 2 summary ---------------------------------------------------
    if df2 is not None:
        merged = df2.merge(
            df1[["patient_id","nct_id","short_circuited","coverage_gated","p_mean","tier"]],
            on=["patient_id","nct_id"],
            how="left",
        )

        h("STAGE 2 — MISTRAL SPOT-CHECK SUMMARY")

        sub("Verdict distribution")
        for v, cnt in merged["mistral_verdict"].value_counts().items():
            lines.append(f"  {v:<15}: {cnt:>6} ({cnt/len(merged):.1%})")

        sub("Agreement distribution")
        for ag, cnt in merged["agreement"].value_counts().items():
            lines.append(f"  {ag:<50}: {cnt:>5} ({cnt/len(merged):.1%})")

        sub("Hard disagreements (Bayesian=FAIL, Mistral=ELIGIBLE)")
        hard_dis = merged[
            merged["agreement"] == "DISAGREE_BAYES_FAIL_MISTRAL_ELIG"
        ]
        lines.append(f"  Count: {len(hard_dis)}")
        if len(hard_dis) > 0:
            lines.append("  Samples (first_fail_text | mistral_explanation[:120]):")
            for _, row in hard_dis.head(10).iterrows():
                ft = df1.loc[
                    (df1["patient_id"]==row["patient_id"]) &
                    (df1["nct_id"]==row["nct_id"]),
                    "first_fail_text"
                ].values
                ft_str = ft[0][:80] if len(ft) > 0 and ft[0] else "(none)"
                expl = (row["mistral_explanation"] or "")[:120].replace("\n", " ")
                lines.append(f"    [{row['patient_id']}/{row['nct_id']}]")
                lines.append(f"      Fail: {ft_str}")
                lines.append(f"      Mistral: {expl}")

        sub("Bayesian=low-prob, Mistral=ELIGIBLE disagreements")
        low_dis = merged[
            merged["agreement"] == "DISAGREE_BAYES_LOW_MISTRAL_ELIG"
        ]
        lines.append(f"  Count: {len(low_dis)}")

        sub("Hallucination candidates (Mistral=NOT ELIGIBLE, Bayesian=high-prob eligible)")
        hall = merged[
            merged["agreement"] == "DISAGREE_BAYES_ELIG_MISTRAL_FAIL"
        ]
        lines.append(f"  Count: {len(hall)}")
        if len(hall) > 0:
            for _, row in hall.head(5).iterrows():
                expl = (row["mistral_explanation"] or "")[:140].replace("\n", " ")
                lines.append(f"    [{row['patient_id']}/{row['nct_id']}] p_mean={row.get('p_mean','?'):.3f}")
                lines.append(f"      Mistral: {expl}")

    # ---- Recommendations ---------------------------------------------------
    h("RECOMMENDATIONS FOR NEXT ITERATION")
    lines.append(
        "Review the fail pattern analysis above. Fix patterns that affect > 5% of\n"
        "all pairs. Patterns affecting < 1% are noise — do not over-fit to them.\n\n"
        "Priority order:\n"
        "  1. Any fail pattern that appears in > 10% of short-circuit cases AND\n"
        "     where Mistral disagrees (ELIGIBLE or UNCERTAIN) > 50% of the time\n"
        "     → routing error, fix the keyword router.\n"
        "  2. Coverage gate triggering > 60% of non-SC pairs → patient schema\n"
        "     missing fields that matter; expand the profile or lower gate threshold.\n"
        "  3. Mistral hallucination rate > 10% in scored pairs → prompt engineering.\n"
        "  4. Patterns affecting < 1% of pairs: document as known limitation only."
    )

    lines.append("")
    lines.append(sep)
    lines.append("END OF REPORT")
    lines.append(sep)

    report = "\n".join(lines)
    with open(STAGE3_PATH, "w") as f:
        f.write(report)
    log.info(f"Stage 3 report written to {STAGE3_PATH}")
    print(report)
    return report


def trials_per_patient_from_df(df: pd.DataFrame, patient_id: str) -> int:
    return int((df["patient_id"] == patient_id).sum())


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Batch evaluation harness — Bayesian + Mistral model comparison"
    )
    p.add_argument(
        "--stage", choices=["1", "2", "3", "all"], default="all",
        help="Which stage(s) to run (default: all)",
    )
    p.add_argument("--trials-per-patient", type=int, default=200)
    p.add_argument("--n-draws",            type=int, default=500,
                   help="PyMC prior predictive draws per pair (default 500)")
    p.add_argument("--mistral-sample-rate", type=float, default=0.10,
                   help="Fraction of SC/gated pairs sent to Mistral (default 0.10)")
    p.add_argument("--coverage-threshold", type=float, default=0.30)
    p.add_argument("--min-criteria",       type=int,   default=5)
    p.add_argument("--seed",               type=int,   default=42)
    return p.parse_args()


def main():
    args = parse_args()
    stages = {"1", "2", "3"} if args.stage == "all" else {args.stage}

    if "1" in stages:
        run_stage1(
            trials_per_patient=args.trials_per_patient,
            n_draws=args.n_draws,
            coverage_threshold=args.coverage_threshold,
            min_criteria=args.min_criteria,
        )

    if "2" in stages:
        run_stage2(
            sample_rate=args.mistral_sample_rate,
            seed=args.seed,
        )

    if "3" in stages:
        run_stage3()


if __name__ == "__main__":
    main()
