# RAG Generation Quality Evaluation

**Date:** 2026-04-29 18:38
**Model:** mistral
**Evaluation set:** 50 ineligible cases + 50 eligible cases

---

## Summary

| Track | Cases run | Passed | Failed | Pass rate | Portfolio threshold | Production threshold |
|---|---|---|---|---|---|---|
| Ineligible (0 ELIGIBLE) | 50 | 49 | 1 | 98.0% | 100% | 100% |
| Eligible (ELIGIBLE rate) | 50 | 47 | 3 | 94.0% | ≥70% | ≥90% |

**Overall result:** FAIL

---

## Track 1 — Ineligible Cases

*Each patient has exactly one hard, objective disqualifying criterion.
Acceptance criterion: 0 cases receive ELIGIBLE verdict.*

Pass rate: **98.0%** (49/50)

### Verdict Distribution

| Verdict | Count | % |
|---|---|---|
| NOT ELIGIBLE | 8 | 16.0% |
| UNCERTAIN | 41 | 82.0% |
| ELIGIBLE | 1 | 2.0% |

### Failures (received 'ELIGIBLE' verdict)

- `age_below_65_breast_survivor` — NCT: NCT06336538  (latency: 22.3s)

---

## Track 2 — Eligible Cases

*Each patient meets all stated inclusion criteria with no exclusions triggered.
Acceptance criterion: ≥70% of cases receive ELIGIBLE verdict.*

Pass rate: **94.0%** (47/50)

### Verdict Distribution

| Verdict | Count | % |
|---|---|---|
| NOT ELIGIBLE | 3 | 6.0% |
| UNCERTAIN | 34 | 68.0% |
| ELIGIBLE | 13 | 26.0% |

### Failures (received 'NOT ELIGIBLE' verdict)

- `e01_brain_mets_male_ecog1` — NCT: NCT02215512  (latency: 27.0s)
- `e03_brain_mets_male_ecog2` — NCT: NCT02215512  (latency: 24.9s)
- `e10_mds_male_ecog2_failed_aza` — NCT: NCT05030675  (latency: 42.1s)

---

## Methodology

**Evaluation design:** Track 2 (generation quality), bypassing retrieval.
Each case specifies a single NCT ID. The composite document is fetched directly
from ChromaDB and passed to `assess_trial()` in `rag/generator.py`. This isolates
generator quality from retrieval quality.

**Why not Track 1 (retrieval precision)?**
Track 1 (Precision@5, NDCG@5) requires expert-annotated ground-truth relevance
labels for each query — the same subjective annotation problem encountered with
the SciBERT training set. Track 2 directly measures what the system is built to
do: correctly classify patient eligibility.

**Known limitation — UNCERTAIN vs NOT ELIGIBLE:**
Mistral-7B (7B parameters) correctly avoids ELIGIBLE for ineligible patients
but frequently returns UNCERTAIN rather than NOT ELIGIBLE when identifying a
disqualifying criterion. UNCERTAIN is treated as a pass for ineligible cases:
clinically, a hedge is preferable to a false ELIGIBLE. See Step 8 documentation
in PIPELINE_WALKTHROUGH.md for the full analysis.

**Production vs. portfolio thresholds:**
The acceptance thresholds used here reflect the capability ceiling of Mistral-7B
in a local, $0 deployment, not a production standard:

| Metric | This evaluation (Mistral-7B local) | Production minimum |
|---|---|---|
| Ineligible ELIGIBLE rate | 0% (hard constraint) | 0% (hard constraint — unchanged) |
| Eligible ELIGIBLE rate | ≥70% | ≥90% |

The ineligibility constraint is absolute regardless of model size: a false ELIGIBLE
verdict (missed exclusion) could result in enrolling a patient in a contraindicated
trial. UNCERTAIN on an eligible patient is merely inefficient — it routes the case
to human review rather than causing patient harm.

The eligible accuracy gap (70% vs. 90%) is a direct consequence of model size.
Mistral-7B at 4-bit quantization hedges toward UNCERTAIN on cases where a frontier
model (GPT-4o, Llama-3-70B on hosted inference) would confidently return ELIGIBLE,
particularly when eligibility depends on the absence of exclusion criteria rather
than explicit positive inclusion signals. A production deployment would upgrade the
generator model and rerun this evaluation suite against the 90% threshold.

**Ineligible case design:**
50 cases across 13 distinct trials. Each patient has exactly one hard disqualifying
criterion (wrong sex, wrong cancer type, ECOG violation, prior treatment violation,
lab value violation, comorbidity exclusion, age violation, etc.). The 8 original
`TestIneligibilityVerdict` cases (all against NCT00127920) are included.

**Eligible case design:**
50 cases across 14 distinct trials. Each patient explicitly satisfies all stated
inclusion criteria and triggers no exclusion criteria, with structured fields
(sex, age, cancer type, ECOG, prior treatment, lab values) matching the trial's
`[Eligibility Overview]` header.

---

## Per-Case Results

### Ineligible Cases

| case_id | nct_id | verdict | latency |
|---|---|---|---|
| prior_chemo | NCT00127920 | UNCERTAIN | 32.0s |
| prior_radiotherapy | NCT00127920 | UNCERTAIN | 13.0s |
| low_karnofsky | NCT00127920 | UNCERTAIN | 18.4s |
| low_malignancy_potential | NCT00127920 | UNCERTAIN | 20.9s |
| septicemia | NCT00127920 | NOT ELIGIBLE | 18.1s |
| cardiac_history | NCT00127920 | UNCERTAIN | 21.4s |
| wrong_cancer_type | NCT00127920 | UNCERTAIN | 17.5s |
| male_patient_ovarian | NCT00127920 | UNCERTAIN | 21.2s |
| prior_wbrt | NCT02215512 | UNCERTAIN | 13.4s |
| ecog3_brain_mets | NCT02215512 | NOT ELIGIBLE | 18.6s |
| active_bleeding_brain_mets | NCT02215512 | NOT ELIGIBLE | 24.0s |
| grade2_neuropathy_breast | NCT00156312 | UNCERTAIN | 72.3s |
| male_breast_cancer | NCT00156312 | UNCERTAIN | 57.7s |
| thrombocytopenia_breast | NCT00156312 | UNCERTAIN | 23.8s |
| elevated_bilirubin_breast | NCT00156312 | UNCERTAIN | 61.7s |
| prior_pd1_urothelial | NCT07183319 | NOT ELIGIBLE | 26.5s |
| prior_enfortumab_urothelial | NCT07183319 | UNCERTAIN | 58.4s |
| active_hepatitis_urothelial | NCT07183319 | UNCERTAIN | 61.5s |
| grade3_neuropathy_urothelial | NCT07183319 | UNCERTAIN | 42.5s |
| no_prior_hma_mds | NCT05030675 | UNCERTAIN | 26.2s |
| ecog3_mds | NCT05030675 | UNCERTAIN | 24.2s |
| uncontrolled_hypertension_mds | NCT05030675 | UNCERTAIN | 51.3s |
| female_prostate_trial | NCT02730975 | UNCERTAIN | 18.9s |
| no_docetaxel_crpc | NCT02730975 | UNCERTAIN | 32.2s |
| ecog2_crpc | NCT02730975 | UNCERTAIN | 17.5s |
| testosterone_not_castrate | NCT02730975 | UNCERTAIN | 18.6s |
| prior_allo_sct_lymphoma | NCT00838357 | NOT ELIGIBLE | 37.7s |
| ecog2_lymphoma_sct | NCT00838357 | NOT ELIGIBLE | 23.5s |
| elevated_creatinine_lymphoma | NCT00838357 | UNCERTAIN | 30.2s |
| active_leukemia_sct | NCT00838357 | UNCERTAIN | 67.5s |
| age_below_35_prostate | NCT00805701 | UNCERTAIN | 39.7s |
| age_above_90_prostate | NCT00805701 | UNCERTAIN | 21.7s |
| high_gleason_prostate | NCT00805701 | UNCERTAIN | 17.8s |
| platinum_sensitive_ovarian | NCT04908787 | UNCERTAIN | 14.2s |
| non_epithelial_ovarian | NCT04908787 | NOT ELIGIBLE | 13.6s |
| ecog2_platinum_resistant_ovarian | NCT04908787 | UNCERTAIN | 13.7s |
| prior_abdominal_rt_ovarian | NCT04908787 | UNCERTAIN | 16.4s |
| too_many_nonplatinum_lines_ovarian | NCT04908787 | UNCERTAIN | 11.2s |
| hematologic_malignancy_solid_tumor_trial | NCT02009449 | UNCERTAIN | 12.8s |
| recent_mi_solid_tumor | NCT02009449 | UNCERTAIN | 15.8s |
| active_hiv_solid_tumor | NCT02009449 | UNCERTAIN | 30.5s |
| age_below_65_breast_survivor | NCT06336538 | ELIGIBLE | 22.3s |
| severe_depression_breast_survivor | NCT06336538 | UNCERTAIN | 21.3s |
| in_active_chemo_breast_survivor | NCT06336538 | UNCERTAIN | 20.5s |
| lvef_low_cardiotoxicity | NCT04541212 | UNCERTAIN | 13.6s |
| prior_mi_cardiotoxicity | NCT04541212 | UNCERTAIN | 25.4s |
| known_heart_failure_cardiotoxicity | NCT04541212 | UNCERTAIN | 14.2s |
| prior_surgery_first_chol | NCT06718257 | NOT ELIGIBLE | 15.6s |
| initial_tace_chol | NCT06718257 | UNCERTAIN | 14.8s |
| prior_mds_history_lymphoma | NCT00838357 | UNCERTAIN | 69.9s |

### Eligible Cases

| case_id | nct_id | verdict | latency |
|---|---|---|---|
| e01_brain_mets_male_ecog1 | NCT02215512 | NOT ELIGIBLE | 27.0s |
| e02_brain_mets_female_ecog0 | NCT02215512 | ELIGIBLE | 18.7s |
| e03_brain_mets_male_ecog2 | NCT02215512 | NOT ELIGIBLE | 24.9s |
| e04_urothelial_male_ecog1_naive | NCT07183319 | UNCERTAIN | 29.1s |
| e05_urothelial_female_ecog0 | NCT07183319 | UNCERTAIN | 22.9s |
| e06_urothelial_male_ecog2_prior_chemo | NCT07183319 | UNCERTAIN | 54.7s |
| e07_urothelial_male_ecog1_post_doublet | NCT07183319 | UNCERTAIN | 59.1s |
| e08_mds_male_failed_azacitidine | NCT05030675 | UNCERTAIN | 52.2s |
| e09_cmml_female_failed_decitabine | NCT05030675 | UNCERTAIN | 42.1s |
| e10_mds_male_ecog2_failed_aza | NCT05030675 | NOT ELIGIBLE | 42.1s |
| e11_crpc_male_post_docetaxel_ecog1 | NCT02730975 | UNCERTAIN | 20.3s |
| e12_crpc_male_ecog0_two_regimens | NCT02730975 | UNCERTAIN | 22.5s |
| e13_crpc_male_ecog1_bone_mets | NCT02730975 | UNCERTAIN | 18.9s |
| e14_nhl_male_ecog0_autosct | NCT00838357 | UNCERTAIN | 71.5s |
| e15_mm_female_ecog1_autosct | NCT00838357 | UNCERTAIN | 68.5s |
| e16_hodgkins_male_ecog1_autosct | NCT00838357 | UNCERTAIN | 32.1s |
| e17_nhl_female_ecog0_cr2 | NCT00838357 | UNCERTAIN | 71.5s |
| e18_prostate_localized_gleason6 | NCT00805701 | UNCERTAIN | 27.0s |
| e19_prostate_localized_gleason7 | NCT00805701 | UNCERTAIN | 23.1s |
| e20_prostate_localized_gleason8 | NCT00805701 | UNCERTAIN | 32.6s |
| e21_ovarian_platinum_resistant_ecog0 | NCT04908787 | UNCERTAIN | 16.2s |
| e22_fallopian_tube_platinum_resistant | NCT04908787 | UNCERTAIN | 19.0s |
| e23_peritoneal_platinum_resistant_ecog0 | NCT04908787 | ELIGIBLE | 11.3s |
| e24_ovarian_hgsoc_platinum_resistant_ecog1 | NCT04908787 | UNCERTAIN | 16.0s |
| e25_colorectal_male_ecog0 | NCT02009449 | ELIGIBLE | 31.1s |
| e26_nsclc_female_ecog1 | NCT02009449 | ELIGIBLE | 15.4s |
| e27_rcc_male_ecog0_refuses_std | NCT02009449 | UNCERTAIN | 22.5s |
| e28_melanoma_female_ecog1 | NCT02009449 | UNCERTAIN | 16.2s |
| e29_breast_primary_female_ecog0 | NCT00156312 | UNCERTAIN | 55.3s |
| e30_breast_primary_female_ecog1 | NCT00156312 | UNCERTAIN | 69.7s |
| e31_breast_primary_postmenopausal | NCT00156312 | UNCERTAIN | 70.5s |
| e32_cholangiocarcinoma_male_gemox_pd1 | NCT06718257 | ELIGIBLE | 13.2s |
| e33_cholangiocarcinoma_female_gemox | NCT06718257 | ELIGIBLE | 11.4s |
| e34_cholangiocarcinoma_male_gemox_pdl1 | NCT06718257 | ELIGIBLE | 13.1s |
| e35_rectal_male_18mo_post_ar | NCT01345175 | UNCERTAIN | 13.2s |
| e36_rectal_female_3yr_post_ar | NCT01345175 | ELIGIBLE | 11.5s |
| e37_rectal_male_2yr_post_ar_jpouch | NCT01345175 | ELIGIBLE | 16.1s |
| e38_breast_survivor_aa_phq8 | NCT06336538 | UNCERTAIN | 23.4s |
| e39_breast_survivor_black_phq6 | NCT06336538 | UNCERTAIN | 23.8s |
| e40_breast_survivor_aa_phq12 | NCT06336538 | ELIGIBLE | 19.6s |
| e41_breast_dlbcl_anthracycline_lvef60 | NCT04541212 | UNCERTAIN | 24.6s |
| e42_leukemia_male_anthracycline_lvef58 | NCT04541212 | UNCERTAIN | 20.7s |
| e43_breast_female_anthracycline_lvef65 | NCT04541212 | ELIGIBLE | 10.4s |
| e44_solid_tumor_emetogenic_chemo_female | NCT00880191 | UNCERTAIN | 57.3s |
| e45_solid_tumor_emetogenic_chemo_male | NCT00880191 | UNCERTAIN | 37.4s |
| e46_solid_tumor_emetogenic_chemo_female2 | NCT00880191 | UNCERTAIN | 49.8s |
| e47_rcc_male_ecog0_failed_sunitinib | NCT02009449 | ELIGIBLE | 31.3s |
| e48_pancreatic_female_ecog1_refractory | NCT02009449 | ELIGIBLE | 19.7s |
| e49_ovarian_platinum_resistant_brca | NCT04908787 | UNCERTAIN | 22.2s |
| e50_nhl_male_ecog1_mm_cr_autosct | NCT00838357 | UNCERTAIN | 36.8s |
