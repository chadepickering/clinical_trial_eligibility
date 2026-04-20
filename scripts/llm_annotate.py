"""
LLM annotation of sampled criteria using the Anthropic API.

Reads the CSV produced by sample_annotation.py, calls Claude to label
each criterion for B2 (objective/subjective) and B3 (observable/unobservable)
with a self-reported confidence score, and writes results back to the CSV.

The rubric is embedded in the system prompt. Each criterion is a separate
API call to keep responses clean and parseable.

Skips rows that already have LLM labels (safe to re-run after interruption).

Usage:
    python scripts/llm_annotate.py
    python scripts/llm_annotate.py --csv data/annotation/sample.csv \
        --model claude-sonnet-4-6 --batch-size 20
"""

import argparse
import json
import os
import time

import anthropic
import pandas as pd
from dotenv import load_dotenv

load_dotenv()


SYSTEM_PROMPT = """You are an expert clinical trial eligibility annotator.

You will be given a single clinical trial eligibility criterion and must label it
on two dimensions:

## B2 — Objective (1) vs Subjective (0)

Core question: Could two independent clinicians evaluate this criterion against a
patient record and always reach the same conclusion?

Label 1 (Objective): A specific, predefined threshold or verifiable clinical state
exists. The answer is yes/no with no judgment required.
  Examples: "Creatinine ≤ 1.5 mg/dL", "ECOG ≤ 2", "HBsAg negative",
            "Histologically confirmed adenocarcinoma"

Label 0 (Subjective): Requires a clinician to make a judgment call. No numeric
threshold or named scale is given.
  Examples: "Adequate hepatic function", "Willing to use contraception",
            "Clinically significant cardiac disease"

Edge cases:
- Compound criteria: if "adequate X" is immediately defined by objective sub-criteria
  (e.g. "adequate renal function defined by creatinine ≤ 1.5×ULN") → label 1
- Named scales without a number stated → label 0
- Binary clinical facts (confirmed diagnoses, serostatus, pathological findings)
  → label 1 even without a numeric threshold
- Symptoms described qualitatively without a validated scale → label 0

## B3 — Observable (1) vs Unobservable (0)

Core question: Could a clinician answer this criterion by reviewing a standard
patient EHR — without asking the patient anything or making predictions about
the future?

Label 1 (Observable): Information exists as a documented fact in a standard EHR —
lab results, diagnoses, imaging, medications, demographics, performance status,
documented history, known allergies.
  Examples: "Hemoglobin ≥ 10 g/dL", "Prior platinum-based chemotherapy",
            "Stage III ovarian cancer", "Known hypersensitivity to bevacizumab"

Label 0 (Unobservable): Requires data typically absent from an EHR — patient intent,
willingness, life expectancy estimates, geographic/logistical constraints,
access to technology, legal capacity.
  Examples: "Willing to use contraception", "Life expectancy ≥ 3 months",
            "Able to attend scheduled visits"

Edge cases:
- Consent and willingness → always 0
- "Adequate organ function" without a defining clause → 0
- "Adequate organ function defined by [lab values]" → 1
- History of a condition → 1
- Life expectancy → always 0
- Pregnancy/reproductive status documented by test or record → 1;
  intent to become pregnant → 0

## Output format

Respond with a single JSON object — no other text:
{
  "b2_label": <0 or 1>,
  "b2_confidence": <float 0.0-1.0>,
  "b3_label": <0 or 1>,
  "b3_confidence": <float 0.0-1.0>,
  "b2_reasoning": "<one sentence>",
  "b3_reasoning": "<one sentence>"
}

Confidence reflects how clearly the rubric resolves the case:
  1.0 = unambiguous
  0.8 = clear with minor caveats
  0.6 = plausible but genuinely uncertain
  below 0.6 = edge case, flag for human review
"""


def annotate_criterion(client: anthropic.Anthropic, text: str, model: str) -> dict:
    """Call Claude to label one criterion. Returns parsed JSON dict."""
    message = client.messages.create(
        model=model,
        max_tokens=256,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": f"Criterion: {text}"}],
    )
    raw = message.content[0].text.strip()
    # Strip markdown code fences if present
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    return json.loads(raw)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv',        default='data/annotation/sample.csv')
    parser.add_argument('--model',      default='claude-sonnet-4-6')
    parser.add_argument('--delay',      type=float, default=0.3,
                        help='seconds between API calls to avoid rate limiting')
    args = parser.parse_args()

    api_key = os.environ.get('ANTHROPIC_API_KEY')
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY environment variable not set")

    client = anthropic.Anthropic(api_key=api_key)
    df = pd.read_csv(args.csv)

    # Only process rows not yet annotated
    todo = df['llm_b2_label'].isna()
    print(f"{todo.sum()} rows to annotate ({(~todo).sum()} already done)")

    errors = 0
    for idx in df[todo].index:
        text = df.at[idx, 'text']
        try:
            result = annotate_criterion(client, text, args.model)
            df.at[idx, 'llm_b2_label']      = int(result['b2_label'])
            df.at[idx, 'llm_b2_confidence']  = float(result['b2_confidence'])
            df.at[idx, 'llm_b3_label']       = int(result['b3_label'])
            df.at[idx, 'llm_b3_confidence']  = float(result['b3_confidence'])
        except Exception as e:
            print(f"  Row {idx} failed: {e}")
            errors += 1
            continue

        if (idx - df[todo].index[0] + 1) % 50 == 0:
            df.to_csv(args.csv, index=False)
            print(f"  {idx - df[todo].index[0] + 1}/{todo.sum()} annotated (checkpoint saved)")

        time.sleep(args.delay)

    df.to_csv(args.csv, index=False)
    print(f"\nDone. {todo.sum() - errors} annotated, {errors} errors.")
    print(f"Results written to {args.csv}")


if __name__ == '__main__':
    main()
