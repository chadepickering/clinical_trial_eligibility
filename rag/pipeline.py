"""
End-to-end RAG orchestration via LlamaIndex.

Combines: query understanding → retrieval → reranking → generation.
Entry point for the Streamlit app and evaluation scripts.
"""


def run_pipeline(patient_profile: dict, query: str) -> dict:
    """
    Returns: {
        "retrieved_trials": [...],
        "generated_summary": str,
        "source_criteria": [...]
    }
    """
    pass
