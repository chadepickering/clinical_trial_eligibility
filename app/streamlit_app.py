"""
Streamlit interface for the Clinical Trial Eligibility Intelligence System.

Pages / sections:
  1. Patient Profile — structured input form
  2. Trial Search — natural language query → RAG retrieval
  3. Eligibility Assessment — Bayesian scorer with uncertainty visualization
  4. Criterion Breakdown — per-criterion explainability table
"""
import streamlit as st


def main():
    st.set_page_config(
        page_title="Clinical Trial Eligibility",
        layout="wide",
    )
    st.title("Clinical Trial Eligibility Intelligence System")
    st.caption("Oncology trial matching with uncertainty quantification")

    # TODO: patient profile sidebar
    # TODO: trial search panel
    # TODO: eligibility assessment with credible interval plot
    # TODO: criterion-by-criterion breakdown table


if __name__ == "__main__":
    main()
