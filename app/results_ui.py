import streamlit as st

# isort: off
from src.definitions import ExperimentResult


def render_download_button(result: ExperimentResult):
    try:
        json_data = result.model_dump_json(indent=2)
        st.download_button(
            label="Download Experiment Results (JSON)",
            data=json_data,
            file_name=f"{result.id}_experiment_results.json",
            mime="application/json",
            use_container_width=True,
            key="download_json_button",
        )
    except Exception as e:
        st.error(f"Error preparing download data: {e}")
