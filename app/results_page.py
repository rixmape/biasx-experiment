import traceback

import streamlit as st

# isort: off
from src.runner import Runner

from .results_ui import render_download_button


def render_results_page():
    settings = st.session_state.get("settings")
    run_requested = st.session_state.get("run_requested", False)
    result = st.session_state.get("experiment_result")

    st.subheader(f"Results for experiment `{settings.experiment_id}`" if settings else "Results")

    if run_requested and settings:
        exp_success = False
        exp_result = None
        with st.spinner("Processing dataset, training model, and analyzing results... This may take a while."):
            try:
                runner = Runner(settings=settings)
                exp_result = runner.run_experiment()
                st.session_state["experiment_result"] = exp_result
                exp_success = True
            except Exception as e:
                st.error(f"An error occurred during the experiment run: {e}", icon="🚨")
                st.exception(traceback.format_exc())
                st.session_state["experiment_result"] = None

        st.session_state["run_requested"] = False

        if exp_success:
            st.success("Experiment completed successfully!")
            result = exp_result
        else:
            st.warning("Experiment run failed. Check errors above. Refresh page to try again.")
            result = None

    result = st.session_state.get("experiment_result")
    settings = st.session_state.get("settings")

    if result and settings:
        render_download_button(result)

    elif not run_requested:
        st.warning("No results found. Please run an experiment from the Settings page.")
