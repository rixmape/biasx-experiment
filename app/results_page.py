import traceback

import streamlit as st

# isort: off
from src.runner import Runner

from .results_ui import (
    display_summary_metrics,
    display_training_history,
    display_performance_metrics,
    display_feature_distributions,
    display_sample_explanations,
)


def render_results_page():
    st.subheader("Experiment Results")

    settings = st.session_state.get("settings")
    run_requested = st.session_state.get("run_requested", False)
    result = st.session_state.get("experiment_result")

    if run_requested and settings:
        st.info(f"Running experiment with ID: `{settings.experiment_id}`...")
        exp_success = False
        exp_result = None
        with st.spinner("Processing dataset, training model, and analyzing results... This may take a while."):
            try:
                runner = Runner(settings=settings)
                exp_result = runner.run_experiment()
                st.session_state["experiment_result"] = exp_result
                st.session_state["image_page_number"] = 0
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
            st.warning("Experiment run failed. Check errors above. Go back to settings to try again.")
            result = None

    result = st.session_state.get("experiment_result")
    settings = st.session_state.get("settings")

    if result and settings:
        st.header(f"Results: Experiment ID `{result.id}`")

        tab_titles = [
            "📊 Bias Analysis",
            "📈 Model Performance",
            "🖼️ Sample Explanations",
        ]
        tab1, tab2, tab3 = st.tabs(tab_titles)

        with tab1:
            display_summary_metrics(result)
            display_feature_distributions(result)

        with tab2:
            display_training_history(result)
            display_performance_metrics(result, settings)

        with tab3:
            display_sample_explanations(result, settings)

    elif not run_requested:
        st.warning("No results found. Please run an experiment from the Settings page.")
