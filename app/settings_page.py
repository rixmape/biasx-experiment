import streamlit as st

# isort: off
from src.settings import AnalysisSettings, DatasetSettings, ExperimentSettings, ModelSettings, OutputSettings, Settings

from .settings_ui import render_analysis_tab_widgets, render_dataset_tab_widgets, render_experiment_tab_widgets, render_model_tab_widgets


def render_settings_page():
    st.subheader("Configure experiment settings")
    st.markdown("Please fill in the required fields and click the :red[Run Experiment] button to start.")

    exp_widgets = {}
    ds_widgets = {}
    mdl_widgets = {}
    an_widgets = {}

    tab1, tab2, tab3, tab4 = st.tabs(["Experiment", "Dataset", "Model", "Analysis"])

    with tab1:
        exp_widgets = render_experiment_tab_widgets()
    with tab2:
        ds_widgets = render_dataset_tab_widgets()
    with tab3:
        mdl_widgets = render_model_tab_widgets()
    with tab4:
        an_widgets = render_analysis_tab_widgets()

    st.divider()

    run_button = st.button("Run Experiment", use_container_width=True, type="primary")

    if run_button:
        try:
            settings = Settings(
                experiment=ExperimentSettings(**exp_widgets),
                dataset=DatasetSettings(**ds_widgets),
                model=ModelSettings(**mdl_widgets),
                analysis=AnalysisSettings(**an_widgets),
                output=OutputSettings(),
            )

            st.session_state["settings"] = settings
            st.session_state["run_requested"] = True
            st.session_state["navigate_to_results"] = True
            st.rerun()

        except Exception as e:
            st.error(f"Error creating settings: {e}")
