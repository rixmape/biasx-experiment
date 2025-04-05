import streamlit as st

# isort: off
from app.results_page import render_results_page
from app.settings_page import render_settings_page


def main():
    st.set_page_config(
        layout="centered",
        page_title="Bias Analysis Experiment",
    )

    st.title("Bias Analysis Experiment")

    default_state = {
        "navigate_to_results": False,
        "run_requested": False,
        "settings": None,
        "experiment_result": None,
    }
    for key, default_value in default_state.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

    if st.session_state.get("navigate_to_results", False):
        render_results_page()
    else:
        render_settings_page()


if __name__ == "__main__":
    main()
