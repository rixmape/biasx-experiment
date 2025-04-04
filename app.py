import streamlit as st

# isort: off
from app.results_page import render_results_page
from app.settings_page import render_settings_page
from src.definitions import Age, Gender, Race


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
        "filter_gender": "All",
        "filter_race": "All",
        "filter_age": "All",
        "show_feature_boxes": False,
    }
    for key, default_value in default_state.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

    if "gender_options" not in st.session_state:
        st.session_state["gender_options"] = ["All"] + [g.name for g in Gender]
    if "race_options" not in st.session_state:
        st.session_state["race_options"] = ["All"] + [r.name for r in Race]
    if "age_options" not in st.session_state:
        st.session_state["age_options"] = ["All"] + [a.name for a in Age]

    if st.session_state.get("navigate_to_results", False):
        render_results_page()
    else:
        render_settings_page()


if __name__ == "__main__":
    main()
