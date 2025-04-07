import math
from typing import Any, Dict

import streamlit as st

from src.definitions import Age, DatasetSource, DemographicAttribute, Feature, Gender, Race


def render_experiment_tab_widgets() -> Dict[str, Any]:
    exp_predict_attribute = st.selectbox(
        "Predict Attribute",
        options=list(DemographicAttribute),
        format_func=lambda x: x.name,
        index=0,
        key="exp_predict_attribute",
    )
    exp_random_seed = st.number_input(
        "Random Seed",
        min_value=0,
        value=42,
        key="exp_random_seed",
    )
    return {
        "predict_attribute": exp_predict_attribute,
        "random_seed": exp_random_seed,
    }


def _render_group_ratios_widget() -> Dict[int, float]:
    """Renders group ratios for the predicted demographic attribute."""
    predict_attribute = st.session_state.get("exp_predict_attribute", DemographicAttribute.GENDER)
    attribute_enum_map = {
        DemographicAttribute.GENDER: Gender,
        DemographicAttribute.RACE: Race,
        DemographicAttribute.AGE: Age,
    }
    target_enum = attribute_enum_map.get(predict_attribute)
    ratio_inputs = {}
    group_ratios_value = None

    num_members = len(target_enum)
    cols = st.columns(num_members)
    default_ratio = 1.0 / num_members

    for i, member in enumerate(target_enum):
        with cols[i]:
            ratio_inputs[member.value] = st.number_input(
                f"{member.name.replace('_', ' ').title()}",
                min_value=0.0,
                max_value=1.0,
                value=default_ratio,
                step=0.01,
                key=f"ds_ratio_{member.name}",
            )

    ratio_sum = sum(ratio_inputs.values())
    if math.isclose(ratio_sum, 1.0, abs_tol=1e-9):
        group_ratios_value = ratio_inputs
    else:
        st.error(f"Entered ratios sum to {ratio_sum:.4f}, not 1.0. Default equal ratios will be used.")
        group_ratios_value = None

    return group_ratios_value


def render_dataset_tab_widgets() -> Dict[str, Any]:
    ds_source_name = st.selectbox(
        "Dataset Source",
        options=list(DatasetSource),
        format_func=lambda x: x.name,
        index=0,
        key="ds_source_name",
    )
    ds_target_size = st.slider(
        "Target Dataset Size",
        min_value=100,
        max_value=20000,
        value=3000,
        step=100,
        key="ds_target_size",
    )
    ds_group_ratios = _render_group_ratios_widget()
    ds_validation_ratio = st.slider(
        "Validation Ratio",
        min_value=0.0,
        max_value=0.95,
        value=0.1,
        step=0.05,
        key="ds_validation_ratio",
    )
    ds_test_ratio = st.slider(
        "Test Ratio",
        min_value=0.0,
        max_value=0.95,
        value=0.2,
        step=0.05,
        key="ds_test_ratio",
    )
    ds_image_size = st.number_input(
        "Image Size (pixels)",
        min_value=32,
        value=48,
        step=4,
        key="ds_image_size",
    )
    ds_use_grayscale = st.checkbox(
        "Use Grayscale Images",
        value=False,
        key="ds_use_grayscale",
    )
    return {
        "source_name": ds_source_name,
        "target_size": ds_target_size,
        "group_ratios": ds_group_ratios,
        "validation_ratio": ds_validation_ratio,
        "test_ratio": ds_test_ratio,
        "image_size": ds_image_size,
        "use_grayscale": ds_use_grayscale,
    }


def render_model_tab_widgets() -> Dict[str, Any]:
    mdl_batch_size = st.number_input(
        "Batch Size",
        min_value=8,
        value=32,
        step=8,
        key="mdl_batch_size",
    )
    mdl_epochs = st.number_input(
        "Epochs",
        min_value=1,
        value=10,
        step=1,
        key="mdl_epochs",
    )
    return {
        "batch_size": mdl_batch_size,
        "epochs": mdl_epochs,
    }


def render_analysis_tab_widgets() -> Dict[str, Any]:
    an_protected_attribute = st.selectbox(
        "Protected Attribute",
        options=list(DemographicAttribute),
        format_func=lambda x: x.name,
        index=0,
        key="an_protected_attribute",
    )
    an_key_feature_threshold = st.slider(
        "Key Feature Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05,
        key="an_key_feature_threshold",
    )
    an_mask_demographic = st.selectbox(
        "Mask Demographic Group",
        options=[None] + list(Gender) + list(Race) + list(Age),
        format_func=lambda x: "None" if x is None else f"{x.__class__.__name__.upper()}.{x.name}",
        key="an_mask_demographic_enum",
        index=0,
    )
    an_mask_features = st.multiselect(
        "Features to Mask",
        options=[None] + list(Feature),
        format_func=lambda x: "None" if x is None else x.name.replace("_", " "),
        key="an_mask_features_enums",
    )
    an_mask_pixel_padding = st.number_input(
        "Mask Pixel Padding",
        min_value=0,
        value=0,
        key="an_mask_pixel_padding",
    )
    return {
        "protected_attribute": an_protected_attribute,
        "key_feature_threshold": an_key_feature_threshold,
        "mask_demographic": an_mask_demographic,
        "mask_features": an_mask_features,
        "mask_pixel_padding": an_mask_pixel_padding,
    }
