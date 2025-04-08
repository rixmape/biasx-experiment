import math
import os

import altair as alt
import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

# isort: off
from src.definitions import DemographicAttribute, ExperimentResult, Explanation


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


def render_bias_metrics(result: ExperimentResult):
    st.subheader(f"Bias Metrics Summary")
    metrics = result.bias_metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(
            label="Demographic Parity",
            value=f"{metrics.demographic_parity:.3f}",
            help="Difference between the largest and smallest selection rates across groups.",
        )
    with col2:
        st.metric(
            label="Equalized Odds",
            value=f"{metrics.equalized_odds:.3f}",
            help="Difference between the largest and smallest true positive rates across groups.",
        )
    with col3:
        st.metric(
            label="Cond. Use Acc. Equality",
            value=f"{metrics.conditional_use_accuracy_equality:.3f}",
            help="Difference between the largest and smallest positive predictive values across groups.",
        )
    with col4:
        st.metric(
            label="Feature Distribution Bias",
            value=f"{metrics.mean_feature_distribution_bias:.3f}",
            help="Average maximum difference in key feature prevalence across groups.",
        )


def render_performance_comparison(result: ExperimentResult):
    st.subheader(f"Performance Metrics by {result.settings['analysis']['protected_attribute'].title()}")
    perf_data = [
        {
            "Group": p.positive_class.name,
            "True Positive Rate": p.tpr,
            "False Positive Rate": p.fpr,
            "Precision": p.ppv,
        }
        for p in result.performance_metrics
    ]

    if perf_data:
        df = pd.DataFrame(perf_data).set_index("Group")
        st.bar_chart(df, horizontal=True)
    else:
        st.warning("No performance metrics data available to display.")


def render_feature_attention_distribution(result: ExperimentResult):
    protected_attribute_name = result.settings["analysis"]["protected_attribute"]
    st.subheader(f"Key Feature Attention Distribution by {protected_attribute_name.title()}")

    distributions = result.feature_distributions
    dist_key_map = {
        "gender": "gender_distributions",
        "race": "race_distributions",
        "age": "age_distributions",
    }
    dist_key = dist_key_map.get(protected_attribute_name)

    if not dist_key or not distributions:
        st.warning(f"Feature distribution data not available for protected attribute '{protected_attribute_name}'.")
        return

    chart_data = []
    for feature_dist in distributions:
        feature_name = feature_dist.feature.name.replace("_", " ").title()
        group_distributions = getattr(feature_dist, dist_key, {})
        if group_distributions:
            for group, value in group_distributions.items():
                chart_data.append(
                    {
                        "Feature": feature_name,
                        "Group": group.name,
                        "Prevalence": value,
                    }
                )

    if chart_data:
        df = pd.DataFrame(chart_data)
        pivot_df = df.pivot(index="Feature", columns="Group", values="Prevalence")
        st.bar_chart(pivot_df, horizontal=True)
    else:
        st.warning("No processed feature distribution data available to display.")


def render_confidence_score_distribution(result: ExperimentResult):
    protected_attribute_name = result.settings["analysis"]["protected_attribute"]
    st.subheader(f"Confidence Score Distribution by {protected_attribute_name.title()}")

    if not result.analyzed_images:
        st.warning("No analyzed image data available to display distributions.")
        return

    data = []
    attr_key = DemographicAttribute(protected_attribute_name).value
    for img in result.analyzed_images:
        group_map = {
            DemographicAttribute.GENDER.value: img.gender.name,
            DemographicAttribute.RACE.value: img.race.name,
            DemographicAttribute.AGE.value: img.age.name,
        }
        data.append(
            {
                "Group": group_map.get(attr_key, "Unknown"),
                "Correct": img.label == img.prediction,
                "Confidence": max(img.confidence_scores) if img.confidence_scores else 0.0,
                "Protected Attribute": protected_attribute_name.title(),
            }
        )

    if not data:
        st.warning("Could not process analyzed image data for confidence scores.")
        return

    df = pd.DataFrame(data)

    def create_horizontal_boxplot(dataframe, title_suffix):
        if dataframe.empty:
            return None
        protected_attr_title = dataframe["Protected Attribute"].iloc[0]
        legend = alt.Legend(title="", orient="bottom", titleOrient="left", offset=10, symbolType="square")
        chart = (
            alt.Chart(dataframe)
            .mark_boxplot(extent="min-max")
            .encode(
                y=alt.Y("Group:N", title=protected_attr_title),
                x=alt.X("Confidence:Q", title="Confidence Score", scale=alt.Scale(zero=False)),
                color=alt.Color("Group:N", title=protected_attr_title, legend=legend),
            )
            .properties(title=f"Confidence Scores ({title_suffix})")
        )
        return chart

    chart_correct = create_horizontal_boxplot(df[df["Correct"] == True], "Correct Predictions")
    chart_incorrect = create_horizontal_boxplot(df[df["Correct"] == False], "Incorrect Predictions")

    if chart_correct:
        st.altair_chart(chart_correct, use_container_width=True)
    else:
        st.info("No data available for correct predictions confidence distribution.")

    if chart_incorrect:
        st.altair_chart(chart_incorrect, use_container_width=True)
    else:
        st.info("No data available for incorrect predictions confidence distribution.")


@st.dialog("Image Explanation Details")
def show_details_dialog(explanation: Explanation):
    st.subheader(f"Image ID: `{explanation.id}`")

    image_path = explanation.image_path
    heatmap_path = explanation.heatmap_path

    original_image = None
    overlay_image = None

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Original Image**")
        if os.path.exists(image_path):
            try:
                original_image = Image.open(image_path)
                st.image(original_image, use_container_width=True)
            except Exception as e:
                st.error(f"Failed to load image: {e}")
        else:
            st.warning("Image file not found.")

    if original_image and heatmap_path and os.path.exists(heatmap_path):
        try:
            heatmap_f16 = np.load(heatmap_path)
            heatmap_f32 = heatmap_f16.astype(np.float32)

            img_rgb_u8 = np.array(original_image.convert("RGB"))
            img_bgr_u8 = cv2.cvtColor(img_rgb_u8, cv2.COLOR_RGB2BGR)
            h, w = img_bgr_u8.shape[:2]

            heatmap_resized = cv2.resize(heatmap_f32, (w, h), interpolation=cv2.INTER_LINEAR)
            heatmap_norm_u8 = cv2.normalize(heatmap_resized, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            heatmap_color_bgr = cv2.applyColorMap(heatmap_norm_u8, cv2.COLORMAP_JET)

            overlay_bgr = cv2.addWeighted(heatmap_color_bgr, 0.5, img_bgr_u8, 0.5, 0)
            overlay_image = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)

        except Exception as e:
            st.error(f"Failed to process heatmap: {e}")
            overlay_image = None

    with col2:
        st.markdown("**Heatmap Overlay**")
        if overlay_image is not None:
            st.image(overlay_image, use_container_width=True)
        elif not heatmap_path:
            st.info("No heatmap data available.")
        elif not os.path.exists(heatmap_path):
            st.warning("Heatmap file not found.")
        else:
            st.warning("Could not generate heatmap.")

    true_label = explanation.label.name
    pred_label = explanation.prediction.name
    status = "✅ Correct" if true_label == pred_label else "❌ Incorrect"
    st.caption(f"**True:** {true_label} | **Pred:** {pred_label} ({status})")

    key_features = [f.feature.name.replace("_", " ").title() for f in explanation.detected_features if f.is_key_feature]
    key_features_str = ", ".join(key_features) if key_features else "None detected"
    st.caption(f"**Key Features:** {key_features_str}")


def render_example_visual_explanations(result: ExperimentResult, max_examples: int = 6):
    st.subheader("Example Visual Explanations")

    if not result.analyzed_images:
        st.warning("No analyzed image data available to display examples.")
        return

    num_images = min(len(result.analyzed_images), max_examples)
    if num_images == 0:
        st.info("No examples to display.")
        return

    cols_per_row = 4
    num_rows = math.ceil(num_images / cols_per_row)
    image_iterator = iter(result.analyzed_images)

    for row_index in range(num_rows):
        cols = st.columns(cols_per_row, gap="small")
        for col_index in range(cols_per_row):
            try:
                explanation: Explanation = next(image_iterator)
                with cols[col_index]:
                    if os.path.exists(explanation.image_path):
                        st.image(explanation.image_path, use_container_width=True)
                    else:
                        st.caption(f"Img {explanation.id} N/A")

                    if st.button(f"See Details", key=f"btn_{explanation.id}", use_container_width=True):
                        show_details_dialog(explanation)

            except StopIteration:
                break
