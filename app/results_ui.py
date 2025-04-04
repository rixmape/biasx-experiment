import os
from typing import List, Optional

import cv2
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from matplotlib.figure import Figure
from PIL import Image

# isort: off
from src.definitions import Age, ExperimentResult, Explanation, FeatureDetails, Gender, Race
from src.settings import Settings

plt.style.use("dark_background")


def reset_page_number():
    if "image_page_number" in st.session_state:
        st.session_state.image_page_number = 0


def display_summary_metrics(result: ExperimentResult):
    st.subheader("Bias Metrics Summary")
    col1, col2, col3, col4 = st.columns(4)
    bias = result.bias_metrics
    col1.metric("Demographic Parity", f"{bias.demographic_parity:.4f}")
    col2.metric("Equalized Odds", f"{bias.equalized_odds:.4f}")
    col3.metric("Cond. Use Acc. Equality", f"{bias.conditional_use_accuracy_equality:.4f}")
    col4.metric("Mean Feature Dist. Bias", f"{bias.mean_feature_distribution_bias:.4f}")


def plot_image_with_heatmap(
    image_info: Explanation,
    base_output_path: str,
    exp_id: str,
    image_size: int,
    use_grayscale: bool,
    show_feature_boxes: bool = False,
    features: Optional[List[FeatureDetails]] = None,
) -> Optional[Figure]:
    fig = None
    try:
        img_filename = f"test_{image_info.image_id}.png"
        img_path_rel = os.path.join(exp_id, "test_images", img_filename)
        img_path_abs = os.path.join(base_output_path, img_path_rel)

        if not os.path.exists(img_path_abs):
            st.warning(f"Image file not found: {img_path_abs}", icon="⚠️")
            return None

        img = Image.open(img_path_abs)
        if use_grayscale:
            img = img.convert("L")
            img_display_np = np.array(img)
            base_cmap = "gray"
        else:
            img = img.convert("RGB")
            img_display_np = np.array(img)
            base_cmap = None

        fig, ax = plt.subplots()
        ax.imshow(img_display_np, cmap=base_cmap)

        if image_info.heatmap_path:
            heatmap_path_abs = os.path.join(base_output_path, image_info.heatmap_path)
            if os.path.exists(heatmap_path_abs):
                heatmap = np.load(heatmap_path_abs).astype(np.float32)
                if heatmap.shape != (image_size, image_size):
                    heatmap = cv2.resize(heatmap, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
                im = ax.imshow(heatmap, cmap="jet", alpha=0.5)
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            else:
                st.warning(f"Heatmap file not found: {heatmap_path_abs}", icon="⚠️")

        if show_feature_boxes and features:
            for feature in features:
                rect = patches.Rectangle(
                    xy=(feature.min_x, feature.min_y),
                    width=feature.max_x - feature.min_x,
                    height=feature.max_y - feature.min_y,
                    linewidth=1.5,
                    edgecolor="black",
                    facecolor="none",
                )
                ax.add_patch(rect)

        ax.axis("off")
        fig.tight_layout()

    except Exception as e:
        st.error(f"Error processing image/heatmap {image_info.image_id}: {e}")
        if fig:
            plt.close(fig)
        return None

    return fig


def display_training_history(result: ExperimentResult):
    st.subheader("Training History")

    history_df_loss = pd.DataFrame(
        {
            "Epoch": range(1, len(result.model.train_loss) + 1),
            "Train Loss": result.model.train_loss,
            "Validation Loss": result.model.val_loss,
        }
    )
    st.line_chart(history_df_loss, x="Epoch", y=["Train Loss", "Validation Loss"], height=300)

    history_df_acc = pd.DataFrame(
        {
            "Epoch": range(1, len(result.model.train_accuracy) + 1),
            "Train Accuracy": result.model.train_accuracy,
            "Validation Accuracy": result.model.val_accuracy,
        }
    )
    st.line_chart(history_df_acc, x="Epoch", y=["Train Accuracy", "Validation Accuracy"], height=300)


def display_performance_metrics(result: ExperimentResult, settings: Settings):
    attribute_name = settings.analysis.protected_attribute.name.capitalize()
    st.subheader(f"Performance Metrics per {attribute_name} Group")
    perf_data = []
    for metrics in result.performance_metrics:
        perf_data.append(
            {
                "Group": metrics.positive_class.name,
                "TPR": metrics.tpr,
                "FPR": metrics.fpr,
                "PPV": metrics.ppv,
                "NPV": metrics.npv,
                "FNR": metrics.fnr,
                "TNR": metrics.tnr,
                "TP": metrics.tp,
                "FP": metrics.fp,
                "TN": metrics.tn,
                "FN": metrics.fn,
            }
        )
    if perf_data:
        perf_df = pd.DataFrame(perf_data)
        st.dataframe(perf_df.round(4))
    else:
        st.info("No performance metrics data available.")


def display_feature_distributions(result: ExperimentResult):
    st.subheader("Key Feature Distribution Bias per Feature")
    dist_data = []
    for dist in result.feature_distributions:
        dist_data.append(
            {
                "Feature": dist.feature.name.replace("_", " ").capitalize(),
                "Distribution Bias": dist.distribution_bias,
            }
        )
    if dist_data:
        dist_df = pd.DataFrame(dist_data).sort_values("Distribution Bias", ascending=False)
        dist_df_chart = dist_df.set_index("Feature")
        if not dist_df_chart.empty:
            st.bar_chart(dist_df_chart, height=300)
        else:
            st.info("No feature distribution data to plot.")
    else:
        st.info("No feature distribution data available.")


def display_sample_explanations(result: ExperimentResult, settings: Settings):
    st.subheader("Sample Image Explanations")
    if not result.analyzed_images:
        st.info("No analyzed images available to display samples.")
        return

    with st.expander("Filters", expanded=True):
        num_samples_max = min(50, len(result.analyzed_images))
        num_samples_default = min(10, num_samples_max)
        num_samples = st.slider("Max Samples in Pool", 1, num_samples_max, num_samples_default, key="num_samples_slider", on_change=reset_page_number)
        selected_gender = st.selectbox("Gender", st.session_state.get("gender_options", ["All"]), key="filter_gender")
        selected_race = st.selectbox("Race", st.session_state.get("race_options", ["All"]), key="filter_race")
        selected_age = st.selectbox("Age Group", st.session_state.get("age_options", ["All"]), key="filter_age")
        show_boxes = st.toggle("Show Feature Boxes", key="show_feature_boxes")

    filtered_images = result.analyzed_images
    if selected_gender != "All":
        filtered_images = [img for img in filtered_images if img.gender.name == selected_gender]
    if selected_race != "All":
        filtered_images = [img for img in filtered_images if img.race.name == selected_race]
    if selected_age != "All":
        filtered_images = [img for img in filtered_images if img.age.name == selected_age]

    filtered_images = filtered_images[:num_samples]

    if not filtered_images:
        st.info("No images match the selected filters or sample size.")
        return

    items_per_page = 4
    page_number = st.session_state.get("image_page_number", 0)
    total_items = len(filtered_images)
    start_index = page_number * items_per_page
    end_index = start_index + items_per_page
    images_to_display = filtered_images[start_index : min(end_index, total_items)]

    cols_per_row = 2
    grid_cols = st.columns(cols_per_row)
    base_output_path = settings.output.base_path

    for i, image_info in enumerate(images_to_display):
        col_idx = i % cols_per_row
        with grid_cols[col_idx]:
            with st.popover(f"Details: {image_info.image_id}", use_container_width=True):
                st.write(f"**True Label:** {image_info.label.name}")
                st.write(f"**Prediction:** {image_info.prediction.name}")
                st.markdown("**Demographics:**")
                st.write(f"- Gender: {image_info.gender.name}")
                st.write(f"- Race: {image_info.race.name}")
                st.write(f"- Age: {image_info.age.name}")

                conf_scores_str = ", ".join([f"{s:.3f}" for s in image_info.confidence_scores])
                st.write(f"**Confidence Scores:** ({conf_scores_str})")

                st.markdown("**Detected Features & Attention:**")
                feature_details_data = []
                for fd in image_info.detected_features:
                    feature_details_data.append(
                        {
                            "Feature": fd.feature.name.replace("_", " ").capitalize(),
                            "Attention": fd.attention_score,
                            "Is Key": fd.is_key_feature,
                        }
                    )
                if feature_details_data:
                    feature_df = pd.DataFrame(feature_details_data).sort_values("Attention", ascending=False)
                    st.dataframe(feature_df.round(4), height=200)
                else:
                    st.info("No features detected for this image.")

            fig = plot_image_with_heatmap(
                image_info=image_info,
                base_output_path=base_output_path,
                exp_id=result.id,
                image_size=settings.dataset.image_size,
                use_grayscale=settings.dataset.use_grayscale,
                show_feature_boxes=show_boxes,
                features=image_info.detected_features,
            )
            if fig:
                st.pyplot(fig, clear_figure=True)
            else:
                st.write(f"Display Error: {image_info.image_id}")

    prev_col, next_col = st.columns([1, 1])
    total_pages = (total_items + items_per_page - 1) // items_per_page

    with prev_col:
        if st.button("<< Previous", disabled=(page_number <= 0), use_container_width=True):
            st.session_state.image_page_number -= 1
            st.rerun()

    with next_col:
        if st.button("Next >>", disabled=(page_number >= total_pages - 1), use_container_width=True):
            st.session_state.image_page_number += 1
            st.rerun()
