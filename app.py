import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from src.definitions import Age, DatasetSource, DemographicAttribute, Feature, Gender, Race
from src.runner import Runner
from src.settings import AnalysisSettings, DatasetSettings, ExperimentSettings, ModelSettings, OutputSettings, Settings

st.set_page_config(layout="wide", page_title="Bias Analysis Experiment")
st.title("Bias Analysis Experiment Runner")

# --- Sidebar Configuration ---
with st.sidebar:
    st.header("Configuration")

    st.subheader("Experiment Settings")
    exp_predict_attribute = st.selectbox(
        "Predict Attribute",
        options=list(DemographicAttribute),
        format_func=lambda x: x.name.capitalize(),
        index=0,
        key="exp_predict_attribute",
    )
    exp_random_seed = st.number_input("Random Seed", min_value=0, value=42, key="exp_random_seed")

    st.subheader("Dataset Settings")
    ds_source_name = st.selectbox(
        "Dataset Source", options=list(DatasetSource), format_func=lambda x: x.name, index=0, key="ds_source_name"
    )
    ds_target_size = st.number_input("Target Dataset Size", min_value=100, value=5000, step=100, key="ds_target_size")
    ds_validation_ratio = st.slider(
        "Validation Ratio", min_value=0.0, max_value=0.9, value=0.1, step=0.05, key="ds_validation_ratio"
    )
    ds_test_ratio = st.slider("Test Ratio", min_value=0.0, max_value=0.9, value=0.2, step=0.05, key="ds_test_ratio")

    if ds_validation_ratio + ds_test_ratio >= 1.0:
        st.error("Sum of Validation and Test ratios must be less than 1.0")
        valid_ratios = False
    else:
        valid_ratios = True

    ds_image_size = st.number_input("Image Size (pixels)", min_value=32, value=48, step=4, key="ds_image_size")
    ds_use_grayscale = st.checkbox("Use Grayscale Images", value=False, key="ds_use_grayscale")

    st.subheader("Model Settings")
    mdl_batch_size = st.number_input("Batch Size", min_value=8, value=64, step=8, key="mdl_batch_size")
    mdl_epochs = st.number_input("Epochs", min_value=1, value=10, step=1, key="mdl_epochs")

    st.subheader("Analysis Settings")
    an_protected_attribute = st.selectbox(
        "Protected Attribute",
        options=list(DemographicAttribute),
        format_func=lambda x: x.name.capitalize(),
        index=0,
        key="an_protected_attribute",
    )
    an_key_feature_threshold = st.slider(
        "Key Feature Threshold", min_value=0.0, max_value=1.0, value=0.5, step=0.05, key="an_key_feature_threshold"
    )

    an_enable_masking = st.checkbox("Enable Feature Masking (During Training)", value=False, key="an_enable_masking")
    an_mask_demographic = None
    an_mask_features = None
    an_mask_pixel_padding = 2
    valid_masking = True

    if an_enable_masking:
        an_mask_demographic_enum = st.selectbox(
            "Mask Demographic Group (Attribute Value)",
            options=[g for g in Gender] + [r for r in Race] + [a for a in Age],
            format_func=lambda x: x.name,
            key="an_mask_demographic_enum",
        )
        an_mask_demographic = an_mask_demographic_enum

        an_mask_features_enums = st.multiselect(
            "Features to Mask",
            options=list(Feature),
            format_func=lambda x: x.name.replace("_", " ").capitalize(),
            key="an_mask_features_enums",
        )
        an_mask_features = an_mask_features_enums

        an_mask_pixel_padding = st.number_input("Mask Pixel Padding", min_value=0, value=2, key="an_mask_pixel_padding")

        if an_mask_features and not an_mask_demographic:
            st.error("If 'Features to Mask' is set, 'Mask Demographic Group' must also be provided.")
            valid_masking = False
        elif not an_mask_features:
            st.warning("No features selected for masking.")
            an_mask_features = None  # Ensure it's None if empty list selected
            # Allow proceeding even if no features selected, masking will just do nothing.

    run_button = st.button("Run Experiment", disabled=not (valid_ratios and valid_masking))


# --- Main Area ---
def load_and_display_heatmap(base_output_path, exp_id, image_info, image_size, use_grayscale):
    try:
        # Construct image path
        img_folder = f"{image_info.label.__class__.__name__.lower()}_images"  # Assumes DatasetSplit.TEST for display
        img_filename = f"test_{image_info.image_id}.png"
        img_path_rel = os.path.join(exp_id, "test_images", img_filename)  # Assuming test split for simplicity
        img_path_abs = os.path.join(base_output_path, img_path_rel)

        if not os.path.exists(img_path_abs):
            st.warning(f"Image file not found: {img_path_abs}")
            return None  # Skip if base image not found

        img = Image.open(img_path_abs)
        if use_grayscale:
            img = img.convert("L")  # Ensure correct mode for grayscale display
            img_display_np = np.array(img)  # Keep as 2D for cmap='gray'
        else:
            img = img.convert("RGB")  # Ensure RGB
            img_display_np = np.array(img)  # Keep as 3D

        # Construct heatmap path
        if image_info.heatmap_path:
            heatmap_path_abs = os.path.join(base_output_path, image_info.heatmap_path)
            if os.path.exists(heatmap_path_abs):
                heatmap = np.load(heatmap_path_abs).astype(np.float32)  # Load and ensure float32
                # Resize heatmap to match image size if necessary (it should match but good practice)
                if heatmap.shape != (image_size, image_size):
                    heatmap = cv2.resize(heatmap, (image_size, image_size))

                fig, ax = plt.subplots()
                ax.imshow(img_display_np, cmap="gray" if use_grayscale else None)
                im = ax.imshow(heatmap, cmap="jet", alpha=0.5)  # Overlay heatmap
                ax.axis("off")
                fig.colorbar(im, ax=ax)
                return fig
            else:
                st.warning(f"Heatmap file not found: {heatmap_path_abs}")
                # Fallback: show just the image if heatmap missing
                fig, ax = plt.subplots()
                ax.imshow(img_display_np, cmap="gray" if use_grayscale else None)
                ax.axis("off")
                return fig
        else:
            # Show just the image if no heatmap path provided
            fig, ax = plt.subplots()
            ax.imshow(img_display_np, cmap="gray" if use_grayscale else None)
            ax.axis("off")
            return fig

    except Exception as e:
        st.error(f"Error loading/displaying image or heatmap {image_info.image_id}: {e}")
        return None


def display_results(result: "ExperimentResult", settings: Settings):
    st.header(f"Results: Experiment ID {result.id}")

    with st.expander("Bias Metrics Summary", expanded=True):
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Demographic Parity", f"{result.bias_metrics.demographic_parity:.4f}")
        col2.metric("Equalized Odds (TPR Diff)", f"{result.bias_metrics.equalized_odds:.4f}")
        col3.metric(
            "Cond. Use Acc. Equality (PPV Diff)", f"{result.bias_metrics.conditional_use_accuracy_equality:.4f}"
        )
        col4.metric("Mean Feature Dist. Bias", f"{result.bias_metrics.mean_feature_distribution_bias:.4f}")

    with st.expander("Model Performance"):
        # Training History Plot
        history_df = pd.DataFrame(
            {
                "Epoch": range(1, len(result.model.train_loss) + 1),
                "Train Loss": result.model.train_loss,
                "Validation Loss": result.model.val_loss,
                "Train Accuracy": result.model.train_accuracy,
                "Validation Accuracy": result.model.val_accuracy,
            }
        )
        st.line_chart(history_df, x="Epoch", y=["Train Loss", "Validation Loss"])
        st.line_chart(history_df, x="Epoch", y=["Train Accuracy", "Validation Accuracy"])
        st.metric(
            "Final Validation Accuracy", f"{result.model.val_accuracy[-1]:.4f}" if result.model.val_accuracy else "N/A"
        )

        # Detailed Performance Metrics per Group
        st.subheader(f"Performance Metrics per {settings.analysis.protected_attribute.name.capitalize()} Group")
        perf_data = []
        for metrics in result.performance_metrics:
            perf_data.append(
                {
                    "Group": metrics.positive_class.name,
                    "TP": metrics.tp,
                    "FP": metrics.fp,
                    "TN": metrics.tn,
                    "FN": metrics.fn,
                    "TPR": metrics.tpr,
                    "FPR": metrics.fpr,
                    "TNR": metrics.tnr,
                    "FNR": metrics.fnr,
                    "PPV": metrics.ppv,
                    "NPV": metrics.npv,
                    "FDR": metrics.fdr,
                }
            )
        perf_df = pd.DataFrame(perf_data)
        st.dataframe(perf_df.round(4))

    with st.expander("Feature Distribution Analysis"):
        st.subheader("Key Feature Distribution Bias per Feature")
        dist_data = []
        for dist in result.feature_distributions:
            dist_data.append(
                {
                    "Feature": dist.feature.name.replace("_", " ").capitalize(),
                    "Distribution Bias": dist.distribution_bias,
                    # Optionally add detailed distributions per group if needed
                }
            )
        dist_df = pd.DataFrame(dist_data).sort_values("Distribution Bias", ascending=False)
        st.dataframe(dist_df.round(4))

        # Consider adding bar charts for visual comparison if useful
        # st.bar_chart(dist_df.set_index('Feature'), y='Distribution Bias')

    with st.expander("Sample Image Explanations"):
        st.subheader("Explanations for Sample Test Images")
        num_samples = st.slider(
            "Number of samples to show",
            min_value=1,
            max_value=min(50, len(result.analyzed_images)),
            value=min(10, len(result.analyzed_images)),
        )
        sample_indices = np.random.choice(len(result.analyzed_images), num_samples, replace=False)

        base_output_path = settings.output.base_path

        for i in sample_indices:
            image_info = result.analyzed_images[i]
            st.markdown(f"--- \n **Image ID:** `{image_info.image_id}`")
            col1, col2 = st.columns(2)

            with col1:
                st.write(f"**True Label:** {image_info.label.name}")
                st.write(f"**Prediction:** {image_info.prediction.name}")
                st.write("**Demographics:**")
                st.write(f"- Gender: {image_info.gender.name}")
                st.write(f"- Race: {image_info.race.name}")
                st.write(f"- Age: {image_info.age.name}")
                conf_scores_str = ", ".join([f"{s:.3f}" for s in image_info.confidence_scores])
                st.write(f"**Confidence Scores:** ({conf_scores_str})")

                feature_details_data = []
                for fd in image_info.detected_features:
                    feature_details_data.append(
                        {
                            "Feature": fd.feature.name.replace("_", " ").capitalize(),
                            "Attention": fd.attention_score,
                            "Is Key": fd.is_key_feature,
                        }
                    )
                feature_df = pd.DataFrame(feature_details_data).sort_values("Attention", ascending=False)
                st.dataframe(feature_df.round(4))

            with col2:
                fig = load_and_display_heatmap(
                    base_output_path, result.id, image_info, settings.dataset.image_size, settings.dataset.use_grayscale
                )
                if fig:
                    st.pyplot(fig)
                else:
                    st.write("Could not display image/heatmap.")


# --- Run Logic ---
if run_button and valid_ratios and valid_masking:
    exp_settings = ExperimentSettings(predict_attribute=exp_predict_attribute, random_seed=exp_random_seed)
    dataset_settings = DatasetSettings(
        source_name=ds_source_name,
        target_size=ds_target_size,
        validation_ratio=ds_validation_ratio,
        test_ratio=ds_test_ratio,
        image_size=ds_image_size,
        use_grayscale=ds_use_grayscale,
    )
    model_settings = ModelSettings(batch_size=mdl_batch_size, epochs=mdl_epochs)
    analysis_settings = AnalysisSettings(
        protected_attribute=an_protected_attribute,
        key_feature_threshold=an_key_feature_threshold,
        mask_demographic=an_mask_demographic,
        mask_features=an_mask_features,
        mask_pixel_padding=an_mask_pixel_padding,
    )
    output_settings = OutputSettings()

    settings = Settings(
        experiment=exp_settings,
        dataset=dataset_settings,
        model=model_settings,
        analysis=analysis_settings,
        output=output_settings,
    )

    st.info(f"Running experiment with ID: {settings.experiment_id}...")
    with st.spinner("Processing dataset, training model, and analyzing results..."):
        try:
            runner = Runner(settings=settings)
            result = runner.run_experiment()
            st.success("Experiment completed successfully!")
            display_results(result, settings)
        except Exception as e:
            st.error(f"An error occurred during the experiment: {e}")
            import traceback

            st.exception(traceback.format_exc())

elif not (valid_ratios and valid_masking):
    st.warning("Please fix the errors in the sidebar configuration before running.")
else:
    st.info("Configure the experiment settings in the sidebar and click 'Run Experiment'.")
