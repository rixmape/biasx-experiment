import logging
import os
import random
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import tensorflow as tf
from pydantic import ValidationError
from tf_keras_vis import ModelVisualization

# isort: off
from .analyzer import Analyzer
from .dataset import Dataset
from .definitions import Age, DemographicAttribute, ExperimentResult, Explanation, Gender, ModelMetadata, Race
from .explainer import Explainer
from .model import Model
from .settings import Settings
from .utils import setup_logger


class Runner:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.logger = self._setup_logger()
        self._set_random_seeds()

        try:
            self.explainer = Explainer(self.settings)
            self.dataset = Dataset(self.settings, self.explainer)
            self.model = Model(self.settings)
            self.analyzer = Analyzer(self.settings)
            self.logger.info("Runner components initialized successfully.")
        except Exception as e:
            self.logger.error(f"Failed to initialize runner components: {e}")
            raise RuntimeError(f"Failed to initialize runner components: {e}") from e

    def _setup_logger(self) -> logging.Logger:
        try:
            log_path = os.path.join(self.settings.output.base_path, self.settings.experiment_id)
            os.makedirs(log_path, exist_ok=True)
            log_file = os.path.join(log_path, f"{self.settings.experiment_id}.log")
            return setup_logger(log_file=log_file, console_log_level=logging.INFO, file_log_level=logging.DEBUG)
        except Exception as e:
            raise RuntimeError(f"Failed to set up logger: {e}") from e

    def _set_random_seeds(self) -> None:
        try:
            seed = self.settings.experiment.random_seed
            random.seed(seed)
            np.random.seed(seed)
            tf.random.set_seed(seed)
            os.environ["PYTHONHASHSEED"] = str(seed)
            self.logger.info(f"Random seeds set to: {seed}")
        except Exception as e:
            self.logger.error(f"Failed to set random seeds: {e}")

    def _get_batch_explanations(
        self,
        batch_df: pd.DataFrame,
        model: tf.keras.Model,
        heatmap_generator: Optional[ModelVisualization],
    ) -> List[Explanation]:

        if batch_df.empty:
            self.logger.warning("Received empty batch DataFrame for explanation.")
            return []
        if not isinstance(model, tf.keras.Model):
            self.logger.error("Invalid Keras model passed to _get_batch_explanations.")
            return []

        target_col = self.settings.experiment.predict_attribute.value
        try:
            target_attr_enum_class = self.analyzer._get_attribute_enum_class(self.settings.experiment.predict_attribute)
        except ValueError as e:
            self.logger.error(f"Invalid predict_attribute in settings: {e}. Cannot determine target enum.")
            return []

        required_cols = ["processed_image", target_col, "image_id", "gender", "race", "age"]
        if not all(col in batch_df.columns for col in required_cols):
            missing = [col for col in required_cols if col not in batch_df.columns]
            self.logger.error(f"Batch DataFrame missing required columns for explanation: {missing}")
            return []

        try:
            images = np.stack(batch_df["processed_image"].values)
        except ValueError as e_stack:
            self.logger.error(f"Failed to stack images in batch: {e_stack}. Check image shapes.")
            return []

        try:
            raw_predictions = model.predict(images, verbose=0)
            predicted_label_indices = raw_predictions.argmax(axis=1)
        except (tf.errors.InvalidArgumentError, ValueError, Exception) as e_pred:
            self.logger.error(f"Model prediction failed for batch: {e_pred}")
            return []

        target_labels = batch_df[target_col].values
        image_ids = batch_df["image_id"].values
        genders = batch_df[DemographicAttribute.GENDER.value].values
        races = batch_df[DemographicAttribute.RACE.value].values
        ages = batch_df[DemographicAttribute.AGE.value].values

        details = []
        batch_size = images.shape[0]
        for i in range(batch_size):
            try:
                image_np = images[i]
                true_target_label_int = int(target_labels[i])
                predicted_label_int = int(predicted_label_indices[i])
                image_id = str(image_ids[i])

                true_label_enum = target_attr_enum_class(true_target_label_int)
                predicted_label_enum = target_attr_enum_class(predicted_label_int)

                gender_enum = Gender(int(genders[i]))
                race_enum = Race(int(races[i]))
                age_enum = Age(int(ages[i]))

                conf_scores_tuple = tuple(raw_predictions[i].tolist())

                detected_features, heatmap_path = self.explainer.generate_explanation(
                    heatmap_generator,
                    model,
                    image_np,
                    true_label_enum,
                    image_id,
                )

                folder = os.path.join(self.settings.output.base_path, self.settings.experiment_id, "test_images")
                image_path = os.path.join(folder, f"test_{image_id}.png")
                if not os.path.exists(image_path):
                    self.logger.warning(f"Test image file not found: {image_path}")

                explanation = Explanation(
                    id=image_id,
                    image_path=image_path,
                    label=true_label_enum,
                    prediction=predicted_label_enum,
                    gender=gender_enum,
                    race=race_enum,
                    age=age_enum,
                    confidence_scores=conf_scores_tuple,
                    heatmap_path=heatmap_path,
                    detected_features=detected_features,
                )
                details.append(explanation)
            except (ValueError, TypeError, IndexError, Exception) as e_loop:
                item_id = str(image_ids[i]) if i < len(image_ids) else f"index {i}"
                self.logger.error(f"Error processing explanation for item {item_id}: {e_loop}")

        return details

    def _get_all_explanations(self, test_df: pd.DataFrame, model: tf.keras.Model) -> List[Explanation]:
        if test_df.empty:
            self.logger.warning("Test DataFrame is empty. Cannot generate explanations.")
            return []
        if not isinstance(model, tf.keras.Model):
            self.logger.error("Invalid Keras model passed to _get_all_explanations.")
            return []

        heatmap_generator = self.explainer.get_heatmap_generator(model)

        all_explanations = []
        num_samples = len(test_df)
        batch_size = self.settings.model.batch_size
        if batch_size <= 0:
            self.logger.error(f"Invalid batch size ({batch_size}). Setting to 32.")
            batch_size = 32

        self.logger.info(f"Generating explanations for {num_samples} test samples...")
        for i in range(0, num_samples, batch_size):
            end_idx = min(i + batch_size, num_samples)
            batch_df = test_df.iloc[i:end_idx]
            try:
                batch_explanations = self._get_batch_explanations(batch_df, model, heatmap_generator)
                if isinstance(batch_explanations, list):  # Check return type
                    all_explanations.extend(batch_explanations)
                else:
                    self.logger.warning(f"_get_batch_explanations returned non-list type: {type(batch_explanations)}")
            except Exception as e:
                self.logger.error(f"Unhandled error processing explanation batch starting at index {i}: {e}")

        self.logger.info(f"Finished generating explanations. Total explanations: {len(all_explanations)}")
        return all_explanations

    def _save_result(
        self,
        model_metadata: ModelMetadata,
        analysis_dict: Dict,
        image_details: List[Explanation],
    ) -> Optional[ExperimentResult]:

        required_analysis_keys = ["bias", "distributions", "performance"]
        if not isinstance(analysis_dict, dict) or not all(k in analysis_dict for k in required_analysis_keys):
            self.logger.error(f"Analysis dictionary is invalid or missing keys: {analysis_dict.keys() if isinstance(analysis_dict, dict) else 'Not a dict'}")
            return None
        if not isinstance(model_metadata, ModelMetadata) or not model_metadata.path:
            self.logger.error("Invalid ModelMetadata provided.")
            return None
        if not isinstance(image_details, list):
            self.logger.error("Invalid image_details (must be a list).")
            return None

        try:
            result = ExperimentResult(
                id=self.settings.experiment_id,
                settings=self.settings.model_dump(mode="json"),
                model=model_metadata,
                bias_metrics=analysis_dict["bias"],
                feature_distributions=analysis_dict["distributions"],
                performance_metrics=analysis_dict["performance"],
                analyzed_images=image_details,
            )
        except (ValidationError, Exception) as e_val:
            self.logger.error(f"Failed to validate and create ExperimentResult object: {e_val}")
            return None

        savepath_dir = os.path.join(self.settings.output.base_path, self.settings.experiment_id)
        try:
            os.makedirs(savepath_dir, exist_ok=True)
        except OSError as e_dir:
            self.logger.error(f"Could not create result directory {savepath_dir}: {e_dir}")
            return None

        filename = f"{self.settings.experiment_id}.json"
        filepath = os.path.join(savepath_dir, filename)

        try:
            json_string = result.model_dump_json(exclude_none=True, indent=2)
            with open(filepath, "w") as f:
                f.write(json_string)
            self.logger.info(f"Experiment result saved successfully to {filepath}")
            return result
        except (IOError, OSError, TypeError, Exception) as e_save:
            self.logger.error(f"Failed to save results JSON to {filepath}: {e_save}")
            return None

    def run_experiment(self) -> Optional[ExperimentResult]:
        self.logger.info(f"Starting experiment run: {self.settings.experiment_id}")
        try:
            train_df, val_df, test_df = self.dataset.prepare_datasets(self.settings.experiment.random_seed)
            if train_df.empty or val_df.empty or test_df.empty:
                self.logger.error("Dataset preparation returned one or more empty DataFrames.")
                raise ValueError("Empty DataFrame(s) returned from dataset preparation.")
            self.logger.info("Dataset preparation completed successfully.")

            model, model_metadata = self.model.get_model_and_history(train_df, val_df)
            if not isinstance(model, tf.keras.Model) or not isinstance(model_metadata, ModelMetadata) or not model_metadata.path:
                self.logger.error("Model training failed to return valid model or metadata.")
                raise ValueError("Invalid model or metadata returned from model training.")
            self.logger.info("Model training completed successfully.")

            explanations = self._get_all_explanations(test_df, model)

            analysis_dict = self.analyzer.get_bias_analysis(explanations)
            required_keys = ["bias", "distributions", "performance"]
            if not isinstance(analysis_dict, dict) or not all(k in analysis_dict for k in required_keys):
                self.logger.error("Bias analysis failed to return a valid dictionary.")
                raise ValueError("Invalid dictionary returned from bias analysis.")
            self.logger.info("Bias analysis completed successfully.")

            result = self._save_result(model_metadata, analysis_dict, explanations)
            if result is None:
                raise RuntimeError("Failed to save the final experiment result.")

            self.logger.info(f"Experiment run {self.settings.experiment_id} completed successfully.")
            return result

        except (ValueError, RuntimeError, Exception) as e_run:
            self.logger.error(f"Experiment run {self.settings.experiment_id} failed: {e_run}", exc_info=True)
            return None
