import os
import random
from typing import Dict, List

import numpy as np
import pandas as pd
import tensorflow as tf
from tf_keras_vis import ModelVisualization

# isort: off
from .analyzer import Analyzer
from .dataset import Dataset
from .definitions import (
    Age,
    DemographicAttribute,
    ExperimentResult,
    Explanation,
    Gender,
    ModelHistory,
    Race,
)
from .explainer import Explainer
from .model import Model
from .settings import Settings


class Runner:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._set_random_seeds()
        self.explainer = Explainer(self.settings)
        self.dataset = Dataset(self.settings, self.explainer)
        self.model = Model(self.settings)
        self.analyzer = Analyzer(self.settings)

    def _set_random_seeds(self) -> None:
        seed = self.settings.experiment.random_seed
        random.seed(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)

    def _get_batch_explanations(
        self,
        batch_df: pd.DataFrame,
        model: tf.keras.Model,
        heatmap_generator: ModelVisualization,
    ) -> List[Explanation]:

        target_col = self.settings.experiment.predict_attribute.value
        target_attr_enum_class = self.analyzer._get_attribute_enum_class(self.settings.experiment.predict_attribute)

        images = np.stack(batch_df["processed_image"].values)
        target_labels = batch_df[target_col].values
        image_ids = batch_df["image_id"].values
        genders = batch_df[DemographicAttribute.GENDER.value].values
        races = batch_df[DemographicAttribute.RACE.value].values
        ages = batch_df[DemographicAttribute.AGE.value].values

        raw_predictions = model.predict(images, verbose=0)
        predicted_label_indices = raw_predictions.argmax(axis=1)

        details = []
        batch_size = images.shape[0]
        for i in range(batch_size):
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

            explanation = Explanation(
                image_id=image_id,
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

        return details

    def _get_all_explanations(
        self,
        test_df: pd.DataFrame,
        model: tf.keras.Model,
    ) -> List[Explanation]:
        heatmap_generator = self.explainer.get_heatmap_generator(model)
        all_explanations = []
        num_samples = len(test_df)
        batch_size = self.settings.model.batch_size

        for i in range(0, num_samples, batch_size):
            end_idx = min(i + batch_size, num_samples)
            batch_df = test_df.iloc[i:end_idx]
            try:
                batch_explanations = self._get_batch_explanations(batch_df, model, heatmap_generator)
                all_explanations.extend(batch_explanations)
            except Exception as e:
                raise RuntimeError(f"Error processing explanation batch starting at index {i}: {e}") from e

        return all_explanations

    def _save_result(
        self,
        model_history: ModelHistory,
        analysis_dict: Dict,
        image_details: List[Explanation],
    ) -> ExperimentResult:

        result = ExperimentResult(
            id=self.settings.experiment_id,
            settings=self.settings.model_dump(mode="json"),
            model_history=model_history,
            bias_metrics=analysis_dict["bias"],
            feature_distributions=analysis_dict["distributions"],
            performance_metrics=analysis_dict["performance"],
            analyzed_images=image_details,
        )

        results_dir = os.path.join(self.settings.output.base_path, self.settings.experiment_id)
        os.makedirs(results_dir, exist_ok=True)
        filename = f"{self.settings.experiment_id}.json"
        path = os.path.join(results_dir, filename)

        try:
            json_string = result.model_dump_json(exclude_none=True, indent=2)
            with open(path, "w") as f:
                f.write(json_string)
        except Exception as e:
            raise RuntimeError(f"Failed to save results to {path}: {e}") from e

        return result

    def run_experiment(self) -> ExperimentResult:
        train_df, val_df, test_df = self.dataset.prepare_datasets(self.settings.experiment.random_seed)

        model, history = self.model.get_model_and_history(train_df, val_df)
        explanations = self._get_all_explanations(test_df, model)
        analysis_dict = self.analyzer.get_bias_analysis(explanations)
        result = self._save_result(history, analysis_dict, explanations)

        return result
