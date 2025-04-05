import logging
from collections import defaultdict
from typing import Dict, List, Type, Union

import numpy as np

# isort: off
from .definitions import Age, AttributePerformanceMetrics, BiasMetrics, DemographicAttribute, Explanation, Feature, FeatureDistribution, Gender, Race
from .settings import Settings
from .utils import LOGGER_NAME

logger = logging.getLogger(LOGGER_NAME)


class Analyzer:
    def __init__(self, settings: Settings):
        self.settings = settings

    def _get_attribute_enum_class(self, attribute: DemographicAttribute) -> Type[Union[Gender, Race, Age]]:
        if attribute == DemographicAttribute.GENDER:
            return Gender
        if attribute == DemographicAttribute.RACE:
            return Race
        if attribute == DemographicAttribute.AGE:
            return Age
        logger.error(f"Invalid DemographicAttribute provided: {attribute}")
        raise ValueError(f"Invalid DemographicAttribute: {attribute}")

    def _validate_explanation(self, detail: Explanation, index: int) -> bool:
        required_attrs = ["label", "prediction", "gender", "race", "age", "detected_features"]
        if not isinstance(detail, Explanation):
            logger.warning(f"Item at index {index} is not an Explanation object (type: {type(detail)}). Skipping.")
            return False
        for attr in required_attrs:
            if not hasattr(detail, attr) or getattr(detail, attr) is None:
                logger.warning(f"Explanation object at index {index} missing or has None value for '{attr}'. Skipping.")
                return False
        if not isinstance(detail.detected_features, list):
            logger.warning(f"Explanation object at index {index} has non-list 'detected_features' (type: {type(detail.detected_features)}). Skipping.")
            return False
        return True

    def _compute_feature_distributions(self, image_details: List[Explanation]) -> List[FeatureDistribution]:
        feature_counts: Dict[Feature, Dict[Union[Gender, Race, Age], int]] = defaultdict(lambda: defaultdict(int))
        gender_totals: Dict[Gender, int] = defaultdict(int)
        race_totals: Dict[Race, int] = defaultdict(int)
        age_totals: Dict[Age, int] = defaultdict(int)

        valid_details_count = 0
        for idx, detail in enumerate(image_details):
            if not self._validate_explanation(detail, idx):
                continue
            valid_details_count += 1

            try:
                gender_totals[detail.gender] += 1
                race_totals[detail.race] += 1
                age_totals[detail.age] += 1
                if detail.detected_features:
                    for feature_detail in detail.detected_features:
                        if hasattr(feature_detail, "is_key_feature") and feature_detail.is_key_feature and hasattr(feature_detail, "feature"):
                            feature_counts[feature_detail.feature][detail.gender] += 1
                            feature_counts[feature_detail.feature][detail.race] += 1
                            feature_counts[feature_detail.feature][detail.age] += 1
            except (AttributeError, TypeError) as e:
                logger.warning(f"Error accessing attributes for explanation at index {idx}: {e}. Skipping detail.")
                continue

        if valid_details_count == 0:
            logger.warning("No valid Explanation objects found to compute feature distributions.")
            return []

        distributions = []
        for feature_enum in Feature:
            try:
                gender_dist: Dict[Gender, float] = {val: feature_counts[feature_enum].get(val, 0) / max(gender_totals.get(val, 1), 1) for val in Gender}
                race_dist: Dict[Race, float] = {val: feature_counts[feature_enum].get(val, 0) / max(race_totals.get(val, 1), 1) for val in Race}
                age_dist: Dict[Age, float] = {val: feature_counts[feature_enum].get(val, 0) / max(age_totals.get(val, 1), 1) for val in Age}
                dist = FeatureDistribution(
                    feature=feature_enum,
                    gender_distributions=gender_dist,
                    race_distributions=race_dist,
                    age_distributions=age_dist,
                )
                distributions.append(dist)
            except Exception as e_dist:
                logger.error(f"Error calculating distribution for feature {feature_enum.name}: {e_dist}")

        logger.info(f"Computed feature distributions for {len(distributions)} features based on {valid_details_count} valid explanations.")
        return distributions

    def _compute_attribute_performance_metrics(
        self,
        positive_class: Union[Gender, Race, Age],
        labels: np.ndarray,
        predictions: np.ndarray,
    ) -> AttributePerformanceMetrics:
        val = positive_class.value
        try:
            is_positive_actual = labels == val
            is_negative_actual = labels != val
            is_positive_pred = predictions == val
            is_negative_pred = predictions != val

            tp = int(np.sum(is_positive_actual & is_positive_pred))
            fn = int(np.sum(is_positive_actual & is_negative_pred))
            fp = int(np.sum(is_negative_actual & is_positive_pred))
            tn = int(np.sum(is_negative_actual & is_negative_pred))

            return AttributePerformanceMetrics(positive_class=positive_class, tp=tp, fp=fp, tn=tn, fn=fn)
        except (TypeError, ValueError) as e:
            logger.error(f"Error computing performance metrics for class {positive_class.name}: {e}. Check label/prediction array contents.")
            return AttributePerformanceMetrics(positive_class=positive_class, tp=0, fp=0, tn=0, fn=0)

    def _compute_bias_metrics(
        self,
        performance_metrics: List[AttributePerformanceMetrics],
        distributions: List[FeatureDistribution],
    ) -> BiasMetrics:
        if not performance_metrics:
            logger.warning("Performance metrics list is empty. Returning default bias metrics.")
            return BiasMetrics(
                demographic_parity=0.0,
                equalized_odds=0.0,
                conditional_use_accuracy_equality=0.0,
                mean_feature_distribution_bias=0.0,
            )

        try:
            select_rates = [(m.tp + m.fp) / max(m.tp + m.fp + m.tn + m.fn, 1) for m in performance_metrics]
            tprs = [m.tpr for m in performance_metrics]
            ppvs = [m.ppv for m in performance_metrics]

            demographic_parity = max(select_rates) - min(select_rates) if select_rates else 0.0
            equalized_odds = max(tprs) - min(tprs) if tprs else 0.0
            conditional_use_accuracy_equality = max(ppvs) - min(ppvs) if ppvs else 0.0

            dist_biases = [dist.distribution_bias for dist in distributions if hasattr(dist, "distribution_bias")]
            mean_feature_distribution_bias = np.mean(dist_biases) if dist_biases else 0.0

            return BiasMetrics(
                demographic_parity=demographic_parity,
                equalized_odds=equalized_odds,
                conditional_use_accuracy_equality=conditional_use_accuracy_equality,
                mean_feature_distribution_bias=mean_feature_distribution_bias,
            )
        except (ValueError, TypeError, Exception) as e:
            logger.error(f"Error calculating bias metrics: {e}. Returning default bias metrics.")
            return BiasMetrics(
                demographic_parity=0.0,
                equalized_odds=0.0,
                conditional_use_accuracy_equality=0.0,
                mean_feature_distribution_bias=0.0,
            )

    def get_bias_analysis(self, image_details: List[Explanation]) -> Dict:
        if not isinstance(image_details, list) or not image_details:
            logger.warning("Input 'image_details' is not a non-empty list. Returning empty analysis.")
            return {
                "distributions": [],
                "performance": [],
                "bias": BiasMetrics(
                    demographic_parity=0.0,
                    equalized_odds=0.0,
                    conditional_use_accuracy_equality=0.0,
                    mean_feature_distribution_bias=0.0,
                ),
            }

        valid_labels = []
        valid_predictions = []
        valid_indices = []

        for idx, detail in enumerate(image_details):
            if self._validate_explanation(detail, idx):
                if hasattr(detail.label, "value") and hasattr(detail.prediction, "value"):
                    valid_labels.append(detail.label.value)
                    valid_predictions.append(detail.prediction.value)
                    valid_indices.append(idx)
                else:
                    logger.warning(f"Explanation at index {idx} missing .value attribute for label or prediction. Skipping.")

        if not valid_labels:
            logger.warning("No valid labels/predictions found in image_details. Returning empty analysis.")
            bias_metrics = BiasMetrics(
                demographic_parity=0.0,
                equalized_odds=0.0,
                conditional_use_accuracy_equality=0.0,
                mean_feature_distribution_bias=0.0,
            )
            return {
                "distributions": [],
                "performance": [],
                "bias": bias_metrics,
            }

        try:
            true_labels = np.array(valid_labels)
            predicted_labels = np.array(valid_predictions)
        except (ValueError, TypeError) as e:
            logger.error(f"Failed to create numpy arrays from labels/predictions: {e}. Returning empty analysis.")
            bias_metrics = BiasMetrics(
                demographic_parity=0.0,
                equalized_odds=0.0,
                conditional_use_accuracy_equality=0.0,
                mean_feature_distribution_bias=0.0,
            )
            return {
                "distributions": [],
                "performance": [],
                "bias": bias_metrics,
            }

        valid_image_details = [image_details[i] for i in valid_indices]
        feature_distributions = self._compute_feature_distributions(valid_image_details)
        attribute_class = self._get_attribute_enum_class(self.settings.analysis.protected_attribute)

        performance_metrics_list = [self._compute_attribute_performance_metrics(val, true_labels, predicted_labels) for val in attribute_class]
        bias_metrics = self._compute_bias_metrics(performance_metrics_list, feature_distributions)

        logger.info("Bias analysis computation completed.")
        return {
            "distributions": feature_distributions,
            "performance": performance_metrics_list,
            "bias": bias_metrics,
        }
