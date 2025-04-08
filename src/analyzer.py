from collections import defaultdict
from typing import Dict, List, Union

import numpy as np

# isort: off
from .definitions import (
    Age,
    AttributePerformanceMetrics,
    BiasMetrics,
    DemographicAttribute,
    Explanation,
    Feature,
    FeatureDistribution,
    Gender,
    Race,
)
from .settings import Settings


class Analyzer:
    def __init__(self, settings: Settings):
        self.settings = settings

    def _get_attribute_enum_class(
        self,
        attribute: DemographicAttribute,
    ) -> Union[Gender, Race, Age]:
        if attribute == DemographicAttribute.GENDER:
            return Gender
        if attribute == DemographicAttribute.RACE:
            return Race
        if attribute == DemographicAttribute.AGE:
            return Age

    def _compute_feature_distributions(
        self,
        image_details: List[Explanation],
    ) -> List[FeatureDistribution]:
        feature_counts: Dict[Feature, Dict[Union[Gender, Race, Age], int]] = defaultdict(lambda: defaultdict(int))
        gender_totals: Dict[Gender, int] = defaultdict(int)
        race_totals: Dict[Race, int] = defaultdict(int)
        age_totals: Dict[Age, int] = defaultdict(int)

        for detail in image_details:
            gender_totals[detail.gender] += 1
            race_totals[detail.race] += 1
            age_totals[detail.age] += 1
            for feature_detail in detail.detected_features:
                if feature_detail.is_key_feature:
                    feature_counts[feature_detail.feature][detail.gender] += 1
                    feature_counts[feature_detail.feature][detail.race] += 1
                    feature_counts[feature_detail.feature][detail.age] += 1

        distributions = []
        for feature_enum in Feature:
            gender_dist: Dict[Gender, float] = {val: feature_counts[feature_enum].get(val, 0) / max(gender_totals.get(val, 0), 1) for val in Gender}  # fmt: skip
            race_dist: Dict[Race, float] = {val: feature_counts[feature_enum].get(val, 0) / max(race_totals.get(val, 0), 1) for val in Race}  # fmt: skip
            age_dist: Dict[Age, float] = {val: feature_counts[feature_enum].get(val, 0) / max(age_totals.get(val, 0), 1) for val in Age}  # fmt: skip

            dist = FeatureDistribution(
                feature=feature_enum,
                gender_distributions=gender_dist,
                race_distributions=race_dist,
                age_distributions=age_dist,
            )
            distributions.append(dist)

        return distributions

    def _compute_attribute_performance_metrics(
        self,
        positive_class: Union[Gender, Race, Age],
        labels: np.ndarray,
        predictions: np.ndarray,
    ) -> AttributePerformanceMetrics:
        val = positive_class.value
        is_positive_actual = labels == val
        is_negative_actual = labels != val
        is_positive_pred = predictions == val
        is_negative_pred = predictions != val

        tp = int(np.sum(is_positive_actual & is_positive_pred))
        fn = int(np.sum(is_positive_actual & is_negative_pred))
        fp = int(np.sum(is_negative_actual & is_positive_pred))
        tn = int(np.sum(is_negative_actual & is_negative_pred))

        return AttributePerformanceMetrics(positive_class=positive_class, tp=tp, fp=fp, tn=tn, fn=fn)

    def _compute_bias_metrics(
        self,
        performance_metrics: List[AttributePerformanceMetrics],
        distributions: List[FeatureDistribution],
    ) -> BiasMetrics:
        select_rates = [(m.tp + m.fp) / max(m.tp + m.fp + m.tn + m.fn, 1) for m in performance_metrics]
        tprs = [m.tpr for m in performance_metrics]
        fprs = [m.fpr for m in performance_metrics]
        ppvs = [m.ppv for m in performance_metrics]

        demographic_parity = max(select_rates) - min(select_rates)
        equalized_odds = max(max(tprs) - min(tprs), max(fprs) - min(fprs))
        conditional_use_accuracy_equality = max(ppvs) - min(ppvs)
        mean_feature_distribution_bias = np.mean([dist.distribution_bias for dist in distributions])

        return BiasMetrics(
            demographic_parity=demographic_parity,
            equalized_odds=equalized_odds,
            conditional_use_accuracy_equality=conditional_use_accuracy_equality,
            mean_feature_distribution_bias=mean_feature_distribution_bias,
        )

    def get_bias_analysis(
        self,
        image_details: List[Explanation],
    ) -> Dict:
        true_labels = np.array([detail.label.value for detail in image_details])
        predicted_labels = np.array([detail.prediction.value for detail in image_details])

        feature_distributions = self._compute_feature_distributions(image_details)
        attribute_class = self._get_attribute_enum_class(self.settings.analysis.protected_attribute)

        performance_metrics_list = [self._compute_attribute_performance_metrics(val, true_labels, predicted_labels) for val in attribute_class]  # fmt: skip

        bias_metrics = self._compute_bias_metrics(performance_metrics_list, feature_distributions)

        return {
            "distributions": feature_distributions,
            "performance": performance_metrics_list,
            "bias": bias_metrics,
        }
