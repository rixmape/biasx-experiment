from collections import defaultdict
from typing import Dict, List, Tuple, Type, Union

import numpy as np

# isort: off
from .definitions import (
    Age,
    AttributePerformanceMetrics,
    BiasMetrics,
    DemographicAttribute,
    DemographicValue,
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
    ) -> DemographicValue:
        if attribute == DemographicAttribute.GENDER:
            return Gender
        if attribute == DemographicAttribute.RACE:
            return Race
        if attribute == DemographicAttribute.AGE:
            return Age

    def _tally_feature_and_demographic_counts(
        self,
        image_details: List[Explanation],
    ) -> Tuple[
        Dict[Feature, Dict[DemographicValue, int]],
        Dict[Type[DemographicValue], Dict[DemographicValue, int]],
    ]:
        feature_counts: Dict[Feature, Dict[DemographicValue, int]] = defaultdict(lambda: defaultdict(int))
        totals = {Gender: defaultdict(int), Race: defaultdict(int), Age: defaultdict(int)}

        for detail in image_details:
            totals[Gender][detail.gender] += 1
            totals[Race][detail.race] += 1
            totals[Age][detail.age] += 1

            for feature_detail in detail.detected_features:
                if feature_detail.is_key_feature:
                    feature = feature_detail.feature
                    feature_counts[feature][detail.gender] += 1
                    feature_counts[feature][detail.race] += 1
                    feature_counts[feature][detail.age] += 1

        return feature_counts, totals

    def _calculate_distribution_for_feature(
        self,
        feature: Feature,
        feature_counts: Dict[Feature, Dict[DemographicValue, int]],
        totals: Dict[Type[DemographicValue], Dict[DemographicValue, int]],
        protected_cls: Type[DemographicValue],
    ) -> FeatureDistribution:
        gender_dist = {g: feature_counts[feature].get(g, 0) / max(totals[Gender].get(g, 0), 1) for g in Gender}
        race_dist = {r: feature_counts[feature].get(r, 0) / max(totals[Race].get(r, 0), 1) for r in Race}
        age_dist = {a: feature_counts[feature].get(a, 0) / max(totals[Age].get(a, 0), 1) for a in Age}

        protected_dists_map = {Gender: gender_dist, Race: race_dist, Age: age_dist}
        protected_values = [protected_dists_map[protected_cls].get(subgroup, 0.0) for subgroup in protected_cls]
        distribution_bias = max(protected_values) - min(protected_values) if protected_values else 0.0

        return FeatureDistribution(
            feature=feature,
            gender_distributions=gender_dist,
            race_distributions=race_dist,
            age_distributions=age_dist,
            distribution_bias=distribution_bias,
        )

    def _compute_feature_distributions(
        self,
        image_details: List[Explanation],
    ) -> List[FeatureDistribution]:
        feature_counts, totals = self._tally_feature_and_demographic_counts(image_details)
        protected_cls = self._get_attribute_enum_class(self.settings.analysis.protected_attribute)
        distributions_list = [self._calculate_distribution_for_feature(feature, feature_counts, totals, protected_cls) for feature in Feature]

        return distributions_list

    def _compute_attribute_performance_metrics(
        self,
        positive_class: DemographicValue,
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
