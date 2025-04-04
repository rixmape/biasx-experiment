from enum import Enum
from typing import Callable, Dict, List, Optional, Tuple, Union

from pydantic import BaseModel, Field, computed_field


class DatasetSource(Enum):
    UTKFACE = "utkface"
    FAIRFACE = "fairface"


class DatasetSplit(Enum):
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"


class Gender(Enum):
    MALE = 0
    FEMALE = 1


class Race(Enum):
    WHITE = 0
    BLACK = 1
    ASIAN = 2
    INDIAN = 3
    OTHERS = 4


class Age(Enum):
    AGE_0_9 = 0
    AGE_10_19 = 1
    AGE_20_29 = 2
    AGE_30_39 = 3
    AGE_40_49 = 4
    AGE_50_59 = 5
    AGE_60_69 = 6
    AGE_70_PLUS = 7


class DemographicAttribute(Enum):
    GENDER = "gender"
    RACE = "race"
    AGE = "age"


class Feature(Enum):
    LEFT_EYE = "left_eye"
    RIGHT_EYE = "right_eye"
    NOSE = "nose"
    LIPS = "lips"
    LEFT_CHEEK = "left_cheek"
    RIGHT_CHEEK = "right_cheek"
    CHIN = "chin"
    FOREHEAD = "forehead"
    LEFT_EYEBROW = "left_eyebrow"
    RIGHT_EYEBROW = "right_eyebrow"


class FeatureDetails(BaseModel):
    feature: Feature = Field(...)
    min_x: int = Field(..., ge=0)
    min_y: int = Field(..., ge=0)
    max_x: int = Field(..., ge=0)
    max_y: int = Field(..., ge=0)
    attention_score: float = Field(..., ge=0.0, le=1.0)
    is_key_feature: bool = Field(...)

    @computed_field
    @property
    def area(self) -> int:
        width = self.max_x - self.min_x
        height = self.max_y - self.min_y
        return max(0, width) * max(0, height)


class AttributePerformanceMetrics(BaseModel):
    positive_class: Union[Gender, Race, Age] = Field(...)
    tp: int = Field(..., ge=0)
    fp: int = Field(..., ge=0)
    tn: int = Field(..., ge=0)
    fn: int = Field(..., ge=0)

    @computed_field
    @property
    def tpr(self) -> float:
        return self.tp / max(self.tp + self.fn, 1)

    @computed_field
    @property
    def fpr(self) -> float:
        return self.fp / max(self.fp + self.tn, 1)

    @computed_field
    @property
    def tnr(self) -> float:
        return self.tn / max(self.tn + self.fp, 1)

    @computed_field
    @property
    def fnr(self) -> float:
        return self.fn / max(self.fn + self.tp, 1)

    @computed_field
    @property
    def ppv(self) -> float:
        return self.tp / max(self.tp + self.fp, 1)

    @computed_field
    @property
    def npv(self) -> float:
        return self.tn / max(self.tn + self.fn, 1)

    @computed_field
    @property
    def fdr(self) -> float:
        return self.fp / max(self.fp + self.tp, 1)

    @computed_field
    @property
    def _for(self) -> float:  # Underscore prefix to avoid keyword conflict
        return self.fn / max(self.fn + self.tn, 1)


class BiasMetrics(BaseModel):
    demographic_parity: float = Field(..., ge=0.0)
    equalized_odds: float = Field(..., ge=0.0)
    conditional_use_accuracy_equality: float = Field(..., ge=0.0)
    mean_feature_distribution_bias: float = Field(..., ge=0.0)


class FeatureDistribution(BaseModel):
    feature: Feature = Field(...)
    gender_distributions: Dict[Gender, float] = Field(...)
    race_distributions: Dict[Race, float] = Field(...)
    age_distributions: Dict[Age, float] = Field(...)

    @computed_field
    @property
    def distribution_bias(self) -> float:
        all_dist_values = (
            list(self.gender_distributions.values())
            + list(self.race_distributions.values())
            + list(self.age_distributions.values())
        )
        max_diff = 0.0
        for i in range(len(all_dist_values)):
            for j in range(i + 1, len(all_dist_values)):
                max_diff = max(max_diff, abs(all_dist_values[i] - all_dist_values[j]))
        return max_diff


class Explanation(BaseModel):
    image_id: str = Field(..., min_length=1)
    label: Union[Gender, Race, Age] = Field(...)
    prediction: Union[Gender, Race, Age] = Field(...)
    gender: Gender = Field(...)
    race: Race = Field(...)
    age: Age = Field(...)
    confidence_scores: Tuple[float, ...] = Field(...)
    heatmap_path: Optional[str] = Field(...)
    detected_features: List[FeatureDetails] = Field(...)


class ModelMetadata(BaseModel):
    path: str = Field(..., min_length=1)
    train_loss: Tuple[float, ...] = Field(...)
    train_accuracy: Tuple[float, ...] = Field(...)
    val_loss: Tuple[float, ...] = Field(...)
    val_accuracy: Tuple[float, ...] = Field(...)


class ExperimentResult(BaseModel):
    id: str = Field(..., min_length=1)
    settings: Dict = Field(...)
    model: ModelMetadata = Field(...)
    bias_metrics: BiasMetrics = Field(...)
    feature_distributions: List[FeatureDistribution] = Field(...)
    performance_metrics: List[AttributePerformanceMetrics] = Field(...)
    analyzed_images: List[Explanation] = Field(...)


ProgressCallback = Optional[Callable[[str], None]]
