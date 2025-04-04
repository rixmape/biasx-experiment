import hashlib
from functools import cached_property
from typing import List, Optional, Union

from pydantic import BaseModel, Field, model_validator

# isort: off
from .definitions import Age, DatasetSource, DemographicAttribute, Feature, Gender, Race


class ExperimentSettings(BaseModel):
    predict_attribute: DemographicAttribute = Field(default=DemographicAttribute.GENDER)
    random_seed: int = Field(default=42, ge=0)


class AnalysisSettings(BaseModel):
    protected_attribute: DemographicAttribute = Field(default=DemographicAttribute.GENDER)
    key_feature_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    mask_demographic: Optional[Union[Gender, Race, Age]] = Field(default=None)
    mask_features: Optional[List[Feature]] = Field(default=None)
    mask_pixel_padding: int = Field(default=2, ge=0)

    @model_validator(mode="after")
    def check_masking_logic(self) -> "AnalysisSettings":
        if self.mask_features:
            if not self.mask_demographic:
                raise ValueError("If 'mask_features' is set, 'mask_demographic' must also be provided.")

        if self.mask_demographic is not None:
            expected_type_map = {
                DemographicAttribute.GENDER: Gender,
                DemographicAttribute.RACE: Race,
                DemographicAttribute.AGE: Age,
            }
            expected_type = expected_type_map.get(self.protected_attribute)
            if expected_type and not isinstance(self.mask_demographic, expected_type):
                raise ValueError(
                    f"'mask_demographic' type ({type(self.mask_demographic).__name__}) is inconsistent "
                    f"with 'protected_attribute' ({self.protected_attribute.name}). "
                    f"Expected type: {expected_type.__name__}"
                )
        return self


class DatasetSettings(BaseModel):
    source_name: DatasetSource = Field(default=DatasetSource.UTKFACE)
    target_size: int = Field(default=5000, gt=0)
    validation_ratio: float = Field(default=0.1, ge=0.0, lt=1.0)
    test_ratio: float = Field(default=0.2, ge=0.0, lt=1.0)
    image_size: int = Field(default=48, gt=0)
    use_grayscale: bool = Field(default=False)

    @model_validator(mode="after")
    def check_split_ratios(self) -> "DatasetSettings":
        if self.validation_ratio + self.test_ratio >= 1.0:
            raise ValueError("The sum of 'validation_ratio' and 'test_ratio' must be less than 1.0.")
        return self


class ModelSettings(BaseModel):
    batch_size: int = Field(default=64, gt=0)
    epochs: int = Field(default=10, gt=0)


class OutputSettings(BaseModel):
    base_path: str = Field(default="outputs", min_length=1)
    log_path: str = Field(default="logs", min_length=1)


class Settings(BaseModel):
    experiment: ExperimentSettings = Field(default_factory=ExperimentSettings)
    analysis: AnalysisSettings = Field(default_factory=AnalysisSettings)
    dataset: DatasetSettings = Field(default_factory=DatasetSettings)
    model: ModelSettings = Field(default_factory=ModelSettings)
    output: OutputSettings = Field(default_factory=OutputSettings)

    @cached_property
    def experiment_id(self) -> str:
        config_json = self.model_dump_json(exclude={"experiment_id"})
        hash_object = hashlib.sha256(config_json.encode())
        return hash_object.hexdigest()[:16]
