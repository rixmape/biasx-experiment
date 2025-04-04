# isort: off
from src.definitions import Age, DatasetSource, DemographicAttribute, Feature, Gender, Race
from src.runner import Runner
from src.settings import AnalysisSettings, Settings, DatasetSettings, ExperimentSettings, ModelSettings, OutputSettings


def main():
    exp_settings = ExperimentSettings(
        predict_attribute=DemographicAttribute.GENDER,
        random_seed=123,
    )

    analysis_settings = AnalysisSettings(
        protected_attribute=DemographicAttribute.RACE,
        key_feature_threshold=0.3,
        mask_demographic=Gender.MALE,
        mask_features=[
            Feature.LEFT_EYE,
            Feature.RIGHT_EYE,
        ],
        mask_pixel_padding=3,
    )

    dataset_settings = DatasetSettings(
        source_name=DatasetSource.UTKFACE,
        target_size=3000,
        validation_ratio=0.1,
        test_ratio=0.2,
        image_size=48,
        use_grayscale=False,  # Better landmark detection with color images
    )

    model_settings = ModelSettings(
        batch_size=32,
        epochs=5,
    )

    output_settings = OutputSettings(
        base_path="outputs",
        log_path="logs",
    )

    settings = Settings(
        experiment=exp_settings,
        analysis=analysis_settings,
        dataset=dataset_settings,
        model=model_settings,
        output=output_settings,
    )

    print(f"Starting experiment run with ID: {settings.experiment_id}")

    try:
        runner = Runner(settings=settings)
        experiment_result = runner.run_experiment()
        print(f"Experiment {experiment_result.id} completed successfully.")
        print(f"Saved results to {settings.output.base_path}/{experiment_result.id}")

    except Exception as e:
        print(f"Experiment failed: {e}")


if __name__ == "__main__":
    main()
