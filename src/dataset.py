import os
from typing import Dict, Tuple

import cv2
import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download
from PIL import Image
from sklearn.model_selection import train_test_split

# isort: off
from .definitions import DatasetSplit, DemographicAttribute, ProgressCallback
from .explainer import Explainer
from .settings import Settings


class Dataset:
    def __init__(self, settings: Settings, explainer: Explainer, progress_callback: ProgressCallback = None):
        self.settings = settings
        self.explainer = explainer
        self.progress_callback = progress_callback

    def _load_raw_dataframe(self) -> pd.DataFrame:
        try:
            repo = f"rixmape/{self.settings.dataset.source_name.value}"
            path = hf_hub_download(
                repo_id=repo,
                filename="data/train-00000-of-00001.parquet",
                repo_type="dataset",
            )
            df = pd.read_parquet(path, columns=["image_id", "image", "gender", "race", "age"])
            return df
        except Exception as e:
            raise RuntimeError(f"Failed to load dataset from {repo}: {e}") from e

    def _process_dataframe(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        df["image_id"] = df["image_id"].astype(str).str[:16]
        df = df.drop_duplicates(subset=["image_id"], keep="first")
        df["image_bytes"] = df["image"].apply(lambda x: x["bytes"])
        df = df.drop(columns=["image"])
        df[["gender", "race", "age"]] = df[["gender", "race", "age"]].astype(int)
        return df

    def _get_sampled_attribute_subset(
        self,
        df: pd.DataFrame,
        attribute_column: str,
        attribute_value: int,
        target_size: int,
        seed: int,
    ) -> pd.DataFrame:
        subset_df = df[df[attribute_column] == attribute_value].copy()
        total_rows = len(subset_df)

        samples = []
        for _, group in subset_df.groupby("strata"):
            group_size = len(group)
            if group_size == 0:
                continue
            group_proportion = group_size / total_rows
            group_sample_size = max(1, round(target_size * group_proportion))
            replacement_needed = group_size < group_sample_size
            sample = group.sample(n=group_sample_size, random_state=seed, replace=replacement_needed)
            samples.append(sample)

        return pd.concat(samples) if samples else pd.DataFrame(columns=subset_df.columns)

    def _sample_dataframe(
        self,
        df: pd.DataFrame,
        seed: int,
    ) -> pd.DataFrame:
        target_col = self.settings.experiment.predict_attribute.value
        strata_cols = [col.value for col in DemographicAttribute if col.value != target_col]

        df["strata"] = df[strata_cols].astype(str).agg("_".join, axis=1)
        unique_values = df[target_col].unique()
        target_size_per_value = max(1, self.settings.dataset.target_size // len(unique_values))

        sampled_subsets = [
            self._get_sampled_attribute_subset(df, target_col, value, target_size_per_value, seed)
            for value in unique_values
        ]
        sampled_subsets = [s for s in sampled_subsets if not s.empty]

        if not sampled_subsets:
            raise ValueError("Sampling resulted in an empty DataFrame. Cannot proceed.")

        combined_df = pd.concat(sampled_subsets)
        combined_df = combined_df.sample(frac=1, random_state=seed).reset_index(drop=True)
        combined_df = combined_df.drop(columns=["strata"])
        return combined_df

    def _split_dataframe(
        self,
        df: pd.DataFrame,
        seed: int,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        stratify_col_name = self.settings.experiment.predict_attribute.value

        train_val_df, test_df = train_test_split(
            df,
            test_size=self.settings.dataset.test_ratio,
            random_state=seed,
            stratify=df[stratify_col_name],
        )

        adjusted_val_ratio = min(1.0, self.settings.dataset.validation_ratio / (1.0 - self.settings.dataset.test_ratio))

        train_df, val_df = train_test_split(
            train_val_df,
            test_size=adjusted_val_ratio,
            random_state=seed,
            stratify=train_val_df[stratify_col_name],
        )

        return train_df, val_df, test_df

    def _preprocess_single_image(
        self,
        image_bytes: bytes,
        purpose: DatasetSplit,
        label_dict: Dict[str, int],
    ) -> np.ndarray:
        image_np = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
        if image_np is None:
            raise ValueError("cv2.imdecode returned None for an image.")
        image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        image_np = cv2.resize(image_np, (self.settings.dataset.image_size, self.settings.dataset.image_size))
        image_np = image_np.astype(np.float32) / 255.0

        if purpose == DatasetSplit.TRAIN and self.settings.analysis.mask_demographic is not None:
            mask_attribute_type_enum = self.settings.analysis.protected_attribute
            mask_attribute_value_enum = self.settings.analysis.mask_demographic
            attribute_key = mask_attribute_type_enum.value
            actual_image_value = label_dict.get(attribute_key)

            if actual_image_value is not None and actual_image_value == mask_attribute_value_enum.value:
                if self.settings.analysis.mask_features:
                    image_np = self.explainer.apply_mask(image_np)

        if self.settings.dataset.use_grayscale:
            if len(image_np.shape) == 3 and image_np.shape[-1] == 3:
                image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
            image_np = np.expand_dims(image_np, axis=-1)

        return image_np.astype(np.float32)

    def _preprocess_dataframe_split(
        self,
        df: pd.DataFrame,
        purpose: DatasetSplit,
    ) -> pd.DataFrame:
        processed_images = []
        for _, row in df.iterrows():
            label_dict = {
                DemographicAttribute.GENDER.value: row["gender"],
                DemographicAttribute.RACE.value: row["race"],
                DemographicAttribute.AGE.value: row["age"],
            }
            processed_image = self._preprocess_single_image(row["image_bytes"], purpose, label_dict)
            processed_images.append(processed_image)

        df_copy = df.copy()
        df_copy.loc[:, "processed_image"] = processed_images
        return df_copy

    def _save_images(
        self,
        df: pd.DataFrame,
        purpose: DatasetSplit,
    ) -> None:
        savepath = os.path.join(
            self.settings.output.base_path,
            self.settings.experiment_id,
            f"{purpose.value.lower()}_images",
        )
        os.makedirs(savepath, exist_ok=True)

        for _, row in df.iterrows():
            img_array = row["processed_image"]
            img_id = row["image_id"]

            img_to_save = (img_array * 255.0).clip(0, 255).astype(np.uint8)

            if img_to_save.shape[-1] == 1:
                pil_img = Image.fromarray(np.squeeze(img_to_save, axis=-1), mode="L")
            elif img_to_save.shape[-1] == 3:
                pil_img = Image.fromarray(img_to_save, mode="RGB")

            filename = f"{purpose.value.lower()}_{img_id}.png"
            filepath = os.path.join(savepath, filename)
            pil_img.save(filepath, format="PNG")

    def prepare_datasets(
        self,
        seed: int,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        raw_df = self._load_raw_dataframe()
        processed_df = self._process_dataframe(raw_df)
        sampled_df = self._sample_dataframe(processed_df, seed)

        train_df, val_df, test_df = self._split_dataframe(sampled_df, seed)

        train_df_processed = self._preprocess_dataframe_split(train_df, DatasetSplit.TRAIN)
        val_df_processed = self._preprocess_dataframe_split(val_df, DatasetSplit.VALIDATION)
        test_df_processed = self._preprocess_dataframe_split(test_df, DatasetSplit.TEST)

        self._save_images(train_df_processed, DatasetSplit.TRAIN)
        self._save_images(val_df_processed, DatasetSplit.VALIDATION)
        self._save_images(test_df_processed, DatasetSplit.TEST)

        return train_df_processed, val_df_processed, test_df_processed
