import logging
import os
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download
from PIL import Image
from sklearn.model_selection import train_test_split

# isort: off
from .definitions import DatasetSplit, DemographicAttribute
from .explainer import Explainer
from .settings import Settings
from .utils import LOGGER_NAME

logger = logging.getLogger(LOGGER_NAME)


class Dataset:
    def __init__(self, settings: Settings, explainer: Explainer):
        self.settings = settings
        self.explainer = explainer

    def _load_raw_dataframe(self) -> pd.DataFrame:
        try:
            repo = f"rixmape/{self.settings.dataset.source_name.value}"
            path = hf_hub_download(
                repo_id=repo,
                filename="data/train-00000-of-00001.parquet",
                repo_type="dataset",
            )
            df = pd.read_parquet(path)
            logger.info(f"Successfully loaded dataset from {repo}")
            return df
        except Exception as e:
            logger.error(f"Failed to load dataset from {repo}: {e}")
            raise RuntimeError(f"Failed to load dataset from {repo}: {e}") from e

    def _process_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        required_cols = ["image_id", "image", "gender", "race", "age"]
        if not all(col in df.columns for col in required_cols):
            missing_cols = [col for col in required_cols if col not in df.columns]
            logger.error(f"DataFrame missing required columns: {missing_cols}")
            raise ValueError(f"DataFrame missing required columns: {missing_cols}")

        try:
            df["image_id"] = df["image_id"].astype(str).str[:16]
            df = df.drop_duplicates(subset=["image_id"], keep="first")
            if "bytes" not in df["image"].iloc[0]:
                logger.error("Column 'image' does not contain 'bytes' key.")
                raise ValueError("Column 'image' does not contain 'bytes' key.")
            df["image_bytes"] = df["image"].apply(lambda x: x["bytes"])
            df = df.drop(columns=["image"])
            df[["gender", "race", "age"]] = df[["gender", "race", "age"]].astype(int)
            logger.info("DataFrame processing completed.")
            return df
        except (KeyError, TypeError, Exception) as e:
            logger.error(f"Error processing DataFrame: {e}")
            raise RuntimeError(f"Error processing DataFrame: {e}") from e

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
        if total_rows == 0:
            return pd.DataFrame(columns=df.columns)

        samples = []
        for _, group in subset_df.groupby("strata"):
            group_size = len(group)
            if group_size == 0:
                continue
            group_proportion = group_size / total_rows
            group_sample_size = max(1, round(target_size * group_proportion))
            replacement_needed = group_size < group_sample_size
            try:
                sample = group.sample(n=group_sample_size, random_state=seed, replace=replacement_needed)
                samples.append(sample)
            except Exception as e:
                logger.warning(f"Could not sample group: {e}")

        return pd.concat(samples) if samples else pd.DataFrame(columns=subset_df.columns)

    def _sample_dataframe(self, df: pd.DataFrame, seed: int) -> pd.DataFrame:
        target_col = self.settings.experiment.predict_attribute.value
        strata_cols = [col.value for col in DemographicAttribute if col.value != target_col]

        df["strata"] = df[strata_cols].astype(str).agg("_".join, axis=1)
        unique_values = df[target_col].unique()
        if len(unique_values) == 0:
            logger.error("Target column has no unique values for sampling.")
            raise ValueError("Target column has no unique values for sampling.")
        target_size_per_value = max(1, self.settings.dataset.target_size // len(unique_values))

        sampled_subsets = [self._get_sampled_attribute_subset(df, target_col, value, target_size_per_value, seed) for value in unique_values]
        sampled_subsets = [s for s in sampled_subsets if not s.empty]

        if not sampled_subsets:
            logger.error("Sampling resulted in no subsets. Cannot proceed.")
            raise ValueError("Sampling resulted in an empty DataFrame. Cannot proceed.")

        combined_df = pd.concat(sampled_subsets)
        if combined_df.empty:
            logger.error("Combined sampled DataFrame is empty after concatenation.")
            raise ValueError("Combined sampled DataFrame is empty after concatenation.")

        combined_df = combined_df.sample(frac=1, random_state=seed).reset_index(drop=True)
        combined_df = combined_df.drop(columns=["strata"])
        logger.info(f"DataFrame sampling completed. Result size: {len(combined_df)}")
        return combined_df

    def _split_dataframe(self, df: pd.DataFrame, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        stratify_col_name = self.settings.experiment.predict_attribute.value
        if stratify_col_name not in df.columns:
            logger.error(f"Stratify column '{stratify_col_name}' not found for splitting.")
            raise ValueError(f"Stratify column '{stratify_col_name}' not found for splitting.")

        try:
            train_val_df, test_df = train_test_split(
                df,
                test_size=self.settings.dataset.test_ratio,
                random_state=seed,
                stratify=df[stratify_col_name],
            )

            if len(train_val_df) == 0:
                logger.warning("Train+Validation DataFrame is empty after first split.")
            if len(test_df) == 0:
                logger.warning("Test DataFrame is empty after first split.")

            adjusted_val_ratio = min(1.0, self.settings.dataset.validation_ratio / (1.0 - self.settings.dataset.test_ratio))

            if len(train_val_df) > 0:
                train_df, val_df = train_test_split(
                    train_val_df,
                    test_size=adjusted_val_ratio,
                    random_state=seed,
                    stratify=train_val_df[stratify_col_name],
                )
                if len(train_df) == 0:
                    logger.warning("Train DataFrame is empty after second split.")
                if len(val_df) == 0:
                    logger.warning("Validation DataFrame is empty after second split.")
            else:
                train_df = pd.DataFrame(columns=df.columns)
                val_df = pd.DataFrame(columns=df.columns)
                logger.warning("Skipping train/validation split as intermediate set was empty.")

        except Exception as e:
            logger.error(f"Error during DataFrame splitting: {e}")
            raise RuntimeError(f"Error during DataFrame splitting: {e}") from e

        logger.info(f"DataFrame splitting completed. Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
        return train_df, val_df, test_df

    def _preprocess_single_image(
        self,
        image_bytes: bytes,
        purpose: DatasetSplit,
        label_dict: Dict[str, int],
        image_id: str,
    ) -> Optional[np.ndarray]:
        try:
            image_np = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
            if image_np is None:
                logger.warning(f"cv2.imdecode returned None for image ID: {image_id}. Skipping.")
                return None
        except (cv2.error, Exception) as e:
            logger.warning(f"Error decoding image ID {image_id}: {e}. Skipping.")
            return None

        try:
            image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
            target_size = (self.settings.dataset.image_size, self.settings.dataset.image_size)
            image_np = cv2.resize(image_np, target_size)

            if image_np.shape[:2] != target_size:
                logger.warning(f"Image {image_id} resize failed. Shape is {image_np.shape[:2]}, expected {target_size}. Skipping.")
                return None

            image_np = image_np.astype(np.float32) / 255.0

            if purpose == DatasetSplit.TRAIN and self.settings.analysis.mask_demographic is not None:
                mask_attribute_type_enum = self.settings.analysis.protected_attribute
                mask_attribute_value_enum = self.settings.analysis.mask_demographic
                attribute_key = mask_attribute_type_enum.value
                actual_image_value = label_dict.get(attribute_key)

                if actual_image_value is not None and actual_image_value == mask_attribute_value_enum.value:
                    if self.settings.analysis.mask_features:
                        try:
                            image_np = self.explainer.apply_mask(image_np)
                        except Exception as e_mask:
                            logger.warning(f"Failed to apply mask to image ID {image_id}: {e_mask}. Using unmasked.")

            if self.settings.dataset.use_grayscale:
                if len(image_np.shape) == 3 and image_np.shape[-1] == 3:
                    image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
                image_np = np.expand_dims(image_np, axis=-1)

            return image_np.astype(np.float32)

        except Exception as e_proc:
            logger.warning(f"Error processing image ID {image_id} after decode: {e_proc}. Skipping.")
            return None

    def _preprocess_dataframe_split(self, df: pd.DataFrame, purpose: DatasetSplit) -> pd.DataFrame:
        processed_images = []
        valid_indices = []
        for index, row in df.iterrows():
            label_dict = {
                DemographicAttribute.GENDER.value: row["gender"],
                DemographicAttribute.RACE.value: row["race"],
                DemographicAttribute.AGE.value: row["age"],
            }
            processed_image = self._preprocess_single_image(row["image_bytes"], purpose, label_dict, row["image_id"])
            if processed_image is not None:
                processed_images.append(processed_image)
                valid_indices.append(index)

        if not valid_indices:
            logger.warning(f"No images successfully processed for split: {purpose.value}. Returning empty DataFrame.")
            return pd.DataFrame(columns=list(df.columns) + ["processed_image"])

        df_processed = df.loc[valid_indices].copy()
        df_processed.loc[:, "processed_image"] = processed_images

        if "processed_image" not in df_processed.columns or len(df_processed["processed_image"]) != len(df_processed):
            logger.error(f"Failed to correctly assign 'processed_image' column for split {purpose.value}.")
            raise RuntimeError(f"Failed to correctly assign 'processed_image' column for split {purpose.value}.")

        logger.info(f"Preprocessing completed for {purpose.value} split. {len(df_processed)} / {len(df)} images valid.")
        return df_processed

    def _save_images(self, df: pd.DataFrame, purpose: DatasetSplit) -> None:
        savepath = os.path.join(self.settings.output.base_path, self.settings.experiment_id, f"{purpose.value.lower()}_images")
        try:
            os.makedirs(savepath, exist_ok=True)
        except OSError as e:
            logger.error(f"Could not create directory {savepath}: {e}. Cannot save images.")
            return

        saved_count = 0
        for _, row in df.iterrows():
            img_array = row["processed_image"]
            img_id = row["image_id"]
            filename = f"{purpose.value.lower()}_{img_id}.png"
            filepath = os.path.join(savepath, filename)

            try:
                img_to_save = (img_array * 255.0).clip(0, 255).astype(np.uint8)
                if img_to_save.shape[-1] == 1:
                    pil_img = Image.fromarray(np.squeeze(img_to_save, axis=-1), mode="L")
                elif img_to_save.shape[-1] == 3:
                    pil_img = Image.fromarray(img_to_save, mode="RGB")
                else:
                    logger.warning(f"Unexpected image shape {img_to_save.shape} for saving image ID {img_id}. Skipping.")
                    continue

                pil_img.save(filepath, format="PNG")
                saved_count += 1
            except (IOError, OSError, Exception) as e:
                logger.warning(f"Could not save image {filename} to {filepath}: {e}")

        logger.info(f"Saved {saved_count} / {len(df)} images for {purpose.value} split to {savepath}")

    def prepare_datasets(self, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        logger.info("Starting dataset preparation...")
        raw_df = self._load_raw_dataframe()
        processed_df = self._process_dataframe(raw_df)
        sampled_df = self._sample_dataframe(processed_df, seed)

        train_df, val_df, test_df = self._split_dataframe(sampled_df, seed)

        train_df_processed = self._preprocess_dataframe_split(train_df, DatasetSplit.TRAIN)
        val_df_processed = self._preprocess_dataframe_split(val_df, DatasetSplit.VALIDATION)
        test_df_processed = self._preprocess_dataframe_split(test_df, DatasetSplit.TEST)

        if train_df_processed.empty or val_df_processed.empty or test_df_processed.empty:
            logger.error("One or more dataset splits are empty after preprocessing. Cannot proceed.")
            raise ValueError("Dataset preparation resulted in one or more empty splits.")

        self._save_images(train_df_processed, DatasetSplit.TRAIN)
        self._save_images(val_df_processed, DatasetSplit.VALIDATION)
        self._save_images(test_df_processed, DatasetSplit.TEST)

        logger.info("Dataset preparation completed successfully.")
        return train_df_processed, val_df_processed, test_df_processed
