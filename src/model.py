import logging
import os
from typing import Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from pandas.api.types import is_numeric_dtype

# isort: off
from .definitions import Age, DemographicAttribute, Gender, ModelMetadata, Race
from .settings import Settings
from .utils import LOGGER_NAME

logger = logging.getLogger(LOGGER_NAME)


class Model:
    def __init__(self, settings: Settings):
        self.settings = settings

    def _get_num_classes(self) -> int:
        target_attr = self.settings.experiment.predict_attribute
        if target_attr == DemographicAttribute.GENDER:
            return len(Gender)
        elif target_attr == DemographicAttribute.RACE:
            return len(Race)
        elif target_attr == DemographicAttribute.AGE:
            return len(Age)
        else:
            logger.error(f"Unknown predict_attribute: {target_attr}")
            raise ValueError(f"Unknown predict_attribute: {target_attr}")

    def _build_model(self) -> tf.keras.Model:
        input_channels = 1 if self.settings.dataset.use_grayscale else 3
        img_size = self.settings.dataset.image_size
        input_shape = (img_size, img_size, input_channels)
        num_classes = self._get_num_classes()

        model = tf.keras.Sequential(name="cnn_model")
        model.add(tf.keras.layers.Input(shape=input_shape, name="input"))

        conv_blocks = [(64, 2), (128, 2), (256, 3)]
        for idx, (filters, layers_count) in enumerate(conv_blocks, start=1):
            for i in range(1, layers_count + 1):
                conv_layer = tf.keras.layers.Conv2D(filters, (3, 3), activation="relu", padding="same", name=f"block{idx}_conv{i}")
                model.add(conv_layer)
            pooling_layer = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name=f"block{idx}_pool")
            model.add(pooling_layer)

        model.add(tf.keras.layers.Flatten(name="flatten"))
        model.add(tf.keras.layers.Dense(512, activation="relu", name="dense"))
        model.add(tf.keras.layers.Dropout(0.5, name="dropout"))
        model.add(tf.keras.layers.Dense(num_classes, activation="softmax", name="output"))

        try:
            model.compile(
                optimizer=tf.keras.optimizers.Adam(0.0001),
                loss="sparse_categorical_crossentropy",
                metrics=["accuracy"],
            )
            logger.info("Model built and compiled successfully.")
        except Exception as e:
            logger.error(f"Failed to compile model: {e}")
            raise RuntimeError(f"Failed to compile model: {e}") from e

        return model

    def save_model(self, model) -> str:
        savepath = os.path.join(self.settings.output.base_path, self.settings.experiment_id)
        try:
            os.makedirs(savepath, exist_ok=True)
        except OSError as e:
            logger.error(f"Could not create directory for saving model {savepath}: {e}")
            raise RuntimeError(f"Could not create directory {savepath}: {e}") from e

        filename = f"{self.settings.experiment_id}.keras"
        filepath = os.path.join(savepath, filename)

        try:
            model.save(filepath)
            logger.info(f"Model saved successfully to {filepath}")
            return filepath
        except (IOError, OSError, Exception) as e:
            logger.error(f"Failed to save model to {filepath}: {e}")
            raise RuntimeError(f"Failed to save model to {filepath}: {e}") from e

    def _validate_dataframe_for_model(self, df: pd.DataFrame, df_name: str) -> bool:
        img_col = "processed_image"
        target_col = self.settings.experiment.predict_attribute.value
        input_channels = 1 if self.settings.dataset.use_grayscale else 3
        expected_shape = (self.settings.dataset.image_size, self.settings.dataset.image_size, input_channels)

        if img_col not in df.columns:
            logger.error(f"'{img_col}' column missing in {df_name} DataFrame.")
            return False
        if target_col not in df.columns:
            logger.error(f"Target column '{target_col}' missing in {df_name} DataFrame.")
            return False
        if not is_numeric_dtype(df[target_col]):
            logger.error(f"Target column '{target_col}' in {df_name} DataFrame is not numeric.")
            return False
        if df.empty or df[img_col].empty:
            logger.error(f"{df_name} DataFrame or its '{img_col}' column is empty.")
            return False
        first_img = df[img_col].iloc[0]
        if not isinstance(first_img, np.ndarray):
            logger.error(f"First element in '{img_col}' of {df_name} is not a numpy array.")
            return False
        if first_img.shape != expected_shape:
            logger.error(f"First image shape in {df_name} {first_img.shape} does not match expected {expected_shape}.")
            return False
        return True

    def get_model_and_history(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
    ) -> Tuple[tf.keras.Model, ModelMetadata]:
        if not self._validate_dataframe_for_model(train_df, "train_df"):
            raise ValueError("Training DataFrame failed validation.")
        if not self._validate_dataframe_for_model(val_df, "val_df"):
            raise ValueError("Validation DataFrame failed validation.")

        target_col_name = self.settings.experiment.predict_attribute.value
        try:
            x_train = np.stack(train_df["processed_image"].values)
            y_train = train_df[target_col_name].values
            x_val = np.stack(val_df["processed_image"].values)
            y_val = val_df[target_col_name].values
            logger.info("Input data stacked successfully.")
        except ValueError as e:
            logger.error(f"Error stacking numpy arrays from DataFrames: {e}. Check image shapes.")
            raise RuntimeError(f"Error stacking numpy arrays: {e}") from e
        except Exception as e_stack:
            logger.error(f"Unexpected error during data stacking: {e_stack}")
            raise RuntimeError(f"Unexpected error during data stacking: {e_stack}") from e_stack

        validation_data = (x_val, y_val)
        model = self._build_model()

        history_callback = None
        try:
            logger.info(f"Starting model training for {self.settings.model.epochs} epochs...")
            history_callback = model.fit(
                x_train,
                y_train,
                validation_data=validation_data,
                epochs=self.settings.model.epochs,
                batch_size=self.settings.model.batch_size,
                verbose=0,
            )
            logger.info("Model training completed.")
        except (tf.errors.InvalidArgumentError, ValueError) as e_fit_val:
            logger.error(f"TensorFlow/ValueError during model.fit: {e_fit_val}. Check data shapes and values.")
            raise RuntimeError(f"Error during model training: {e_fit_val}") from e_fit_val
        except Exception as e_fit:
            logger.error(f"Unexpected error during model.fit: {e_fit}")
            raise RuntimeError(f"Unexpected error during model training: {e_fit}") from e_fit

        model_path = self.save_model(model)

        history_keys = ["loss", "accuracy", "val_loss", "val_accuracy"]
        if history_callback is None or not all(k in history_callback.history for k in history_keys) or not all(history_callback.history[k] for k in history_keys):
            logger.warning("Training history is incomplete or missing. Using empty tuples for metadata.")
            metadata = ModelMetadata(path=model_path, train_loss=(), train_accuracy=(), val_loss=(), val_accuracy=())
        else:
            metadata = ModelMetadata(
                path=model_path,
                train_loss=tuple(history_callback.history.get("loss", [])),
                train_accuracy=tuple(history_callback.history.get("accuracy", [])),
                val_loss=tuple(history_callback.history.get("val_loss", [])),
                val_accuracy=tuple(history_callback.history.get("val_accuracy", [])),
            )
            logger.info("Model metadata created successfully.")

        return model, metadata
