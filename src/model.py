import os
from typing import Tuple

import numpy as np
import pandas as pd
import tensorflow as tf

# isort: off
from .definitions import Age, DemographicAttribute, Gender, ModelMetadata, ProgressCallback, Race
from .settings import Settings


class Model:
    def __init__(self, settings: Settings, progress_callback: ProgressCallback = None):
        self.settings = settings
        self.progress_callback = progress_callback

    def _get_num_classes(self) -> int:
        target_attr = self.settings.experiment.predict_attribute
        if target_attr == DemographicAttribute.GENDER:
            return len(Gender)
        elif target_attr == DemographicAttribute.RACE:
            return len(Race)
        elif target_attr == DemographicAttribute.AGE:
            return len(Age)

    def _build_model(self) -> tf.keras.Model:
        input_channels = 1 if self.settings.dataset.use_grayscale else 3
        input_shape = (
            self.settings.dataset.image_size,
            self.settings.dataset.image_size,
            input_channels,
        )
        num_classes = self._get_num_classes()

        model = tf.keras.Sequential(name="cnn_model")
        model.add(tf.keras.layers.Input(shape=input_shape, name="input"))

        conv_blocks = [(64, 2), (128, 2), (256, 3)]
        for idx, (filters, layers_count) in enumerate(conv_blocks, start=1):
            for i in range(1, layers_count + 1):
                conv_layer = tf.keras.layers.Conv2D(
                    filters,
                    (3, 3),
                    activation="relu",
                    padding="same",
                    name=f"block{idx}_conv{i}",
                )
                model.add(conv_layer)
            pooling_layer = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name=f"block{idx}_pool")
            model.add(pooling_layer)

        model.add(tf.keras.layers.Flatten(name="flatten"))
        model.add(tf.keras.layers.Dense(512, activation="relu", name="dense"))
        model.add(tf.keras.layers.Dropout(0.5, name="dropout"))
        model.add(tf.keras.layers.Dense(num_classes, activation="softmax", name="output"))

        model.compile(
            optimizer=tf.keras.optimizers.Adam(0.0001),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        return model

    def save_model(self, model) -> str:
        savepath = os.path.join(self.settings.output.base_path, self.settings.experiment_id)
        os.makedirs(savepath, exist_ok=True)
        filename = f"{self.settings.experiment_id}.keras"
        filepath = os.path.join(savepath, filename)

        try:
            model.save(filepath)
            return filepath
        except Exception as e:
            raise RuntimeError(f"Failed to save model to {filepath}: {e}")

    def get_model_and_history(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
    ) -> Tuple[tf.keras.Model, ModelMetadata]:

        x_train = np.stack(train_df["processed_image"].values)
        y_train = train_df[self.settings.experiment.predict_attribute.value].values

        x_val = np.stack(val_df["processed_image"].values)
        y_val = val_df[self.settings.experiment.predict_attribute.value].values
        validation_data = (x_val, y_val)

        model = self._build_model()
        model_path = self.save_model(model)

        history_callback = model.fit(
            x_train,
            y_train,
            validation_data=validation_data,
            epochs=self.settings.model.epochs,
            batch_size=self.settings.model.batch_size,
            verbose=0,
        )

        metadata = ModelMetadata(
            path=model_path,
            train_loss=tuple(history_callback.history.get("loss", [])),
            train_accuracy=tuple(history_callback.history.get("accuracy", [])),
            val_loss=tuple(history_callback.history.get("val_loss", [])),
            val_accuracy=tuple(history_callback.history.get("val_accuracy", [])),
        )

        return model, metadata
