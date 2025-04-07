import json
import os
import random
from pathlib import Path
from typing import Any, Callable, Dict, Tuple, Union

import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from huggingface_hub import hf_hub_download
from keras.api.applications import VGG16, VGG19, InceptionResNetV2, InceptionV3, ResNet50V2, ResNet101V2, ResNet152V2, Xception
from keras.api.callbacks import EarlyStopping
from keras.api.layers import Dense, Dropout, GlobalAveragePooling2D
from keras.api.models import Model
from keras.api.optimizers import Adam
from sklearn.model_selection import train_test_split

SEED = 42
DATASET_NAME = "utkface"
NUM_SAMPLES = 5000
VAL_RATIO = 0.2
TEST_RATIO = 0.1
IMG_SIZE = 75
NUM_CHANNELS = 3
NUM_CLASSES = 2

USE_TRANSFER_LEARNING = True
DENSE_UNITS = 512
DROPOUT = 0.5
LEARNING_RATE = 0.001
BATCH_SIZE = 64
EPOCHS = 10

DS_REPO = "rixmape/utkface"
DS_FILE = "data/train-00000-of-00001.parquet"
OUT_DIR_NAME = "gender_classifiers"
RESULTS_FILENAME = "model_performance.json"

ARCHITECTURES: Dict[str, Callable] = {
    "VGG16": VGG16,
    "VGG19": VGG19,
    "ResNet50V2": ResNet50V2,
    "ResNet101V2": ResNet101V2,
    "ResNet152V2": ResNet152V2,
    "InceptionV3": InceptionV3,
    "InceptionResNetV2": InceptionResNetV2,
    "Xception": Xception,
}


def set_seeds() -> None:
    """Sets random seeds using global SEED for reproducibility."""
    os.environ["PYTHONHASHSEED"] = str(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)
    tf.get_logger().setLevel("ERROR")


def load_sampled_data() -> pd.DataFrame:
    """Downloads, loads, filters, and samples data from Hugging Face Hub."""
    path = Path(hf_hub_download(repo_id=DS_REPO, filename=DS_FILE, repo_type="dataset"))
    df = pd.read_parquet(path)
    df = df[df["age"] > 0]
    return df.sample(n=NUM_SAMPLES, random_state=SEED)


def split_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Splits DataFrame into stratified train, validation, and test sets."""
    df_train_val, df_test = train_test_split(df, test_size=TEST_RATIO, random_state=SEED, stratify=df["gender"])
    adj_val_ratio = VAL_RATIO / (1.0 - TEST_RATIO)
    df_train, df_val = train_test_split(df_train_val, test_size=adj_val_ratio, random_state=SEED, stratify=df_train_val["gender"])
    return df_train, df_val, df_test


@tf.function
def parse_and_preprocess_tf(image_bytes: tf.Tensor, label: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    """Decodes, resizes, normalizes images, and one-hot encodes labels."""
    image = tf.io.decode_image(image_bytes, channels=NUM_CHANNELS, expand_animations=False)
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    image = tf.ensure_shape(image, (IMG_SIZE, IMG_SIZE, NUM_CHANNELS))
    image = tf.cast(image, tf.float32) / 255.0
    label = tf.one_hot(tf.cast(label, tf.int32), depth=NUM_CLASSES)
    return image, label


def create_tf_dataset(df: pd.DataFrame, shuffle: bool = False) -> tf.data.Dataset:
    """Creates a batched and prefetched TensorFlow Dataset from DataFrame."""
    img_bytes_list = [img_dict["bytes"] for img_dict in df["image"]]
    ds = tf.data.Dataset.from_tensor_slices((img_bytes_list, df["gender"].tolist()))
    ds = ds.map(parse_and_preprocess_tf, num_parallel_calls=tf.data.AUTOTUNE)
    if shuffle:
        ds = ds.shuffle(buffer_size=len(df), seed=SEED)
    return ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


def build_model(architecture_fn: Callable) -> Model:
    """Builds and compiles a Keras model using a specified base architecture."""
    base = architecture_fn(weights="imagenet" if USE_TRANSFER_LEARNING else None, include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, NUM_CHANNELS))
    base.trainable = not USE_TRANSFER_LEARNING
    x = GlobalAveragePooling2D()(base.output)
    x = Dense(DENSE_UNITS, activation="relu")(x)
    x = Dropout(DROPOUT)(x)
    outputs = Dense(NUM_CLASSES, activation="softmax")(x)
    model = Model(inputs=base.input, outputs=outputs)
    model.compile(optimizer=Adam(learning_rate=LEARNING_RATE), loss="categorical_crossentropy", metrics=["accuracy"])
    return model


def train_and_evaluate_model(model: Model, train_ds: tf.data.Dataset, val_ds: tf.data.Dataset, test_ds: tf.data.Dataset) -> Dict[str, Any]:
    """Trains the model, evaluates on test set, and returns serializable performance."""
    early_stopping = EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)
    history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=[early_stopping], verbose=0)
    test_loss, test_acc = model.evaluate(test_ds, verbose=0)
    serializable_history = {k: [float(val) for val in v] for k, v in history.history.items()}
    return {
        "epoch_history": serializable_history,
        "test_metrics": {"loss": float(test_loss), "accuracy": float(test_acc)},
    }


def save_keras_model(model: Model, filepath: Path) -> None:
    """Saves the Keras model, creating parent directories if needed."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    model.save(filepath)
    print(f"Model saved to: {filepath}")


def save_results_json(filepath: Path, results_data: Dict[str, Any]) -> None:
    """Saves performance results dictionary to a JSON file."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(results_data, f, indent=4)
    print(f"Results JSON saved to: {filepath}")


def _plot_metric(df_full, metric_name, axes):
    """Helper function to plot training and validation metrics."""
    metric_df = df_full[df_full["metric"] == metric_name]
    sns.lineplot(data=metric_df[metric_df["type"] == "Training"], x="epoch", y="value", hue="model", ax=axes[0], legend=False, marker="o", markersize=4)
    sns.lineplot(data=metric_df[metric_df["type"] == "Validation"], x="epoch", y="value", hue="model", ax=axes[1], legend=True, marker="o", markersize=4)

    y_min, y_max = metric_df["value"].min(), metric_df["value"].max()
    lower_bound = max(0.0, y_min) if metric_name == "Accuracy" else y_min
    padding = (y_max - lower_bound) * 0.05 if (y_max - lower_bound) > 1e-6 else 0.1
    final_lower = max(0.0, lower_bound - padding) if metric_name == "Accuracy" else (lower_bound - padding)
    final_upper = y_max + padding

    axes[0].set_ylim(final_lower, final_upper)
    axes[1].set_ylim(final_lower, final_upper)

    axes[0].set_ylabel(f"Training {metric_name}")
    axes[1].set_ylabel(f"Validation {metric_name}")
    axes[1].set_xlabel("Epoch")
    axes[1].legend(fontsize="small")
    axes[0].tick_params(axis="y", rotation=90)
    axes[1].tick_params(axis="y", rotation=90)


def plot_model_performance(all_results: Dict[str, Any]) -> None:
    """Plots training and validation performance metrics for all models."""
    plot_data = [
        entry
        for model_name, results in all_results.items()
        for epoch, loss, acc, val_loss, val_acc in zip(
            range(1, len(results["epoch_history"]["loss"]) + 1),
            results["epoch_history"]["loss"],
            results["epoch_history"]["accuracy"],
            results["epoch_history"]["val_loss"],
            results["epoch_history"]["val_accuracy"],
        )
        for entry in [
            {"model": model_name, "epoch": epoch, "metric": "Loss", "type": "Training", "value": loss},
            {"model": model_name, "epoch": epoch, "metric": "Accuracy", "type": "Training", "value": acc},
            {"model": model_name, "epoch": epoch, "metric": "Loss", "type": "Validation", "value": val_loss},
            {"model": model_name, "epoch": epoch, "metric": "Accuracy", "type": "Validation", "value": val_acc},
        ]
    ]

    df = pd.DataFrame(plot_data)
    df["value"] = pd.to_numeric(df["value"])

    sns.set_theme(style="whitegrid")

    fig_loss, axes_loss = plt.subplots(nrows=2, ncols=1, figsize=(8, 5), sharex=True)
    _plot_metric(df, "Loss", axes_loss)
    fig_loss.tight_layout()

    fig_acc, axes_acc = plt.subplots(nrows=2, ncols=1, figsize=(8, 5), sharex=True)
    _plot_metric(df, "Accuracy", axes_acc)
    fig_acc.tight_layout()

    plt.show()


def main() -> None:
    """Executes the ML pipeline: data handling, multi-model training, eval, saving."""
    set_seeds()
    out_dir = Path(OUT_DIR_NAME)
    results_filepath = out_dir / RESULTS_FILENAME
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading and preparing data...")
    df_sampled = load_sampled_data()
    df_train, df_val, df_test = split_data(df_sampled)
    print(f"Data split: Train={len(df_train)}, Val={len(df_val)}, Test={len(df_test)}")

    print("Creating TensorFlow datasets...")
    train_ds = create_tf_dataset(df_train, shuffle=True)
    val_ds = create_tf_dataset(df_val)
    test_ds = create_tf_dataset(df_test)

    all_results = {}

    print("\nStarting model training loop...")
    for name, arch_fn in ARCHITECTURES.items():
        print(f"--- Training {name} ---")
        model = build_model(arch_fn)
        model_filepath = out_dir / f"{name.lower()}_{DATASET_NAME}_gender_classifier.keras"
        results = train_and_evaluate_model(model, train_ds, val_ds, test_ds)
        all_results[name] = results
        save_keras_model(model, model_filepath)
        keras.backend.clear_session()
        print(f"--- Finished {name} --- Test Accuracy: {results['test_metrics']['accuracy']:.4f} ---")

    save_results_json(results_filepath, all_results)
    plot_model_performance(all_results)
    print(f"\nPipeline complete. Models and results saved in directory: '{out_dir}'")


if __name__ == "__main__":
    main()
