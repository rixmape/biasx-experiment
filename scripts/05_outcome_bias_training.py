import json
import os
import random
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from huggingface_hub import hf_hub_download
from sklearn.model_selection import train_test_split

MALE_RATIO = 0.9
TARGET_TRAIN_SAMPLES = 5000

FEMALE_RATIO = 1.0 - MALE_RATIO
SEED = 42
DATASET_NAME = "utkface"
DS_REPO = "rixmape/utkface"
DS_FILE = "data/train-00000-of-00001.parquet"

VAL_RATIO = 0.1
TEST_RATIO = 0.2
IMG_SIZE = 48
NUM_CHANNELS = 3
NUM_CLASSES = 2
BATCH_SIZE = 64
EPOCHS = 10
LEARNING_RATE = 0.0001
DENSE_UNITS = 512
DROPOUT = 0.5

OUT_DIR_NAME = "outputs"
MODEL_FILENAME_TEMPLATE = "gender_skew_{male_ratio:.0f}_{female_ratio:.0f}.keras"
RESULTS_FILENAME_TEMPLATE = "gender_skew_{male_ratio:.0f}_{female_ratio:.0f}.json"


def set_seeds() -> None:
    os.environ["PYTHONHASHSEED"] = str(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)
    tf.get_logger().setLevel("ERROR")
    print(f"Seeds set to {SEED}")


def load_raw_dataframe() -> pd.DataFrame:
    print(f"Loading dataset from Hugging Face: {DS_REPO}")
    try:
        path = Path(hf_hub_download(repo_id=DS_REPO, filename=DS_FILE, repo_type="dataset"))
        df = pd.read_parquet(path, columns=["image", "gender", "race", "age"])
        print(f"Loaded {len(df)} raw records.")
        return df
    except Exception as e:
        raise RuntimeError(f"Failed to load dataset from {DS_REPO}: {e}") from e


def _stratified_sample_group(group: pd.DataFrame, target_size: int, seed: int) -> pd.DataFrame:
    total_rows = len(group)
    if total_rows == 0 or target_size == 0:
        return pd.DataFrame(columns=group.columns)

    samples = []
    for _, sub_group in group.groupby("strata"):
        sub_group_size = len(sub_group)
        if sub_group_size == 0:
            continue
        sub_group_proportion = sub_group_size / total_rows
        stratum_sample_size = max(1, round(target_size * sub_group_proportion))

        replacement_needed = sub_group_size < stratum_sample_size
        sample = sub_group.sample(n=stratum_sample_size, random_state=seed, replace=replacement_needed)
        samples.append(sample)

    return pd.concat(samples) if samples else pd.DataFrame(columns=group.columns)


def create_skewed_train_set(df_train_orig: pd.DataFrame) -> pd.DataFrame:
    print(f"Creating skewed training set ({MALE_RATIO*100:.0f}% Male, {FEMALE_RATIO*100:.0f}% Female)...")
    print(f"Target total training samples: {TARGET_TRAIN_SAMPLES}")

    target_male_samples = int(TARGET_TRAIN_SAMPLES * MALE_RATIO)
    target_female_samples = TARGET_TRAIN_SAMPLES - target_male_samples
    print(f"Target male samples: {target_male_samples}")
    print(f"Target female samples: {target_female_samples}")

    df_male = df_train_orig[df_train_orig["gender"] == 0].copy()
    df_female = df_train_orig[df_train_orig["gender"] == 1].copy()

    df_male["strata"] = df_male[["race", "age"]].astype(str).agg("_".join, axis=1)
    df_female["strata"] = df_female[["race", "age"]].astype(str).agg("_".join, axis=1)

    print(f"Resampling males (target {target_male_samples}) while stratifying by race/age...")
    sampled_males = _stratified_sample_group(df_male, target_male_samples, SEED)
    print(f"Resampling females (target {target_female_samples}) while stratifying by race/age...")
    sampled_females = _stratified_sample_group(df_female, target_female_samples, SEED)

    if sampled_males.empty and sampled_females.empty:
        raise ValueError("Sampling resulted in an empty DataFrame for both genders. Cannot proceed.")
    if sampled_males.empty:
        print("Warning: Male sampling resulted in an empty DataFrame.")
    if sampled_females.empty:
        print("Warning: Female sampling resulted in an empty DataFrame.")

    df_train_skewed = pd.concat([sampled_males, sampled_females]).reset_index(drop=True)
    df_train_skewed = df_train_skewed.drop(columns=["strata"])

    df_train_skewed = df_train_skewed.sample(frac=1, random_state=SEED).reset_index(drop=True)

    print(f"Created skewed training set with {len(df_train_skewed)} samples.")
    actual_male_ratio = len(df_train_skewed[df_train_skewed["gender"] == 0]) / len(df_train_skewed) if len(df_train_skewed) > 0 else 0
    print(f"Actual Male Ratio in skewed set: {actual_male_ratio:.4f}")
    return df_train_skewed


@tf.function
def parse_and_preprocess_tf(image_bytes: tf.Tensor, label: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    image = tf.io.decode_image(image_bytes, channels=NUM_CHANNELS, expand_animations=False)
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    image = tf.ensure_shape(image, (IMG_SIZE, IMG_SIZE, NUM_CHANNELS))
    image = tf.cast(image, tf.float32) / 255.0
    label = tf.cast(label, tf.int32)
    return image, label


def create_tf_dataset(df: pd.DataFrame, batch_size: int, shuffle: bool = False) -> tf.data.Dataset:
    img_bytes_list = [img_dict["bytes"] for img_dict in df["image"] if isinstance(img_dict, dict) and "bytes" in img_dict]
    if len(img_bytes_list) != len(df):
        raise ValueError("Mismatch between DataFrame length and extracted image bytes. Check 'image' column format.")

    ds = tf.data.Dataset.from_tensor_slices((img_bytes_list, df["gender"].tolist()))
    ds = ds.map(parse_and_preprocess_tf, num_parallel_calls=tf.data.AUTOTUNE)
    if shuffle:
        ds = ds.shuffle(buffer_size=len(df), seed=SEED)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def build_model() -> tf.keras.Model:
    input_shape = (IMG_SIZE, IMG_SIZE, NUM_CHANNELS)

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
    model.add(tf.keras.layers.Dense(DENSE_UNITS, activation="relu", name="dense"))
    model.add(tf.keras.layers.Dropout(DROPOUT, name="dropout"))
    model.add(tf.keras.layers.Dense(NUM_CLASSES, activation="softmax", name="output"))

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def train_and_evaluate_model(model: tf.keras.Model, train_ds: tf.data.Dataset, val_ds: tf.data.Dataset, test_ds: tf.data.Dataset, epochs: int) -> Dict[str, Any]:
    print("Starting model training...")
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True, verbose=1)
    history = model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=[early_stopping], verbose=2)
    print("Training finished.")

    print("Evaluating model on test set...")
    test_loss, test_acc = model.evaluate(test_ds, verbose=1)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")

    serializable_history = {k: [float(val) for val in v] for k, v in history.history.items()}
    return {
        "epoch_history": serializable_history,
        "test_metrics": {"loss": float(test_loss), "accuracy": float(test_acc)},
    }


def save_results(model: tf.keras.Model, results_data: Dict[str, Any], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    male_ratio_perc = int(MALE_RATIO * 100)
    female_ratio_perc = int(FEMALE_RATIO * 100)
    model_filename = MODEL_FILENAME_TEMPLATE.format(male_ratio=male_ratio_perc, female_ratio=female_ratio_perc)
    model_filepath = out_dir / model_filename
    model.save(model_filepath)
    print(f"Model saved to: {model_filepath}")

    results_filename = RESULTS_FILENAME_TEMPLATE.format(male_ratio=male_ratio_perc, female_ratio=female_ratio_perc)
    results_filepath = out_dir / results_filename
    try:
        results_data["script_parameters"] = {
            "MALE_RATIO": MALE_RATIO,
            "TARGET_TRAIN_SAMPLES": TARGET_TRAIN_SAMPLES,
            "SEED": SEED,
            "VAL_RATIO": VAL_RATIO,
            "TEST_RATIO": TEST_RATIO,
            "IMG_SIZE": IMG_SIZE,
            "EPOCHS": EPOCHS,
            "BATCH_SIZE": BATCH_SIZE,
            "LEARNING_RATE": LEARNING_RATE,
        }
        with open(results_filepath, "w") as f:
            json.dump(results_data, f, indent=4)
        print(f"Results JSON saved to: {results_filepath}")
    except TypeError as e:
        print(f"Error saving results to JSON (potential serialization issue): {e}")
        print("Attempting to save without unserializable parts...")
        try:
            simplified_results = {"test_metrics": results_data.get("test_metrics")}
            simplified_results["script_parameters"] = results_data.get("script_parameters")
            with open(results_filepath.with_suffix(".simplified.json"), "w") as f:
                json.dump(simplified_results, f, indent=4)
            print(f"Simplified results saved to: {results_filepath.with_suffix('.simplified.json')}")
        except Exception as final_e:
            print(f"Could not save simplified results either: {final_e}")


def main() -> None:
    set_seeds()
    out_dir = Path(OUT_DIR_NAME)

    df_full = load_raw_dataframe()

    print("Splitting data into initial train/validation/test sets (stratified by gender)...")
    df_train_val, df_test = train_test_split(df_full, test_size=TEST_RATIO, random_state=SEED, stratify=df_full["gender"])
    adj_val_ratio = VAL_RATIO / (1.0 - TEST_RATIO) if (1.0 - TEST_RATIO) > 0 else 0
    df_train_orig, df_val = train_test_split(df_train_val, test_size=adj_val_ratio, random_state=SEED, stratify=df_train_val["gender"])
    print(f"Original split sizes: Train={len(df_train_orig)}, Val={len(df_val)}, Test={len(df_test)}")

    df_train_skewed = create_skewed_train_set(df_train_orig)

    print("Creating TensorFlow datasets...")
    try:
        train_ds_skewed = create_tf_dataset(df_train_skewed, BATCH_SIZE, shuffle=True)
        val_ds = create_tf_dataset(df_val, BATCH_SIZE, shuffle=False)
        test_ds = create_tf_dataset(df_test, BATCH_SIZE, shuffle=False)
        print("TensorFlow datasets created successfully.")
    except ValueError as e:
        print(f"Error creating TF datasets: {e}")
        return

    print("Building CNN model...")
    model = build_model()
    model.summary()

    results = train_and_evaluate_model(model, train_ds_skewed, val_ds, test_ds, EPOCHS)

    print("Saving model and results...")
    save_results(model, results, out_dir)

    print(f"\nOutcome bias scenario script finished. Outputs saved in '{out_dir.resolve()}'")


if __name__ == "__main__":
    main()
