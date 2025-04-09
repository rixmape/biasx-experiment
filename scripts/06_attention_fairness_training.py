import json
import os
import random
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import tensorflow as tf
from huggingface_hub import hf_hub_download
from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.vision.face_landmarker import FaceLandmarker, FaceLandmarkerOptions, FaceLandmarkerResult
from sklearn.model_selection import train_test_split


class Gender(Enum):
    MALE = 0
    FEMALE = 1


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


SEED = 42
DATASET_NAME = "utkface"
DS_REPO = "rixmape/utkface"
DS_FILE = "data/train-00000-of-00001.parquet"

MASK_GENDER: Gender = Gender.MALE
MASK_FEATURES: List[Feature] = [Feature.LEFT_EYE, Feature.RIGHT_EYE]
MASK_PIXEL_PADDING = 2

TARGET_TRAIN_SAMPLES = 5000
VAL_RATIO = 0.1
TEST_RATIO = 0.2
IMG_SIZE = 48
NUM_CHANNELS = 3
NUM_CLASSES = len(Gender)
BATCH_SIZE = 64
EPOCHS = 10
LEARNING_RATE = 0.0001
DENSE_UNITS = 512
DROPOUT = 0.5

OUT_DIR_NAME = "outputs"
MODEL_FILENAME_TEMPLATE = "attention_fairness_{mask_gender}_{mask_feature_names}.keras"
RESULTS_FILENAME_TEMPLATE = "attention_fairness_{mask_gender}_{mask_feature_names}.json"

FACE_LANDMARKER: Optional[FaceLandmarker] = None
FEATURE_INDICES_MAP: Optional[Dict[Feature, List[int]]] = None


def set_seeds() -> None:
    """Sets random seeds for reproducibility."""
    os.environ["PYTHONHASHSEED"] = str(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)
    tf.get_logger().setLevel("ERROR")


def _load_mediapipe_utils():
    """Loads the MediaPipe landmarker model and feature map."""
    global FACE_LANDMARKER, FEATURE_INDICES_MAP
    if FACE_LANDMARKER is not None and FEATURE_INDICES_MAP is not None:
        return

    landmarker_model_path = hf_hub_download(
        repo_id="rixmape/fairness-analysis-experiment",
        filename="mediapipe_landmarker.task",
        repo_type="model",
    )
    landmarker_options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=landmarker_model_path),
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
        num_faces=1,
    )
    FACE_LANDMARKER = FaceLandmarker.create_from_options(landmarker_options)

    map_path = hf_hub_download(
        repo_id="rixmape/fairness-analysis-experiment",
        filename="landmark_map.json",
        repo_type="model",
    )
    with open(map_path, "r") as f:
        raw_map = json.load(f)

    FEATURE_INDICES_MAP = {}
    for key_str, value in raw_map.items():
        feature_enum = Feature(key_str)
        FEATURE_INDICES_MAP[feature_enum] = value


def load_raw_dataframe() -> pd.DataFrame:
    """Loads the raw dataset from Hugging Face."""
    path = Path(hf_hub_download(repo_id=DS_REPO, filename=DS_FILE, repo_type="dataset"))
    df = pd.read_parquet(path, columns=["image", "gender", "race", "age"])
    df["gender"] = df["gender"].astype(int)
    return df


def split_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Splits the dataframe into train, validation, and test sets."""
    df_train_val, df_test = train_test_split(df, test_size=TEST_RATIO, random_state=SEED, stratify=df["gender"])
    adj_val_ratio = VAL_RATIO / (1.0 - TEST_RATIO)
    df_train, df_val = train_test_split(df_train_val, test_size=adj_val_ratio, random_state=SEED, stratify=df_train_val["gender"])
    return df_train, df_val, df_test


def _get_pixel_landmarks(image_np: np.ndarray) -> List[Tuple[int, int]]:
    """Detects face landmarks and returns their pixel coordinates."""
    if FACE_LANDMARKER is None:
        _load_mediapipe_utils()

    if image_np.dtype != np.uint8:
        if image_np.dtype in [np.float32, np.float64]:
            image_uint8 = (image_np * 255).clip(0, 255).astype(np.uint8)
        else:
            image_uint8 = image_np.astype(np.uint8)
    else:
        image_uint8 = image_np

    if len(image_uint8.shape) == 2:
        image_uint8 = cv2.cvtColor(image_uint8, cv2.COLOR_GRAY2RGB)
    elif image_uint8.shape[-1] == 1:
        image_uint8 = cv2.cvtColor(image_uint8, cv2.COLOR_GRAY2RGB)

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_uint8)
    detection_result: Optional[FaceLandmarkerResult] = FACE_LANDMARKER.detect(mp_image)

    if not detection_result or not detection_result.face_landmarks:
        return []

    landmarks = detection_result.face_landmarks[0]
    img_height, img_width = image_uint8.shape[:2]
    pixel_coords = [(int(pt.x * img_width), int(pt.y * img_height)) for pt in landmarks]
    return pixel_coords


def _get_feature_bbox(pixel_coords: List[Tuple[int, int]], feature: Feature, img_height: int, img_width: int) -> Optional[Tuple[int, int, int, int]]:
    """Calculates the bounding box for a given feature."""
    if FEATURE_INDICES_MAP is None:
        _load_mediapipe_utils()

    indices = FEATURE_INDICES_MAP.get(feature)
    if not indices or not pixel_coords or max(indices) >= len(pixel_coords):
        return None

    points = [pixel_coords[i] for i in indices]
    min_x = min(p[0] for p in points)
    min_y = min(p[1] for p in points)
    max_x = max(p[0] for p in points)
    max_y = max(p[1] for p in points)

    min_x_pad = max(0, min_x - MASK_PIXEL_PADDING)
    min_y_pad = max(0, min_y - MASK_PIXEL_PADDING)
    max_x_pad = min(img_width - 1, max_x + MASK_PIXEL_PADDING)
    max_y_pad = min(img_height - 1, max_y + MASK_PIXEL_PADDING)

    if min_x_pad >= max_x_pad or min_y_pad >= max_y_pad:
        return None

    return (min_x_pad, min_y_pad, max_x_pad, max_y_pad)


def apply_feature_mask(image_np: np.ndarray) -> np.ndarray:
    """Applies masks to specified features using MediaPipe landmarks."""
    pixel_coords = _get_pixel_landmarks(image_np)
    if not pixel_coords:
        return image_np

    masked_image = image_np.copy()
    img_height, img_width = image_np.shape[:2]

    for feature_enum in MASK_FEATURES:
        bbox = _get_feature_bbox(pixel_coords, feature_enum, img_height, img_width)
        if bbox:
            min_x, min_y, max_x, max_y = bbox
            masked_image[min_y : max_y + 1, min_x : max_x + 1] = 0

    return masked_image


@tf.function
def parse_and_preprocess_tf(image_bytes: tf.Tensor, label: tf.Tensor, apply_masking: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    """Decodes, resizes, conditionally masks, normalizes images, and returns tensors."""
    image = tf.io.decode_image(image_bytes, channels=NUM_CHANNELS, expand_animations=False)
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    image = tf.ensure_shape(image, (IMG_SIZE, IMG_SIZE, NUM_CHANNELS))

    def mask_py_func(img_tensor, should_mask):
        if should_mask:
            img_np = img_tensor.numpy()
            masked_img_np = apply_feature_mask(img_np)
            return tf.convert_to_tensor(masked_img_np, dtype=tf.float32)
        else:
            return tf.cast(img_tensor, dtype=tf.float32)

    image_processed = tf.py_function(func=mask_py_func, inp=[image, apply_masking], Tout=tf.float32)
    image_processed.set_shape((IMG_SIZE, IMG_SIZE, NUM_CHANNELS))

    image_normalized = image_processed / 255.0
    label_int = tf.cast(label, tf.int32)
    return image_normalized, label_int


def create_tf_dataset(df: pd.DataFrame, batch_size: int, shuffle: bool = False, is_train_set: bool = False) -> tf.data.Dataset:
    """Creates a TensorFlow Dataset with conditional masking."""
    img_bytes_list = []
    labels_list = []
    mask_flags = []

    for _, row in df.iterrows():
        if isinstance(row["image"], dict) and "bytes" in row["image"]:
            img_bytes_list.append(row["image"]["bytes"])
            labels_list.append(row["gender"])
            should_mask = is_train_set and (row["gender"] == MASK_GENDER.value)
            mask_flags.append(should_mask)

    ds = tf.data.Dataset.from_tensor_slices((img_bytes_list, labels_list, mask_flags))

    ds = ds.map(lambda img, lbl, mask: parse_and_preprocess_tf(img, lbl, mask), num_parallel_calls=tf.data.AUTOTUNE)

    if shuffle:
        ds = ds.shuffle(buffer_size=len(df), seed=SEED)

    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def build_model() -> tf.keras.Model:
    """Builds the CNN model architecture."""
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
    """Trains the model and evaluates it on the test set."""
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True, verbose=0)
    history = model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=[early_stopping], verbose=0)
    test_loss, test_acc = model.evaluate(test_ds, verbose=0)
    serializable_history = {k: [float(val) for val in v] for k, v in history.history.items()}
    return {
        "epoch_history": serializable_history,
        "test_metrics": {"loss": float(test_loss), "accuracy": float(test_acc)},
    }


def save_results(model: tf.keras.Model, results_data: Dict[str, Any], out_dir: Path) -> None:
    """Saves the trained model and evaluation results."""
    out_dir.mkdir(parents=True, exist_ok=True)
    mask_gender_name = MASK_GENDER.name.lower()
    mask_feature_names = "_".join(sorted([f.name.lower() for f in MASK_FEATURES]))

    model_filename = MODEL_FILENAME_TEMPLATE.format(mask_gender=mask_gender_name, mask_feature_names=mask_feature_names)
    model_filepath = out_dir / model_filename
    model.save(model_filepath)

    results_filename = RESULTS_FILENAME_TEMPLATE.format(mask_gender=mask_gender_name, mask_feature_names=mask_feature_names)
    results_filepath = out_dir / results_filename
    results_data["script_parameters"] = {
        "MASK_GENDER": MASK_GENDER.name,
        "MASK_FEATURES": [f.name for f in MASK_FEATURES],
        "MASK_PIXEL_PADDING": MASK_PIXEL_PADDING,
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


def main() -> None:
    set_seeds()
    _load_mediapipe_utils()
    out_dir = Path(OUT_DIR_NAME)

    df_full = load_raw_dataframe()
    if len(df_full) > TARGET_TRAIN_SAMPLES * 2:
        df_sampled = df_full.sample(n=int(TARGET_TRAIN_SAMPLES / (1 - TEST_RATIO - VAL_RATIO)), random_state=SEED)
    else:
        df_sampled = df_full

    df_train, df_val, df_test = split_data(df_sampled)

    train_ds = create_tf_dataset(df_train, BATCH_SIZE, shuffle=True, is_train_set=True)
    val_ds = create_tf_dataset(df_val, BATCH_SIZE, shuffle=False, is_train_set=False)
    test_ds = create_tf_dataset(df_test, BATCH_SIZE, shuffle=False, is_train_set=False)

    model = build_model()
    results = train_and_evaluate_model(model, train_ds, val_ds, test_ds, EPOCHS)
    save_results(model, results, out_dir)


if __name__ == "__main__":
    main()
