import os
import random
from pathlib import Path
from typing import Dict, Tuple

import cv2
import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from huggingface_hub import hf_hub_download
from tf_keras_vis.gradcam import Gradcam
from tf_keras_vis.gradcam_plus_plus import GradcamPlusPlus
from tf_keras_vis.saliency import Saliency
from tf_keras_vis.scorecam import Scorecam

SEED = 42
DATASET_NAME = "utkface"
NUM_SAMPLES = 6
IMG_SIZE = 75
NUM_CHANNELS = 3
NUM_CLASSES = 2

DS_REPO = "rixmape/utkface"
DS_FILE = "data/train-00000-of-00001.parquet"
MODEL_PATH = "/content/vgg16_utkface_gender_classifier.keras"


def set_seeds() -> None:
    """Sets random seeds using global SEED for reproducibility."""
    os.environ["PYTHONHASHSEED"] = str(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)
    tf.get_logger().setLevel("ERROR")


def load_sampled_data() -> pd.DataFrame:
    """Downloads (if needed), loads parquet, and samples data using global constants."""
    local_path = Path(hf_hub_download(repo_id=DS_REPO, filename=DS_FILE, repo_type="dataset"))
    df = pd.read_parquet(local_path)
    df = df[df["age"] > 0]
    return df.sample(n=NUM_SAMPLES, random_state=SEED)


@tf.function
def preprocess_for_vis_tf(image_bytes: tf.Tensor, label: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """Decodes, resizes, and normalizes images using TensorFlow, returning normalized, original, and label tensors."""
    image = tf.io.decode_image(image_bytes, channels=NUM_CHANNELS, expand_animations=False, dtype=tf.uint8)
    image_resized = tf.image.resize(image, [IMG_SIZE, IMG_SIZE], method=tf.image.ResizeMethod.BILINEAR)
    image_resized_uint8 = tf.cast(image_resized, tf.uint8)
    image_normalized = tf.cast(image_resized_uint8, tf.float32) / 255.0
    return image_normalized, image_resized_uint8, tf.cast(label, tf.int32)


def load_and_preprocess_for_vis(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Creates TF dataset, preprocesses images via TF, and returns NumPy arrays for visualization."""
    img_bytes_list = [img_dict["bytes"] for img_dict in df["image"]]
    labels_list = df["gender"].tolist()
    ds = tf.data.Dataset.from_tensor_slices((img_bytes_list, labels_list))
    ds = ds.map(preprocess_for_vis_tf, num_parallel_calls=tf.data.AUTOTUNE)
    images_normalized_np = np.stack([item[0].numpy() for item in ds])
    images_original_np = np.stack([item[1].numpy() for item in ds])
    labels_np = np.array([item[2].numpy() for item in ds], dtype=np.int32)
    return images_normalized_np, labels_np, images_original_np


def generate_attention_maps(model: keras.Model, images_normalized: np.ndarray, labels: np.ndarray) -> Dict[str, np.ndarray]:
    """Generates multiple types of attention maps using tf-keras-vis."""
    print("Generating attention maps...")
    attention_maps = {}

    target_scores = lambda output: tuple(output[i, l] for i, l in enumerate(labels))
    replace_to_linear = lambda m: setattr(m.layers[-1], "activation", keras.activations.linear)

    saliency = Saliency(model, model_modifier=None, clone=True)
    gradcam = Gradcam(model, model_modifier=replace_to_linear, clone=True)
    gradcam_plus_plus = GradcamPlusPlus(model, model_modifier=replace_to_linear, clone=True)
    scorecam = Scorecam(model, model_modifier=replace_to_linear, clone=True)

    attention_maps["SmoothGrad"] = saliency(target_scores, images_normalized, smooth_samples=20, smooth_noise=0.20)
    attention_maps["GradCAM"] = gradcam(target_scores, images_normalized, penultimate_layer=-1)
    attention_maps["GradCAM++"] = gradcam_plus_plus(target_scores, images_normalized, penultimate_layer=-1)
    attention_maps["ScoreCAM"] = scorecam(target_scores, images_normalized, penultimate_layer=-1, max_N=10)

    return attention_maps


def visualize_attention(images_original: np.ndarray, attention_maps: Dict[str, np.ndarray]) -> None:
    """Creates and displays a grid visualizing images and attention maps with row labels."""
    print("Creating visualization grid...")
    print("Creating visualization grid...")
    num_images = images_original.shape[0]
    map_names = ["Original"] + list(attention_maps.keys())
    num_rows = len(map_names)

    fig, axes = plt.subplots(nrows=num_rows, ncols=num_images, figsize=(num_images * 2, num_rows * 2), squeeze=False)

    for r, map_name in enumerate(map_names):
        maps_for_row = attention_maps.get(map_name)

        for c in range(num_images):
            ax = axes[r, c]
            img_display = images_original[c]

            if maps_for_row is None:
                ax.imshow(img_display)
            else:
                current_map = maps_for_row[c]
                if "Saliency" in map_name or "SmoothGrad" in map_name:
                    ax.imshow(current_map, cmap="viridis")
                else:
                    heatmap_rgb = np.uint8(plt.cm.jet(current_map.astype(np.float32))[..., :3] * 255)
                    overlay = cv2.addWeighted(img_display, 0.6, heatmap_rgb, 0.4, 0)
                    ax.imshow(overlay)

            ax.set_xticks([])
            ax.set_yticks([])
            ax.spines[:].set_visible(False)

            if c == 0:
                ax.set_ylabel(map_name)

    plt.tight_layout()
    plt.show()


def main() -> None:
    """Executes the visualization pipeline for multiple attention methods."""
    set_seeds()
    print("Loading and preparing data...")
    df_sampled = load_sampled_data()
    images_processed_np, labels_np, images_original_np = load_and_preprocess_for_vis(df_sampled)
    print(f"Loaded {len(df_sampled)} samples for visualization.")

    model_path = Path(MODEL_PATH)
    print(f"Loading model from {model_path}...")
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    model = keras.models.load_model(model_path)

    maps_dict = generate_attention_maps(model, images_processed_np, labels_np)
    visualize_attention(images_original_np, maps_dict)
    print("Script finished.")


if __name__ == "__main__":
    main()
