import json
import os
import random
import warnings
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from huggingface_hub import hf_hub_download
from keras.api.layers import AveragePooling2D, Conv2D, Dense, Dropout, Flatten, GlobalAveragePooling2D, GlobalMaxPooling2D, Input, MaxPooling2D
from keras.api.models import Model
from sklearn.model_selection import train_test_split
from tf_keras_vis.gradcam_plus_plus import GradcamPlusPlus
from tf_keras_vis.utils.scores import CategoricalScore

cfg: Dict[str, Any] = {
    "dataset_name": "utkface",
    "ds_repo": "rixmape/utkface",
    "ds_file": "data/train-00000-of-00001.parquet",
    "num_samples_train_val": 2000,
    "num_samples_vis": 8,
    "img_size": 75,
    "num_channels": 3,
    "num_classes": 2,
    "dense_units": 512,
    "dropout": 0.5,
    "learning_rate": 0.0001,
    "batch_size": 32,
    "epochs": 10,
    "val_ratio": 0.2,
    "seed": 42,
    "output_dir": "outputs",
}
OUTPUT_DIR = Path(cfg["output_dir"])

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
tf.get_logger().setLevel("ERROR")
os.environ["PYTHONHASHSEED"] = str(cfg["seed"])
random.seed(cfg["seed"])
np.random.seed(cfg["seed"])
tf.random.set_seed(cfg["seed"])
print(f"Seeds set to {cfg['seed']}")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def download_dataset(repo_id: str, filename: str, n: int, seed: int) -> pd.DataFrame:
    """Downloads dataset from Hugging Face Hub and samples it."""
    print(f"Loading {repo_id} and sampling {n} images...")
    local_path = Path(hf_hub_download(repo_id=repo_id, filename=filename, repo_type="dataset"))
    df = pd.read_parquet(local_path)
    return df.sample(n=n, random_state=seed) if 0 < n < len(df) else df


@tf.function
def parse_and_preprocess(image_bytes: tf.Tensor, label: tf.Tensor, img_size: int, num_channels: int, num_classes: int) -> Tuple[tf.Tensor, tf.Tensor]:
    """Decodes, resizes, normalizes image and one-hot encodes label for training."""
    image = tf.io.decode_image(image_bytes, channels=num_channels, expand_animations=False)
    image = tf.image.resize(image, [img_size, img_size])
    image = tf.ensure_shape(image, (img_size, img_size, num_channels))
    image_normalized = tf.cast(image, tf.float32) / 255.0
    label_one_hot = tf.one_hot(tf.cast(label, tf.int32), depth=num_classes)
    return image_normalized, label_one_hot


@tf.function
def parse_and_preprocess_for_vis(image_bytes: tf.Tensor, label: tf.Tensor, img_size: int, num_channels: int) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """Decodes, resizes image (keeping uint8 and normalized versions), returns integer label."""
    image = tf.io.decode_image(image_bytes, channels=num_channels, expand_animations=False)
    image_resized = tf.image.resize(image, [img_size, img_size], method=tf.image.ResizeMethod.BILINEAR)
    image_resized_uint8 = tf.cast(image_resized, tf.uint8)
    image_resized_uint8 = tf.ensure_shape(image_resized_uint8, (img_size, img_size, num_channels))
    image_normalized = tf.cast(image_resized_uint8, tf.float32) / 255.0
    return image_normalized, image_resized_uint8, tf.cast(label, tf.int32)


def create_tf_dataset(
    df: pd.DataFrame,
    preprocess_func: Callable,
    batch_size: int,
    seed: int,
    shuffle: bool = False,
    batch: bool = True,
    **kwargs: Any,
) -> tf.data.Dataset:
    """Creates a tf.data.Dataset from a DataFrame."""
    ds = tf.data.Dataset.from_tensor_slices((df["image"].apply(lambda x: x["bytes"]).tolist(), df["gender"].tolist()))
    map_func = lambda img, lbl: preprocess_func(img, lbl, cfg["img_size"], cfg["num_channels"], **kwargs)
    ds = ds.map(map_func, num_parallel_calls=tf.data.AUTOTUNE)
    if shuffle:
        ds = ds.shuffle(buffer_size=len(df), seed=seed, reshuffle_each_iteration=False)
    if batch:
        ds = ds.batch(batch_size)
    return ds.prefetch(tf.data.AUTOTUNE)


def get_fixed_vis_samples(df_full: pd.DataFrame, cfg_dict: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Prepares a fixed set of samples for consistent visualization."""
    df_vis = df_full.sample(n=cfg_dict["num_samples_vis"], random_state=cfg_dict["seed"])
    vis_ds = create_tf_dataset(df_vis, parse_and_preprocess_for_vis, 1, cfg_dict["seed"], shuffle=False, batch=False)
    print(f"Preparing {cfg_dict['num_samples_vis']} fixed samples for visualization...")
    norm_list, orig_list, label_list = [], [], []
    for norm_img, orig_img, label_int in vis_ds:
        norm_list.append(norm_img.numpy())
        orig_list.append(orig_img.numpy())
        label_list.append(label_int.numpy())
    return np.array(norm_list), np.array(orig_list), np.array(label_list)


def build_vgg_block(x: tf.Tensor, filters: int, num_convs: int, block_num: int) -> tf.Tensor:
    """Builds a VGG convolutional block."""
    for i in range(num_convs):
        x = Conv2D(filters, (3, 3), activation="relu", padding="same", name=f"block{block_num}_conv{i+1}")(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name=f"block{block_num}_pool")(x)
    return x


def build_vgg_base(input_shape: Tuple[int, int, int], num_blocks: int) -> Model:
    """Builds the VGG convolutional base with a variable number of blocks."""
    inputs = Input(shape=input_shape)
    x = inputs
    filters = [64, 128, 256, 512, 512]
    convs = [2, 2, 3, 3, 3]
    for i in range(num_blocks):
        x = build_vgg_block(x, filters[i], convs[i], block_num=i + 1)
    return Model(inputs=inputs, outputs=x, name=f"vgg_base_{num_blocks}blocks")


def build_model_with_pooling_base(input_shape: Tuple[int, int, int]) -> Model:
    """Builds a fixed 3-block VGG base, stopping before the block3 pool."""
    inputs = Input(shape=input_shape)
    x = build_vgg_block(inputs, 64, 2, 1)
    x = build_vgg_block(x, 128, 2, 2)
    x = Conv2D(256, (3, 3), activation="relu", padding="same", name="block3_conv1")(x)
    x = Conv2D(256, (3, 3), activation="relu", padding="same", name="block3_conv2")(x)
    x = Conv2D(256, (3, 3), activation="relu", padding="same", name="block3_conv3")(x)
    return Model(inputs=inputs, outputs=x, name="vgg_base_3blocks_pooling_exp")


def add_pooling_and_classifier(base_output: tf.Tensor, pooling_type: str, cfg_dict: Dict[str, Any]) -> tf.Tensor:
    """Adds a specified pooling layer and classifier head to the base output."""
    pooling_layers = {
        "max_pooling": MaxPooling2D((2, 2), name="exp_pool"),
        "avg_pooling": AveragePooling2D((2, 2), name="exp_pool"),
        "global_avg_pooling": GlobalAveragePooling2D(name="exp_global_avg_pool"),
        "global_max_pooling": GlobalMaxPooling2D(name="exp_global_max_pool"),
        "flatten": Flatten(name="exp_flatten"),
    }
    pool_layer = pooling_layers[pooling_type]
    x = pool_layer(base_output)
    if pooling_type in ["max_pooling", "avg_pooling"]:
        x = Flatten(name="exp_flatten_after_pool")(x)

    x = Dense(cfg_dict["dense_units"], activation="relu", name="classifier_dense")(x)
    x = Dropout(cfg_dict["dropout"], name="classifier_dropout")(x)
    return Dense(cfg_dict["num_classes"], activation="softmax", name="classifier_output")(x)


def build_experimental_model(cfg_dict: Dict[str, Any], num_blocks: int = 3, pooling_type: str = "gap") -> Model:
    """Builds and compiles an experimental model based on depth or pooling variant."""
    input_shape = (cfg_dict["img_size"], cfg_dict["img_size"], cfg_dict["num_channels"])
    if "pooling" in pooling_type.lower() or pooling_type == "flatten":
        base = build_model_with_pooling_base(input_shape)
        outputs = add_pooling_and_classifier(base.output, pooling_type, cfg_dict)
        name = f"model_3block_{pooling_type}"
    else:
        base = build_vgg_base(input_shape, num_blocks)
        outputs = add_pooling_and_classifier(base.output, "global_avg_pooling", cfg_dict)
        name = f"model_{num_blocks}block_gap"

    model = Model(inputs=base.input, outputs=outputs, name=name)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=cfg_dict["learning_rate"]),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train_model(
    model: Model,
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
    cfg_dict: Dict[str, Any],
    model_name: str,
) -> Model:
    """Trains the model with early stopping and saves the best weights."""
    print(f"\n--- Training {model_name} ---")
    model_path = OUTPUT_DIR / f"{model_name}.keras"
    callbacks = [keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True, verbose=0)]
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=cfg_dict["epochs"],
        callbacks=callbacks,
        verbose=0,
    )
    print(f"Finished training. Saving model to {model_path}")
    model.save(model_path)
    return model


def get_target_layer_name(model: Model, num_blocks: Optional[int] = None) -> str:
    """Determines the target convolutional layer name for CAM generation."""
    if num_blocks:
        layer_map = {2: "block2_conv2", 3: "block3_conv3", 4: "block4_conv3", 5: "block5_conv3"}
        return model.get_layer(layer_map[num_blocks]).name
    return model.get_layer("block3_conv3").name


def generate_cam_batch(
    model: Model,
    images_norm: np.ndarray,
    labels_int: np.ndarray,
    target_layer_name: str,
) -> np.ndarray:
    """Generates Grad-CAM++ heatmaps for a batch of images."""
    print(f"Generating GradCAM++ for {len(images_norm)} images using {target_layer_name}")
    score = CategoricalScore(labels_int.tolist())

    def model_modifier(m: Model) -> Model:
        m.layers[-1].activation = keras.activations.linear
        return m

    cam_visualizer = GradcamPlusPlus(model, model_modifier=model_modifier, clone=False)
    cams = cam_visualizer(score, images_norm, penultimate_layer=target_layer_name)
    return cams.numpy() if hasattr(cams, "numpy") else np.array(cams)


def calculate_concentration_ratio(cam: np.ndarray, top_percent: float = 10.0) -> float:
    """Calculates the ratio of activation in the top X% pixels to the total activation."""
    flat_cam = cam.flatten()
    total_activation = np.sum(flat_cam) + 1e-7
    top_k = max(1, int(len(flat_cam) * top_percent / 100))
    top_activation = np.sum(np.sort(flat_cam)[::-1][:top_k])
    return np.clip(float(top_activation / total_activation), 0.0, 1.0)


def calculate_metrics(results_dict: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Dict[str, Any]]:
    """Calculates mean and std deviation of concentration ratios for each variant."""
    print("Calculating concentration ratio metrics...")
    metrics: Dict[str, Dict[str, Any]] = {}
    for variant, results in results_dict.items():
        ratios = [calculate_concentration_ratio(r["activation"]) for r in results if r.get("activation") is not None]
        if ratios:
            metrics[variant] = {"mean": float(np.mean(ratios)), "std": float(np.std(ratios)), "values": ratios}
            print(f"Variant '{variant}': Mean Ratio = {metrics[variant]['mean']:.4f} +/- {metrics[variant]['std']:.4f}")
        else:
            metrics[variant] = {"mean": 0.0, "std": 0.0, "values": []}
            print(f"Warning: No valid CAMs found for variant '{variant}'")
    return metrics


def display_grid(
    results_dict: Dict[str, List[Dict[str, Any]]],
    title: str,
    filename: str,
    samples_per_row: int,
) -> None:
    """Displays and saves a grid of images overlaid with CAM heatmaps."""
    valid_variants = {k: v for k, v in results_dict.items() if v and len(v) >= samples_per_row}
    if not valid_variants:
        print("No valid variants with enough samples to display.")
        return
    num_variants = len(valid_variants)
    fig, axes = plt.subplots(num_variants, samples_per_row, figsize=(samples_per_row * 2, num_variants * 2), squeeze=False)

    for r_idx, (variant, results) in enumerate(sorted(valid_variants.items())):
        for c_idx in range(samples_per_row):
            ax = axes[r_idx, c_idx]
            img_data = results[c_idx]["image"]
            cam_data = results[c_idx]["activation"]

            if cam_data is None:
                continue

            img = np.array(img_data) if isinstance(img_data, list) else img_data
            cam = np.array(cam_data) if isinstance(cam_data, list) else cam_data

            cam_norm = (cam - np.min(cam)) / (np.max(cam) - np.min(cam) + 1e-6)
            heatmap = np.uint8(plt.cm.jet(cam_norm)[..., :3] * 255)
            overlay = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)
            ax.imshow(overlay)
            ax.set_xticks([])
            ax.set_yticks([])
            if c_idx == 0:
                ax.set_ylabel(f"{variant}", rotation=0, labelpad=30, va="center")
            if r_idx == 0:
                ax.set_title(f"S{c_idx+1}")

    plt.suptitle(title)
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    plt.subplots_adjust(wspace=0.05, hspace=0.05)
    output_path = OUTPUT_DIR / filename
    print(f"Saving image grid to {output_path}")
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_metrics(metrics: Dict[str, Dict[str, Any]], title: str, filename: str) -> None:
    """Plots and saves the concentration ratio metrics as a bar chart."""
    valid_metrics = {k: v for k, v in metrics.items() if v["values"]}
    if not valid_metrics:
        print("No metrics to plot.")
        return
    names = list(valid_metrics.keys())
    means = [valid_metrics[n]["mean"] for n in names]
    stds = [valid_metrics[n]["std"] for n in names]

    plt.figure(figsize=(8, 5))
    plt.bar(names, means, yerr=stds, capsize=4, alpha=0.8)
    plt.ylim(0, min(1.05, max(means) * 1.1 + max(stds) * 1.1) if means else 1.05)
    plt.ylabel("Concentration Ratio")
    plt.title(f"{title} - Metrics")
    plt.xticks(rotation=15, ha="right")
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()
    output_path = OUTPUT_DIR / f"{filename.split('.')[0]}_metrics.png"
    print(f"Saving metrics plot to {output_path}")
    plt.savefig(output_path, dpi=200)
    plt.close()


def run_experiment(
    exp_name: str,
    variants: List[Any],
    build_params_func: Callable[[Any], Dict[str, Any]],
    cfg_dict: Dict[str, Any],
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
    vis_norm: np.ndarray,
    vis_orig: np.ndarray,
    vis_labels: np.ndarray,
) -> Dict[str, Any]:
    """Runs a full experiment for a set of variants, returning results and metrics."""
    print(f"\n===== Running {exp_name} Experiment =====")
    results: Dict[str, List[Dict[str, Any]]] = {}
    trained_models: Dict[Any, Model] = {}

    vis_orig_list = vis_orig.tolist()

    for variant in variants:
        params = build_params_func(variant)
        model = build_experimental_model(cfg_dict, **params)
        model_name = model.name
        model = train_model(model, train_ds, val_ds, cfg_dict, model_name)
        trained_models[variant] = model

        target_layer = get_target_layer_name(model, params.get("num_blocks"))
        cams = generate_cam_batch(trained_models[variant], vis_norm, vis_labels, target_layer)
        cams_list = cams.tolist() if cams.ndim > 1 else [None] * len(vis_orig_list)

        results[variant] = [{"image": vis_orig_list[i], "activation": cams_list[i] if i < len(cams_list) else None} for i in range(len(vis_orig_list))]
        keras.backend.clear_session()

    title = f"Effect of {exp_name} on Grad-CAM++ Granularity"
    grid_filename = f"{exp_name.lower().replace(' ', '_')}_experiment_images.png"
    display_grid(results, title, grid_filename, cfg_dict["num_samples_vis"])
    metrics = calculate_metrics(results)
    plot_metrics(metrics, title, grid_filename)

    final_results_for_json = {}
    for variant, res_list in results.items():
        final_results_for_json[variant] = []
        for item in res_list:
            activation_data = item.get("activation")

            if isinstance(activation_data, np.ndarray):
                activation_data = activation_data.tolist()
            elif not isinstance(activation_data, (list, type(None))):
                activation_data = None
            final_results_for_json[variant].append({"image": item["image"], "activation": activation_data})

    return {"results": final_results_for_json, "metrics": metrics}


if __name__ == "__main__":
    all_experiment_results: Dict[str, Dict[str, Any]] = {}

    df_full = download_dataset(cfg["ds_repo"], cfg["ds_file"], cfg["num_samples_train_val"], cfg["seed"])
    vis_norm, vis_orig, vis_labels = get_fixed_vis_samples(df_full, cfg)
    print(f"Vis shapes: Norm={vis_norm.shape}, Orig={vis_orig.shape}, Labels={vis_labels.shape}")

    train_idx, val_idx = train_test_split(df_full.index, test_size=cfg["val_ratio"], random_state=cfg["seed"], stratify=df_full["gender"])
    df_train, df_val = df_full.loc[train_idx], df_full.loc[val_idx]

    train_ds = create_tf_dataset(df_train, parse_and_preprocess, cfg["batch_size"], cfg["seed"], shuffle=True, batch=True, num_classes=cfg["num_classes"])
    val_ds = create_tf_dataset(df_val, parse_and_preprocess, cfg["batch_size"], cfg["seed"], shuffle=False, batch=True, num_classes=cfg["num_classes"])
    print(f"Train/Val sizes: {len(df_train)}/{len(df_val)}")

    depth_results = run_experiment(
        "Architecture Depth",
        list(range(2, 6)),
        lambda nb: {"num_blocks": nb, "pooling_type": "gap"},
        cfg,
        train_ds,
        val_ds,
        vis_norm,
        vis_orig,
        vis_labels,
    )
    all_experiment_results["architecture_depth"] = depth_results

    pooling_results = run_experiment(
        "Pooling Strategy",
        ["flatten", "max_pooling", "avg_pooling", "global_avg_pooling", "global_max_pooling"],
        lambda pt: {"num_blocks": 3, "pooling_type": pt},
        cfg,
        train_ds,
        val_ds,
        vis_norm,
        vis_orig,
        vis_labels,
    )
    all_experiment_results["pooling_strategy"] = pooling_results

    json_output_path = OUTPUT_DIR / "experiment_summary.json"
    print(f"\nSaving experiment results and metrics to {json_output_path}")
    with open(json_output_path, "w") as f:
        json.dump(all_experiment_results, f, indent=4)

    print("\n===== Experiments Complete =====")
    print(f"Outputs saved in: {OUTPUT_DIR.resolve()}")
