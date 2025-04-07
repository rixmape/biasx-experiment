import os
from pathlib import Path

import tensorflow as tf
import visualkeras

IMG_SIZE = 48
NUM_CHANNELS = 3
NUM_CLASSES = 2
OUTPUT_DIR_NAME = "outputs"


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
    model.add(tf.keras.layers.Dense(512, activation="relu", name="dense"))
    model.add(tf.keras.layers.Dropout(0.5, name="dropout"))
    model.add(tf.keras.layers.Dense(NUM_CLASSES, activation="softmax", name="output"))

    return model


def main() -> None:
    print("Building the CNN model...")
    model = build_model()
    model.summary()

    print("\nGenerating model architecture visualization using visualkeras...")
    output_dir = Path(OUTPUT_DIR_NAME)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "cnn_model_architecture.png"

    visualkeras.layered_view(model, to_file=str(output_path), legend=True, max_z=100, draw_volume=True, spacing=20)
    print(f"\nModel visualization saved successfully to: {output_path.resolve()}")


if __name__ == "__main__":
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    tf.get_logger().setLevel("ERROR")
    main()
