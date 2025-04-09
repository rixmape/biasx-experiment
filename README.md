# Fariness Experiment

An experimental framework for conducting machine learning experiments focused on analyzing fairness in facial recognition models.

## Core Features

* **Dataset Handling:** Loads and prepares facial datasets (e.g., UTKFace), including stratified sampling and train/validation/test splitting.
* **Image Preprocessing:** Resizes, normalizes, and optionally applies masking to facial features based on configuration.
* **Model Training:** Builds and trains a configurable Convolutional Neural Network (CNN) using TensorFlow/Keras for demographic attribute prediction (e.g., gender, race, age).
* **Visual Explanations:** Generates GradCAM++ heatmaps to visualize model attention and calculates attention scores for detected facial features using MediaPipe.
* **Fairness Violation Analysis:** Computes various fairness violation metrics (e.g., Demographic Parity, Equalized Odds) and analyzes feature distributions across demographic groups.
* **Structured Output:** Saves comprehensive experiment results, including configuration, training history, fairness violation metrics, detailed explanations per image, and image/heatmap artifacts.

## Installation

Clone the repository and install the package using pip:

```bash
git clone https://github.com/rixmape/fairness-experiment
cd fairness-experiment
pip install .
```
