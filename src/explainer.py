import json
import logging
import os
from typing import Dict, List, Optional, Tuple

import mediapipe as mp
import numpy as np
import tensorflow as tf
from huggingface_hub import hf_hub_download
from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.vision.face_landmarker import FaceLandmarker, FaceLandmarkerOptions, FaceLandmarkerResult
from tf_keras_vis.gradcam_plus_plus import GradcamPlusPlus

# isort: off
from .definitions import DemographicValue, Feature, FeatureDetails
from .settings import Settings

logger = logging.getLogger(__name__)


class Explainer:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.landmarker = self._load_landmarker()
        self.feature_indices_map = self._load_feature_indices_map()

    def _load_landmarker(self) -> FaceLandmarker:
        try:
            model_path = hf_hub_download(
                repo_id="rixmape/fairness-analysis-experiment",
                filename="mediapipe_landmarker.task",
                repo_type="model",
            )
            options = FaceLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=model_path),
                output_face_blendshapes=False,
                output_facial_transformation_matrixes=False,
                num_faces=1,
            )
            landmarker = FaceLandmarker.create_from_options(options)
            return landmarker
        except Exception as e:
            raise RuntimeError(f"Failed to load landmarker model: {e}") from e

    def _load_feature_indices_map(self) -> Dict[Feature, List[int]]:
        try:
            map_path = hf_hub_download(
                repo_id="rixmape/fairness-analysis-experiment",
                filename="landmark_map.json",
                repo_type="model",
            )
            with open(map_path, "r") as f:
                raw_map = json.load(f)
            feature_map = {Feature(key): value for key, value in raw_map.items()}
            return feature_map
        except Exception as e:
            raise RuntimeError(f"Failed to load feature indices map: {e}") from e

    def _get_pixel_landmarks(
        self,
        image_np: np.ndarray,
    ) -> List[Tuple[int, int]]:
        if image_np.dtype in [np.float32, np.float64]:
            image_uint8 = (image_np * 255).clip(0, 255).astype(np.uint8)
        else:
            image_uint8 = image_np.astype(np.uint8)

        if len(image_uint8.shape) != 3 or image_uint8.shape[-1] != 3:
            image_uint8 = np.stack([image_uint8] * 3, axis=-1)

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_uint8)
        detection_result: Optional[FaceLandmarkerResult] = self.landmarker.detect(mp_image)

        if not detection_result or not detection_result.face_landmarks:
            return []

        landmarks = detection_result.face_landmarks[0]
        img_size_h, img_size_w = image_np.shape[:2]
        pixel_coords = [(int(pt.x * img_size_w), int(pt.y * img_size_h)) for pt in landmarks]
        return pixel_coords

    def _get_feature_bbox(
        self,
        pixel_coords: List[Tuple[int, int]],
        feature: Feature,
        img_height: int,
        img_width: int,
    ) -> Optional[FeatureDetails]:
        indices = self.feature_indices_map.get(feature)
        if not indices or not pixel_coords:
            return None
        if max(indices) >= len(pixel_coords):
            return None

        try:
            points = [pixel_coords[i] for i in indices]
            min_x = min(x for x, y in points)
            min_y = min(y for x, y in points)
            max_x = max(x for x, y in points)
            max_y = max(y for x, y in points)
        except IndexError:
            return None

        pad = self.settings.analysis.mask_pixel_padding
        min_x_pad = max(0, min_x - pad)
        min_y_pad = max(0, min_y - pad)
        max_x_pad = min(img_width, max_x + pad)
        max_y_pad = min(img_height, max_y + pad)

        if min_x_pad >= max_x_pad or min_y_pad >= max_y_pad:
            return None

        return FeatureDetails(
            feature=feature,
            min_x=min_x_pad,
            min_y=min_y_pad,
            max_x=max_x_pad,
            max_y=max_y_pad,
            attention_score=0.0,
            is_key_feature=False,
        )

    def apply_mask(
        self,
        image_np: np.ndarray,
    ) -> np.ndarray:
        pixel_coords = self._get_pixel_landmarks(image_np)
        if not pixel_coords:
            return image_np

        masked_image = image_np.copy()
        img_height, img_width = image_np.shape[:2]

        for feature_enum in self.settings.analysis.mask_features:
            feature_details = self._get_feature_bbox(pixel_coords, feature_enum, img_height, img_width)
            if feature_details:
                masked_image[feature_details.min_y : feature_details.max_y, feature_details.min_x : feature_details.max_x] = 0

        return masked_image

    def get_features(
        self,
        image_np: np.ndarray,
    ) -> List[FeatureDetails]:
        pixel_coords = self._get_pixel_landmarks(image_np)
        if not pixel_coords:
            return []

        img_height, img_width = image_np.shape[:2]
        detected_features: List[FeatureDetails] = []

        for feature_enum in self.feature_indices_map.keys():
            feature_details = self._get_feature_bbox(pixel_coords, feature_enum, img_height, img_width)
            if feature_details:
                detected_features.append(feature_details)

        return detected_features

    def _calculate_heatmap(
        self,
        heatmap_generator: GradcamPlusPlus,
        model: tf.keras.Model,
        image_np: np.ndarray,
        label: DemographicValue,
    ) -> np.ndarray:
        target_class = lambda output: output[0][label.value]
        image_batch = np.expand_dims(image_np.astype(np.float32), axis=0)
        target_layer = "block3_conv3"

        if target_layer not in [layer.name for layer in model.layers]:
            raise ValueError(f"Target layer '{target_layer}' not found for heatmap generation.")

        try:
            heatmap = heatmap_generator(target_class, image_batch, penultimate_layer=target_layer)[0]
        except Exception as e:
            raise RuntimeError(f"Heatmap generation via GradCAM++ failed: {e}") from e

        min_val, max_val = np.min(heatmap), np.max(heatmap)
        if max_val <= min_val:
            logger.warning(f"Invalid heatmap range (min={min_val}, max={max_val}).")

        normalized_heatmap = (heatmap - min_val) / (max_val - min_val)
        return normalized_heatmap.astype(np.float32)

    def _calculate_single_feature_attention(
        self,
        feature: FeatureDetails,
        heatmap: np.ndarray,
    ) -> float:
        heatmap_height, heatmap_width = heatmap.shape[:2]

        min_y, max_y = max(0, feature.min_y), min(heatmap_height, feature.max_y)
        min_x, max_x = max(0, feature.min_x), min(heatmap_width, feature.max_x)

        if min_y >= max_y or min_x >= max_x:
            logger.warning(f"Invalid feature region for attention: box=({min_x}, {min_y}, {max_x}, {max_y})")
            return 0.0

        feature_attention_region = heatmap[min_y:max_y, min_x:max_x]

        if feature_attention_region.size == 0:
            logger.warning(f"Empty feature region for attention: box=({min_x}, {min_y}, {max_x}, {max_y})")
            return 0.0

        attention = np.nanmean(feature_attention_region)

        if np.isnan(attention):
            logger.warning(f"Feature region for {feature.feature.name} contained only NaN values.")
            return 0.0

        return float(attention)

    def _save_heatmap(
        self,
        heatmap: np.ndarray,
        image_id: str,
    ) -> Optional[str]:
        savepath = os.path.join(self.settings.output.base_path, "heatmaps")
        os.makedirs(savepath, exist_ok=True)
        filename = f"heatmap_{image_id}.npy"
        filepath = os.path.join(savepath, filename)

        try:
            np.save(filepath, heatmap.astype(np.float16))
            return filepath
        except Exception as e:
            logger.error(f"Failed to save heatmap to {filepath}: {e}")

    def _compute_feature_details(
        self,
        features: List[FeatureDetails],
        heatmap: np.ndarray,
    ) -> List[FeatureDetails]:
        for feature_detail in features:
            attention_score = self._calculate_single_feature_attention(feature_detail, heatmap)
            is_key = attention_score >= self.settings.analysis.key_feature_threshold
            feature_detail.attention_score = float(attention_score)
            feature_detail.is_key_feature = bool(is_key)
        return features

    def get_heatmap_generator(
        self,
        model: tf.keras.Model,
    ) -> GradcamPlusPlus:
        try:
            replace_to_linear = lambda m: setattr(m.layers[-1], "activation", tf.keras.activations.linear)
            generator = GradcamPlusPlus(model, model_modifier=replace_to_linear, clone=True)
            return generator
        except Exception as e:
            raise RuntimeError(f"Failed to create GradCAM++ generator: {e}") from e

    def generate_explanation(
        self,
        heatmap_generator: GradcamPlusPlus,
        model: tf.keras.Model,
        image_np: np.ndarray,
        label: DemographicValue,
        image_id: str,
    ) -> Tuple[List[FeatureDetails], Optional[str]]:
        heatmap = self._calculate_heatmap(heatmap_generator, model, image_np, label)
        heatmap_path = self._save_heatmap(heatmap, image_id)
        detected_features = self.get_features(image_np)
        feature_details_with_attention = self._compute_feature_details(detected_features, heatmap)

        return feature_details_with_attention, heatmap_path
