import json
import logging
import os
from typing import Dict, List, Optional, Tuple, Union

import mediapipe as mp
import numpy as np
import tensorflow as tf
from huggingface_hub import hf_hub_download
from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.vision.face_landmarker import FaceLandmarker, FaceLandmarkerOptions, FaceLandmarkerResult
from tf_keras_vis.gradcam_plus_plus import GradcamPlusPlus

# isort: off
from .definitions import Age, Feature, FeatureDetails, Gender, Race
from .settings import Settings
from .utils import LOGGER_NAME

logger = logging.getLogger(LOGGER_NAME)

EXPECTED_LANDMARKS = 478  # Mediapipe standard


class Explainer:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.landmarker = self._load_landmarker()
        self.feature_indices_map = self._load_feature_indices_map()

    def _load_landmarker(self) -> FaceLandmarker:
        try:
            model_path = hf_hub_download(repo_id="rixmape/biasx-models", filename="mediapipe_landmarker.task", repo_type="model")
            options = FaceLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=model_path),
                output_face_blendshapes=False,
                output_facial_transformation_matrixes=False,
                num_faces=1,
            )
            landmarker = FaceLandmarker.create_from_options(options)
            logger.info("MediaPipe FaceLandmarker loaded successfully.")
            return landmarker
        except Exception as e:
            logger.error(f"Failed to load landmarker model: {e}")
            raise RuntimeError(f"Failed to load landmarker model: {e}") from e

    def _load_feature_indices_map(self) -> Dict[Feature, List[int]]:
        try:
            map_path = hf_hub_download(repo_id="rixmape/biasx-models", filename="landmark_map.json", repo_type="model")
            with open(map_path, "r") as f:
                raw_map = json.load(f)
            feature_map = {Feature(key): value for key, value in raw_map.items()}
            logger.info("Landmark feature map loaded successfully.")
            return feature_map
        except (json.JSONDecodeError, KeyError, ValueError, Exception) as e:
            logger.error(f"Failed to load or parse feature indices map: {e}")
            raise RuntimeError(f"Failed to load feature indices map: {e}") from e

    def _get_pixel_landmarks(self, image_np: np.ndarray) -> List[Tuple[int, int]]:
        try:
            if image_np.dtype in [np.float32, np.float64]:
                image_uint8 = (image_np * 255).clip(0, 255).astype(np.uint8)
            else:
                image_uint8 = image_np.astype(np.uint8)

            if len(image_uint8.shape) != 3 or image_uint8.shape[-1] != 3:
                if len(image_uint8.shape) == 2:
                    image_uint8 = np.stack([image_uint8] * 3, axis=-1)
                elif len(image_uint8.shape) == 3 and image_uint8.shape[-1] == 1:
                    image_uint8 = np.concatenate([image_uint8] * 3, axis=-1)
                else:
                    logger.warning(f"Unsupported image shape for landmark detection: {image_uint8.shape}")
                    return []

            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_uint8)
            detection_result: Optional[FaceLandmarkerResult] = self.landmarker.detect(mp_image)

            if not detection_result or not detection_result.face_landmarks or not detection_result.face_landmarks[0]:
                logger.warning("No face landmarks detected.")
                return []

            landmarks = detection_result.face_landmarks[0]
            if len(landmarks) != EXPECTED_LANDMARKS:
                logger.warning(f"Unexpected number of landmarks detected: {len(landmarks)}, expected {EXPECTED_LANDMARKS}.")

            img_size_h, img_size_w = image_np.shape[:2]

            try:
                pixel_coords = [(int(pt.x * img_size_w), int(pt.y * img_size_h)) for pt in landmarks]
                return pixel_coords
            except (AttributeError, TypeError, Exception) as e_coord:
                logger.error(f"Error converting landmark coordinates: {e_coord}")
                return []

        except Exception as e_detect:
            logger.error(f"Error during landmark detection: {e_detect}")
            return []

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

        num_landmarks = len(pixel_coords)
        if any(not (0 <= i < num_landmarks) for i in indices):
            logger.warning(f"Invalid indices {indices} for feature {feature.name} with {num_landmarks} landmarks.")
            return None

        try:
            points = [pixel_coords[i] for i in indices]
            min_x = min(x for x, y in points)
            min_y = min(y for x, y in points)
            max_x = max(x for x, y in points)
            max_y = max(y for x, y in points)
        except (IndexError, ValueError, Exception) as e:
            logger.error(f"Error calculating bbox for feature {feature.name}: {e}")
            return None

        pad = self.settings.analysis.mask_pixel_padding
        min_x_pad = max(0, min_x - pad)
        min_y_pad = max(0, min_y - pad)
        max_x_pad = min(img_width - 1, max_x + pad)
        max_y_pad = min(img_height - 1, max_y + pad)

        if min_x_pad >= max_x_pad or min_y_pad >= max_y_pad:
            logger.warning(f"Invalid bbox calculated for {feature.name}: ({min_x_pad}, {min_y_pad}, {max_x_pad}, {max_y_pad}).")
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

    def apply_mask(self, image_np: np.ndarray) -> np.ndarray:
        pixel_coords = self._get_pixel_landmarks(image_np)
        if not pixel_coords:
            logger.warning("No landmark coordinates available for masking.")
            return image_np

        masked_image = image_np.copy()
        img_height, img_width = image_np.shape[:2]

        if not self.settings.analysis.mask_features:
            return masked_image

        for feature_enum in self.settings.analysis.mask_features:
            feature_details = self._get_feature_bbox(pixel_coords, feature_enum, img_height, img_width)
            if feature_details:
                try:
                    masked_image[
                        feature_details.min_y : feature_details.max_y,
                        feature_details.min_x : feature_details.max_x,
                    ] = 0
                except IndexError:
                    logger.warning(f"IndexError applying mask for {feature_enum.name}. Bbox: {feature_details}. Image shape: {masked_image.shape}")
            else:
                logger.warning(f"Could not get bbox for masking feature {feature_enum.name}.")

        return masked_image

    def get_features(self, image_np: np.ndarray) -> List[FeatureDetails]:
        pixel_coords = self._get_pixel_landmarks(image_np)
        if not pixel_coords:
            logger.warning("No landmark coordinates available for feature extraction.")
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
        label: Union[Gender, Race, Age],
    ) -> Optional[np.ndarray]:
        try:
            target_class = lambda output: output[0][label.value]
            image_batch = np.expand_dims(image_np.astype(np.float32), axis=0)
            target_layer_name = next((layer.name for layer in reversed(model.layers) if isinstance(layer, tf.keras.layers.Conv2D)), None)

            if not target_layer_name:
                logger.error("Could not find a Conv2D layer for GradCAM++ target.")
                return None

            heatmap = heatmap_generator(target_class, image_batch, penultimate_layer=target_layer_name)[0]

            if heatmap is None:
                logger.warning(f"GradCAM++ generator returned None heatmap for label {label.name}.")
                return None
            if not isinstance(heatmap, np.ndarray) or heatmap.ndim != 2:
                logger.warning(f"GradCAM++ returned invalid heatmap type/dims: {type(heatmap)}, {heatmap.ndim if isinstance(heatmap, np.ndarray) else 'N/A'}")
                return None

            min_val, max_val = np.min(heatmap), np.max(heatmap)
            epsilon = 1e-8
            if max_val - min_val < epsilon:
                logger.warning(f"Invalid heatmap range (min={min_val}, max={max_val} nearly equal). Returning zero map.")
                return np.zeros_like(heatmap, dtype=np.float32)

            normalized_heatmap = (heatmap - min_val) / (max_val - min_val + epsilon)
            return normalized_heatmap.astype(np.float32)

        except (ValueError, FloatingPointError, tf.errors.InvalidArgumentError, Exception) as e:
            logger.error(f"Heatmap generation via GradCAM++ failed for label {label.name}: {e}")
            return None

    def _calculate_single_feature_attention(self, feature: FeatureDetails, heatmap: np.ndarray) -> float:
        heatmap_height, heatmap_width = heatmap.shape[:2]

        min_y, max_y = int(feature.min_y), int(feature.max_y)
        min_x, max_x = int(feature.min_x), int(feature.max_x)

        min_y = max(0, min_y)
        min_x = max(0, min_x)
        max_y = min(heatmap_height, max_y)
        max_x = min(heatmap_width, max_x)

        if min_y >= max_y or min_x >= max_x:
            logger.warning(f"Invalid feature region after clamping for {feature.feature.name}: box=({min_x}, {min_y}, {max_x}, {max_y}). Returning 0.0.")
            return 0.0

        try:
            feature_attention_region = heatmap[min_y:max_y, min_x:max_x]
            if feature_attention_region.size == 0:
                logger.warning(f"Empty feature region slice for {feature.feature.name}. Returning 0.0.")
                return 0.0
            attention_score = float(np.mean(feature_attention_region))
            if not (0.0 <= attention_score <= 1.0):
                logger.warning(f"Calculated attention score {attention_score} for {feature.feature.name} out of [0, 1] range. Clamping.")
                attention_score = np.clip(attention_score, 0.0, 1.0)
            return attention_score
        except (ValueError, TypeError, Exception) as e:
            logger.error(f"Error calculating mean attention for {feature.feature.name}: {e}")
            return 0.0

    def _save_heatmap(self, heatmap: np.ndarray, image_id: str) -> Optional[str]:
        savepath_dir = os.path.join(self.settings.output.base_path, self.settings.experiment_id, "heatmaps")
        try:
            if not os.path.exists(savepath_dir):
                os.makedirs(savepath_dir, exist_ok=True)
        except OSError as e_dir:
            logger.error(f"Could not create heatmap directory {savepath_dir}: {e_dir}")
            return None

        filename = f"heatmap_{image_id}.npy"
        filepath = os.path.join(savepath_dir, filename)

        try:
            np.save(filepath, heatmap.astype(np.float16))
            return filepath
        except (IOError, OSError, Exception) as e_save:
            logger.error(f"Failed to save heatmap to {filepath}: {e_save}")
            return None

    def _compute_feature_details(
        self,
        features: List[FeatureDetails],
        heatmap: Optional[np.ndarray],
    ) -> List[FeatureDetails]:
        if heatmap is None:
            logger.warning("Heatmap is None, cannot compute feature attention scores.")
            for f in features:
                f.attention_score = 0.0
                f.is_key_feature = False
            return features

        for feature_detail in features:
            attention_score = self._calculate_single_feature_attention(feature_detail, heatmap)
            is_key = attention_score >= self.settings.analysis.key_feature_threshold
            feature_detail.attention_score = attention_score
            feature_detail.is_key_feature = bool(is_key)
        return features

    def get_heatmap_generator(self, model: tf.keras.Model) -> GradcamPlusPlus:
        try:
            replace_to_linear = lambda m: setattr(m.layers[-1], "activation", tf.keras.activations.linear)
            generator = GradcamPlusPlus(model, model_modifier=replace_to_linear, clone=True)
            logger.info("GradCAM++ generator created.")
            return generator
        except Exception as e:
            logger.error(f"Failed to create GradCAM++ generator: {e}")
            raise RuntimeError(f"Failed to create GradCAM++ generator: {e}") from e

    def generate_explanation(
        self,
        heatmap_generator: GradcamPlusPlus,
        model: tf.keras.Model,
        image_np: np.ndarray,
        label: Union[Gender, Race, Age],
        image_id: str,
    ) -> Tuple[List[FeatureDetails], Optional[str]]:
        heatmap = self._calculate_heatmap(heatmap_generator, model, image_np, label)
        heatmap_path = self._save_heatmap(heatmap, image_id) if heatmap is not None else None

        detected_features = self.get_features(image_np)
        feature_details_with_attention = self._compute_feature_details(detected_features, heatmap)

        return feature_details_with_attention, heatmap_path
