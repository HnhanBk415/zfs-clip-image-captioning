"""Frozen CLIP image encoding and optional image-ID feature lookup."""

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


class ImageEncoder:
    def __init__(self, models, config):
        self.models = models
        self.config = config
        self.device = models.device
        self.feature_cache = None
        self.feature_index = None
        self._load_optional_cache()

    @staticmethod
    def _feature_tensor(value):
        if torch.is_tensor(value):
            return value
        if hasattr(value, "pooler_output"):
            return value.pooler_output
        if isinstance(value, (tuple, list)) and value:
            return value[0]
        raise TypeError(f"Unsupported CLIP feature output: {type(value)}")

    def _load_optional_cache(self):
        if self.config.feature_cache_path is None:
            print("CLIP feature cache: disabled; raw images will be encoded.")
            return

        cache_path = Path(self.config.feature_cache_path)
        if not cache_path.is_file():
            raise FileNotFoundError(
                f"Configured feature cache does not exist: {cache_path}"
            )

        payload = torch.load(
            cache_path,
            map_location="cpu",
            weights_only=False,
        )
        if not isinstance(payload, dict):
            raise RuntimeError("Feature cache must be a dictionary.")

        if "image_ids" in payload and "features" in payload:
            image_ids = list(payload["image_ids"])
            features = torch.as_tensor(payload["features"])
            if len(image_ids) != features.shape[0]:
                raise RuntimeError("Cache image_ids/features length mismatch.")
            if len(image_ids) != len(set(image_ids)):
                raise RuntimeError("Feature cache contains duplicate image IDs.")
            cache_model = payload.get("clip_model")
            if cache_model is not None and cache_model != self.config.clip_model:
                raise RuntimeError(
                    f"Cache CLIP model {cache_model!r} does not match "
                    f"{self.config.clip_model!r}."
                )
            self.feature_cache = features
            self.feature_index = {
                image_id: row_index
                for row_index, image_id in enumerate(image_ids)
            }
        else:
            tensor_mapping = {
                str(image_id): torch.as_tensor(feature)
                for image_id, feature in payload.items()
                if torch.is_tensor(feature)
                or isinstance(feature, (list, tuple, np.ndarray))
            }
            if not tensor_mapping:
                raise RuntimeError(
                    "Unsupported cache schema. Expected image_ids + features "
                    "or an image_id -> feature mapping."
                )
            self.feature_cache = tensor_mapping
            self.feature_index = None

        print("CLIP feature cache loaded:", cache_path)

    def _cached_feature(self, image_id):
        if self.feature_cache is None:
            return None
        if self.feature_index is not None:
            if image_id not in self.feature_index:
                raise KeyError(f"Image ID is absent from feature cache: {image_id}")
            row_index = self.feature_index[image_id]
            feature = self.feature_cache[row_index]
        else:
            if image_id not in self.feature_cache:
                raise KeyError(f"Image ID is absent from feature cache: {image_id}")
            feature = self.feature_cache[image_id]
        return torch.as_tensor(feature)

    def encode(self, image, image_id):
        cached = self._cached_feature(image_id)
        if cached is None:
            inputs = self.models.clip_processor(
                images=image,
                return_tensors="pt",
            )
            pixel_values = inputs["pixel_values"].to(
                device=self.device,
                dtype=self.models.dtype,
            )
            with torch.no_grad():
                raw_feature = self.models.clip_model.get_image_features(
                    pixel_values=pixel_values
                )
                feature = self._feature_tensor(raw_feature)
        else:
            feature = cached.to(
                device=self.device,
                dtype=self.models.dtype,
            )
            if feature.ndim == 1:
                feature = feature.unsqueeze(0)

        if feature.ndim != 2 or feature.shape[0] != 1:
            raise AssertionError(
                f"Image feature must have shape [1, D], got {tuple(feature.shape)}"
            )
        if not torch.isfinite(feature).all():
            raise AssertionError("Image feature contains NaN/Inf.")

        feature = F.normalize(feature, dim=-1).detach()
        norm = feature.norm(dim=-1)
        if not torch.allclose(
            norm,
            torch.ones_like(norm),
            atol=1e-4,
            rtol=1e-4,
        ):
            raise AssertionError(f"Image feature norm is {norm.item()}, expected 1.")

        if self.config.debug:
            print("E_img shape:", tuple(feature.shape))
            print("E_img norm:", norm.item())
        return feature
