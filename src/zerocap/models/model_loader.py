"""Frozen GPT-2/CLIP loader and official forbidden-token artifact."""

import hashlib
import os
import urllib.request
from pathlib import Path

import numpy as np
import torch
from transformers import CLIPModel, CLIPProcessor, GPT2LMHeadModel, GPT2TokenizerFast

from ..runtime.storage import PredictionStore


class ZeroCapModels:
    def __init__(self, config):
        self.config = config
        requested_device = torch.device(config.device)
        self.device = (
            torch.device("cuda", torch.cuda.current_device())
            if requested_device.type == "cuda"
            else requested_device
        )
        self.dtype = getattr(torch, config.model_dtype)

        self.gpt_tokenizer = GPT2TokenizerFast.from_pretrained(
            config.gpt_model
        )
        if self.gpt_tokenizer.eos_token_id is None:
            raise RuntimeError("GPT-2 tokenizer has no EOS token.")
        self.gpt_tokenizer.pad_token = self.gpt_tokenizer.eos_token

        self.gpt_model = GPT2LMHeadModel.from_pretrained(
            config.gpt_model,
            dtype=self.dtype,
            attn_implementation="eager",
        ).to(self.device)
        self.clip_processor = CLIPProcessor.from_pretrained(
    config.clip_model,
    use_fast=False,
)
        self.clip_model = CLIPModel.from_pretrained(
            config.clip_model,
            dtype=self.dtype,
            attn_implementation="eager",
        ).to(self.device)

        self.gpt_model.config.use_cache = True
        self.gpt_model.eval()
        self.clip_model.eval()
        self.gpt_model.requires_grad_(False)
        self.clip_model.requires_grad_(False)

        self._assert_models()
        self.forbidden_token_ids = self._load_forbidden_tokens()

    def _assert_models(self):
        if self.gpt_model.name_or_path != self.config.gpt_model:
            raise AssertionError("Loaded GPT-2 model ID does not match config.")
        if self.clip_model.name_or_path != self.config.clip_model:
            raise AssertionError("Loaded CLIP model ID does not match config.")
        if self.gpt_model.training or self.clip_model.training:
            raise AssertionError("Both models must be in eval mode.")
        if any(parameter.requires_grad for parameter in self.gpt_model.parameters()):
            raise AssertionError("GPT-2 is not fully frozen.")
        if any(parameter.requires_grad for parameter in self.clip_model.parameters()):
            raise AssertionError("CLIP is not fully frozen.")
        if next(self.gpt_model.parameters()).device != self.device:
            raise AssertionError("GPT-2 is on the wrong device.")
        if next(self.clip_model.parameters()).device != self.device:
            raise AssertionError("CLIP is on the wrong device.")
        if len(self.gpt_tokenizer) != 50257:
            raise AssertionError(
                "Official forbidden token IDs require the standard GPT-2 vocab "
                "of 50,257 entries."
            )

    def _validate_forbidden_array(self, path):
        path = Path(path)
        with path.open("rb") as handle:
            if handle.read(6) != b"\x93NUMPY":
                raise RuntimeError("Forbidden-token file is not a valid NPY artifact.")
        values = np.load(path, allow_pickle=False)
        if values.ndim != 1:
            raise RuntimeError("Forbidden-token array must be one-dimensional.")
        if not np.issubdtype(values.dtype, np.integer):
            raise RuntimeError("Forbidden-token array must contain integer IDs.")
        values = values.astype(np.int64, copy=False)
        if values.size == 0:
            raise RuntimeError("Forbidden-token array is empty.")
        if np.unique(values).size != values.size:
            raise RuntimeError("Forbidden-token array contains duplicate IDs.")
        if values.min() < 0 or values.max() >= len(self.gpt_tokenizer):
            raise RuntimeError(
                "Forbidden-token IDs are incompatible with the GPT-2 tokenizer."
            )
        return values

    def _load_forbidden_tokens(self):
        artifact_dir = Path(self.config.run_dir) / "artifacts"
        artifact_dir.mkdir(parents=True, exist_ok=True)
        artifact_path = artifact_dir / "forbidden_tokens.npy"
        metadata_path = artifact_dir / "forbidden_tokens_metadata.json"

        if not artifact_path.exists():
            temporary = artifact_path.with_suffix(".npy.part")
            request = urllib.request.Request(
                self.config.forbidden_tokens_url,
                headers={"User-Agent": "ZeroCap-Colab-Prototype/1.0"},
            )
            try:
                with urllib.request.urlopen(request, timeout=60) as response:
                    payload = response.read()
                with temporary.open("wb") as handle:
                    handle.write(payload)
                os.replace(temporary, artifact_path)
            except Exception as exc:
                if temporary.exists():
                    temporary.unlink()
                raise RuntimeError(
                    "Failed to download official ZeroCap forbidden_tokens.npy. "
                    "Generation is stopped; suppression will not be skipped."
                ) from exc

        try:
            values = self._validate_forbidden_array(artifact_path)
        except Exception as exc:
            raise RuntimeError(
                f"Cached forbidden-token artifact is invalid: {artifact_path}. "
                "Remove this run directory and Run all again."
            ) from exc

        sha256 = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
        metadata = {
            "source_url": self.config.forbidden_tokens_url,
            "sha256": sha256,
            "count": int(values.size),
            "tokenizer": self.config.gpt_model,
            "vocab_size": len(self.gpt_tokenizer),
        }
        PredictionStore._atomic_json(metadata_path, metadata)
        print(
            "Forbidden tokens:",
            int(values.size),
            "SHA-256:",
            sha256,
        )
        return torch.as_tensor(
            values,
            dtype=torch.long,
            device=self.device,
        )
