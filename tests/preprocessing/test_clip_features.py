"""Tests for CLIP feature preprocessing."""

import importlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

from PIL import Image


fake_import_modules = {
    "kagglehub": ModuleType("kagglehub"),
    "torch": ModuleType("torch"),
    "transformers": ModuleType("transformers"),
}
fake_import_modules["transformers"].CLIPModel = object
fake_import_modules["transformers"].CLIPProcessor = object

with patch.dict(sys.modules, fake_import_modules):
    clip_features_module = importlib.import_module(
        "src.clipcap.preprocessing.clip_features"
    )

_get_feature_tensor = clip_features_module._get_feature_tensor
extract_clip_features = clip_features_module.extract_clip_features
load_split_image_ids = clip_features_module.load_split_image_ids
run_clip_feature_extraction = clip_features_module.run_clip_feature_extraction


class FakeTensor:
    def __init__(self, rows):
        self.rows = [list(row) for row in rows]

    @property
    def shape(self):
        return (len(self.rows), len(self.rows[0]))

    def to(self, _device):
        return self

    def detach(self):
        return self

    def cpu(self):
        return self


class FakeTorch:
    class cuda:
        @staticmethod
        def is_available():
            return False

    class _NoGrad:
        def __enter__(self):
            return None

        def __exit__(self, _exception_type, _exception, _traceback):
            return False

    def __init__(self):
        self.saved_data = None
        self.saved_path = None

    def device(self, name):
        return name

    def no_grad(self):
        return self._NoGrad()

    def cat(self, tensors, dim=0):
        if dim != 0:
            raise AssertionError("FakeTorch only supports concatenation on dim 0")
        rows = [row for tensor in tensors for row in tensor.rows]
        return FakeTensor(rows)

    def isfinite(self, _tensor):
        return SimpleNamespace(all=lambda: True)

    def save(self, data, path):
        self.saved_data = data
        self.saved_path = Path(path)
        self.saved_path.write_bytes(b"fake torch artifact")


class FakeProcessor:
    def __call__(self, images, return_tensors):
        if return_tensors != "pt":
            raise AssertionError("Expected PyTorch processor output")
        rows = [[image.getpixel((0, 0))[0]] for image in images]
        return {"pixel_values": FakeTensor(rows)}


class FakeParameter:
    requires_grad = True


class FakeClipModel:
    def __init__(self):
        self.config = SimpleNamespace(projection_dim=2)
        self.parameter = FakeParameter()
        self.selected_device = None
        self.evaluation_mode = False

    def to(self, device):
        self.selected_device = device
        return self

    def parameters(self):
        return [self.parameter]

    def eval(self):
        self.evaluation_mode = True
        return self

    def get_image_features(self, pixel_values):
        rows = [[row[0], row[0] + 1] for row in pixel_values.rows]
        return SimpleNamespace(pooler_output=FakeTensor(rows))


class ClipFeaturesTest(unittest.TestCase):
    def test_configured_extraction_calls_directory_setup(self):
        expected = {"status": "ok"}
        with (
            patch.object(
                clip_features_module,
                "setup_clipcap_directories",
            ) as setup_directories,
            patch.object(
                clip_features_module,
                "extract_clip_features",
                return_value=expected,
            ),
        ):
            result = run_clip_feature_extraction(
                dataset_path="dataset-cache",
                show_progress=False,
            )

        setup_directories.assert_called_once_with()
        self.assertEqual(result, expected)

    def test_split_ids_are_sorted_and_disjoint(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            split_dir = Path(temporary_directory)
            self._write_json(split_dir / "train.json", {"c.jpg": [], "a.jpg": []})
            self._write_json(split_dir / "val.json", {"b.jpg": []})
            self._write_json(split_dir / "test.json", {"d.jpg": []})

            image_ids = load_split_image_ids(split_dir)

            self.assertEqual(image_ids, ["a.jpg", "b.jpg", "c.jpg", "d.jpg"])

            self._write_json(split_dir / "test.json", {"a.jpg": []})
            with self.assertRaises(ValueError):
                load_split_image_ids(split_dir)

    def test_extraction_keeps_notebook_order_shape_and_schema(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            split_dir = root / "splits"
            images_dir = root / "Images"
            feature_dir = root / "features"
            split_dir.mkdir()
            images_dir.mkdir()

            self._write_json(split_dir / "train.json", {"c.jpg": []})
            self._write_json(split_dir / "val.json", {"a.jpg": []})
            self._write_json(split_dir / "test.json", {"b.jpg": []})
            for image_id, value in (("a.jpg", 10), ("b.jpg", 20), ("c.jpg", 30)):
                Image.new("RGB", (2, 2), color=(value, 0, 0)).save(
                    images_dir / image_id
                )

            fake_torch = FakeTorch()
            fake_model = FakeClipModel()
            with patch.object(clip_features_module, "torch", fake_torch):
                result = extract_clip_features(
                    split_dir=split_dir,
                    images_dir=images_dir,
                    feature_dir=feature_dir,
                    model_name="test-clip",
                    batch_size=2,
                    processor=FakeProcessor(),
                    clip_model=fake_model,
                    show_progress=False,
                )

            self.assertEqual(result["image_ids"], ["a.jpg", "b.jpg", "c.jpg"])
            self.assertEqual(result["features"].rows, [[10, 11], [20, 21], [30, 31]])
            self.assertEqual(result["clip_model"], "test-clip")
            self.assertEqual(result["feature_dim"], 2)
            self.assertEqual(set(result), {"image_ids", "features", "clip_model", "feature_dim"})
            self.assertFalse(fake_model.parameter.requires_grad)
            self.assertTrue(fake_model.evaluation_mode)
            self.assertEqual(fake_model.selected_device, "cpu")
            self.assertEqual(fake_torch.saved_data, result)
            self.assertEqual(fake_torch.saved_path, feature_dir / "clip_features.pt")

            direct_tensor = FakeTensor([[1, 2]])
            self.assertIs(_get_feature_tensor(direct_tensor), direct_tensor)

    @staticmethod
    def _write_json(path, data):
        path.write_text(json.dumps(data), encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
