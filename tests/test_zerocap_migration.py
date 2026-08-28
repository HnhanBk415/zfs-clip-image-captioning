"""Static regression checks for the notebook-to-module ZeroCap migration."""

import ast
import builtins
import json
import symtable
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = (
    PROJECT_ROOT
    / "notebook"
    / "zerocap"
    / "zerocap_gpt2_base_colab.ipynb"
)

MIGRATED_CLASSES = {
    "ZeroCapRunConfig": (
        8,
        PROJECT_ROOT / "src" / "config" / "zerocap_config.py",
    ),
    "GenerationResult": (12, PROJECT_ROOT / "src" / "zerocap" / "types.py"),
    "ContextStepResult": (12, PROJECT_ROOT / "src" / "zerocap" / "types.py"),
    "BeamState": (12, PROJECT_ROOT / "src" / "zerocap" / "types.py"),
    "GenerationTrace": (12, PROJECT_ROOT / "src" / "zerocap" / "types.py"),
    "Flickr8kData": (10, PROJECT_ROOT / "src" / "zerocap" / "data.py"),
    "PredictionStore": (14, PROJECT_ROOT / "src" / "zerocap" / "storage.py"),
    "ZeroCapModels": (16, PROJECT_ROOT / "src" / "zerocap" / "model_loader.py"),
    "ImageEncoder": (18, PROJECT_ROOT / "src" / "zerocap" / "image_encoder.py"),
    "CLIPGuidance": (20, PROJECT_ROOT / "src" / "zerocap" / "clip_guidance.py"),
    "ContextOptimizer": (
        22,
        PROJECT_ROOT / "src" / "zerocap" / "context_optimizer.py",
    ),
    "ZeroCapDecoder": (24, PROJECT_ROOT / "src" / "zerocap" / "decoding.py"),
    "ZeroCapGenerator": (26, PROJECT_ROOT / "src" / "zerocap" / "generator.py"),
    "ZeroCapCaptioner": (28, PROJECT_ROOT / "src" / "zerocap" / "captioner.py"),
}

MIGRATED_MODULES = sorted(
    {
        module_path
        for _, module_path in MIGRATED_CLASSES.values()
    }
    | {
        PROJECT_ROOT / "src" / "zerocap" / "data.py",
        PROJECT_ROOT / "src" / "zerocap" / "experiment.py",
        PROJECT_ROOT / "scripts" / "run_zerocap.py",
    }
)


def _class_node(source, class_name):
    matches = [
        node
        for node in ast.parse(source).body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    ]
    assert len(matches) == 1, f"Expected one {class_name}, found {len(matches)}"
    return matches[0]


def test_migrated_class_bodies_match_verified_notebook():
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    for class_name, (cell_index, module_path) in MIGRATED_CLASSES.items():
        notebook_source = "".join(notebook["cells"][cell_index]["source"])
        module_source = module_path.read_text(encoding="utf-8")
        notebook_node = _class_node(notebook_source, class_name)
        module_node = _class_node(module_source, class_name)
        assert ast.dump(module_node, include_attributes=False) == ast.dump(
            notebook_node,
            include_attributes=False,
        ), f"{class_name} diverged from the verified notebook"


def test_locked_baseline_defaults_are_present():
    module_path = PROJECT_ROOT / "src" / "config" / "zerocap_config.py"
    config_class = _class_node(
        module_path.read_text(encoding="utf-8"),
        "ZeroCapRunConfig",
    )
    defaults = {}
    for node in config_class.body:
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            if isinstance(node.value, ast.Constant):
                defaults[node.target.id] = node.value.value

    assert defaults["gpt_model"] == "openai-community/gpt2"
    assert defaults["clip_model"] == "openai/clip-vit-base-patch32"
    assert defaults["prompt"] == "Image of a"
    assert defaults["prepend_bos_token"] is True
    assert defaults["top_k"] == 512
    assert defaults["inner_iterations"] == 5
    assert defaults["beam_size"] == 5
    assert defaults["max_new_tokens"] == 15
    assert defaults["clip_temperature"] == 0.01
    assert defaults["clip_loss_scale"] == 1.0
    assert defaults["fluency_weight"] == 0.2
    assert defaults["step_size"] == 0.3
    assert defaults["grad_norm_factor"] == 0.9
    assert defaults["fusion_factor"] == 0.99
    assert defaults["final_rerank_mode"] == "clip_only"
    assert defaults["final_rerank_text_mode"] == "caption_only"


def test_migrated_modules_have_no_unbound_global_names():
    builtin_names = set(dir(builtins)) | {"__file__", "__name__"}

    def walk_tables(table):
        yield table
        for child in table.get_children():
            yield from walk_tables(child)

    for module_path in MIGRATED_MODULES:
        source = module_path.read_text(encoding="utf-8")
        module_table = symtable.symtable(source, str(module_path), "exec")
        bound_at_module = {
            symbol.get_name()
            for symbol in module_table.get_symbols()
            if symbol.is_assigned() or symbol.is_imported()
        }
        referenced_globals = {
            symbol.get_name()
            for table in walk_tables(module_table)
            for symbol in table.get_symbols()
            if symbol.is_referenced() and symbol.is_global()
        }
        missing = referenced_globals - bound_at_module - builtin_names
        assert not missing, f"{module_path.name} has unbound globals: {sorted(missing)}"
