from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import torch
import yaml


BUNDLE_FORMAT = "branko_pretrained_bundle"
BUNDLE_VERSION = 1


def load_yaml(path: str | Path) -> dict:
    file_path = Path(path)
    with file_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def save_yaml(data: dict, path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False)


def load_config(path: str | Path) -> dict:
    return normalize_config(load_yaml(path))


def save_config(config: dict, path: str | Path) -> None:
    save_yaml(normalize_config(config), path)


def ensure_dir(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def normalize_merge_index_name(value: str | None, default: str) -> str:
    normalized = str(value or default).strip().lower()
    if normalized in {"uniform", "random"}:
        return "uniform"
    if normalized in {"half", "center", "center_window", "l/2", "midpoint"}:
        return "half"
    raise ValueError(f"Unsupported merge-index strategy '{value}'. Use 'half' or 'uniform'.")


def normalize_data_config(data_config: dict) -> dict:
    if not isinstance(data_config, dict):
        raise ValueError("Expected 'data' to be a dictionary-like config section.")

    return {
        "train_path": data_config.get("train_path"),
        "val_path": data_config.get("val_path"),
        "batch_size": int(data_config.get("batch_size", 1)),
        "num_workers": int(data_config.get("num_workers", 0)),
        "train_merge_index": normalize_merge_index_name(
            data_config.get("train_merge_index"),
            default="uniform",
        ),
        "val_merge_index": normalize_merge_index_name(
            data_config.get("val_merge_index"),
            default="half",
        ),
    }


def normalize_config(config: dict) -> dict:
    """Normalize the public YAML config layout used by this repo."""

    if not isinstance(config, dict):
        raise ValueError("Expected a dictionary-like config.")

    normalized = dict(config)
    if "data" in normalized:
        normalized["data"] = normalize_data_config(normalized["data"])
    return normalized


def resolve_architecture_config(
    config: dict,
    map_location: str | torch.device = "cpu",
) -> dict:
    """Fill in architecture sections from an init_model bundle when needed."""

    normalized = normalize_config(config)
    if "model" in normalized:
        return normalized

    init_model = normalized.get("init_model")
    if not init_model:
        raise ValueError(
            "Config is missing a 'model' section. "
            "Provide either 'model' directly or 'init_model' pointing to a bundled B-RANKO model file."
        )

    try:
        bundled_config, _, _ = load_model_bundle(init_model, map_location=map_location)
    except ValueError as exc:
        raise ValueError(
            "Config is missing a 'model' section and the 'init_model' file does not contain one. "
            "Lightning checkpoints can initialize weights, but the config must define the model architecture."
        ) from exc

    resolved = dict(normalized)
    resolved["model"] = bundled_config["model"]
    if "inference" not in resolved and "inference" in bundled_config:
        resolved["inference"] = bundled_config["inference"]
    return normalize_config(resolved)


def build_release_config(config: dict) -> dict:
    """Keep only the config sections needed by bundled release checkpoints."""

    normalized = normalize_config(config)
    if "model" not in normalized or not isinstance(normalized["model"], dict):
        raise ValueError("Expected config to contain a dictionary-like 'model' section.")

    release_config = {"model": dict(normalized["model"])}
    if "inference" in normalized:
        inference_config = normalized["inference"]
        if not isinstance(inference_config, dict):
            raise ValueError("Expected 'inference' to be a dictionary-like config section.")
        release_config["inference"] = dict(inference_config)

    return release_config


def clean_state_dict(raw_state: Any) -> dict[str, torch.Tensor]:
    state_dict = raw_state.get("state_dict", raw_state) if isinstance(raw_state, dict) else raw_state
    if not isinstance(state_dict, dict):
        raise ValueError("Expected a checkpoint with a state dict.")

    cleaned_state = {}
    for key, value in state_dict.items():
        if key.startswith("model."):
            cleaned_state[key[len("model."):]] = value
        elif key.startswith("module.model."):
            cleaned_state[key[len("module.model."):]] = value
        elif key.startswith("module."):
            cleaned_state[key[len("module."):]] = value
        else:
            cleaned_state[key] = value
    return cleaned_state


def load_model_state(model_path: str | Path, map_location: str | torch.device = "cpu") -> dict[str, torch.Tensor]:
    raw_state = load_torch_file(model_path, map_location=map_location)
    if isinstance(raw_state, dict) and raw_state.get("format") == BUNDLE_FORMAT:
        return clean_state_dict(raw_state["state_dict"])
    return clean_state_dict(raw_state)


def load_model_bundle(
    model_path: str | Path,
    map_location: str | torch.device = "cpu",
) -> tuple[dict, dict[str, torch.Tensor], dict]:
    raw_bundle = load_torch_file(model_path, map_location=map_location)
    if not isinstance(raw_bundle, dict):
        raise ValueError("Expected a dictionary-like bundle.")
    if raw_bundle.get("format") != BUNDLE_FORMAT:
        raise ValueError(
            f"File '{model_path}' is not a bundled B-RANKO model file."
        )
    if "config" not in raw_bundle or "state_dict" not in raw_bundle:
        raise ValueError("Bundled model is missing 'config' or 'state_dict'.")

    config = normalize_config(raw_bundle["config"])
    state_dict = clean_state_dict(raw_bundle["state_dict"])
    metadata = dict(raw_bundle.get("metadata", {}))
    return config, state_dict, metadata


def load_model_config(
    model_path: str | Path,
    config_path: str | Path | None = None,
    map_location: str | torch.device = "cpu",
) -> dict:
    model_path = Path(model_path)

    try:
        config, _, _ = load_model_bundle(model_path, map_location=map_location)
        return config
    except ValueError:
        pass

    resolved_config_path = Path(config_path) if config_path is not None else model_path.with_name("config.yaml")
    if not resolved_config_path.exists():
        raise ValueError(
            f"File '{model_path}' is not a bundled B-RANKO model file and no config was found. "
            "For raw Lightning checkpoints, pass --config or keep config.yaml next to the checkpoint."
        )

    return resolve_architecture_config(load_config(resolved_config_path), map_location=map_location)


def save_model_bundle(
    output_path: str | Path,
    config: dict,
    state_dict: dict[str, torch.Tensor],
    metadata: dict | None = None,
) -> Path:
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    bundle = {
        "format": BUNDLE_FORMAT,
        "version": BUNDLE_VERSION,
        "config": build_release_config(config),
        "state_dict": state_dict,
        "metadata": metadata or {},
    }
    torch.save(bundle, destination)
    return destination


def merge_state_dicts(*state_dicts: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    merged = {}
    for state_dict in state_dicts:
        merged.update(state_dict)
    return merged


def load_torch_file(model_path: str | Path, map_location: str | torch.device = "cpu") -> Any:
    try:
        return torch.load(model_path, map_location=map_location, weights_only=True)
    except pickle.UnpicklingError:
        pass
    return torch.load(model_path, map_location=map_location, weights_only=False)
