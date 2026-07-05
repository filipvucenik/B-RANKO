#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import lightning.pytorch as pl
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from branko import (
    BidirectionalRNAWrapper,
    SequenceDataModule,
    ensure_dir,
    load_config,
    load_model_state,
    resolve_architecture_config,
    save_config,
    save_model_bundle,
)
from branko.lightning import build_trainer


def resolve_training_config(config: dict) -> dict:
    """Fill in architecture sections from a bundled model file when needed."""
    return resolve_architecture_config(config)


def build_data_module(config: dict) -> SequenceDataModule:
    data_config = config["data"]
    return SequenceDataModule(
        train_path=data_config["train_path"],
        val_path=data_config["val_path"],
        batch_size=int(data_config["batch_size"]),
        num_workers=int(data_config["num_workers"]),
        train_merge_index=data_config.get("train_merge_index", "uniform"),
        val_merge_index=data_config.get("val_merge_index", "half"),
    )


def initialize_model(model: BidirectionalRNAWrapper, config: dict) -> None:
    init_model = config.get("init_model")
    init_checkpoint = config.get("init_checkpoint")
    if init_model and init_checkpoint:
        raise ValueError("Config can specify only one of 'init_model' or 'init_checkpoint'.")

    init_path = init_model or init_checkpoint
    if init_path:
        state_dict = load_model_state(init_path, map_location="cpu")
        model.model.load_state_dict(state_dict, strict=bool(config.get("strict_init", True)))


def export_final_model(model: BidirectionalRNAWrapper, config: dict, output_dir: Path) -> Path:
    export_name = config.get("export_model_name", "branko_final.ckpt")
    export_path = output_dir / export_name
    metadata = {
        "source": "scripts/train.py",
        "stage": "final",
    }
    return save_model_bundle(
        output_path=export_path,
        config=config,
        state_dict=model.model.state_dict(),
        metadata=metadata,
    )


def run_training(config: dict) -> None:
    config = resolve_training_config(config)
    if config.get("seed") is not None:
        pl.seed_everything(int(config["seed"]))

    torch.set_float32_matmul_precision("high")

    output_dir = ensure_dir(config["output_dir"])
    save_config(config, output_dir / "config.yaml")

    data_module = build_data_module(config)
    model = BidirectionalRNAWrapper(config)
    initialize_model(model, config)

    trainer = build_trainer(config, output_dir)
    trainer.fit(model=model, datamodule=data_module, ckpt_path=config.get("resume_from_checkpoint"))

    if config.get("export_model", True):
        export_path = export_final_model(model, config, output_dir)
        print(f"Saved bundled model to {export_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train or fine-tune B-RANKO from a YAML config.")
    parser.add_argument("--config", required=True, help="Path to a YAML config file.")
    args = parser.parse_args()
    run_training(load_config(args.config))


if __name__ == "__main__":
    main()
