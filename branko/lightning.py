from __future__ import annotations

from pathlib import Path

import lightning as pl
import torch
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR, SequentialLR

from .model import BidirectionalRNAModel
from .utils import load_model_bundle, load_model_config, load_model_state


def load_checkpoint_state(checkpoint_path: str | Path) -> dict[str, torch.Tensor]:
    return load_model_state(checkpoint_path, map_location="cpu")


class BidirectionalRNAWrapper(pl.LightningModule):
    def __init__(self, config: dict) -> None:
        super().__init__()
        self.config = config
        self.model = BidirectionalRNAModel(config)
        self.pad_token_id = self.model.pad_token_id
        self.vocab_size = self.model.vocab_size
        self.loss_fn = torch.nn.CrossEntropyLoss(ignore_index=self.pad_token_id, reduction="none")

    def forward(
        self,
        tokens: torch.Tensor,
        shared_attention_mask: torch.Tensor,
        left_query_index: int | None = None,
        right_query_index: int | None = None,
    ) -> dict[str, torch.Tensor]:
        return self.model(
            tokens=tokens,
            shared_attention_mask=shared_attention_mask,
            left_query_index=left_query_index,
            right_query_index=right_query_index,
        )

    def initialize_from_checkpoint(self, checkpoint_path: str | Path, strict: bool = True) -> None:
        state_dict = load_checkpoint_state(checkpoint_path)
        self.model.load_state_dict(state_dict, strict=strict)

    @classmethod
    def from_pretrained(
        cls,
        model_path: str | Path,
        device: str | torch.device | None = None,
        strict: bool = True,
    ) -> "BidirectionalRNAWrapper":
        config, state_dict, _ = load_model_bundle(model_path, map_location="cpu")
        module = cls(config)
        module.model.load_state_dict(state_dict, strict=strict)
        if device is not None:
            module = module.to(device)
        module.eval()
        return module

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        device: str | torch.device | None = None,
        strict: bool = True,
        config_path: str | Path | None = None,
    ) -> "BidirectionalRNAWrapper":
        config = load_model_config(checkpoint_path, config_path=config_path, map_location="cpu")
        state_dict = load_model_state(checkpoint_path, map_location="cpu")
        module = cls(config)
        module.model.load_state_dict(state_dict, strict=strict)
        if device is not None:
            module = module.to(device)
        module.eval()
        return module

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        return self.shared_step(batch, stage="train")

    def validation_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        return self.shared_step(batch, stage="val")

    def configure_optimizers(self) -> dict:
        training = self.config["training"]
        optimizer = AdamW(
            self.parameters(),
            lr=float(training["learning_rate"]),
            weight_decay=float(training.get("weight_decay", 0.01)),
            betas=tuple(training.get("betas", [0.9, 0.999])),
        )

        warmup_steps = int(training.get("warmup_steps", 0))
        cosine_steps = int(training.get("cosine_steps", 0))
        min_learning_rate = float(training.get("min_learning_rate", training["learning_rate"]))
        start_learning_rate = float(training["learning_rate"])

        if warmup_steps == 0 and cosine_steps == 0:
            return {"optimizer": optimizer}

        schedulers = []
        milestones = []

        if warmup_steps > 0:
            schedulers.append(
                LambdaLR(
                    optimizer,
                    lr_lambda=lambda step: min(1.0, (step + 1) / (warmup_steps + 1)),
                )
            )
            milestones.append(warmup_steps)

        if cosine_steps > 0:
            schedulers.append(
                CosineAnnealingLR(
                    optimizer,
                    T_max=cosine_steps,
                    eta_min=min_learning_rate,
                )
            )
            milestones.append(warmup_steps + cosine_steps)

        schedulers.append(
            LambdaLR(
                optimizer,
                lr_lambda=lambda step: min_learning_rate / start_learning_rate,
            )
        )

        scheduler = SequentialLR(
            optimizer=optimizer,
            schedulers=schedulers,
            milestones=milestones,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }

    def shared_step(self, batch: dict, stage: str) -> torch.Tensor:
        sequences = batch["sequences"]
        merge_index = batch["merge_index"].to(sequences.device)
        shared_attention_mask = batch["shared_attention_mask"].to(sequences.device)
        batch_size, seq_len = sequences.shape

        logits = self(tokens=sequences, shared_attention_mask=shared_attention_mask)
        left_logits = logits["left"]
        right_logits = logits["right"]

        left_targets = torch.full_like(sequences, self.pad_token_id)
        left_targets[:, :-1] = sequences[:, 1:]

        right_targets = torch.full_like(sequences, self.pad_token_id)
        right_targets[:, 1:] = sequences[:, :-1]

        left_loss = self.loss_fn(
            left_logits.view(-1, self.vocab_size),
            left_targets.reshape(-1),
        ).view(batch_size, seq_len)
        right_loss = self.loss_fn(
            right_logits.view(-1, self.vocab_size),
            right_targets.reshape(-1),
        ).view(batch_size, seq_len)

        positions = torch.arange(seq_len, device=sequences.device).view(1, seq_len)
        left_mask = (positions < merge_index.view(batch_size, 1)) & left_targets.ne(self.pad_token_id)
        right_mask = (positions >= merge_index.view(batch_size, 1)) & right_targets.ne(self.pad_token_id)

        total_loss = (left_loss * left_mask).sum() + (right_loss * right_mask).sum()
        total_tokens = left_mask.sum() + right_mask.sum()
        loss = total_loss / total_tokens.clamp_min(1)

        left_predictions = left_logits.argmax(dim=-1)
        right_predictions = right_logits.argmax(dim=-1)
        correct = ((left_predictions == left_targets) & left_mask).sum() + (
            (right_predictions == right_targets) & right_mask
        ).sum()
        accuracy = correct.float() / total_tokens.clamp_min(1).float()
        perplexity = torch.exp(loss)

        metrics = {
            f"{stage}/loss": loss,
            f"{stage}/accuracy": accuracy,
            f"{stage}/perplexity": perplexity,
            f"{stage}/merge_index": merge_index.float().mean(),
        }

        if stage == "train":
            self.log_dict(
                {f"{name}_step": value for name, value in metrics.items()},
                batch_size=batch_size,
                on_step=True,
                on_epoch=False,
                sync_dist=True,
            )
            self.log_dict(
                {f"{name}_epoch": value for name, value in metrics.items()},
                batch_size=batch_size,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            )
        else:
            self.log_dict(
                metrics,
                batch_size=batch_size,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            )

        return loss


def build_trainer(config: dict, output_dir: str | Path) -> pl.Trainer:
    output_dir = Path(output_dir)
    trainer_config = config["trainer"]

    callbacks = [
        ModelCheckpoint(
            dirpath=output_dir,
            filename="branko-epoch{epoch:02d}-step{step}",
            every_n_epochs=1,
            save_last=True,
            save_top_k=-1,
        ),
        LearningRateMonitor(logging_interval="step"),
    ]

    checkpoint_every_n_steps = int(trainer_config.get("checkpoint_every_n_steps", 0) or 0)
    if checkpoint_every_n_steps > 0:
        callbacks.append(
            ModelCheckpoint(
                dirpath=output_dir,
                filename="branko-step-epoch{epoch:02d}-step{step}",
                every_n_train_steps=checkpoint_every_n_steps,
                save_top_k=-1,
            )
        )

    logger = CSVLogger(save_dir=str(output_dir), name="logs")

    return pl.Trainer(
        default_root_dir=str(output_dir),
        accelerator=trainer_config.get("accelerator", "auto"),
        devices=trainer_config.get("devices", 1),
        precision=trainer_config.get("precision", "32-true"),
        max_epochs=int(trainer_config.get("max_epochs", 1)),
        max_steps=int(trainer_config.get("max_steps", -1)),
        accumulate_grad_batches=int(trainer_config.get("accumulate_grad_batches", 1)),
        log_every_n_steps=int(trainer_config.get("log_every_n_steps", 50)),
        gradient_clip_val=float(trainer_config.get("gradient_clip_val", 1.0)),
        check_val_every_n_epoch=int(trainer_config.get("check_val_every_n_epoch", 1)),
        callbacks=callbacks,
        logger=logger,
    )
