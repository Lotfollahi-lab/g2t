"""
Training loop for ScGG.

Handles section-based training, learning rate scheduling, checkpointing,
and periodic evaluation. Designed for the section-sampling paradigm where
each training step processes cells from one section.
"""

import math
import torch
import torch.nn as nn
import numpy as np
import logging
import time
import json
from pathlib import Path
from typing import Optional, Dict, List, Any

from ..model.scgg import ScGG
from ..data.dataset import (
    SpatialTranscriptomicsDataset,
    create_cell_batches,
)
from .losses import ScGGLoss

logger = logging.getLogger(__name__)


class Trainer:
    """ScGG trainer.

    Args:
        model: ScGG model.
        train_dataset: Training dataset.
        val_dataset: Optional validation dataset.
        config: Full configuration dict.
        device: Torch device.
    """

    def __init__(
        self,
        model: ScGG,
        train_dataset: SpatialTranscriptomicsDataset,
        val_dataset: Optional[SpatialTranscriptomicsDataset],
        config: dict,
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    ):
        self.model = model.to(device)
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.config = config
        self.device = device

        train_cfg = config["training"]

        # Loss
        loss_cfg = train_cfg.get("loss", {})
        self.criterion = ScGGLoss(
            lambda_contrastive=loss_cfg.get("contrastive_spatial", 0.1),
            temperature=loss_cfg.get("contrastive_temperature", 0.1),
            n_negatives=loss_cfg.get("contrastive_n_negatives", 64),
        )

        # Optimizer
        if train_cfg["optimizer"] == "adamw":
            self.optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=train_cfg["lr"],
                weight_decay=train_cfg["weight_decay"],
            )
        else:
            self.optimizer = torch.optim.Adam(
                model.parameters(),
                lr=train_cfg["lr"],
                weight_decay=train_cfg["weight_decay"],
            )

        # Estimate actual training steps per epoch (mini-batches across all sections)
        batch_size = train_cfg["batch_size"]
        steps_per_epoch = 0
        for s in train_dataset.sections:
            n_cells = len(train_dataset.section_indices[s])
            steps_per_epoch += max(1, math.ceil(n_cells / batch_size))

        total_steps = train_cfg["epochs"] * steps_per_epoch
        warmup_steps = train_cfg.get("warmup_epochs", 10) * steps_per_epoch

        logger.info(f"Scheduler: {steps_per_epoch} steps/epoch, "
                     f"{total_steps} total steps, {warmup_steps} warmup steps")

        # Combined warmup + cosine decay via LambdaLR
        if train_cfg["scheduler"] == "cosine":
            def lr_lambda(step):
                if step < warmup_steps:
                    return step / max(warmup_steps, 1)
                progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
                return 0.5 * (1.0 + math.cos(math.pi * progress))

            self.scheduler = torch.optim.lr_scheduler.LambdaLR(
                self.optimizer, lr_lambda=lr_lambda
            )
        else:
            self.scheduler = None

        self.warmup_steps = warmup_steps
        self.grad_clip = train_cfg.get("grad_clip", 1.0)
        self.global_step = 0
        self.best_val_loss = float("inf")

        # Checkpointing
        self.checkpoint_dir = Path(train_cfg.get("checkpoint_dir", "./checkpoints"))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Logging intervals
        self.log_every = train_cfg.get("log_every", 100)
        self.eval_every = train_cfg.get("eval_every", 5)
        self.save_every = train_cfg.get("save_every", 10)

        # Wandb
        self.use_wandb = train_cfg.get("wandb", False)
        self.wandb = None
        if self.use_wandb:
            self._init_wandb(train_cfg, config)

    def _init_wandb(self, train_cfg: dict, config: dict):
        """Initialize Weights & Biases logging."""
        try:
            import wandb

            wandb.init(
                project=train_cfg.get("wandb_project", "scgg"),
                name=train_cfg.get("wandb_run_name", None),
                config=config,
                tags=train_cfg.get("wandb_tags", []),
            )
            # Watch model for gradient and parameter histograms
            wandb.watch(self.model, log="gradients", log_freq=self.log_every)
            self.wandb = wandb
            logger.info(f"Wandb initialized: project={wandb.run.project}, run={wandb.run.name}")
        except ImportError:
            logger.warning("wandb not installed. Install with: pip install wandb")
            self.use_wandb = False

    def _log_wandb(self, metrics: Dict[str, float], prefix: str = "train"):
        """Log metrics to wandb if enabled."""
        if not self.use_wandb or self.wandb is None:
            return
        payload = {f"{prefix}/{k}": v for k, v in metrics.items()}
        payload["global_step"] = self.global_step
        payload["lr"] = self.optimizer.param_groups[0]["lr"]
        self.wandb.log(payload, step=self.global_step)

    def train(self):
        """Run full training loop."""
        train_cfg = self.config["training"]
        n_epochs = train_cfg["epochs"]
        batch_size = train_cfg["batch_size"]

        n_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info(f"Starting training for {n_epochs} epochs")
        logger.info(f"  Sections: {len(self.train_dataset)}")
        logger.info(f"  Batch size: {batch_size} cells")
        logger.info(f"  Device: {self.device}")
        logger.info(f"  Parameters: {n_params:,}")

        epoch_timer = time.time()

        for epoch in range(n_epochs):
            t0 = time.time()
            epoch_metrics = self._train_epoch(epoch, batch_size)
            epoch_time = time.time() - t0

            # Log epoch summary
            logger.info(
                f"Epoch {epoch + 1}/{n_epochs} | "
                f"Loss: {epoch_metrics['total_loss']:.4f} | "
                f"FM: {epoch_metrics['fm_loss']:.4f} | "
                f"LR: {self.optimizer.param_groups[0]['lr']:.2e} | "
                f"Time: {epoch_time:.1f}s"
            )

            # Wandb epoch-level logging
            epoch_metrics["epoch_time_s"] = epoch_time
            epoch_metrics["epoch"] = epoch + 1
            self._log_wandb(epoch_metrics, prefix="train/epoch")

            # Evaluation
            if self.val_dataset is not None and (epoch + 1) % self.eval_every == 0:
                val_metrics = self._validate(batch_size)
                logger.info(
                    f"  Val Loss: {val_metrics['total_loss']:.4f} | "
                    f"Val FM: {val_metrics['fm_loss']:.4f}"
                )

                self._log_wandb(val_metrics, prefix="val")

                if val_metrics["total_loss"] < self.best_val_loss:
                    self.best_val_loss = val_metrics["total_loss"]
                    self._save_checkpoint(epoch, is_best=True)
                    if self.use_wandb and self.wandb is not None:
                        self.wandb.run.summary["best_val_loss"] = self.best_val_loss
                        self.wandb.run.summary["best_epoch"] = epoch + 1

            # Save checkpoint
            if (epoch + 1) % self.save_every == 0:
                self._save_checkpoint(epoch)

        total_time = time.time() - epoch_timer
        logger.info(f"Training complete! Total time: {total_time / 60:.1f} min")

        if self.use_wandb and self.wandb is not None:
            self.wandb.run.summary["total_training_time_min"] = total_time / 60
            self.wandb.finish()

    def _train_epoch(self, epoch: int, batch_size: int) -> Dict[str, float]:
        """Train for one epoch (iterate over all sections)."""
        self.model.train()

        epoch_losses = []
        section_order = np.random.permutation(len(self.train_dataset))

        for sec_idx in section_order:
            section_data = self.train_dataset[sec_idx]

            # Get section-level expression for section encoder
            section_expr = section_data["gene_expr"].to(self.device)

            # Split section into cell mini-batches
            mini_batches = create_cell_batches(
                section_data, batch_size=batch_size, shuffle=True
            )

            # Get ground truth graph for contrastive loss
            gt_key = section_data["gt_graph_key"]
            gt_graph = self.train_dataset.gt_graphs.get(gt_key, None)

            for batch in mini_batches:
                metrics = self._train_step(
                    batch, section_expr, gt_graph
                )
                epoch_losses.append(metrics)

        # Aggregate epoch metrics
        avg_metrics = {}
        for key in epoch_losses[0]:
            avg_metrics[key] = np.mean([m[key] for m in epoch_losses])

        return avg_metrics

    def _train_step(
        self,
        batch: Dict,
        section_expr: torch.Tensor,
        gt_graph: Optional[Any],
    ) -> Dict[str, float]:
        """Single training step on a mini-batch of cells."""
        gene_expr = batch["gene_expr"].to(self.device)
        coords = batch["coords"].to(self.device)
        k_target_val = batch["k_target"]
        batch_size = gene_expr.shape[0]

        # Encode
        cell_embed, section_embed = self.model.encode(gene_expr, section_expr)

        # k target
        k_target = torch.full(
            (batch_size,), k_target_val, dtype=torch.long, device=self.device
        )

        # Flow matching loss
        fm_loss, fm_metrics = self.model.flow.compute_loss(
            z_1=coords,
            cell_embed=cell_embed,
            section_embed=section_embed,
            k_target=k_target,
        )

        # Combined loss with optional contrastive term
        batch_indices = batch.get("batch_indices_in_section", None)
        total_loss, all_metrics = self.criterion(
            fm_loss=fm_loss,
            cell_embeddings=cell_embed,
            spatial_adj=gt_graph,
            batch_indices=batch_indices,
        )

        # Backprop
        self.optimizer.zero_grad()
        total_loss.backward()

        if self.grad_clip > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.grad_clip
            )
            all_metrics["grad_norm"] = grad_norm.item()

        self.optimizer.step()

        if self.scheduler is not None:
            self.scheduler.step()

        self.global_step += 1

        # Step-level logging
        if self.global_step % self.log_every == 0:
            lr = self.optimizer.param_groups[0]["lr"]
            logger.info(
                f"  Step {self.global_step} | Loss: {all_metrics['total_loss']:.4f} | "
                f"FM: {all_metrics['fm_loss']:.4f} | LR: {lr:.2e}"
            )
            # Wandb step-level logging
            self._log_wandb(all_metrics, prefix="train/step")

        return all_metrics

    @torch.no_grad()
    def _validate(self, batch_size: int) -> Dict[str, float]:
        """Run validation over all validation sections."""
        self.model.eval()
        val_losses = []

        for sec_idx in range(len(self.val_dataset)):
            section_data = self.val_dataset[sec_idx]
            section_expr = section_data["gene_expr"].to(self.device)

            mini_batches = create_cell_batches(
                section_data, batch_size=batch_size, shuffle=False
            )

            gt_key = section_data["gt_graph_key"]
            gt_graph = self.val_dataset.gt_graphs.get(gt_key, None)

            for batch in mini_batches:
                gene_expr = batch["gene_expr"].to(self.device)
                coords = batch["coords"].to(self.device)
                k_val = batch["k_target"]
                bs = gene_expr.shape[0]

                cell_embed, section_embed = self.model.encode(
                    gene_expr, section_expr
                )
                k_target = torch.full(
                    (bs,), k_val, dtype=torch.long, device=self.device
                )

                fm_loss, _ = self.model.flow.compute_loss(
                    z_1=coords,
                    cell_embed=cell_embed,
                    section_embed=section_embed,
                    k_target=k_target,
                )

                _, metrics = self.criterion(fm_loss=fm_loss)
                val_losses.append(metrics)

        avg = {}
        for key in val_losses[0]:
            avg[key] = np.mean([m[key] for m in val_losses])

        return avg

    def _save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint."""
        state = {
            "epoch": epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_val_loss": self.best_val_loss,
            "config": self.config,
        }

        path = self.checkpoint_dir / f"checkpoint_epoch{epoch + 1}.pt"
        torch.save(state, path)
        logger.info(f"Saved checkpoint: {path}")

        if is_best:
            best_path = self.checkpoint_dir / "best_model.pt"
            torch.save(state, best_path)
            logger.info(f"Saved best model: {best_path}")

        # Log checkpoint as wandb artifact
        if self.use_wandb and self.wandb is not None and is_best:
            artifact = self.wandb.Artifact(
                f"model-best", type="model",
                description=f"Best model at epoch {epoch + 1}",
            )
            artifact.add_file(str(best_path))
            self.wandb.log_artifact(artifact)

    def load_checkpoint(self, path: str):
        """Load model from checkpoint."""
        state = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(state["model_state_dict"])
        self.optimizer.load_state_dict(state["optimizer_state_dict"])
        self.global_step = state["global_step"]
        self.best_val_loss = state.get("best_val_loss", float("inf"))
        logger.info(f"Loaded checkpoint from epoch {state['epoch'] + 1}")
