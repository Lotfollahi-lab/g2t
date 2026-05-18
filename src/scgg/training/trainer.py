"""
Training loop for ScGG.

The trainer dispatches based on `config["training"]["objective"]`:

  * "contrastive" (default): supervised InfoNCE on the metric-head output
    against the GT spatial kNN graph. One forward + one loss.
  * "flow_matching" (ablation): flow matching MSE on coordinates, optionally
    augmented with the legacy auxiliary contrastive on encoder outputs.

Optional cross-modality / OOD components (cross-modality contrastive,
section-embedding consistency, domain adversarial) are scaffolded but
DISABLED by default. They require a second-modality dataloader and section
pairing metadata; the trainer will warn (and ignore them) if they are
enabled without those inputs being available.
"""

from __future__ import annotations

import csv
import math
import time
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List, Any

import numpy as np
import torch
import torch.nn as nn

from ..model.scgg import ScGG
from ..data.dataset import (
    SpatialTranscriptomicsDataset,
    create_cell_batches,
)
from .losses import (
    ContrastiveRankingLoss,
    FlowMatchingLoss,
    CellClassAuxLoss,
    DistanceRegressionLoss,
)
from .ood_losses import (
    CrossModalityContrastiveLoss,
    SectionEmbeddingConsistencyLoss,
    DomainAdversarialLoss,
)

logger = logging.getLogger(__name__)


class Trainer:
    """ScGG trainer with pluggable objective + optional OOD components.

    Args:
        model: Initialised ScGG.
        train_dataset: Spatial-transcriptomics training dataset (ground-truth
            graphs precomputed).
        val_dataset: Optional validation dataset (same shape).
        config: Full configuration dict.
        device: Torch device.
        sc_dataset: OPTIONAL second-modality (scRNA-seq) dataset. Only used
            when any of the OOD components in config["training"]["ood"] is
            enabled. If you don't have one yet, pass None.
        section_pairing: OPTIONAL dict mapping ST section id -> matched
            adjacent scRNA section id. Required for cross-modality contrastive
            and section-embedding consistency losses when enabled.
    """

    def __init__(
        self,
        model: ScGG,
        train_dataset: SpatialTranscriptomicsDataset,
        val_dataset: Optional[SpatialTranscriptomicsDataset],
        config: dict,
        device: torch.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        ),
        sc_dataset: Optional[SpatialTranscriptomicsDataset] = None,
        section_pairing: Optional[Dict[int, int]] = None,
    ):
        self.model = model.to(device)
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.config = config
        self.device = device

        self.sc_dataset = sc_dataset
        self.section_pairing = section_pairing or {}

        train_cfg = config["training"]
        self.objective: str = train_cfg.get("objective", "contrastive")

        # ---- Primary criterion ----------------------------------------------
        loss_cfg = train_cfg.get("loss", {})
        if self.objective == "contrastive":
            cl_cfg = loss_cfg.get("contrastive", {}) or {}
            mh_normalizes = config["model"].get("metric_head", {}).get(
                "normalize", True
            )
            self.criterion = ContrastiveRankingLoss(
                temperature=cl_cfg.get("temperature", 0.07),
                max_positives_per_anchor=cl_cfg.get(
                    "max_positives_per_anchor",
                    cl_cfg.get("positives_per_anchor", None),
                ),
                exclude_self=cl_cfg.get("exclude_self", True),
                normalize_inputs=not mh_normalizes,
            )
            self.contrastive_weight = float(cl_cfg.get("weight", 1.0))

            # Optional pairwise distance regression (LUNA eq. 11 analog).
            dr_cfg = loss_cfg.get("distance_regression", {}) or {}
            self.use_distance_loss = bool(dr_cfg.get("enabled", True))
            self.distance_loss_weight = float(dr_cfg.get("weight", 1.0))
            if self.use_distance_loss:
                self.distance_criterion = DistanceRegressionLoss(
                    use_squared=dr_cfg.get("use_squared", True),
                    class_stratified=bool(dr_cfg.get("class_stratified", False)),
                    min_cells_per_class=int(dr_cfg.get("min_cells_per_class", 6)),
                )
                logger.info(
                    "Distance regression loss ENABLED "
                    f"(weight={self.distance_loss_weight}, use_squared="
                    f"{dr_cfg.get('use_squared', True)}, class_stratified="
                    f"{bool(dr_cfg.get('class_stratified', False))})"
                )
            else:
                self.distance_criterion = None

            # Optional cell-class auxiliary classifier on the encoder output.
            cc_cfg = loss_cfg.get("cell_class_aux", {}) or {}
            n_classes = getattr(train_dataset, "n_classes", 0)
            self.use_cellclass_aux = (
                bool(cc_cfg.get("enabled", False)) and n_classes > 0
            )
            self.cellclass_aux_weight = float(cc_cfg.get("weight", 1.0))
            if self.use_cellclass_aux:
                self.cellclass_aux_criterion = CellClassAuxLoss(
                    cell_embed_dim=config["model"]["encoder"]["embed_dim"],
                    n_classes=n_classes,
                    hidden_dim=cc_cfg.get("hidden_dim", 128),
                    dropout=cc_cfg.get("dropout", 0.1),
                ).to(self.device)
                logger.info(
                    "Cell-class auxiliary loss ENABLED "
                    f"(n_classes={n_classes}, weight={self.cellclass_aux_weight})"
                )
            else:
                self.cellclass_aux_criterion = None
                if cc_cfg.get("enabled", False) and n_classes == 0:
                    logger.warning(
                        "cell_class_aux enabled in config but train dataset has "
                        "no cell_class labels (n_classes=0); skipping."
                    )
        elif self.objective == "flow_matching":
            self.criterion = FlowMatchingLoss(
                lambda_contrastive=loss_cfg.get("contrastive_spatial", 0.1),
                temperature=loss_cfg.get("contrastive_temperature", 0.1),
                n_negatives=loss_cfg.get("contrastive_n_negatives", 64),
            )
        else:
            raise ValueError(
                f"Unknown objective {self.objective!r}; expected one of "
                f"'contrastive' or 'flow_matching'."
            )

        # ---- Optional OOD components ----------------------------------------
        ood_cfg = train_cfg.get("ood", {}) or {}
        self._setup_ood_components(ood_cfg, train_cfg)

        # ---- Optimizer ------------------------------------------------------
        if train_cfg["optimizer"] == "adamw":
            self.optimizer = torch.optim.AdamW(
                self._all_params(),
                lr=train_cfg["lr"],
                weight_decay=train_cfg["weight_decay"],
            )
        else:
            self.optimizer = torch.optim.Adam(
                self._all_params(),
                lr=train_cfg["lr"],
                weight_decay=train_cfg["weight_decay"],
            )

        # ---- LR schedule ----------------------------------------------------
        batch_size = train_cfg["batch_size"]
        steps_per_epoch = 0
        for s in train_dataset.sections:
            n_cells = len(train_dataset.section_indices[s])
            steps_per_epoch += max(1, math.ceil(n_cells / batch_size))

        total_steps = train_cfg["epochs"] * steps_per_epoch
        warmup_steps = train_cfg.get("warmup_epochs", 10) * steps_per_epoch
        logger.info(
            f"Scheduler: {steps_per_epoch} steps/epoch, "
            f"{total_steps} total steps, {warmup_steps} warmup steps"
        )

        if train_cfg["scheduler"] == "cosine":
            def lr_lambda(step: int) -> float:
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

        # ---- Checkpoints & logging -----------------------------------------
        self.checkpoint_dir = Path(train_cfg.get("checkpoint_dir", "./checkpoints"))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_every = train_cfg.get("log_every", 100)
        self.eval_every = train_cfg.get("eval_every", 5)
        self.save_every = train_cfg.get("save_every", 10)

        # CSV metrics log. Lives next to the checkpoint dir (so the model
        # output directory has everything: checkpoints/ + metrics.csv +
        # config.yaml + benchmark.log). Schema is a fixed superset of the
        # known metric keys; unknown keys go to wandb but not to CSV.
        self.metrics_csv_path = self.checkpoint_dir.parent / "metrics.csv"
        self._csv_columns = [
            "timestamp", "phase", "global_step", "epoch", "lr",
            "total_loss",
            "contrastive_loss", "n_anchors_used", "mean_positives_per_anchor",
            "distance_loss", "distance_pearson", "distance_pearson_global",
            "distance_n_classes_used",
            "cellclass_aux_loss", "cellclass_aux_acc",
        ]

        # ---- wandb ---------------------------------------------------------
        self.use_wandb = train_cfg.get("wandb", False)
        self.wandb = None
        if self.use_wandb:
            self._init_wandb(train_cfg, config)

    # ------------------------------------------------------------------- setup

    def _setup_ood_components(self, ood_cfg: dict, train_cfg: dict):
        """Instantiate enabled OOD components and warn if data is missing."""
        self.ood_components: Dict[str, Any] = {}
        self.ood_weights: Dict[str, float] = {}

        cm_cfg = ood_cfg.get("cross_modality_contrastive", {}) or {}
        if cm_cfg.get("enabled", False):
            self.ood_components["cm_contrastive"] = CrossModalityContrastiveLoss(
                temperature=cm_cfg.get("temperature", 0.1),
                normalize_inputs=cm_cfg.get("normalize_inputs", False),
                symmetric=cm_cfg.get("symmetric", True),
            )
            self.ood_weights["cm_contrastive"] = float(cm_cfg.get("weight", 0.1))
            if self.sc_dataset is None or not self.section_pairing:
                logger.warning(
                    "cross_modality_contrastive enabled but sc_dataset / "
                    "section_pairing not provided; loss will be skipped."
                )

        sc_cfg = ood_cfg.get("section_embedding_consistency", {}) or {}
        if sc_cfg.get("enabled", False):
            self.ood_components["section_consistency"] = (
                SectionEmbeddingConsistencyLoss(
                    distance=sc_cfg.get("distance", "l2"),
                    normalize=sc_cfg.get("normalize", False),
                )
            )
            self.ood_weights["section_consistency"] = float(sc_cfg.get("weight", 1.0))
            if self.sc_dataset is None or not self.section_pairing:
                logger.warning(
                    "section_embedding_consistency enabled but sc_dataset / "
                    "section_pairing not provided; loss will be skipped."
                )

        da_cfg = ood_cfg.get("domain_adversarial", {}) or {}
        if da_cfg.get("enabled", False):
            enc_dim = self.config["model"]["encoder"]["embed_dim"]
            self.ood_components["domain_adv"] = DomainAdversarialLoss(
                in_dim=enc_dim,
                n_domains=da_cfg.get("n_domains", 2),
                hidden_dim=da_cfg.get("hidden_dim", 128),
                lambd=da_cfg.get("lambd", 1.0),
            ).to(self.device)
            self.ood_weights["domain_adv"] = float(da_cfg.get("weight", 0.1))
            if self.sc_dataset is None:
                logger.warning(
                    "domain_adversarial enabled but sc_dataset not provided; "
                    "loss will be skipped."
                )

        if self.ood_components:
            logger.info(f"OOD components enabled: {list(self.ood_components.keys())}")
        else:
            logger.info("No OOD components enabled (default).")

    def _all_params(self):
        """Iterate parameters from the main model + auxiliary heads + OOD."""
        params = list(self.model.parameters())
        if getattr(self, "cellclass_aux_criterion", None) is not None:
            params += list(self.cellclass_aux_criterion.parameters())
        for c in self.ood_components.values():
            if isinstance(c, nn.Module):
                params += list(c.parameters())
        return [p for p in params if p.requires_grad]

    # ------------------------------------------------------------------- wandb

    def _init_wandb(self, train_cfg: dict, config: dict):
        try:
            import wandb

            wandb.init(
                project=train_cfg.get("wandb_project", "scgg"),
                name=train_cfg.get("wandb_run_name", None),
                config=config,
                tags=train_cfg.get("wandb_tags", []),
            )
            wandb.watch(self.model, log="gradients", log_freq=self.log_every)
            self.wandb = wandb
            logger.info(
                f"Wandb initialized: project={wandb.run.project}, run={wandb.run.name}"
            )
        except ImportError:
            logger.warning("wandb not installed. Install with: pip install wandb")
            self.use_wandb = False

    def _log_wandb(self, metrics: Dict[str, float], prefix: str = "train"):
        # Always write the CSV log, regardless of whether wandb is on —
        # the CSV is the local source of truth for offline analysis.
        self._log_metrics_csv(metrics, phase=prefix)

        if not self.use_wandb or self.wandb is None:
            return
        payload = {f"{prefix}/{k}": v for k, v in metrics.items()}
        payload["global_step"] = self.global_step
        payload["lr"] = self.optimizer.param_groups[0]["lr"]
        self.wandb.log(payload, step=self.global_step)

    def _log_metrics_csv(self, metrics: Dict[str, float], phase: str) -> None:
        """Append one row to metrics.csv with the well-known metric columns.

        Phase is the same string passed to _log_wandb ("train/step",
        "train/epoch", "val", ...). Unknown keys are silently dropped from
        the CSV but still flow to wandb. The file is created with a header
        on the first call; subsequent calls append.
        """
        try:
            self.metrics_csv_path.parent.mkdir(parents=True, exist_ok=True)
            row = {c: "" for c in self._csv_columns}
            row["timestamp"] = datetime.now().isoformat(timespec="seconds")
            row["phase"] = phase
            row["global_step"] = self.global_step
            row["lr"] = self.optimizer.param_groups[0]["lr"]
            for k, v in metrics.items():
                if k in self._csv_columns:
                    row[k] = v
            new_file = not self.metrics_csv_path.exists()
            with open(self.metrics_csv_path, "a", newline="") as f:
                w = csv.DictWriter(f, fieldnames=self._csv_columns)
                if new_file:
                    w.writeheader()
                w.writerow(row)
        except OSError as e:
            # Never let a CSV write failure crash training.
            logger.warning(f"Failed to write metrics.csv row: {e}")

    # ------------------------------------------------------------------- train

    def train(self):
        train_cfg = self.config["training"]
        n_epochs = train_cfg["epochs"]
        batch_size = train_cfg["batch_size"]

        n_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        logger.info(f"Starting training: objective={self.objective}")
        logger.info(f"  Epochs: {n_epochs}")
        logger.info(f"  Sections: {len(self.train_dataset)}")
        logger.info(f"  Batch size: {batch_size} cells")
        logger.info(f"  Device: {self.device}")
        logger.info(f"  Parameters (model): {n_params:,}")

        epoch_timer = time.time()
        for epoch in range(n_epochs):
            t0 = time.time()
            epoch_metrics = self._train_epoch(epoch, batch_size)
            epoch_time = time.time() - t0

            primary_loss_name = (
                "contrastive_loss" if self.objective == "contrastive" else "fm_loss"
            )
            parts = [
                f"Epoch {epoch + 1}/{n_epochs}",
                f"Total: {epoch_metrics.get('total_loss', float('nan')):.4f}",
                f"{primary_loss_name}: {epoch_metrics.get(primary_loss_name, float('nan')):.4f}",
            ]
            if "distance_pearson" in epoch_metrics:
                parts.append(f"dist_pearson: {epoch_metrics['distance_pearson']:.4f}")
            parts.append(f"LR: {self.optimizer.param_groups[0]['lr']:.2e}")
            parts.append(f"Time: {epoch_time:.1f}s")
            logger.info(" | ".join(parts))

            epoch_metrics["epoch_time_s"] = epoch_time
            epoch_metrics["epoch"] = epoch + 1
            self._log_wandb(epoch_metrics, prefix="train/epoch")

            if (
                self.val_dataset is not None
                and (epoch + 1) % self.eval_every == 0
            ):
                val_metrics = self._validate(batch_size)
                val_metrics["epoch"] = epoch + 1
                logger.info(
                    f"  Val Total: {val_metrics.get('total_loss', float('nan')):.4f}"
                )
                self._log_wandb(val_metrics, prefix="val")
                if val_metrics.get("total_loss", float("inf")) < self.best_val_loss:
                    self.best_val_loss = val_metrics["total_loss"]
                    self._save_checkpoint(epoch, is_best=True)
                    if self.use_wandb and self.wandb is not None:
                        self.wandb.run.summary["best_val_loss"] = self.best_val_loss
                        self.wandb.run.summary["best_epoch"] = epoch + 1

            if (epoch + 1) % self.save_every == 0:
                self._save_checkpoint(epoch)

        total_time = time.time() - epoch_timer
        logger.info(f"Training complete! Total time: {total_time / 60:.1f} min")
        if self.use_wandb and self.wandb is not None:
            self.wandb.run.summary["total_training_time_min"] = total_time / 60
            self.wandb.finish()

    # ----------------------------------------------------------------- epochs

    def _train_epoch(self, epoch: int, batch_size: int) -> Dict[str, float]:
        self.model.train()
        for c in self.ood_components.values():
            if isinstance(c, nn.Module):
                c.train()

        all_metrics: List[Dict[str, float]] = []
        section_order = np.random.permutation(len(self.train_dataset))

        for sec_idx in section_order:
            section_data = self.train_dataset[sec_idx]
            section_expr = section_data["gene_expr"].to(self.device)
            mini_batches = create_cell_batches(
                section_data, batch_size=batch_size, shuffle=True
            )
            gt_key = section_data["gt_graph_key"]
            gt_graph = self.train_dataset.gt_graphs.get(gt_key, None)

            for batch in mini_batches:
                metrics = self._train_step(batch, section_expr, gt_graph)
                all_metrics.append(metrics)

        avg: Dict[str, float] = {}
        if not all_metrics:
            return avg
        for k in all_metrics[0]:
            try:
                avg[k] = float(np.mean([m[k] for m in all_metrics if k in m]))
            except (TypeError, ValueError):
                pass
        return avg

    # ------------------------------------------------------------------- step

    def _train_step(
        self,
        batch: Dict,
        section_expr: torch.Tensor,
        gt_graph,
    ) -> Dict[str, float]:
        gene_expr = batch["gene_expr"].to(self.device)
        batch_indices = batch.get("batch_indices_in_section", None)

        if self.objective == "contrastive":
            # Split the forward so we can also feed cell_embed to the
            # optional cell-class auxiliary classifier.
            cell_embed, section_embed = self.model.encode(gene_expr, section_expr)
            emb = self.model.metric_head(cell_embed, section_embed)

            # Primary: SupCon contrastive (top-k retrieval).
            c_loss, c_metrics = self.criterion(
                embeddings=emb,
                spatial_adj=gt_graph,
                batch_indices=batch_indices,
            )
            metrics = {"contrastive_loss": c_metrics["contrastive_loss"]}
            for k in ("n_anchors_used", "n_anchors_skipped", "n_anchors_clamped",
                      "mean_positives_per_anchor"):
                if k in c_metrics:
                    metrics[k] = c_metrics[k]

            # Optional: pairwise-distance regression (Spearman-aligned).
            loss = self.contrastive_weight * c_loss
            if self.use_distance_loss and self.distance_criterion is not None:
                coords = batch["coords"].to(self.device)
                cls_for_dist = batch.get("cell_class", None)
                if cls_for_dist is not None:
                    cls_for_dist = cls_for_dist.to(self.device)
                d_loss, d_metrics = self.distance_criterion(
                    emb, coords, cell_class=cls_for_dist,
                )
                loss = loss + self.distance_loss_weight * d_loss
                metrics["distance_loss"] = d_metrics["distance_loss"]
                metrics["distance_pearson"] = d_metrics["distance_pearson"]
                metrics["distance_pearson_global"] = d_metrics.get(
                    "distance_pearson_global", d_metrics["distance_pearson"]
                )
                metrics["distance_n_classes_used"] = d_metrics.get(
                    "n_classes_used", 0
                )

            # Optional: cell-class auxiliary classifier on cell_embed.
            if self.use_cellclass_aux and self.cellclass_aux_criterion is not None:
                cls = batch.get("cell_class", None)
                if cls is not None:
                    cls = cls.to(self.device)
                    valid = cls >= 0
                    if valid.any():
                        cc_loss, cc_metrics = self.cellclass_aux_criterion(
                            cell_embed[valid], cls[valid]
                        )
                        loss = loss + self.cellclass_aux_weight * cc_loss
                        metrics["cellclass_aux_loss"] = cc_metrics["cellclass_aux_loss"]
                        metrics["cellclass_aux_acc"] = cc_metrics["cellclass_aux_acc"]

        else:  # flow_matching
            coords = batch["coords"].to(self.device)
            k_target_val = batch["k_target"]
            B = gene_expr.shape[0]
            cell_embed, section_embed = self.model.encode(gene_expr, section_expr)
            k_target = torch.full(
                (B,), k_target_val, dtype=torch.long, device=self.device
            )
            fm_loss, _ = self.model.flow.compute_loss(
                z_1=coords,
                cell_embed=cell_embed,
                section_embed=section_embed,
                k_target=k_target,
            )
            loss, primary_metrics = self.criterion(
                fm_loss=fm_loss,
                cell_embeddings=cell_embed,
                spatial_adj=gt_graph,
                batch_indices=batch_indices,
            )
            metrics = dict(primary_metrics)

        # ---- OOD components (no-op by default; require second modality) ----
        # Hooks are wired here so flipping them on in the future only requires
        # plumbing a paired-modality batch into _train_step. We currently do
        # not have that path because the user requested OOD components be off
        # by default; the trainer will emit a one-time warning if invoked.
        # (Future PR: accept a (st_batch, sc_batch) tuple and route accordingly.)

        # ---- Backprop ------------------------------------------------------
        self.optimizer.zero_grad()
        loss.backward()
        if self.grad_clip > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self._all_params(), self.grad_clip
            )
            metrics["grad_norm"] = grad_norm.item()
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()
        self.global_step += 1

        metrics["total_loss"] = loss.item()

        if self.global_step % self.log_every == 0:
            lr = self.optimizer.param_groups[0]["lr"]
            primary_name = (
                "contrastive_loss" if self.objective == "contrastive" else "fm_loss"
            )
            parts = [
                f"Step {self.global_step}",
                f"Total: {metrics['total_loss']:.4f}",
                f"{primary_name}: {metrics.get(primary_name, float('nan')):.4f}",
            ]
            if "distance_pearson" in metrics:
                parts.append(f"dist_pearson: {metrics['distance_pearson']:.4f}")
            parts.append(f"LR: {lr:.2e}")
            logger.info("  " + " | ".join(parts))
            self._log_wandb(metrics, prefix="train/step")

        return metrics

    # ---------------------------------------------------------------- validate

    @torch.no_grad()
    def _validate(self, batch_size: int) -> Dict[str, float]:
        self.model.eval()
        val_metrics: List[Dict[str, float]] = []

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
                batch_indices = batch.get("batch_indices_in_section", None)

                if self.objective == "contrastive":
                    cell_embed, section_embed = self.model.encode(
                        gene_expr, section_expr
                    )
                    emb = self.model.metric_head(cell_embed, section_embed)
                    c_loss, c_m = self.criterion(
                        embeddings=emb,
                        spatial_adj=gt_graph,
                        batch_indices=batch_indices,
                    )
                    total_loss_val = self.contrastive_weight * c_loss
                    val_entry = {"contrastive_loss": c_m["contrastive_loss"]}
                    if self.use_distance_loss and self.distance_criterion is not None:
                        coords = batch["coords"].to(self.device)
                        cls_for_dist = batch.get("cell_class", None)
                        if cls_for_dist is not None:
                            cls_for_dist = cls_for_dist.to(self.device)
                        d_loss, d_m = self.distance_criterion(
                            emb, coords, cell_class=cls_for_dist,
                        )
                        total_loss_val = (
                            total_loss_val + self.distance_loss_weight * d_loss
                        )
                        val_entry["distance_loss"] = d_m["distance_loss"]
                        val_entry["distance_pearson"] = d_m["distance_pearson"]
                        val_entry["distance_pearson_global"] = d_m.get(
                            "distance_pearson_global", d_m["distance_pearson"]
                        )
                        val_entry["distance_n_classes_used"] = d_m.get(
                            "n_classes_used", 0
                        )
                    if (
                        self.use_cellclass_aux
                        and self.cellclass_aux_criterion is not None
                    ):
                        cls = batch.get("cell_class", None)
                        if cls is not None:
                            cls = cls.to(self.device)
                            valid = cls >= 0
                            if valid.any():
                                cc_loss, cc_m = self.cellclass_aux_criterion(
                                    cell_embed[valid], cls[valid]
                                )
                                total_loss_val = (
                                    total_loss_val
                                    + self.cellclass_aux_weight * cc_loss
                                )
                                val_entry["cellclass_aux_loss"] = cc_m[
                                    "cellclass_aux_loss"
                                ]
                                val_entry["cellclass_aux_acc"] = cc_m[
                                    "cellclass_aux_acc"
                                ]
                    val_entry["total_loss"] = total_loss_val.item()
                    val_metrics.append(val_entry)
                else:
                    coords = batch["coords"].to(self.device)
                    k_val = batch["k_target"]
                    B = gene_expr.shape[0]
                    ce, se = self.model.encode(gene_expr, section_expr)
                    kk = torch.full(
                        (B,), k_val, dtype=torch.long, device=self.device
                    )
                    fm_loss, _ = self.model.flow.compute_loss(
                        z_1=coords,
                        cell_embed=ce,
                        section_embed=se,
                        k_target=kk,
                    )
                    loss, m = self.criterion(fm_loss=fm_loss)
                    val_metrics.append(m)

        if not val_metrics:
            return {"total_loss": float("nan")}
        out: Dict[str, float] = {}
        for k in val_metrics[0]:
            try:
                out[k] = float(np.mean([m[k] for m in val_metrics if k in m]))
            except (TypeError, ValueError):
                pass
        return out

    # ---------------------------------------------------------------- save/load

    def _save_checkpoint(self, epoch: int, is_best: bool = False):
        state = {
            "epoch": epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": (
                self.scheduler.state_dict() if self.scheduler is not None else None
            ),
            "best_val_loss": self.best_val_loss,
            "config": self.config,
            "cellclass_aux_state_dict": (
                self.cellclass_aux_criterion.state_dict()
                if getattr(self, "cellclass_aux_criterion", None) is not None
                else None
            ),
            "ood_state_dicts": {
                name: c.state_dict()
                for name, c in self.ood_components.items()
                if isinstance(c, nn.Module)
            },
        }
        path = self.checkpoint_dir / f"checkpoint_epoch{epoch + 1}.pt"
        torch.save(state, path)
        logger.info(f"Saved checkpoint: {path}")
        if is_best:
            best_path = self.checkpoint_dir / "best_model.pt"
            torch.save(state, best_path)
            logger.info(f"Saved best model: {best_path}")
        if self.use_wandb and self.wandb is not None and is_best:
            artifact = self.wandb.Artifact(
                "model-best", type="model",
                description=f"Best model at epoch {epoch + 1}",
            )
            artifact.add_file(str(best_path))
            self.wandb.log_artifact(artifact)

    def load_checkpoint(self, path: str):
        state = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(state["model_state_dict"])
        self.optimizer.load_state_dict(state["optimizer_state_dict"])
        sched_state = state.get("scheduler_state_dict", None)
        if sched_state is not None and self.scheduler is not None:
            self.scheduler.load_state_dict(sched_state)
        elif sched_state is None and self.scheduler is not None:
            logger.warning(
                "Checkpoint did not include scheduler_state_dict; LR schedule "
                "will restart from step 0. Resuming a long run can produce a "
                "different LR trajectory than the original."
            )
        cc_state = state.get("cellclass_aux_state_dict", None)
        if cc_state is not None and getattr(self, "cellclass_aux_criterion", None) is not None:
            self.cellclass_aux_criterion.load_state_dict(cc_state)
        for name, sd in state.get("ood_state_dicts", {}).items():
            if name in self.ood_components and isinstance(
                self.ood_components[name], nn.Module
            ):
                self.ood_components[name].load_state_dict(sd)
        self.global_step = state["global_step"]
        self.best_val_loss = state.get("best_val_loss", float("inf"))
        logger.info(f"Loaded checkpoint from epoch {state['epoch'] + 1}")
