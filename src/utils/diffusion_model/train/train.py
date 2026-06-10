import math

import torch
import wandb

from utils.data.dataholder import DataHolder
from utils.data.misc import to_batch


def _apply_o2_augmentation(
    batched_data: DataHolder,
    augment_rotation: bool,
    augment_reflection: bool,
) -> DataHolder:
    """Apply an independent random O(2) transform to each slice's
    positions, in-place on the DataHolder's positions tensor.

    O(2) = rotations + (optionally) reflections. We sample one matrix
    per slice in the batch and apply it as ``x' = x @ Rᵀ``. Gene
    features and cell metadata are scalars and stay untouched.

    Padding cells (where node_mask is False) have positions == 0 and
    are unaffected by the linear transform; no extra masking needed.

    The loss this feeds into is the pairwise-distance MSE, which is
    O(2)-invariant. So augmenting input AND target by the same
    matrix is a no-op for the loss value, but forces the network
    to be invariant to the input frame — equivalent in spirit to
    EGNN's built-in equivariance but applied to the LUNA transformer.
    """
    if not (augment_rotation or augment_reflection):
        return batched_data

    pos = batched_data.positions                  # (B, N, 2)
    B = pos.shape[0]
    device = pos.device
    dtype = pos.dtype

    # Rotation matrices R(θ). θ ~ U(0, 2π) per slice.
    if augment_rotation:
        theta = (
            torch.rand(B, device=device, dtype=dtype) * (2.0 * math.pi)
        )
    else:
        theta = torch.zeros(B, device=device, dtype=dtype)
    c, s = torch.cos(theta), torch.sin(theta)
    # R (B, 2, 2). Build columnwise to avoid a stack/transpose.
    R = torch.stack(
        [
            torch.stack([c, -s], dim=-1),
            torch.stack([s,  c], dim=-1),
        ],
        dim=-2,
    )                                              # (B, 2, 2)

    if augment_reflection:
        # Flip = diag(±1, 1), Bernoulli(0.5) per slice. det(flip) = ±1.
        flip = torch.where(
            torch.rand(B, device=device) < 0.5,
            torch.tensor(-1.0, device=device, dtype=dtype),
            torch.tensor( 1.0, device=device, dtype=dtype),
        )
        # Compose: rot · flip. flip diag matrix as (B, 2, 2).
        F = torch.zeros(B, 2, 2, device=device, dtype=dtype)
        F[:, 0, 0] = flip
        F[:, 1, 1] = 1.0
        R = R @ F

    # Apply per-slice transform: x' = x @ Rᵀ for each slice.
    # einsum is the cleanest way to batch this without a transpose dance.
    pos_aug = torch.einsum("bnd,bed->bne", pos, R)  # (B, N, 2)

    # Construct a new DataHolder (DataHolder is roughly a dataclass —
    # safer than mutating in-place). DataHolder.__init__ requires
    # ``diffusion_time`` as a positional arg; at this point in the
    # pipeline (between to_batch and apply_noise) it's typically
    # None on the input — passing it through preserves whatever
    # the upstream stage set.
    return DataHolder(
        node_features=batched_data.node_features,
        positions=pos_aug,
        diffusion_time=getattr(batched_data, "diffusion_time", None),
        cell_class=batched_data.cell_class,
        cell_ID=getattr(batched_data, "cell_ID", None),
        t_int=getattr(batched_data, "t_int", None),
        t=getattr(batched_data, "t", None),
        node_mask=batched_data.node_mask,
    ).mask()


def _apply_gene_augmentation(
    batched_data: DataHolder,
    dropout_p: float,
    noise_sigma: float,
    mixup_alpha: float,
) -> DataHolder:
    """Train-time augmentation on gene-expression INPUT.

    Three independent mechanisms (each off when its parameter is 0):

      * ``dropout_p`` ∈ [0, 1] — per-(cell, gene-channel) Bernoulli
        zero-out. Forces the model to be robust to "gene wasn't
        detected this cell" noise. 0.1-0.2 is typical.
      * ``noise_sigma`` ≥ 0 — additive Gaussian noise (multiplied by
        the batch-wise stddev of non-padding gene values so the
        regularizer is scale-invariant). Forces robustness to
        measurement noise. 0.05-0.1 typical.
      * ``mixup_alpha`` > 0 — cell-level mixup. For each cell sample a
        partner cell from the same slice (uniform over real cells),
        blend gene vectors with λ ~ Beta(α, α). λ=0.5 (α large)
        gives strong mixup; λ near 0 or 1 (α small) gives mild.
        Position TARGETS are NOT mixed — the model still learns to
        predict the original cell's position from the mixed-gene
        input, which is a "denoising / disentanglement" auxiliary
        signal. 0.2-0.4 typical for α.

    Padding cells (node_mask=False) are untouched. The augmentation
    runs INSIDE training_step, so it's free at validation/test/
    inference time — the model trained under augmented input
    generalises better but pays nothing at eval.

    The three mechanisms compose linearly when more than one is
    active — dropout → noise → mixup, applied in that order. In
    practice you typically pick one as the primary regularizer.
    """
    if dropout_p <= 0 and noise_sigma <= 0 and mixup_alpha <= 0:
        return batched_data

    nf = batched_data.node_features                   # (B, N, G)
    node_mask = batched_data.node_mask                # (B, N)
    mask_3d = node_mask.unsqueeze(-1)                 # (B, N, 1)

    # --- (1) Bernoulli dropout on gene channels ---------------------
    if dropout_p > 0:
        # Per-(cell, channel) draw. Drops the gene value to 0; the
        # downstream gene encoder sees this as "this gene wasn't
        # measured for this cell."
        drop = (torch.rand_like(nf) < dropout_p) & mask_3d
        nf = nf.masked_fill(drop, 0.0)

    # --- (2) Additive Gaussian noise scaled by batch stddev ---------
    if noise_sigma > 0:
        # Scale by stddev of real (non-padding) gene values so the
        # absolute noise level is consistent across batches /
        # datasets with different normalisation. Computed under
        # no_grad — it's a scalar adjustment to the noise, not a
        # learnable scale.
        with torch.no_grad():
            real = nf[mask_3d.expand_as(nf)]
            scale = real.float().std().clamp_min(1e-6)
        noise = torch.randn_like(nf) * (noise_sigma * scale)
        noise = noise * mask_3d.to(noise.dtype)        # zero noise on padding
        nf = nf + noise

    # --- (3) Cell-level within-slice mixup --------------------------
    if mixup_alpha > 0:
        B, N, G = nf.shape
        # Sample a per-(slice, cell) partner index uniformly. We use
        # a within-slice permutation so partner cells live in the
        # SAME spatial context as the source cell — different slices
        # may have different cell-type distributions, and mixing
        # across slices could pull the model toward an average frame
        # that doesn't match any real slice.
        partner_idx = torch.stack([
            torch.randperm(N, device=nf.device) for _ in range(B)
        ], dim=0)                                       # (B, N)
        # Gather partner features.
        partner_nf = nf.gather(
            1, partner_idx.unsqueeze(-1).expand(-1, -1, G),
        )                                               # (B, N, G)
        # λ ~ Beta(α, α), one per cell.
        beta_dist = torch.distributions.Beta(
            torch.tensor(float(mixup_alpha)),
            torch.tensor(float(mixup_alpha)),
        )
        lam = beta_dist.sample((B, N, 1)).to(nf.device).to(nf.dtype)
        # Mix. Padding cells are zero on both sides → still zero.
        nf = lam * nf + (1.0 - lam) * partner_nf
        nf = nf * mask_3d.to(nf.dtype)

    return DataHolder(
        node_features=nf,
        positions=batched_data.positions,
        diffusion_time=getattr(batched_data, "diffusion_time", None),
        cell_class=batched_data.cell_class,
        cell_ID=getattr(batched_data, "cell_ID", None),
        t_int=getattr(batched_data, "t_int", None),
        t=getattr(batched_data, "t", None),
        node_mask=batched_data.node_mask,
    ).mask()


def training_step_func(self, data: DataHolder, i: int) -> torch.Tensor:
    """
    Training step for a single batch.

    Parameters:
    - data: Batch of input data.
    - i: Index of the current batch.

    Returns:
    - torch.Tensor: Loss for the current batch.
    """
    # Get the current learning rate and log it if using WandB
    lr = self.optimizers().param_groups[0]["lr"]
    if wandb.run:
        wandb.log({"LR": lr}, commit=False)

    # Set the model to train mode
    self.model.train()

    # Preprocess the input data
    batched_data = to_batch(data)

    # Train-time O(2) augmentation. Defaults OFF (see
    # configs/train/default.yaml). When ON, this rotates BOTH the
    # input positions AND the targets (because the SAME tensor feeds
    # both apply_noise's x_0 and the loss's true_positions), so the
    # loss value is unchanged but the network sees a different frame
    # each step. Free regularisation against orientation overfitting
    # on small-N-slice datasets (cortex has ~6 slices).
    train_cfg = getattr(self.cfg, "train", None)
    if train_cfg is not None:
        aug_rot = bool(getattr(train_cfg, "augment_rotation", False))
        aug_ref = bool(getattr(train_cfg, "augment_reflection", False))
        if aug_rot or aug_ref:
            batched_data = _apply_o2_augmentation(batched_data, aug_rot, aug_ref)

        # Gene-expression augmentation (dropout / noise / mixup on
        # node_features). Three independent knobs, each off when 0;
        # see ``_apply_gene_augmentation`` docstring. Composes with
        # O(2) augmentation above — positions are augmented by O(2),
        # genes by this block. Free at inference (train-only).
        gene_dropout = float(
            getattr(train_cfg, "gene_augment_dropout_p", 0.0)
        )
        gene_noise = float(
            getattr(train_cfg, "gene_augment_noise_sigma", 0.0)
        )
        gene_mixup = float(
            getattr(train_cfg, "gene_augment_mixup_alpha", 0.0)
        )
        if gene_dropout > 0 or gene_noise > 0 or gene_mixup > 0:
            batched_data = _apply_gene_augmentation(
                batched_data,
                dropout_p=gene_dropout,
                noise_sigma=gene_noise,
                mixup_alpha=gene_mixup,
            )

    # Auxiliary gene reconstruction: with probability 1 (every step
    # when enabled), mask out a fraction of each cell's gene values
    # before noising. The masked entries become reconstruction
    # targets for ``self.gene_recon_head`` at the end of the
    # forward. Active iff the LightningModule built the head.
    gene_recon_mask = None
    gene_recon_targets = None
    if getattr(self, "gene_recon_head", None) is not None:
        mask_ratio = float(self._gene_recon_mask_ratio)
        with torch.no_grad():
            # Per-gene-element Bernoulli mask. Apply only to real
            # cells (padding cells are already zeros).
            gene_recon_mask = (
                torch.rand_like(batched_data.node_features) < mask_ratio
            ) & batched_data.node_mask.unsqueeze(-1)
            gene_recon_targets = batched_data.node_features.clone()
        # Mask the input by zeroing masked positions. The decoder
        # has to recover them from gene CONTEXT (other cells' genes
        # via the attention machinery).
        masked_features = batched_data.node_features.clone()
        masked_features[gene_recon_mask] = 0.0
        batched_data_for_noise = batched_data.copy()
        batched_data_for_noise.node_features = masked_features
    else:
        batched_data_for_noise = batched_data

    # MixFlow-style informed prior (gated by prior_head; default off).
    # Predict each cell's coarse CANONICAL-frame position from its
    # features and (a) train that head with a geodesic MSE to the true
    # canonical position, (b) hand the prediction (DETACHED) to
    # apply_noise so the FM source Gaussian is re-centred there. Detach
    # = the head trains only via the geodesic term, the FM only sees a
    # warm-started prior (no FM→prior feedback loop).
    prior_mean = None
    prior_geo = None
    if getattr(self, "prior_head", None) is not None:
        from utils.data.canonicalize import canonicalize_cloud
        _feat = batched_data.node_features
        _m = batched_data.node_mask.unsqueeze(-1).to(_feat.dtype)
        prior_mean = self.prior_head(_feat) * _m                  # (B,N,2)
        with torch.no_grad():
            _x0_canon = canonicalize_cloud(
                batched_data.positions, batched_data.node_mask)
        _sq = ((prior_mean - _x0_canon) ** 2) * _m
        prior_geo = _sq.sum() / (_m.sum() * 2.0 + 1e-8)

    z_t = self.noise_model.apply_noise(
        batched_data_for_noise,
        prior_mean=(prior_mean.detach() if prior_mean is not None else None),
    )

    # Stash TRUE positions on the Lightning module so the
    # CoarseToFineWrapper (if active) can use them for teacher
    # forcing in its coarse-stage centroid conditioning. The wrapper
    # reads this attribute in self.forward; non-c2f models ignore it.
    self._c2f_true_positions = batched_data.positions

    # Self-conditioning two-pass training (Chen 2023). With
    # probability self._self_cond_prob: do a first forward with
    # zero self-cond, detach the predicted positions, and feed them
    # as self-cond for the second forward. Loss is on the second
    # pass. With probability (1 - prob): single-pass forward with
    # zero self-cond.
    self_cond_enabled = bool(getattr(self, "_self_cond_enabled", False))
    if self_cond_enabled:
        sc_prob = float(getattr(self, "_self_cond_prob", 0.5))
        do_two_pass = bool(torch.rand(1).item() < sc_prob)
    else:
        do_two_pass = False

    try:
        if do_two_pass:
            with torch.no_grad():
                # First pass: zero self-cond (the field is
                # interpreted as "no prior info").
                z_t._self_cond_x0 = torch.zeros_like(z_t.positions)
                first_pred = self.forward(z_t)
            # Second pass: condition on first prediction (detached).
            z_t._self_cond_x0 = first_pred.positions.detach()
            pred = self.forward(z_t)
        else:
            if self_cond_enabled:
                z_t._self_cond_x0 = torch.zeros_like(z_t.positions)
            pred = self.forward(z_t)
    finally:
        # Clear the stash so validation/test/inference forward calls
        # don't accidentally see stale true positions.
        self._c2f_true_positions = None

    # Compute the training loss
    loss, tl_log_dict = self.train_loss(
        masked_pred=pred, masked_true=batched_data, log=i % self.log_every_steps == 0
    )
    loss = loss

    # learned_regression informed prior: add the geodesic MSE that trains
    # the prior head (keeps μ near the true canonical position, so the FM
    # only makes small corrections — MixFlow's geodesic term).
    if prior_geo is not None:
        loss = loss + self._prior_geo_weight * prior_geo
        if wandb.run and (i % self.log_every_steps == 0):
            wandb.log({"train_loss/prior_geodesic":
                       float(prior_geo.detach().item())}, commit=False)

    # Auxiliary gene-reconstruction loss. Head input: concat of the
    # backbone's per-cell output node_features and the predicted
    # position. Loss is masked MSE on the original gene values at
    # the masked positions only.
    if gene_recon_mask is not None and gene_recon_mask.any():
        head_input = torch.cat([pred.node_features, pred.positions], dim=-1)
        recon = self.gene_recon_head(head_input)                     # (B, N, n_genes)
        sq_err = (recon - gene_recon_targets) ** 2
        # Only compare at masked positions.
        masked_sq = sq_err * gene_recon_mask.to(sq_err.dtype)
        n_masked = gene_recon_mask.sum().clamp_min(1).to(sq_err.dtype)
        recon_loss = masked_sq.sum() / n_masked
        weighted = float(self._gene_recon_weight) * recon_loss
        loss = loss + weighted
        if wandb.run and (i % self.log_every_steps == 0):
            wandb.log({
                "train_loss/gene_reconstruction": float(recon_loss.detach().item()),
                "train_loss/gene_reconstruction_weighted": float(weighted.detach().item()),
            }, commit=False)

    # Log the training loss and metrics if available
    if tl_log_dict is not None:
        self.log_dict(tl_log_dict, batch_size=self.BS)

    # Log epoch metrics for training loss
    tle_log = self.train_loss.log_epoch_metrics()
    self.log_dict(tle_log, batch_size=self.BS)

    # Log the epoch number if using WandB
    if wandb.run:
        wandb.log({"epoch": self.current_epoch}, commit=False)
    return loss


def on_train_epoch_end_func(self) -> None:
    """
    Callback function called at the end of each training epoch.

    Returns:
    - None
    """
    pass


def on_train_epoch_start_func(self) -> None:
    """
    Callback function called at the start of each training epoch.

    Returns:
    - None
    """
    # Reset training loss and metrics for the new epoch
    self.train_loss.reset()
