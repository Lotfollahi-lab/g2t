"""EDM (Euclidean Distance Matrix) output head — scGG fundamental method #1.

Motivation
----------
LUNA generates 2D coordinates X ∈ R^{N×2} and depends on the pairwise-
distance loss (and train-time rotation augmentation) to fake rotation
invariance. The network has to learn to produce coordinates that are
internally consistent in SOME frame — a frame the network effectively
invents.

The EDM head reframes the output: rather than coordinates, we emit
per-cell embeddings h_i ∈ R^k, which induce a pairwise squared-
distance matrix D_ij = ‖h_i − h_j‖² that lies on the Euclidean
Distance Matrix manifold by construction. D is invariant to the
network's choice of frame because no frame is chosen — only relative
geometry is emitted. Coordinates are recovered post-hoc by classical
MDS (eigendecomposition of the double-centred D) and Procrustes-
aligned to the current x_t frame so the FM/DDPM trajectory stays
consistent across reverse-process steps.

Properties
----------
- Translation/rotation/reflection invariance hold BY CONSTRUCTION
  (D depends only on relative geometry, MDS produces a frame, the
  Procrustes step optionally aligns it to a reference).
- Supervision signal scales as O(N²) (every cell pair) instead of
  O(N) (every cell). On a 6-slice, 7k-cells-per-slice dataset like
  cortex, that's a 7000× increase in supervised quantities.
- Composable with any framework (diffusion / flow_matching /
  regression) and any inner backbone (luna_transformer / egnn).

Trade-offs
----------
- The MDS step is O(N³) per slice (eigendecomposition of an N×N
  matrix). For N ≤ 10k this is fine on GPU; beyond that consider
  truncated MDS or skipping the alignment.
- The eigh / svd backward passes are numerically delicate when
  eigenvalues are close (Procrustes degenerates near isotropic
  layouts). In practice this isn't an issue on real tissue data
  where layout is anisotropic.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from utils.data.dataholder import DataHolder


# ---------------------------------------------------------------------------
# Differentiable classical MDS + Procrustes alignment
# ---------------------------------------------------------------------------


def _classical_mds_2d(D_sq: torch.Tensor) -> torch.Tensor:
    """Differentiable classical MDS to 2D from a squared-distance matrix.

    Args:
        D_sq: (n, n) symmetric, nonneg squared-distance matrix.

    Returns:
        (n, 2) MDS coordinates. Frame is the eigenframe of the
        double-centred Gram matrix — defined up to reflection/rotation.
    """
    n = D_sq.shape[0]
    device = D_sq.device
    dtype = D_sq.dtype

    # Double-centring: B = -0.5 * J D J where J = I - 1/n.
    one = torch.ones((n, n), device=device, dtype=dtype) / n
    J = torch.eye(n, device=device, dtype=dtype) - one
    B = -0.5 * J @ D_sq @ J
    # Symmetrise (numerical safety — D_sq may have minor asymmetry).
    B = 0.5 * (B + B.T)

    # Eigendecomposition (ascending). Top 2 = last 2.
    evals, evecs = torch.linalg.eigh(B)
    e2 = evals[-2:].clamp(min=1e-12)              # (2,) nonneg
    v2 = evecs[:, -2:]                            # (n, 2)
    # MDS coords: V * sqrt(Λ).
    return v2 * torch.sqrt(e2).unsqueeze(0)       # (n, 2)


def _procrustes_align(x_src: torch.Tensor, x_ref: torch.Tensor) -> torch.Tensor:
    """Orthogonal Procrustes: find the rotation+reflection R that minimises
    ‖x_src @ R − x_ref‖_F. Returns x_src @ R.

    Both inputs assumed to be mean-centred. Reflection is allowed
    (full O(2) alignment, not just SO(2)) — biologically symmetric
    tissue + we just want the canonical frame.

    Derivation: with M = x_src^T @ x_ref and SVD M = U Σ Vᵀ, the
    closed-form solution is R = U Vᵀ (Schönemann 1966). PyTorch's
    ``linalg.svd`` returns ``Vh = Vᵀ`` directly, so the formula is
    ``R = U @ Vh``. (Earlier revision had ``R = Vh.T @ U.T = V Uᵀ``,
    which is R-transposed — applied the INVERSE rotation and broke
    the smoke test.)
    """
    M = x_src.T @ x_ref                            # (2, 2)
    U, _S, Vh = torch.linalg.svd(M)
    R = U @ Vh                                      # (2, 2)
    return x_src @ R


# ---------------------------------------------------------------------------
# Output wrapper
# ---------------------------------------------------------------------------


class EDMOutputWrapper(nn.Module):
    """Wraps an inner backbone with an EDM output head.

    The inner backbone is expected to expose the same DataHolder-in /
    DataHolder-out interface as ``models.model.Model``. After the inner
    forward, we project ``pred.node_features`` to per-cell embeddings
    ``h ∈ R^k``, compute the pairwise squared-distance matrix, and
    (optionally) overwrite ``pred.positions`` with the MDS-recovered
    canonical layout aligned to the input frame.

    The predicted distance matrix is stashed on ``pred.edm_D`` so the
    LossFunction's new ``edm_distance_mse`` component can read it
    directly — independent of whatever MDS/positions path runs.
    """

    def __init__(
        self,
        inner_model: nn.Module,
        inner_out_dim: int,
        embed_dim: int = 8,
        mds_align: bool = True,
        anisotropic_gating: bool = False,
        mds_align_gradient: bool = False,
    ) -> None:
        super().__init__()
        self.inner_model = inner_model
        self.embed_dim = int(embed_dim)
        self.mds_align = bool(mds_align)
        self.anisotropic_gating = bool(anisotropic_gating)
        # When False (the legacy default), the MDS-aligned positions
        # are produced under ``torch.no_grad()`` and overwrite
        # ``pred.positions`` with a gradient-FREE tensor. That's
        # conservative for backward stability (eigh's backward has
        # 1/(λ_i - λ_j) terms that NaN at near-degenerate eigenvalues),
        # but it SILENTLY silences any loss component computed on
        # ``pred.positions`` (sinkhorn, shape_matching, etc.) —
        # the loss value still moves because the underlying D_sq
        # improves under edm_distance_mse, but the gradient w.r.t.
        # the model weights is zero through this path. Result: all
        # runs comparing different position-space auxiliary losses
        # produce IDENTICAL final weights (only edm_distance_mse
        # drives training).
        #
        # When True, MDS runs WITH gradient and a NaN-guard hook is
        # registered on the output. The hook intercepts NaN/Inf
        # gradients (from eigh backward's pathological cases) and
        # replaces them with zeros — so a single slice with
        # near-degenerate eigvals at most loses its auxiliary-loss
        # gradient contribution rather than poisoning the whole
        # backward graph. The common case (well-separated eigvals)
        # gets clean gradient flow.
        self.mds_align_gradient = bool(mds_align_gradient)

        # Projector: per-cell (inner-features + position) → embedding.
        # We concat positions to inner features so the head sees the
        # current x_t frame as additional context — useful because the
        # backbone has already produced a position estimate that the
        # head can refine. inner_out_dim is typically 4 (LUNA's
        # output_features_to_pos_dims), + 2D position = 6.
        in_dim = int(inner_out_dim) + 2
        hidden = max(32, 2 * self.embed_dim)
        self.projector = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, self.embed_dim),
        )

        # Architectural extension #1 — anisotropic / Mahalanobis gating.
        # Learnable (k, k) matrix W such that M = Wᵀ W is positive
        # semi-definite by construction. Then:
        #     D_ij = (h_i − h_j)ᵀ M (h_i − h_j) = ‖W (h_i − h_j)‖²
        # Initialise W = I so the initial behaviour is byte-identical
        # to the isotropic D = ‖h_i − h_j‖² (Mahalanobis with M = I).
        # The optimizer is then free to deviate from identity if some
        # embedding dimensions are more spatially-informative than
        # others. See the config block for the full motivation.
        if self.anisotropic_gating:
            self.gating_W = nn.Parameter(torch.eye(self.embed_dim))
        else:
            # Register as None (not a Parameter) so state_dicts of
            # anisotropic-off and anisotropic-on runs differ
            # predictably (one has the key, the other doesn't).
            self.gating_W = None

        # Propagate the c2f teacher-forcing marker through the wrapper
        # chain. The LightningModule's forward checks
        # ``hasattr(self.model, "_c2f_uses_true_positions")`` (OR a
        # class-name substring) to decide whether to pass
        # ``true_positions=...`` as a kwarg. When EDM wraps c2f, the
        # class-name check fails (type(self.model) is EDMOutputWrapper)
        # so we need this attribute to keep teacher-forcing alive.
        if (
            hasattr(inner_model, "_c2f_uses_true_positions")
            or "CoarseToFineWrapper" in type(inner_model).__name__
        ):
            self._c2f_uses_true_positions = True

    # Some wrappers (CoarseToFineWrapper) take ``true_positions`` as a
    # kwarg during training. We pass kwargs through transparently.
    def forward(self, data: DataHolder, **kwargs) -> DataHolder:
        pred = self.inner_model(data, **kwargs)

        # Project per-cell features+positions to k-D embedding.
        feat_in = torch.cat([pred.node_features, pred.positions], dim=-1)
        h = self.projector(feat_in)                # (B, N, k)
        # Zero out padding cells so they don't contaminate distances.
        mask = data.node_mask.to(h.dtype).unsqueeze(-1)
        h = h * mask                                # (B, N, k)

        # Pairwise squared distances:
        #   isotropic case   (default): D_ij = ‖h_i − h_j‖²
        #   anisotropic case (#1 flag): D_ij = ‖W (h_i − h_j)‖²
        #                              = (h_i − h_j)ᵀ Wᵀ W (h_i − h_j)
        # where M = Wᵀ W is the learned Mahalanobis kernel. Init W = I
        # makes the anisotropic case start byte-identical to the
        # isotropic one; the optimizer is free to deviate.
        diff = h.unsqueeze(2) - h.unsqueeze(1)     # (B, N, N, k)
        if self.gating_W is not None:
            # Apply W to each diff vector: scaled[..., j] = sum_l W[j, l] diff[..., l]
            # einsum is the cleanest way to express this batch-of-3D-arrays
            # × (k, k) matrix contraction.
            scaled = torch.einsum("bnmk,jk->bnmj", diff, self.gating_W)
            D_sq = (scaled * scaled).sum(dim=-1)   # (B, N, N)
        else:
            D_sq = (diff * diff).sum(dim=-1)       # (B, N, N)

        # Mask padding rows/cols to zero (so they don't enter the loss).
        m1 = data.node_mask                        # (B, N)
        m2 = m1.unsqueeze(2) * m1.unsqueeze(1)     # (B, N, N) bool→0/1
        D_sq = D_sq * m2.to(D_sq.dtype)

        # Stash on pred for the loss to read. Plain attribute set on
        # the DataHolder is fine — it's a regular Python object.
        pred.edm_D = D_sq
        pred.edm_h = h

        if self.mds_align:
            if self.mds_align_gradient:
                # GRADIENT-CARRYING MDS path. The MDS-aligned positions
                # have a live autograd connection back to ``D_sq``, so
                # any loss computed on ``pred.positions`` (sinkhorn,
                # shape_matching, pairwise_distance_mse-on-positions)
                # actually trains the model. Without this flag, those
                # losses are silently no-ops — their values move
                # because ``D_sq`` improves under edm_distance_mse,
                # but their gradient w.r.t. the weights is zero, so
                # they contribute nothing to training. Enabling this
                # flag is the difference between "loss curves on
                # wandb look reasonable" and "the auxiliary loss
                # actually drives the model toward what it measures".
                #
                # Why it was off by default: eigh's backward formula
                # has ``1/(λ_i − λ_j)`` terms that diverge at near-
                # degenerate eigenvalues. We install a NaN-guard hook
                # on the MDS output below that replaces any NaN/Inf
                # gradient component with zero — so a single
                # degenerate slice at most loses its own auxiliary-
                # loss gradient contribution rather than poisoning the
                # whole backward graph. The common case (well-
                # separated eigvals on real point clouds) flows
                # through cleanly.
                new_pos = self._mds_align_positions(
                    D_sq, pred.positions, data.node_mask,
                )
                new_pos = new_pos * data.node_mask.unsqueeze(-1).to(new_pos.dtype)
                # Hook MUST be registered on a tensor that
                # requires_grad. Skip in eval / under no_grad context
                # (e.g. inference), where the tensor won't have grad.
                if new_pos.requires_grad:
                    def _nan_to_zero(grad):
                        # Replace NaN / ±Inf with 0. nan_to_num is
                        # out-of-place and autograd-compatible.
                        return torch.nan_to_num(
                            grad, nan=0.0, posinf=0.0, neginf=0.0,
                        )
                    new_pos.register_hook(_nan_to_zero)
            else:
                # Legacy detached path. The MDS step uses
                # ``torch.linalg.eigh``, whose backward formula has
                # ``1/(λ_i − λ_j)`` terms that diverge to ±inf when
                # eigenvalues are close. Even paths that produce a
                # gradient of 0 at the OUTPUT (e.g. the cluster_balance
                # fallback ``pred.positions.sum() * 0.0``) then compute
                # ``0 × inf = NaN`` in autograd's chain-rule traversal,
                # which poisons the entire backward. Detaching makes
                # this slice gradient-free.
                # NOTE: under this path, auxiliary position-space
                # losses (sinkhorn, shape_matching, etc.) do NOT
                # train the model — their gradient through pred.positions
                # is identically zero. Set mds_align_gradient=True to
                # restore that gradient signal.
                with torch.no_grad():
                    new_pos = self._mds_align_positions(
                        D_sq.detach(), pred.positions.detach(), data.node_mask,
                    )
                new_pos = new_pos * data.node_mask.unsqueeze(-1).to(new_pos.dtype)
            pred.positions = new_pos

        return pred

    # ------------------------------------------------------------------
    # MDS + Procrustes per slice (B is typically 1 for our datasets)
    # ------------------------------------------------------------------
    def _mds_align_positions(
        self,
        D_sq: torch.Tensor,
        x_ref: torch.Tensor,
        node_mask: torch.Tensor,
    ) -> torch.Tensor:
        """For each slice, compute classical MDS from D_sq, then
        Procrustes-align to x_ref's frame. Returns (B, N, 2).

        Per-slice loop because n_valid varies across the batch. B=1
        for our small-slice datasets so the loop overhead is moot.
        Constructs each slice as a fresh tensor and stacks at the
        end to keep autograd happy (no in-place index assignment
        into a single pre-allocated output tensor).
        """
        B, N, _ = x_ref.shape
        aligned_list = []
        for b in range(B):
            m = node_mask[b]
            valid_idx = torch.nonzero(m, as_tuple=False).squeeze(-1)
            n_valid = int(valid_idx.numel())
            if n_valid < 3:
                # Not enough points for MDS — fall back to ref frame.
                aligned_list.append(x_ref[b])
                continue
            D_v = D_sq[b].index_select(0, valid_idx).index_select(1, valid_idx)
            try:
                x_mds = _classical_mds_2d(D_v)            # (n_valid, 2)
            except Exception:
                aligned_list.append(x_ref[b])
                continue
            x_ref_v = x_ref[b].index_select(0, valid_idx)  # (n_valid, 2)
            # Centre both (x_ref is already mean-zero from DataHolder
            # but be defensive).
            x_ref_c = x_ref_v - x_ref_v.mean(dim=0, keepdim=True)
            x_mds_c = x_mds - x_mds.mean(dim=0, keepdim=True)
            # framework=regression zeroes the input positions, so
            # x_ref_c is all zeros — Procrustes SVD on a 2x2 zero
            # matrix is degenerate (singular values all zero, arbitrary
            # U/V). In that case skip alignment and use the raw MDS
            # eigenframe. Metric-wise this is fine (Spearman / pairwise
            # MSE are frame-invariant); visualization frame is
            # arbitrary but consistent across samples for a given
            # checkpoint. Threshold: 1e-8 covers both genuine zero and
            # negligible numerical noise from mean-subtraction.
            ref_norm = x_ref_c.abs().mean().item()
            if ref_norm < 1e-8:
                x_aligned = x_mds_c
            else:
                try:
                    x_aligned = _procrustes_align(x_mds_c, x_ref_c)  # (n_valid, 2)
                except Exception:
                    aligned_list.append(x_ref[b])
                    continue
            # Scatter back to (N, 2) without breaking the graph:
            # build a zero tensor + index_copy (the out-of-place
            # variant), which autograd treats as a fresh node.
            padded = torch.zeros(N, 2, device=x_ref.device, dtype=x_ref.dtype)
            padded = padded.index_copy(0, valid_idx, x_aligned)
            aligned_list.append(padded)
        return torch.stack(aligned_list, dim=0)
