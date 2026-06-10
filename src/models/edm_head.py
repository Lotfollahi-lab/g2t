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

import contextlib
from typing import Optional

import torch
import torch.nn as nn

from utils.data.dataholder import DataHolder


# ---------------------------------------------------------------------------
# Differentiable classical MDS + Procrustes alignment
# ---------------------------------------------------------------------------


def _classical_mds_2d(
    D_sq: torch.Tensor, tikhonov_eps: float = 1e-6,
) -> torch.Tensor:
    """Differentiable classical MDS to 2D from a squared-distance matrix.

    Args:
        D_sq: (n, n) symmetric, nonneg squared-distance matrix.
        tikhonov_eps: Per-eigenvalue spacing target for the
            Tikhonov regularization, expressed as a fraction of
            ``max(|B|)``. The diagonal of B gets a strictly-
            increasing perturbation ``arange(n) * tikhonov_eps * scale``
            (NO `/n` divisor) so adjacent eigenvalues are separated
            by at least ``tikhonov_eps * scale`` after the bump.
            Default 1e-6 = "MDS precision priority". Bump to 1e-4
            or 1e-3 when running with ``mds_align_gradient=true``
            on large clouds (n ≥ 5000) — eigh's backward formula
            contains ``1/(λ_i − λ_j)`` terms that diverge to ±Inf
            at degenerate eigenvalues, and the 1e-6 default gives
            spacing too tight (~1e-6 · scale) for fp32 chain-rule
            traversal to stay finite when a position-based loss
            (sinkhorn / chamfer / shape) puts a gradient sink on
            the MDS output.

    Returns:
        (n, 2) MDS coordinates. Frame is the eigenframe of the
        double-centred Gram matrix — defined up to reflection/rotation.

    Note (2026-05-27): the prior implementation divided the
    perturbation by ``n``, which at cortex's n≈7000 gave
    per-eigenvalue spacing of ~1.4e-10. autograd's eigvec-Jacobian
    then contained 1/(λ_i − λ_j) terms ≈ 7×10⁹ — high enough that
    chain-rule traversal overflowed fp32 when ANY loss put a
    gradient sink on the MDS output. The new version drops the
    `/n` divisor; precision impact stays bounded because the
    top-2 eigenvectors are the most-separated pair so their
    perturbation is dominated by the data signal, not by the
    Tikhonov tag.
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

    # Tikhonov regularization to break eigenvalue degeneracy.
    # Per-eigenvalue spacing is ``tikhonov_eps * scale`` where
    # scale = max(|B|). No `/n` divisor — that's the fix for the
    # 2026-05-27 NaN bug.
    scale = B.diag().abs().max().clamp_min(1.0)
    eps_reg = float(tikhonov_eps) * scale
    reg = torch.arange(n, device=device, dtype=dtype) * eps_reg
    B = B + torch.diag(reg)

    # Eigendecomposition (ascending). Top 2 = last 2.
    evals, evecs = torch.linalg.eigh(B)
    e2 = evals[-2:].clamp(min=1e-12)              # (2,) nonneg
    v2 = evecs[:, -2:]                            # (n, 2)
    # MDS coords: V * sqrt(Λ).
    return v2 * torch.sqrt(e2).unsqueeze(0)       # (n, 2)


def _classical_mds_2d_lobpcg(
    D_sq: torch.Tensor, tikhonov_eps: float = 1e-6,
) -> torch.Tensor:
    """Truncated classical MDS via ``torch.lobpcg`` — top-2 eigenpairs
    only.

    Faster alternative to ``_classical_mds_2d`` (which uses
    ``torch.linalg.eigh`` and computes ALL N eigenpairs). For MDS we
    only need the top-2 (largest) eigenpairs; lobpcg with k=2 is
    O(N²·iters) vs eigh's O(N³), giving a 10-20× speedup at N≈46k.

    IMPORTANT: ``torch.lobpcg`` does NOT have a stable autograd
    backward path on all torch versions. This function is therefore
    intended for use ONLY when no gradient flows through MDS — i.e.
    ``mds_align_gradient=False`` AND/OR no position-side loss is
    active. The EDMOutputWrapper enforces this gating when
    ``mds_solver='lobpcg'``.

    Same Tikhonov / clamp / sqrt structure as ``_classical_mds_2d``
    so the two are byte-comparable up to numerical convergence of
    lobpcg.
    """
    n = D_sq.shape[0]
    device = D_sq.device
    dtype = D_sq.dtype

    # Same double-centring as the eigh path.
    one = torch.ones((n, n), device=device, dtype=dtype) / n
    J = torch.eye(n, device=device, dtype=dtype) - one
    B = -0.5 * J @ D_sq @ J
    B = 0.5 * (B + B.T)

    # Same Tikhonov (no /n divisor — see _classical_mds_2d docstring).
    scale = B.diag().abs().max().clamp_min(1.0)
    eps_reg = float(tikhonov_eps) * scale
    reg = torch.arange(n, device=device, dtype=dtype) * eps_reg
    B = B + torch.diag(reg)

    # lobpcg: top-2 (largest=True). Returns (eigvals, eigvecs) with
    # eigvals in descending order on the first axis.
    evals, evecs = torch.lobpcg(B, k=2, largest=True)
    # evals: (2,), evecs: (n, 2). Clamp + sqrt for MDS coords.
    e2 = evals.clamp(min=1e-12)
    return evecs * torch.sqrt(e2).unsqueeze(0)                    # (n, 2)


def _procrustes_align(x_src: torch.Tensor, x_ref: torch.Tensor) -> torch.Tensor:
    """Orthogonal Procrustes: find the rotation+reflection R that minimises
    ‖x_src @ R − x_ref‖_F. Returns x_src @ R with R DETACHED.

    Both inputs assumed to be mean-centred. Reflection is allowed
    (full O(2) alignment, not just SO(2)) — biologically symmetric
    tissue + we just want the canonical frame.

    Derivation: with M = x_src^T @ x_ref and SVD M = U Σ Vᵀ, the
    closed-form solution is R = U Vᵀ (Schönemann 1966). PyTorch's
    ``linalg.svd`` returns ``Vh = Vᵀ`` directly, so the formula is
    ``R = U @ Vh``.

    Gradient pattern: R is computed UNDER ``torch.no_grad()`` and
    detached before ``x_src @ R``. Gradient w.r.t. ``x_src`` flows
    as if R were a constant rotation:
        d L / d x_src = (d L / d (x_src @ R)) @ R^T
    "Fixed-structure differentiable alignment" — R is recomputed
    fresh every forward from the CURRENT x_src and x_ref, so the
    alignment stays current; we just don't let gradient
    back-propagate through the SVD that produced R.

    Why we detach R (2026-05-27 finding)
    ------------------------------------
    SVD's backward formula has ``1/(σ_i² − σ_j²)`` terms that
    diverge at degenerate or near-degenerate singular values. The
    earlier snapshot-threshold check (M_max, σ_min/σ_max,
    (σ_max−σ_min)/σ_max < 1e-3) was NECESSARY-but-not-sufficient:
    it caught only obvious degenerate cases. The much more common
    failure was the STEADY-STATE regime once the model converges
    enough that MDS layout ≈ reference layout — M is then nearly a
    scaled identity, σ_max ≈ σ_min, and the SVD backward is
    intrinsically ill-conditioned. Observed wandb signature: bound
    loss values (sinkhorn ~0.015, chamfer same), NaN gradient
    raised by ``on_after_backward``.

    Detaching R sidesteps the SVD-backward entirely. Geometric
    meaning preserved: we still rotate MDS output to align with the
    reference frame; we just don't propagate gradient through that
    rotation choice. The MDS layout is intrinsically
    rotation-ambiguous anyway (every rotation of the eigvecs gives
    the same D), so there's no learnable signal in the rotation —
    detaching it doesn't lose information. Same fix already applied
    to ``metrics.loss_function._procrustes_align_2d`` (bug #105)
    and ``models.knn_graph_head._procrustes_align``.
    """
    # Compute the entire alignment under no_grad. Forward arithmetic
    # is identical; only the autograd graph changes. The M-near-zero
    # short-circuit is kept for defense-in-depth: when M is exactly
    # zero the SVD's forward itself can return NaN in U/V on some
    # builds (not just its backward), so we skip the SVD call.
    with torch.no_grad():
        M_detached = (x_src.detach().T @ x_ref.detach())     # (2, 2)
        M_max = M_detached.abs().max()
        if (not torch.isfinite(M_max).item()) or M_max.item() < 1e-6:
            return x_src
        # SVD's own forward is well-defined for non-zero M even at
        # σ_max ≈ σ_min — only its BACKWARD has the divergence,
        # which we sidestep by being under no_grad here.
        U, _S, Vh = torch.linalg.svd(M_detached)
        R = (U @ Vh).detach()                                 # (2, 2)
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
        mds_tikhonov_eps: float = 1e-6,
        mds_align_train: bool = True,
        mds_dtype: str = "fp64",
        mds_solver: str = "eigh",
        skip_edm_D_train: bool = False,
        fp32_geometry: bool = True,
    ) -> None:
        super().__init__()
        self.inner_model = inner_model
        self.embed_dim = int(embed_dim)
        self.mds_align = bool(mds_align)
        self.anisotropic_gating = bool(anisotropic_gating)
        # Sparse-training fast path. When True, the forward pass does NOT
        # materialise the (B, N, N) squared-distance matrix ``D_sq``
        # during TRAINING — it only sets ``pred.edm_h`` (the small
        # (B, N, k) embedding). This is the enabler for the
        # O(N·k) ``sparse_local_distance`` loss, which reads ``edm_h``
        # and computes distances on each cell's k-NN only, never on the
        # full pairwise matrix. The EDM head's ``D_sq`` is otherwise the
        # last remaining O(N²) term in the training step (the backbone
        # can already be sub-quadratic, e.g. geomattn_localglobal), so
        # skipping it is what actually buys end-to-end scalability.
        # The FM Euler step reads ``pred.edm_h`` (not ``edm_D``), so the
        # generative path is unaffected; at INFERENCE (eval mode) the
        # full ``D_sq`` + MDS always run so ``pred.positions`` stays the
        # canonical layout. SAFETY: only meaningful when no loss reads
        # ``pred.edm_D`` at train time — i.e. ``edm_distance_mse`` is off
        # (``model.edm.loss_weight=0``) and ``sparse_local_distance`` is
        # the active distance loss. The diffusion_model auto-config
        # raises a loud error otherwise.
        self.skip_edm_D_train = bool(skip_edm_D_train)
        # Surgical fp32 island for the geometry-sensitive head. When True
        # (default), the projector → pairwise-distance → MDS path runs in
        # fp32 even under a bf16-mixed autocast, while the backbone stays
        # bf16. Rationale: bf16's ~3 significant digits coarsen predicted
        # distances and scramble the fine neighbour orderings the per-cell
        # Spearman metric measures — a global-bf16 run regressed badly, so
        # we keep the distance math fp32. No-op under pure-fp32 training
        # (autocast is off, so the disabled-autocast context does nothing).
        self.fp32_geometry = bool(fp32_geometry)
        # MDS solver knobs.
        # ``mds_dtype``:
        #   "fp64" (default) — promote D_v to float64 before eigh.
        #     Required when mds_align_gradient=True because eigh's
        #     backward formula needs fp64 precision to keep
        #     1/(λ_i − λ_j) finite at near-degenerate spectra.
        #   "fp32" — skip the promotion. ~2× faster; safe ONLY when
        #     no backward goes through eigh (i.e. mds_align_gradient
        #     False AND/OR mds_align_train False).
        # ``mds_solver``:
        #   "eigh" (default) — torch.linalg.eigh, computes ALL N
        #     eigenpairs (O(N³)). Has autograd backward.
        #   "lobpcg" — torch.lobpcg, top-2 eigenpairs only
        #     (O(N²·iter)). 10-20× faster at large N. NO stable
        #     backward — safe only when mds_align_gradient=False.
        self.mds_dtype = str(mds_dtype).lower()
        self.mds_solver = str(mds_solver).lower()
        if self.mds_dtype not in ("fp32", "fp64"):
            raise ValueError(
                f"mds_dtype must be 'fp32' or 'fp64'; got "
                f"{self.mds_dtype!r}"
            )
        if self.mds_solver not in ("eigh", "lobpcg"):
            raise ValueError(
                f"mds_solver must be 'eigh' or 'lobpcg'; got "
                f"{self.mds_solver!r}"
            )
        # Fail-loud if mds_align_gradient is True AND user requested
        # a backward-unsafe setting. eigh-fp32 backward can produce
        # NaN; lobpcg backward isn't stable on all torch versions.
        if bool(mds_align_gradient):
            if self.mds_dtype == "fp32":
                raise ValueError(
                    "model.edm.mds_dtype='fp32' is unsafe when "
                    "mds_align_gradient=True — eigh's backward needs "
                    "fp64 precision to avoid NaN at near-degenerate "
                    "eigenvalues. Either disable mds_align_gradient "
                    "or use mds_dtype='fp64' (default)."
                )
            if self.mds_solver == "lobpcg":
                raise ValueError(
                    "model.edm.mds_solver='lobpcg' is unsafe when "
                    "mds_align_gradient=True — lobpcg has no stable "
                    "autograd backward. Either disable "
                    "mds_align_gradient or use mds_solver='eigh' "
                    "(default)."
                )
        # Whether to run MDS+Procrustes during TRAINING. Set False to
        # skip the O(N³) eigh during training when no loss reads
        # ``pred.positions`` (typical for EDM-only runs where
        # ``edm_distance_mse`` is the sole position-related loss — it
        # reads pred.edm_D directly, not pred.positions). MDS still
        # runs at inference (eval mode) regardless. Default True
        # preserves the historic behaviour for runs that DO use
        # position-side losses (sinkhorn / chamfer / shape /
        # pairwise_distance_mse). Massive speedup at CNS scale:
        # eigh on a 46k×46k matrix is multiple seconds per step.
        self.mds_align_train = bool(mds_align_train)
        # Per-eigenvalue Tikhonov perturbation in classical MDS.
        # Forwarded to ``_classical_mds_2d`` so the strength is set
        # per-run rather than hardcoded. See the docstring in that
        # function for the math; in short, this controls the minimum
        # spacing between adjacent eigenvalues of B before eigh, and
        # 1/(λ_i − λ_j) terms in eigh's backward are bounded by
        # 1/(tikhonov_eps · scale). At cortex's n≈7000, default
        # 1e-6 gives backward magnitudes ~1e6/scale that can
        # overflow fp32 in chain rule; bump to 1e-4 or 1e-3 when
        # using ``mds_align_gradient=true`` with a position-based
        # loss. Default kept at 1e-6 for backward-compat with
        # mds_align_gradient=false runs (where MDS-backward isn't
        # exercised anyway, so precision wins).
        self.mds_tikhonov_eps = float(mds_tikhonov_eps)
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
    def _fp32_ctx(self, ref: torch.Tensor):
        """Autocast-disabled (fp32) context for the geometry-sensitive
        ops, active when ``fp32_geometry`` is on. Under a bf16-mixed
        autocast this keeps the projector → pairwise-distance → MDS path
        in fp32 (autocast otherwise downcasts Linear/bmm to bf16
        regardless of input dtype, coarsening the distances the per-cell
        rank metric depends on). A no-op in pure-fp32 training."""
        if self.fp32_geometry:
            return torch.autocast(device_type=ref.device.type, enabled=False)
        return contextlib.nullcontext()

    def forward(self, data: DataHolder, **kwargs) -> DataHolder:
        pred = self.inner_model(data, **kwargs)

        # Project per-cell features+positions to k-D embedding.
        # fp32 ISLAND (see _fp32_ctx + fp32_geometry): cast inputs to fp32
        # and disable autocast so the projector runs fp32 even under
        # bf16-mixed; the backbone above keeps its bf16 speed.
        with self._fp32_ctx(pred.node_features):
            feat = pred.node_features
            pos = pred.positions
            if self.fp32_geometry:
                feat = feat.float()
                pos = pos.float()
            feat_in = torch.cat([feat, pos], dim=-1)
            h = self.projector(feat_in)            # (B, N, k)
        # Zero out padding cells so they don't contaminate distances.
        mask = data.node_mask.to(h.dtype).unsqueeze(-1)
        h = h * mask                                # (B, N, k)

        # Sparse-training fast path — skip the full (B, N, N) matrix.
        # See the ``skip_edm_D_train`` comment in __init__. ``edm_D`` is
        # set to None so any loss that still reads it (via the
        # ``getattr(masked_pred, "edm_D", None)`` guard) degrades to its
        # gradient-zero fallback rather than crashing. The active
        # distance loss (sparse_local_distance) reads ``edm_h`` instead.
        if self.training and self.skip_edm_D_train:
            pred.edm_h = h
            pred.edm_D = None
            return pred

        # Pairwise squared distances:
        #   isotropic case   (default): D_ij = ‖h_i − h_j‖²
        #   anisotropic case (#1 flag): D_ij = ‖W (h_i − h_j)‖²
        #                              = (h_i − h_j)ᵀ Wᵀ W (h_i − h_j)
        # where M = Wᵀ W is the learned Mahalanobis kernel. Init W = I
        # makes the anisotropic case start byte-identical to the
        # isotropic one; the optimizer is free to deviate.
        # Memory-efficient pairwise squared distances.
        #
        # Naive formulation ``diff = h.u(2) - h.u(1); (diff*diff).sum(-1)``
        # materialises a (B, N, N, k) tensor of size 4·B·N²·k bytes.
        # At CNS scale (N ≈ 46k, k = 8) that's 68 GB FOR THE FORWARD
        # ALONE, plus an equal-size activation stash for backward —
        # OOMs even an H200 (139 GB) at batch_size=1.
        #
        # Algebraic identity: ||a − b||² = ||a||² + ||b||² − 2·a·b.
        # The (B, N, k)·(B, k, N) → (B, N, N) matmul is computed
        # without any intermediate tensor larger than the final
        # output (8.5 GB at N=46k vs 68 GB). 8× peak-memory reduction;
        # same forward value, same gradient direction. The matmul's
        # backward only stashes the inputs (B, N, k) which is small.
        #
        # Anisotropic case: apply W to h first, then use the same
        # identity on hW. Mathematically equivalent to the
        # (h_i − h_j)ᵀ Wᵀ W (h_i − h_j) formulation.
        # fp32 ISLAND — the distance matmul (bmm) is autocast-downcast to
        # bf16 otherwise; h is already fp32 from the projector island.
        with self._fp32_ctx(h):
            if self.gating_W is not None:
                # h @ W^T per cell: (B, N, k) @ (k, k) → (B, N, k_out)
                hW = torch.einsum("bnk,jk->bnj", h, self.gating_W)   # (B, N, k)
            else:
                hW = h
            norms = (hW * hW).sum(dim=-1)                            # (B, N)
            # outer sum + −2·dot. clamp_min(0) guards against fp32
            # round-off producing tiny negatives when h_i ≈ h_j.
            D_sq = (
                norms.unsqueeze(2) + norms.unsqueeze(1)
                - 2.0 * torch.bmm(hW, hW.transpose(1, 2))
            ).clamp_min(0.0)                                          # (B, N, N)

        # Mask padding rows/cols to zero (so they don't enter the loss).
        m1 = data.node_mask                        # (B, N)
        m2 = m1.unsqueeze(2) * m1.unsqueeze(1)     # (B, N, N) bool→0/1
        D_sq = D_sq * m2.to(D_sq.dtype)

        # Stash on pred for the loss to read. Plain attribute set on
        # the DataHolder is fine — it's a regular Python object.
        pred.edm_D = D_sq
        pred.edm_h = h

        # Skip MDS entirely during TRAINING when no loss reads
        # pred.positions. The wrapper's auto-config in
        # ``diffusion_model.py`` disables ``pairwise_distance_mse``
        # whenever EDM is on, and the other position-side losses
        # (sinkhorn / chamfer / shape) default off. With only
        # ``edm_distance_mse`` active (which reads ``pred.edm_D``,
        # not ``pred.positions``), the MDS step is pure overhead —
        # an O(N³) eigh per slice per step. At CNS scale (N≈46k)
        # this dominates training wall time. Set
        # ``mds_align_train=false`` to skip it during training; at
        # inference (eval mode) MDS always runs so pred.positions
        # is the canonical layout for downstream metrics + plots.
        skip_mds_for_training = self.training and not self.mds_align_train
        if self.mds_align and not skip_mds_for_training:
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
                # The eigh inside ``_classical_mds_2d`` has a
                # backward formula containing ``1/(λ_i − λ_j)`` terms
                # that diverge at near-degenerate eigenvalues. We
                # stabilise this AT THE SOURCE (Tikhonov-style
                # strictly-increasing diagonal perturbation inside
                # ``_classical_mds_2d``) — not via a backward hook
                # that hides NaN gradients. Silent NaN-to-zero was
                # tried in an earlier revision; it produced
                # wrong-but-finite gradients with no user signal,
                # exactly the silent-silencing pattern the codebase
                # now explicitly rejects. If a NaN gradient still
                # appears here (e.g., from a new auxiliary loss with
                # an unstable backward), the
                # LightningModule.on_after_backward detector raises
                # a loud RuntimeError naming the offending parameter.
                # fp32 ISLAND — MDS double-centering bmm + eigh on fp32.
                with self._fp32_ctx(D_sq):
                    new_pos = self._mds_align_positions(
                        D_sq, pred.positions, data.node_mask,
                    )
                new_pos = new_pos * data.node_mask.unsqueeze(-1).to(new_pos.dtype)
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
                # fp32 ISLAND — MDS double-centering bmm + eigh on fp32.
                with torch.no_grad(), self._fp32_ctx(D_sq):
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
            # Dtype: fp64 (default) for backward stability of eigh
            # when mds_align_gradient=True; fp32 for ~2× speed when
            # no backward goes through eigh. The constructor's
            # fail-loud guards prevent fp32 + mds_align_gradient=True
            # (which would NaN). See the __init__ comment for the
            # full mathematics.
            if self.mds_dtype == "fp64":
                D_v_proc = D_v.to(torch.float64)
            else:  # fp32
                D_v_proc = D_v if D_v.dtype == torch.float32 else D_v.to(torch.float32)
            # Solver: eigh (default, full N×N decomposition) or
            # lobpcg (top-2 only, much faster at large N).
            mds_fn = (
                _classical_mds_2d_lobpcg if self.mds_solver == "lobpcg"
                else _classical_mds_2d
            )
            try:
                x_mds_proc = mds_fn(                  # (n_valid, 2)
                    D_v_proc, tikhonov_eps=self.mds_tikhonov_eps,
                )
                # Cast back to the surrounding graph's dtype.
                x_mds = x_mds_proc.to(D_v.dtype)
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
