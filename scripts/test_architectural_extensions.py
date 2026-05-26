#!/usr/bin/env python
"""Lite smoke tests for the five architectural extensions added in
the 2026-05-26 batch:

  #1 Bilinear / anisotropic gating in the EDM head
  #2 DiT-style backbone
  #3 Perceiver-style backbone
  #4 True EDM diffusion (FM in h-space)
  #5 Latent diffusion (scaffold; just checks the stub raises cleanly)

Each test runs the actual code path with realistic-ish synthetic data
on the CPU. Doesn't require scanpy / torch_geometric / hydra — only
torch + the DataHolder dataclass.

Run from the repo root:

    python scripts/test_architectural_extensions.py

Exits 0 on pass, non-zero on the first failure.
"""

from __future__ import annotations

import sys
from pathlib import Path


def _setup_path() -> None:
    here = Path(__file__).resolve()
    scgg_src = here.parent.parent / "src"
    if not scgg_src.exists():
        raise FileNotFoundError(f"scgg/src not found at {scgg_src}")
    sys.path.insert(0, str(scgg_src))


_setup_path()

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from utils.data.dataholder import DataHolder  # noqa: E402


# ---------------------------------------------------------------------------
# Stub inner model — same as scripts/test_edm_knn_lite.py uses
# ---------------------------------------------------------------------------


class _StubInner(nn.Module):
    """Mimics LUNA Model's DataHolder-in / DataHolder-out contract with a
    trivial Linear projection. Used to feed the EDMOutputWrapper's
    anisotropic gating test (which doesn't depend on the inner backbone)
    without dragging in scanpy through models/model.py's imports.
    """

    def __init__(self, in_features: int, out_features: int = 4):
        super().__init__()
        self.feat_proj = nn.Linear(in_features, out_features)
        self.pos_proj = nn.Linear(in_features, 2)

    def forward(self, data: DataHolder, **kwargs):
        out_features = self.feat_proj(data.node_features)
        out_positions = self.pos_proj(data.node_features)
        m = data.node_mask.unsqueeze(-1).to(out_features.dtype)
        out_features = out_features * m
        out_positions = out_positions * m
        out_positions = out_positions - out_positions.mean(dim=1, keepdim=True)
        out_positions = out_positions * m
        return DataHolder(
            node_features=out_features,
            positions=out_positions,
            diffusion_time=data.diffusion_time,
            cell_class=data.cell_class,
            cell_ID=data.cell_ID,
            t_int=data.t_int,
            t=data.t,
            node_mask=data.node_mask,
        )


def _build_data(B: int = 1, N: int = 24, gene_dim: int = 16) -> DataHolder:
    """Build a DataHolder matching ``to_batch``'s output shape:
    cell_class is (B, N, 1) — the apply_mask broadcast in
    DataHolder.mask() requires the trailing singleton dim."""
    torch.manual_seed(0)
    return DataHolder(
        node_features=torch.randn(B, N, gene_dim),
        positions=torch.randn(B, N, 2) * 0.3,
        diffusion_time=torch.rand(B, 1),
        cell_class=torch.zeros(B, N, 1, dtype=torch.long),
        cell_ID=torch.arange(N).repeat(B, 1).unsqueeze(-1),
        t_int=torch.zeros(B, 1, dtype=torch.long),
        t=torch.rand(B, 1),
        node_mask=torch.ones(B, N, dtype=torch.bool),
    )


# ---------------------------------------------------------------------------
# #1 — Bilinear / anisotropic gating in the EDM head
# ---------------------------------------------------------------------------


def test_anisotropic_gating_init_is_identity() -> None:
    """With W initialised to I, the anisotropic gating path produces
    EXACTLY the same D as the isotropic path. Catches regressions
    in the init or the einsum contraction."""
    from models.edm_head import EDMOutputWrapper

    B, N = 1, 20
    inner = _StubInner(in_features=16, out_features=4)
    wrap_iso = EDMOutputWrapper(
        inner_model=inner, inner_out_dim=4, embed_dim=8,
        mds_align=False, anisotropic_gating=False,
    )
    wrap_aniso = EDMOutputWrapper(
        inner_model=inner, inner_out_dim=4, embed_dim=8,
        mds_align=False, anisotropic_gating=True,
    )
    # Share projector weights so the two wrappers produce the same h.
    wrap_aniso.projector.load_state_dict(wrap_iso.projector.state_dict())

    data = _build_data(B=B, N=N)
    with torch.no_grad():
        pred_iso = wrap_iso(data)
        pred_aniso = wrap_aniso(data)
    err = (pred_iso.edm_D - pred_aniso.edm_D).abs().max().item()
    assert err < 1e-5, (
        f"anisotropic gating with W=I should match isotropic; max diff {err}"
    )


def test_anisotropic_gating_D_is_symmetric_nonneg() -> None:
    """For any (random) W, D = ‖W(h_i - h_j)‖² is symmetric and >= 0."""
    from models.edm_head import EDMOutputWrapper

    B, N = 1, 20
    inner = _StubInner(in_features=16, out_features=4)
    wrap = EDMOutputWrapper(
        inner_model=inner, inner_out_dim=4, embed_dim=8,
        mds_align=False, anisotropic_gating=True,
    )
    # Perturb W away from identity.
    with torch.no_grad():
        wrap.gating_W.add_(torch.randn_like(wrap.gating_W) * 0.5)
    data = _build_data(B=B, N=N)
    pred = wrap(data)
    D = pred.edm_D
    assert torch.allclose(D, D.transpose(-1, -2), atol=1e-5), \
        "D not symmetric under anisotropic gating"
    assert (D >= -1e-6).all(), "D should be non-negative"


def test_anisotropic_gating_gradient_flows_to_W() -> None:
    """W receives non-zero gradient when the loss depends on D."""
    from models.edm_head import EDMOutputWrapper

    B, N = 1, 20
    inner = _StubInner(in_features=16, out_features=4)
    wrap = EDMOutputWrapper(
        inner_model=inner, inner_out_dim=4, embed_dim=6,
        mds_align=False, anisotropic_gating=True,
    )
    data = _build_data(B=B, N=N)
    pred = wrap(data)
    # Synthetic loss on D.
    target_d = torch.cdist(data.positions[0], data.positions[0])
    pred_d = (pred.edm_D[0] + 1e-8).sqrt()
    loss = ((pred_d - target_d) ** 2).mean()
    loss.backward()
    g = wrap.gating_W.grad
    assert g is not None and g.abs().sum().item() > 0, \
        "no gradient reached the anisotropic gating W"


# ---------------------------------------------------------------------------
# #2 — DiT-style backbone
# ---------------------------------------------------------------------------


def _build_full_data(B: int = 2, N: int = 24, gene_dim: int = 16) -> DataHolder:
    """Variant with padding cells so we can check mask handling."""
    torch.manual_seed(1)
    nm = torch.ones(B, N, dtype=torch.bool)
    nm[0, -4:] = False  # 4 padding cells in slice 0
    return DataHolder(
        node_features=torch.randn(B, N, gene_dim),
        positions=torch.randn(B, N, 2) * 0.3,
        diffusion_time=torch.rand(B, 1),
        cell_class=torch.zeros(B, N, 1, dtype=torch.long),
        cell_ID=torch.arange(N).repeat(B, 1).unsqueeze(-1),
        t_int=torch.zeros(B, 1, dtype=torch.long),
        t=torch.rand(B, 1),
        node_mask=nm,
    )


def _build_input_output_dims(gene_dim: int = 16, out_features: int = 4):
    input_dims = {
        "node_features_dimensions": gene_dim,
        "diffusion_time_dimensions": 1,
    }
    hidden_mlp_dims = {"X": 32, "y": 32, "pos": 16}
    hidden_dims = {
        "dx": 32, "dy": 1, "num_heads": 4,
        "dim_ffX": 32, "dim_ffy": 32, "dd": 16,
        "output_features_to_pos_dims": out_features,
    }
    output_dims = {
        "node_features_dimensions": gene_dim,
        "diffusion_time_dimensions": 0,
    }
    return input_dims, hidden_mlp_dims, hidden_dims, output_dims


class _DiTCfg:
    """Mimic an omegaconf DictConfig with .get / attribute access."""

    def __init__(self, **kw):
        self.__dict__.update(kw)

    def get(self, k, default=None):
        return getattr(self, k, default)


def test_dit_backbone_forward_shapes() -> None:
    from models.dit_backbone import DiTBackbone

    B, N = 2, 24
    input_dims, hidden_mlp_dims, hidden_dims, output_dims = (
        _build_input_output_dims(gene_dim=16, out_features=4)
    )
    model = DiTBackbone(
        input_dims=input_dims,
        n_layers=2,
        hidden_mlp_dims=hidden_mlp_dims,
        hidden_dims=hidden_dims,
        output_dims=output_dims,
        dit_cfg=_DiTCfg(hidden_dim=32, n_layers=2, n_heads=4,
                        mlp_ratio=2, time_embed_dim=64),
    )
    data = _build_full_data(B=B, N=N)
    pred = model(data)
    assert pred.node_features.shape == (B, N, 4), \
        f"node_features shape {pred.node_features.shape}"
    assert pred.positions.shape == (B, N, 2), \
        f"positions shape {pred.positions.shape}"
    assert torch.isfinite(pred.node_features).all()
    assert torch.isfinite(pred.positions).all()


def test_dit_backbone_padding_mask_zeros_output() -> None:
    """Padding cells must end up with zero output features and zero
    positions (matches LUNA Model's masked-output contract)."""
    from models.dit_backbone import DiTBackbone

    B, N = 2, 24
    input_dims, hidden_mlp_dims, hidden_dims, output_dims = (
        _build_input_output_dims(gene_dim=16, out_features=4)
    )
    model = DiTBackbone(
        input_dims=input_dims, n_layers=2,
        hidden_mlp_dims=hidden_mlp_dims, hidden_dims=hidden_dims,
        output_dims=output_dims,
        dit_cfg=_DiTCfg(hidden_dim=32, n_layers=2, n_heads=4,
                        mlp_ratio=2, time_embed_dim=64),
    )
    data = _build_full_data(B=B, N=N)
    pred = model(data)
    # Slice 0 has padding at indices 20..23 → those rows should be zero.
    pad_features = pred.node_features[0, 20:].abs().max().item()
    pad_positions = pred.positions[0, 20:].abs().max().item()
    assert pad_features < 1e-5, f"padding features not zero: {pad_features}"
    assert pad_positions < 1e-5, f"padding positions not zero: {pad_positions}"


def test_dit_backbone_gradient_flow() -> None:
    from models.dit_backbone import DiTBackbone

    B, N = 1, 20
    input_dims, hidden_mlp_dims, hidden_dims, output_dims = (
        _build_input_output_dims(gene_dim=16, out_features=4)
    )
    model = DiTBackbone(
        input_dims=input_dims, n_layers=2,
        hidden_mlp_dims=hidden_mlp_dims, hidden_dims=hidden_dims,
        output_dims=output_dims,
        dit_cfg=_DiTCfg(hidden_dim=32, n_layers=2, n_heads=4,
                        mlp_ratio=2, time_embed_dim=64),
    )
    data = _build_data(B=B, N=N)
    pred = model(data)
    # Synthetic loss on pairwise distances of pred positions.
    d = torch.cdist(pred.positions[0], pred.positions[0])
    d_true = torch.cdist(data.positions[0], data.positions[0])
    loss = ((d - d_true) ** 2).mean()
    loss.backward()
    total_grad = sum(
        p.grad.abs().sum().item() for p in model.parameters() if p.grad is not None
    )
    assert total_grad > 0, "no gradient flowed through DiT backbone"


# ---------------------------------------------------------------------------
# #3 — Perceiver backbone
# ---------------------------------------------------------------------------


def test_perceiver_backbone_forward_shapes() -> None:
    from models.perceiver_backbone import PerceiverBackbone

    B, N = 2, 24
    input_dims, hidden_mlp_dims, hidden_dims, output_dims = (
        _build_input_output_dims(gene_dim=16, out_features=4)
    )
    model = PerceiverBackbone(
        input_dims=input_dims, n_layers=2,
        hidden_mlp_dims=hidden_mlp_dims, hidden_dims=hidden_dims,
        output_dims=output_dims,
        perceiver_cfg=_DiTCfg(
            n_anchors=8, anchor_dim=32, cell_dim=32,
            n_anchor_blocks=2, n_heads=4, mlp_ratio=2,
            time_embed_dim=64,
        ),
    )
    data = _build_full_data(B=B, N=N)
    pred = model(data)
    assert pred.node_features.shape == (B, N, 4)
    assert pred.positions.shape == (B, N, 2)
    assert torch.isfinite(pred.node_features).all()
    assert torch.isfinite(pred.positions).all()


def test_perceiver_backbone_padding_mask_zeros_output() -> None:
    from models.perceiver_backbone import PerceiverBackbone

    B, N = 2, 24
    input_dims, hidden_mlp_dims, hidden_dims, output_dims = (
        _build_input_output_dims(gene_dim=16, out_features=4)
    )
    model = PerceiverBackbone(
        input_dims=input_dims, n_layers=2,
        hidden_mlp_dims=hidden_mlp_dims, hidden_dims=hidden_dims,
        output_dims=output_dims,
        perceiver_cfg=_DiTCfg(
            n_anchors=8, anchor_dim=32, cell_dim=32,
            n_anchor_blocks=2, n_heads=4, mlp_ratio=2,
            time_embed_dim=64,
        ),
    )
    data = _build_full_data(B=B, N=N)
    pred = model(data)
    pad_features = pred.node_features[0, 20:].abs().max().item()
    pad_positions = pred.positions[0, 20:].abs().max().item()
    assert pad_features < 1e-5, f"padding features not zero: {pad_features}"
    assert pad_positions < 1e-5, f"padding positions not zero: {pad_positions}"


def test_perceiver_anchor_tokens_receive_gradient() -> None:
    """The learnable anchor parameter should receive gradient — if it
    doesn't, the whole anchor pathway is dead code."""
    from models.perceiver_backbone import PerceiverBackbone

    B, N = 1, 20
    input_dims, hidden_mlp_dims, hidden_dims, output_dims = (
        _build_input_output_dims(gene_dim=16, out_features=4)
    )
    model = PerceiverBackbone(
        input_dims=input_dims, n_layers=2,
        hidden_mlp_dims=hidden_mlp_dims, hidden_dims=hidden_dims,
        output_dims=output_dims,
        perceiver_cfg=_DiTCfg(
            n_anchors=8, anchor_dim=32, cell_dim=32,
            n_anchor_blocks=2, n_heads=4, mlp_ratio=2,
            time_embed_dim=64,
        ),
    )
    data = _build_data(B=B, N=N)
    pred = model(data)
    # Force the loss to depend on the predicted positions (which depend
    # on the anchor tokens via the cell-stream output projection).
    loss = pred.positions.pow(2).mean()
    loss.backward()
    assert model.anchor_tokens.grad is not None, "no grad on anchor_tokens"
    assert model.anchor_tokens.grad.abs().sum().item() > 0, (
        "anchor_tokens.grad is exactly zero — the gradient path through "
        "anchors is broken"
    )


# ---------------------------------------------------------------------------
# #4 — True EDM diffusion (FM in h-space)
# ---------------------------------------------------------------------------


def _build_edm_fm_cfg(
    framework: str = "flow_matching",
    embed_dim: int = 4,
    n_steps: int = 5,
):
    """Minimal nested-namespace cfg for EDMFlowMatchingModel.__init__."""

    class _N:
        pass

    cfg = _N()
    cfg.model = _N()
    cfg.model.edm = _N()
    cfg.model.edm.enabled = True
    cfg.model.edm.embed_dim = embed_dim
    cfg.model.edm.diffuse_in_embed_space = True
    cfg.model.flow_matching = _N()
    cfg.model.flow_matching.n_sampling_steps = n_steps
    cfg.model.flow_matching.eps_t = 1e-3
    cfg.model.flow_matching.prediction = "x0"
    return cfg


def test_edm_fm_apply_noise_shapes() -> None:
    from utils.diffusion_model.diffusion.edm_fm_model import EDMFlowMatchingModel

    nm = EDMFlowMatchingModel(_build_edm_fm_cfg(embed_dim=4))
    B, N = 2, 16
    data = _build_data(B=B, N=N)
    z_t = nm.apply_noise(data)
    assert z_t.positions.shape == (B, N, 2), \
        f"positions alias should still be 2D; got {z_t.positions.shape}"
    h_t = getattr(z_t, "_edm_h_t", None)
    h_0 = getattr(z_t, "_edm_h_0", None)
    assert h_t is not None and h_t.shape == (B, N, 4), f"h_t shape {h_t.shape if h_t is not None else None}"
    assert h_0 is not None and h_0.shape == (B, N, 4)
    # h_0 = (true_pos, 0_pad) — first 2 dims are true positions.
    err = (h_0[..., :2] - data.positions).abs().max().item()
    assert err < 1e-5, f"h_0 should embed true positions in first 2 dims; err {err}"
    assert (h_0[..., 2:].abs() < 1e-5).all(), "h_0 padding should be zero"


def test_edm_fm_k2_matches_standard_fm_trajectory_signature() -> None:
    """With embed_dim=2 the lift is trivial (h_0 == true positions) and
    h_t[..., :2] == h_t (no padding). So the 2D alias the backbone
    sees is exactly the same trajectory the existing FlowMatchingModel
    feeds it. Sanity check: at k=2, x_t = (1-t)·x_0 + t·noise."""
    from utils.diffusion_model.diffusion.edm_fm_model import EDMFlowMatchingModel

    nm = EDMFlowMatchingModel(_build_edm_fm_cfg(embed_dim=2))
    torch.manual_seed(42)
    B, N = 1, 16
    data = _build_data(B=B, N=N)
    # apply_noise samples its own t and noise internally; just verify
    # the output is a valid linear-FM mixture: each row of x_t lies on
    # the line from x_0 to some noise sample (we can't recover noise
    # exactly without seeding control, but we CAN check h_0 == x_0).
    z_t = nm.apply_noise(data)
    h_0 = z_t._edm_h_0
    assert torch.allclose(h_0, data.positions, atol=1e-5), \
        "at embed_dim=2 the lift should be identity"


def test_edm_fm_sample_limit_dist_finite() -> None:
    from utils.diffusion_model.diffusion.edm_fm_model import EDMFlowMatchingModel

    nm = EDMFlowMatchingModel(_build_edm_fm_cfg(embed_dim=4))
    B, N = 1, 16
    node_features = torch.randn(B, N, 8)
    node_mask = torch.ones(B, N, dtype=torch.bool)
    z = nm.sample_limit_dist(
        node_features=node_features,
        node_mask=node_mask,
        cell_ID=torch.arange(N).repeat(B, 1).unsqueeze(-1),
        cell_class=torch.zeros(B, N, dtype=torch.long),
    )
    assert z.positions.shape == (B, N, 2)
    assert torch.isfinite(z.positions).all()
    h_t = getattr(z, "_edm_h_t", None)
    assert h_t is not None and h_t.shape == (B, N, 4)
    assert torch.isfinite(h_t).all()


def test_edm_fm_euler_step_at_s0_recovers_h0_pred() -> None:
    """At s_int=0 (the final reverse step), the FM Euler update should
    return exactly h_0_pred (modulo masking). This is the trivial
    s=0 limit of the convex-combination form: h_s = (0/t)·h_t +
    (t/t)·h_0_pred = h_0_pred.
    """
    from utils.diffusion_model.diffusion.edm_fm_model import EDMFlowMatchingModel

    nm = EDMFlowMatchingModel(_build_edm_fm_cfg(embed_dim=4, n_steps=5))
    B, N = 1, 12
    h_t = torch.randn(B, N, 4)
    h0_pred = torch.randn(B, N, 4)
    z_t = DataHolder(
        node_features=torch.zeros(B, N, 1),
        positions=h_t[..., :2],
        diffusion_time=torch.tensor([[0.4]]),
        cell_class=torch.zeros(B, N, 1, dtype=torch.long),
        cell_ID=torch.arange(N).repeat(B, 1).unsqueeze(-1),
        t_int=torch.tensor([[400]]),
        t=torch.tensor([[0.4]]),
        node_mask=torch.ones(B, N, dtype=torch.bool),
    )
    z_t._edm_h_t = h_t.clone()
    pred = DataHolder(
        node_features=torch.zeros(B, N, 1),
        positions=h0_pred[..., :2],
        diffusion_time=torch.tensor([[0.4]]),
        cell_class=torch.zeros(B, N, 1, dtype=torch.long),
        cell_ID=torch.arange(N).repeat(B, 1).unsqueeze(-1),
        t_int=torch.tensor([[400]]),
        t=torch.tensor([[0.4]]),
        node_mask=torch.ones(B, N, dtype=torch.bool),
    )
    pred.edm_h = h0_pred.clone()

    z_s = nm.sample_zs_from_zt_and_pred(z_t, pred, torch.tensor(0))
    err = (z_s._edm_h_t - h0_pred).abs().max().item()
    assert err < 1e-5, f"at s=0 the FM step should give h_0_pred; err {err}"


def test_edm_fm_requires_edm_enabled() -> None:
    """Without cfg.model.edm.enabled, EDMFlowMatchingModel raises a clear error."""
    from utils.diffusion_model.diffusion.edm_fm_model import EDMFlowMatchingModel

    cfg = _build_edm_fm_cfg()
    cfg.model.edm.enabled = False
    try:
        EDMFlowMatchingModel(cfg)
    except ValueError as e:
        assert "edm.enabled=true" in str(e)
        return
    raise AssertionError("EDMFlowMatchingModel did not raise without edm.enabled")


# ---------------------------------------------------------------------------
# #5 — Latent diffusion scaffold raises clearly
# ---------------------------------------------------------------------------


def test_latent_diffusion_is_real_now() -> None:
    """Latent diffusion was a scaffold in earlier revisions; it's now
    a full implementation. The previous "raises NotImplementedError"
    test was retired — see scripts/test_latent_diffusion.py for the
    real test suite that exercises encoder + denoiser + decoder +
    noise model + loss components end-to-end. Here we just confirm
    the module can be imported AND that the noise model requires a
    cfg (the old no-args constructor that raised is gone)."""
    from utils.diffusion_model.diffusion.latent_diffusion_model import (
        LatentDiffusionModel,
    )
    try:
        LatentDiffusionModel()
    except TypeError:
        # Constructor now requires a cfg — expected.
        return
    raise AssertionError(
        "LatentDiffusionModel() should require a cfg argument now; "
        "the no-args stub has been replaced with a real implementation. "
        "If this passed silently, the constructor signature has drifted."
    )


# ---------------------------------------------------------------------------
# #6 — Shape-matching loss: variants + per-slice eigenvalue diagnostics
# ---------------------------------------------------------------------------


def _shape_loss_cfg(variant: str = "eigvals", weight: float = 0.1) -> dict:
    """Minimal cfg dict for LossFunction with only shape_matching on.
    Uses nested ``dict``s — LossFunction._cfg_get falls back to .get()
    for dicts so we don't need OmegaConf to test."""
    return {
        "model": {
            "loss": {
                "pairwise_distance_mse": {"enabled": False, "weight": 0.0},
                "shape_matching": {
                    "enabled": True,
                    "weight": weight,
                    "variant": variant,
                },
            },
        },
    }


def _shape_two_clouds(
    aniso_true: float = 4.0, aniso_pred: float = 1.0, n: int = 64,
):
    """Build a single-batch DataHolder pair where the true positions
    are a stretched ellipse (aspect ratio = aniso_true) and the
    predictions are stretched by aniso_pred. Both clouds share x,y
    layout, just scaled along x.
    """
    from utils.data.dataholder import DataHolder
    torch.manual_seed(123)
    # Base: 2-D gaussian.
    base = torch.randn(1, n, 2)
    # Stretch x by the requested factor.
    true_pos = base.clone()
    true_pos[..., 0] *= aniso_true
    pred_pos = base.clone()
    pred_pos[..., 0] *= aniso_pred
    mask = torch.ones(1, n, dtype=torch.bool)
    masked_true = DataHolder(
        node_features=torch.zeros(1, n, 4),
        positions=true_pos,
        diffusion_time=torch.zeros(1, 1),
        cell_class=torch.zeros(1, n, 1, dtype=torch.long),
        cell_ID=torch.arange(n).unsqueeze(0).unsqueeze(-1),
        t_int=torch.zeros(1, 1, dtype=torch.long),
        t=torch.zeros(1, 1),
        node_mask=mask,
    )
    # Predictions need gradient so we can assert backward works.
    pred_pos.requires_grad_(True)
    masked_pred = DataHolder(
        node_features=torch.zeros(1, n, 4),
        positions=pred_pos,
        diffusion_time=torch.zeros(1, 1),
        cell_class=torch.zeros(1, n, 1, dtype=torch.long),
        cell_ID=torch.arange(n).unsqueeze(0).unsqueeze(-1),
        t_int=torch.zeros(1, 1, dtype=torch.long),
        t=torch.zeros(1, 1),
        node_mask=mask,
    )
    return masked_pred, masked_true


def test_shape_eigvals_default_behaviour() -> None:
    """The ``eigvals`` variant is the original behaviour: MSE on
    sorted eigenvalues. Must (a) return a finite scalar tensor,
    (b) be > 0 when predictions are isotropic and truth is anisotropic,
    (c) backprop to the prediction tensor.
    """
    from metrics.loss_function import LossFunction
    loss_fn = LossFunction(_shape_loss_cfg(variant="eigvals"))
    pred, true = _shape_two_clouds(aniso_true=4.0, aniso_pred=1.0)
    val = loss_fn._compute_shape_matching(pred, true)
    assert val.ndim == 0, f"expected scalar, got shape {val.shape}"
    assert torch.isfinite(val).item(), f"eigvals loss not finite: {val}"
    assert val.item() > 0.0, (
        f"eigvals loss should be > 0 for aniso pred vs iso truth; got {val}"
    )
    val.backward()
    assert pred.positions.grad is not None, (
        "eigvals variant didn't reach pred.positions"
    )


def test_shape_ratio_is_scale_invariant() -> None:
    """The ``ratio`` variant must NOT depend on overall scale: doubling
    both pred and true should leave the loss unchanged (up to fp noise).
    This is the whole point of the variant — distinguishes shape
    anisotropy from absolute size.
    """
    from metrics.loss_function import LossFunction
    loss_fn = LossFunction(_shape_loss_cfg(variant="ratio"))
    # Reference: aniso=4 truth, aniso=1 pred (isotropic).
    pred1, true1 = _shape_two_clouds(aniso_true=4.0, aniso_pred=1.0)
    val1 = loss_fn._compute_shape_matching(pred1, true1).item()
    # Scale both by 100×. Eigenvalues scale by 100² = 10000, but the
    # RATIO λ_min/λ_max is invariant.
    pred2, true2 = _shape_two_clouds(aniso_true=4.0, aniso_pred=1.0)
    pred2.positions.data.mul_(100.0)
    true2.positions.data.mul_(100.0)
    val2 = loss_fn._compute_shape_matching(pred2, true2).item()
    rel_err = abs(val1 - val2) / (abs(val1) + 1e-12)
    assert rel_err < 1e-3, (
        f"ratio variant not scale-invariant: val1={val1:.6f} vs "
        f"val2={val2:.6f} (rel err {rel_err:.3e}). The ratio loss "
        f"should depend only on shape, not size."
    )


def test_shape_ratio_separates_anisotropies() -> None:
    """Sanity: an isotropic prediction against a strongly anisotropic
    truth should produce a LARGER ratio-loss than a mildly anisotropic
    prediction against the same truth.
    """
    from metrics.loss_function import LossFunction
    loss_fn = LossFunction(_shape_loss_cfg(variant="ratio"))
    pred_iso, true_aniso = _shape_two_clouds(aniso_true=4.0, aniso_pred=1.0)
    pred_mild, _        = _shape_two_clouds(aniso_true=4.0, aniso_pred=3.0)
    val_iso  = loss_fn._compute_shape_matching(pred_iso,  true_aniso).item()
    val_mild = loss_fn._compute_shape_matching(pred_mild, true_aniso).item()
    assert val_iso > val_mild, (
        f"ratio variant should penalise isotropic prediction more than "
        f"mildly-anisotropic; got iso={val_iso:.4f} vs mild={val_mild:.4f}"
    )


def test_shape_covariance_variant_finite_and_backprops() -> None:
    """Covariance Frobenius variant: finite scalar, > 0 on mismatched
    clouds, and gradient reaches the prediction tensor.
    """
    from metrics.loss_function import LossFunction
    loss_fn = LossFunction(_shape_loss_cfg(variant="covariance"))
    pred, true = _shape_two_clouds(aniso_true=4.0, aniso_pred=1.0)
    val = loss_fn._compute_shape_matching(pred, true)
    assert torch.isfinite(val).item(), f"covariance loss not finite: {val}"
    assert val.item() > 0.0, (
        f"covariance loss should be > 0 on mismatched clouds; got {val}"
    )
    val.backward()
    assert pred.positions.grad is not None, (
        "covariance variant didn't reach pred.positions"
    )


def test_shape_diagnostics_populated_and_summarised() -> None:
    """After a compute, ``_last_shape_diagnostics`` carries one entry
    per non-degenerate slice with the documented keys, and
    ``_summarise_shape_diagnostics`` flattens to the expected wandb
    scalar names.
    """
    from metrics.loss_function import LossFunction
    loss_fn = LossFunction(_shape_loss_cfg(variant="eigvals"))
    pred, true = _shape_two_clouds(aniso_true=4.0, aniso_pred=1.0)
    _ = loss_fn._compute_shape_matching(pred, true)
    assert len(loss_fn._last_shape_diagnostics) == 1, (
        f"expected one diagnostic entry for the single batch slice; "
        f"got {len(loss_fn._last_shape_diagnostics)}"
    )
    entry = loss_fn._last_shape_diagnostics[0]
    for k in ("eig_min_true", "eig_max_true", "ratio_true",
              "eig_min_pred", "eig_max_pred", "ratio_pred"):
        assert k in entry, f"diagnostic stash missing key {k!r}: {entry}"
    # The truth was stretched 4× along x → λ_max_true should be much
    # bigger than λ_min_true; pred was isotropic → ratio_pred ≈ 1.
    assert entry["eig_max_true"] > 4 * entry["eig_min_true"], (
        f"truth eigvals look wrong: {entry}"
    )
    assert entry["ratio_pred"] > 0.5, (
        f"pred is isotropic, ratio should be near 1.0; got {entry}"
    )
    assert entry["ratio_true"] < 0.2, (
        f"truth is 4×-stretched, ratio should be small; got {entry}"
    )
    # Now check the wandb-flattening helper.
    out = loss_fn._summarise_shape_diagnostics("val_shape")
    expected_keys = {
        "val_shape/eig_min_true_mean", "val_shape/eig_max_true_mean",
        "val_shape/ratio_true_mean",   "val_shape/eig_min_pred_mean",
        "val_shape/eig_max_pred_mean", "val_shape/ratio_pred_mean",
        "val_shape/eig_min_mae",       "val_shape/eig_max_mae",
        "val_shape/ratio_mae",
    }
    missing = expected_keys - set(out.keys())
    assert not missing, f"summary missing keys: {missing}"
    # ratio_mae must equal |ratio_pred - ratio_true| (single slice).
    expected_ratio_mae = abs(entry["ratio_pred"] - entry["ratio_true"])
    assert abs(out["val_shape/ratio_mae"] - expected_ratio_mae) < 1e-6, (
        f"ratio_mae mismatch: got {out['val_shape/ratio_mae']:.6f}, "
        f"expected {expected_ratio_mae:.6f}"
    )


def test_shape_diagnostics_empty_when_disabled() -> None:
    """When shape_matching is OFF, ``_summarise_shape_diagnostics``
    must return ``{}`` (no stale entries from a prior call leak into
    wandb). Equivalent for the disabled case: never populated.
    """
    from metrics.loss_function import LossFunction
    # Empty cfg → shape_matching defaults to disabled.
    loss_fn = LossFunction({})
    # Stash starts empty.
    out = loss_fn._summarise_shape_diagnostics("val_shape")
    assert out == {}, f"expected empty dict when disabled; got {out}"


def test_shape_invalid_variant_raises() -> None:
    """Misconfigured ``variant`` must fail loud at construction time —
    we want the strict-mode behaviour the rest of the loss module
    already follows."""
    from metrics.loss_function import LossFunction
    try:
        LossFunction(_shape_loss_cfg(variant="trace"))
    except ValueError as e:
        assert "shape_matching.variant" in str(e), (
            f"error message should name the offending field; got {e}"
        )
        return
    raise AssertionError(
        "invalid variant should raise ValueError at __init__"
    )


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def main() -> int:
    tests = [
        ("#1 EDM anisotropic gating: W=I matches isotropic",
         test_anisotropic_gating_init_is_identity),
        ("#1 EDM anisotropic gating: D stays symmetric/non-neg with random W",
         test_anisotropic_gating_D_is_symmetric_nonneg),
        ("#1 EDM anisotropic gating: W receives gradient",
         test_anisotropic_gating_gradient_flows_to_W),
        ("#2 DiT backbone: forward shapes",
         test_dit_backbone_forward_shapes),
        ("#2 DiT backbone: padding mask zeros output",
         test_dit_backbone_padding_mask_zeros_output),
        ("#2 DiT backbone: gradient flow end-to-end",
         test_dit_backbone_gradient_flow),
        ("#3 Perceiver: forward shapes",
         test_perceiver_backbone_forward_shapes),
        ("#3 Perceiver: padding mask zeros output",
         test_perceiver_backbone_padding_mask_zeros_output),
        ("#3 Perceiver: anchor tokens receive gradient",
         test_perceiver_anchor_tokens_receive_gradient),
        ("#4 EDM-FM: apply_noise shapes + h_0 padding",
         test_edm_fm_apply_noise_shapes),
        ("#4 EDM-FM: k=2 lift is identity",
         test_edm_fm_k2_matches_standard_fm_trajectory_signature),
        ("#4 EDM-FM: sample_limit_dist finite",
         test_edm_fm_sample_limit_dist_finite),
        ("#4 EDM-FM: Euler step at s=0 recovers h_0_pred",
         test_edm_fm_euler_step_at_s0_recovers_h0_pred),
        ("#4 EDM-FM: raises without edm.enabled",
         test_edm_fm_requires_edm_enabled),
        ("#5 Latent diffusion is now a full implementation (no stub)",
         test_latent_diffusion_is_real_now),
        ("#6 Shape loss: eigvals variant (default) finite + backprops",
         test_shape_eigvals_default_behaviour),
        ("#6 Shape loss: ratio variant is scale-invariant",
         test_shape_ratio_is_scale_invariant),
        ("#6 Shape loss: ratio variant separates aniso magnitudes",
         test_shape_ratio_separates_anisotropies),
        ("#6 Shape loss: covariance variant finite + backprops",
         test_shape_covariance_variant_finite_and_backprops),
        ("#6 Shape loss: per-slice diagnostics + wandb summary",
         test_shape_diagnostics_populated_and_summarised),
        ("#6 Shape loss: diagnostics empty when disabled",
         test_shape_diagnostics_empty_when_disabled),
        ("#6 Shape loss: invalid variant raises at __init__",
         test_shape_invalid_variant_raises),
    ]
    n_pass = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  ✓ {name}")
            n_pass += 1
        except Exception as e:
            print(f"  ✗ {name}: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            return 1
    print(f"\n{n_pass}/{len(tests)} tests passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
