#!/usr/bin/env python
"""Regression tests for the audit-pass fixes (2026-05-27).

Each test guards against a specific bug that was previously verified
and fixed. Failing one of these tests means a regression of the
corresponding bug.

  * B1.1 — DDPM sampler no longer reseeds the global CPU RNG mid-call.
  * B2.4 — `sample_zs_from_zt` returns `(z_s, pred)`; self-cond
           stash uses `pred.positions` (training contract).
  * B2.5 — EDM-FM noise is mean-removed (centroid stays at 0).
  * B2.6 — `max_diffusion_steps` branches for framework=latent_diffusion
           (LDM uses its own n_diffusion_steps, not the DDPM default).
  * B2.7 — `_edm_h_t` is in the forward()'s propagation list.
  * B2.8 — `log_epoch_metrics` does NOT advance PH `_ph_step_counter`.
  * B2.9 — PH dim=1 unpaired-feature penalty uses (b−d)²/2,
           not (b² + d²).
  * B3.13 — FM `sample_zs_from_zt_and_pred` preserves cell_class /
            cell_ID through the sampling chain.
  * B3.14 — gene_reconstruction × framework=latent_diffusion raises.

Run from the repo root:

    python scripts/test_audit_fixes.py
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

from utils.data.dataholder import DataHolder  # noqa: E402


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


def _build_data(B: int = 1, N: int = 24, gene_dim: int = 16) -> DataHolder:
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
# B1.1 — DDPM sampler doesn't reseed
# ---------------------------------------------------------------------------


def test_b1_1_ddpm_sampler_does_not_reseed() -> None:
    """The DDPM ``sample_limit_dist`` must not contain
    ``torch.manual_seed`` — that reseeds the global CPU RNG
    mid-sampler, defeating multi-sample inference ensembling.

    We can't import noise_model directly here because it pulls
    scanpy through utils.data.load, which is intentionally not in
    the lite test venv. Read the source file as text instead — the
    test is a static check on whether the bad call has crept back
    into the function body.
    """
    import re

    src_path = (
        Path(__file__).resolve().parent.parent / "src"
        / "utils" / "diffusion_model" / "diffusion" / "noise_model.py"
    )
    src = src_path.read_text()
    # Find the sample_limit_dist function body and check no
    # manual_seed inside it.
    m = re.search(
        r"def sample_limit_dist\(.*?\)(.*?)(?=\n    def |\nclass )",
        src, re.DOTALL,
    )
    assert m is not None, "couldn't locate sample_limit_dist in noise_model.py"
    body = m.group(1)
    # Strip out the docstring block + comments to avoid false positives
    # on the explanatory note we wrote there.
    body_no_strings = re.sub(r'""".*?"""', "", body, flags=re.DOTALL)
    body_no_comments = re.sub(r"#.*", "", body_no_strings)
    assert "manual_seed" not in body_no_comments, (
        "regression: torch.manual_seed appears INSIDE the "
        "executable body of NoiseModel.sample_limit_dist — "
        "reseeding the global CPU RNG mid-sampler breaks "
        "multi-sample inference variance."
    )


# ---------------------------------------------------------------------------
# B2.4 — sample_zs_from_zt returns (z_s, pred); self-cond uses pred
# ---------------------------------------------------------------------------


def test_b2_4_sample_zs_from_zt_returns_pred() -> None:
    """The sampler helper must return a 2-tuple of ``(z_s, pred)`` so
    callers can stash pred.positions.detach() as self-cond — the
    canonical training-time contract. Text inspection."""
    src_path = (
        Path(__file__).resolve().parent.parent / "src"
        / "utils" / "diffusion_model" / "sample" / "sample.py"
    )
    src = src_path.read_text()
    assert "return z_s, pred" in src, (
        "regression: sample_zs_from_zt no longer returns "
        "(z_s, pred). The self-cond inference path silently falls "
        "back to using z_s.positions, which differs from the "
        "training-time x_0_pred contract."
    )
    assert "z_s, pred = sample_zs_from_zt" in src, (
        "regression: iterate_sampling no longer unpacks "
        "(z_s, pred) from sample_zs_from_zt."
    )
    assert "pred.positions.detach()" in src, (
        "regression: iterate_sampling no longer stashes "
        "pred.positions.detach() as the self-cond input."
    )


# ---------------------------------------------------------------------------
# B2.5 — EDM-FM noise is mean-removed
# ---------------------------------------------------------------------------


def test_b2_5_edm_fm_noise_mean_centered() -> None:
    """After EDMFlowMatchingModel.apply_noise, the noise term in h_t
    is mean-removed per slice — verified by checking that the
    h_t - (1-t)*h_0 contribution has near-zero centroid over real
    cells."""
    from utils.diffusion_model.diffusion.edm_fm_model import EDMFlowMatchingModel

    class _Cfg:
        class model:
            class edm:
                embed_dim = 8
                enabled = True

    nm = EDMFlowMatchingModel.__new__(EDMFlowMatchingModel)
    nm.embed_dim = 8
    nm.eps_t = 1e-3
    nm.n_sampling_steps = 5
    nm.max_diffusion_steps = 5
    nm.sampler = "euler"
    nm.noise_schedule = "linear"
    nm.prediction = "x0"

    B, N = 2, 32
    data = _build_data(B=B, N=N, gene_dim=4)
    # Mean-centre true positions so h_0[..., :2] has zero centroid;
    # then any centroid drift in h_t is attributable to noise.
    data.positions = data.positions - data.positions.mean(dim=1, keepdim=True)

    out = nm.apply_noise(data)
    # h_t centroid over real cells must be near zero (translation-
    # invariance preserved through the FM trajectory).
    h_t = out._edm_h_t                                          # (B, N, k)
    mean_per_slice = h_t.mean(dim=1)                            # (B, k)
    err = mean_per_slice.abs().max().item()
    assert err < 1e-5, (
        f"regression: h_t centroid not zero after apply_noise; "
        f"max abs {err:.3e}. EDM-FM noise mean-removal may be off."
    )

    # Same check for sample_limit_dist.
    z_T = nm.sample_limit_dist(
        node_features=data.node_features,
        node_mask=data.node_mask,
        cell_ID=data.cell_ID,
        cell_class=data.cell_class,
    )
    h_T = z_T._edm_h_t
    err_T = h_T.mean(dim=1).abs().max().item()
    assert err_T < 1e-5, (
        f"regression: h_T centroid not zero in sample_limit_dist; "
        f"max abs {err_T:.3e}."
    )


# ---------------------------------------------------------------------------
# B2.6 — max_diffusion_steps branches for framework=latent_diffusion
# ---------------------------------------------------------------------------


def test_b2_6_max_diffusion_steps_branches_for_ldm() -> None:
    """Verify __init__ has a latent_diffusion branch for
    max_diffusion_steps that references n_diffusion_steps. Same
    text-based inspection as B1.1 to avoid importing diffusion_model
    (which pulls pytorch_lightning)."""
    src_path = (
        Path(__file__).resolve().parent.parent / "src" / "diffusion_model.py"
    )
    src = src_path.read_text()
    assert 'framework == "latent_diffusion"' in src, (
        "regression: diffusion_model.py no longer has a "
        "latent_diffusion branch."
    )
    # The LDM branch must use n_diffusion_steps (LDM's own knob),
    # not cfg.model.diffusion_steps (DDPM default).
    # Roughly: look for a block that pairs the LDM branch with
    # n_diffusion_steps. The fix added an `elif framework == "latent_diffusion"`
    # block that reads `getattr(ldm_cfg, "n_diffusion_steps", ...)`.
    assert "n_diffusion_steps" in src, (
        "regression: diffusion_model.py doesn't reference "
        "n_diffusion_steps (the LDM-specific step count)."
    )


# ---------------------------------------------------------------------------
# B2.7 — _edm_h_t is in forward's propagation list
# ---------------------------------------------------------------------------


def test_b2_7_edm_h_t_in_propagation_list() -> None:
    src_path = (
        Path(__file__).resolve().parent.parent / "src" / "diffusion_model.py"
    )
    src = src_path.read_text()
    assert '"_edm_h_t"' in src, (
        "regression: _edm_h_t no longer in the DataHolder.copy() "
        "propagation list in diffusion_model.py."
    )


# ---------------------------------------------------------------------------
# B2.8 — log_epoch_metrics does NOT advance PH counter
# ---------------------------------------------------------------------------


def test_b2_8_log_epoch_metrics_uses_stash_not_recompute() -> None:
    """log_epoch_metrics must read self._last_per_component, NOT call
    self.compute_loss again. Recomputation re-runs side effects
    (PH counter ++, sinkhorn re-samples negatives, etc.)."""
    import inspect
    from metrics.loss_function import LossFunction

    src = inspect.getsource(LossFunction.log_epoch_metrics)
    # Must read the stash.
    assert "_last_per_component" in src, (
        "regression: log_epoch_metrics no longer reads "
        "_last_per_component — likely recomputes via compute_loss "
        "and advances stateful counters."
    )
    # Must NOT call compute_loss.
    assert "self.compute_loss" not in src, (
        "regression: log_epoch_metrics calls self.compute_loss again "
        "— this re-runs every component including stateful ones."
    )


def test_b2_8_forward_stashes_per_component() -> None:
    """LossFunction.forward must populate self._last_per_component
    so log_epoch_metrics can read it. Verify by running a forward
    and checking the stash."""
    from metrics.loss_function import LossFunction

    cfg = {
        "model": {
            "loss": {
                "pairwise_distance_mse": {"enabled": True, "weight": 1.0},
            },
        },
    }
    lf = LossFunction(cfg)
    pred_pos = torch.randn(1, 8, 2, requires_grad=True)
    true_pos = torch.randn(1, 8, 2)
    mask = torch.ones(1, 8, dtype=torch.bool)
    masked_pred = DataHolder(
        node_features=torch.zeros(1, 8, 2),
        positions=pred_pos,
        diffusion_time=torch.zeros(1, 1),
        cell_class=torch.zeros(1, 8, 1, dtype=torch.long),
        cell_ID=torch.arange(8).unsqueeze(0).unsqueeze(-1),
        t_int=torch.zeros(1, 1, dtype=torch.long),
        t=torch.zeros(1, 1),
        node_mask=mask,
    )
    masked_true = DataHolder(
        node_features=torch.zeros(1, 8, 2),
        positions=true_pos,
        diffusion_time=torch.zeros(1, 1),
        cell_class=torch.zeros(1, 8, 1, dtype=torch.long),
        cell_ID=torch.arange(8).unsqueeze(0).unsqueeze(-1),
        t_int=torch.zeros(1, 1, dtype=torch.long),
        t=torch.zeros(1, 1),
        node_mask=mask,
    )
    _ = lf.forward(masked_pred, masked_true, train_stage=True, log=False)
    assert hasattr(lf, "_last_per_component"), (
        "forward() didn't stash _last_per_component"
    )
    assert "pairwise_distance_mse" in lf._last_per_component, (
        f"per-component stash missing pairwise_distance_mse: "
        f"{lf._last_per_component}"
    )

    # Now call log_epoch_metrics — it must NOT recompute, AND must
    # return non-empty.
    out = lf.log_epoch_metrics()
    assert "train_epoch/pairwise_distance_mse" in out, (
        f"log_epoch_metrics returned without the pairwise key: {out}"
    )


# ---------------------------------------------------------------------------
# B2.9 — PH dim=1 unpaired-feature penalty uses (b-d)^2/2
# ---------------------------------------------------------------------------


def test_b2_9_ph_dim1_unpaired_penalty_formula() -> None:
    """The PH dim=1 loss uses W2-correct diagonal slack for unpaired
    features. Text-inspection check on the source file (gudhi is
    not in the lite venv)."""
    src_path = (
        Path(__file__).resolve().parent.parent / "src"
        / "metrics" / "loss_function.py"
    )
    src = src_path.read_text()
    has_pred_diag = (
        "pred_sorted[:, 0] - pred_sorted[:, 1]" in src
        and "* 0.5" in src
    )
    has_true_diag = "true_sorted[:, 0] - true_sorted[:, 1]" in src
    assert has_pred_diag and has_true_diag, (
        "regression: PH dim=1 loss doesn't compute the (b-d)^2 / 2 "
        "diagonal-projection penalty for unpaired features; falls "
        "back to over-penalising with (b^2 + d^2)."
    )
    assert "both_real" in src, (
        "regression: PH dim=1 loss doesn't distinguish real-real "
        "from real-pad matches."
    )


# ---------------------------------------------------------------------------
# B3.13 — FM sample step preserves cell_class / cell_ID
# ---------------------------------------------------------------------------


def test_b3_13_fm_sample_preserves_cell_class() -> None:
    """sample_zs_from_zt_and_pred for FM must forward cell_class /
    cell_ID — otherwise backbones that condition on cell_class lose
    that signal after the first sampling step. Text-inspection
    (flow_matching_model pulls scanpy via utils.data.load)."""
    src_path = (
        Path(__file__).resolve().parent.parent / "src"
        / "utils" / "diffusion_model" / "diffusion" / "flow_matching_model.py"
    )
    src = src_path.read_text()
    # Find the sample_zs_from_zt_and_pred function.
    import re
    m = re.search(
        r"def sample_zs_from_zt_and_pred\(.*?\)(.*?)(?=\n    def |\nclass )",
        src, re.DOTALL,
    )
    assert m is not None, "couldn't locate sample_zs_from_zt_and_pred"
    body = m.group(1)
    assert "cell_class=" in body and "cell_ID=" in body, (
        "regression: FM sample_zs_from_zt_and_pred no longer forwards "
        "cell_class / cell_ID. After the first sampling step, the "
        "chain loses cell-class conditioning."
    )


# ---------------------------------------------------------------------------
# B3.14 — gene_recon × framework=latent_diffusion raises
# ---------------------------------------------------------------------------


def test_edm_mds_gradient_flag_default_silences_pos_grad() -> None:
    """Default behaviour (mds_align_gradient=False): an auxiliary loss
    computed on ``pred.positions`` produces zero gradient on the
    projector weights — this is the bug surface that made all
    loss-config-only runs converge to identical models.

    Test setup: combine the position loss with a 0-coefficient edm_D
    anchor so backward CAN run (otherwise pred.positions
    has requires_grad=False under default mode and .backward errors).
    The anchor contributes zero gradient by construction.
    """
    import torch.nn as nn
    from utils.data.dataholder import DataHolder
    from models.edm_head import EDMOutputWrapper

    class _Stub(nn.Module):
        def __init__(self):
            super().__init__()
            self.feat_proj = nn.Linear(8, 4)
            self.pos_proj = nn.Linear(8, 2)
        def forward(self, data, **_):
            f = self.feat_proj(data.node_features)
            p = self.pos_proj(data.node_features)
            return DataHolder(
                node_features=f, positions=p,
                diffusion_time=data.diffusion_time,
                cell_class=data.cell_class, cell_ID=data.cell_ID,
                t_int=data.t_int, t=data.t, node_mask=data.node_mask,
            )

    torch.manual_seed(0)
    stub = _Stub()
    wrap = EDMOutputWrapper(
        inner_model=stub, inner_out_dim=4, embed_dim=4,
        mds_align=True, anisotropic_gating=False,
        mds_align_gradient=False,  # default
    )
    data = _build_data(B=1, N=24, gene_dim=8)
    pred = wrap(data)
    # The actual demonstration of the bug: positions has no grad_fn.
    assert not pred.positions.requires_grad, (
        "pred.positions should be DETACHED under default mds_align_gradient=False"
    )
    # Wire up a backward through pred.edm_D (which DOES have grad) so
    # we can confirm gradient FROM positions specifically is zero,
    # while gradient from edm_D is non-zero.
    loss = (pred.positions ** 2).sum() + pred.edm_D.sum() * 0.0
    wrap.zero_grad()
    loss.backward()
    proj_grad = sum(
        (p.grad.abs().sum().item() if p.grad is not None else 0.0)
        for p in wrap.projector.parameters()
    )
    # 0 * edm_D.sum() contributes zero gradient; positions contributes
    # zero gradient because it's detached. So total must be zero.
    assert proj_grad == 0.0, (
        f"projector should have ZERO gradient from pred.positions "
        f"loss under default (detached) MDS; got {proj_grad:.6e}. "
        f"If this test fails, the detached-MDS default is no longer "
        f"silencing position losses — verify mds_align_gradient flag."
    )


def test_edm_mds_gradient_flag_true_restores_pos_grad() -> None:
    """With mds_align_gradient=True, an auxiliary loss on
    ``pred.positions`` produces NON-zero gradient on the projector —
    that's the entire point of the flag."""
    import torch.nn as nn
    from utils.data.dataholder import DataHolder
    from models.edm_head import EDMOutputWrapper

    class _Stub(nn.Module):
        def __init__(self):
            super().__init__()
            self.feat_proj = nn.Linear(8, 4)
            self.pos_proj = nn.Linear(8, 2)
        def forward(self, data, **_):
            f = self.feat_proj(data.node_features)
            p = self.pos_proj(data.node_features)
            return DataHolder(
                node_features=f, positions=p,
                diffusion_time=data.diffusion_time,
                cell_class=data.cell_class, cell_ID=data.cell_ID,
                t_int=data.t_int, t=data.t, node_mask=data.node_mask,
            )

    torch.manual_seed(0)
    stub = _Stub()
    wrap = EDMOutputWrapper(
        inner_model=stub, inner_out_dim=4, embed_dim=4,
        mds_align=True, anisotropic_gating=False,
        mds_align_gradient=True,
    )
    data = _build_data(B=1, N=24, gene_dim=8)
    pred = wrap(data)
    loss = (pred.positions ** 2).sum()
    wrap.zero_grad()
    loss.backward()
    proj_grad = sum(
        (p.grad.abs().sum().item() if p.grad is not None else 0.0)
        for p in wrap.projector.parameters()
    )
    assert proj_grad > 0.0, (
        f"projector should receive non-zero gradient from "
        f"pred.positions when mds_align_gradient=True; got "
        f"{proj_grad:.6e}. The fix is broken."
    )


def test_edm_silent_silencing_guard_raises() -> None:
    """LossFunction must refuse to start when EDM is on with
    detached MDS AND a position-space auxiliary loss is enabled —
    that combination silently silences the gradient.
    """
    from metrics.loss_function import LossFunction

    cfg = {
        "model": {
            "edm": {
                "enabled": True,
                "mds_align": True,
                "mds_align_gradient": False,  # the silent-silencing combo
            },
            "loss": {
                "shape_matching": {
                    "enabled": True, "weight": 0.1, "variant": "eigvals",
                },
            },
        },
    }
    try:
        LossFunction(cfg)
    except ValueError as e:
        msg = str(e)
        assert "Silent-silencing" in msg and "shape_matching" in msg, (
            f"guard fired but wrong message: {msg}"
        )
        # Mentions the three fixes.
        assert "mds_align_gradient=true" in msg, (
            "fix-suggestion 1 missing from error message"
        )
        return
    raise AssertionError(
        "LossFunction should raise on silent-silencing combo "
        "(EDM detached + shape_matching enabled) but didn't"
    )


def test_edm_silent_silencing_guard_passes_when_mds_grad_on() -> None:
    """The guard MUST NOT trip when mds_align_gradient=True — that
    combination is the supported one. Otherwise users can't enable
    auxiliary losses with EDM at all."""
    from metrics.loss_function import LossFunction
    cfg = {
        "model": {
            "edm": {
                "enabled": True,
                "mds_align": True,
                "mds_align_gradient": True,
            },
            "loss": {
                "shape_matching": {"enabled": True, "weight": 0.1},
                "sinkhorn": {"enabled": True, "weight": 0.1},
            },
        },
    }
    LossFunction(cfg)  # must NOT raise


def test_edm_silent_silencing_guard_passes_without_edm() -> None:
    """The guard MUST NOT trip when EDM is off. Without EDM the
    detach doesn't happen and gradient flows normally."""
    from metrics.loss_function import LossFunction
    cfg = {
        "model": {
            "edm": {"enabled": False},
            "loss": {
                "shape_matching": {"enabled": True, "weight": 0.1},
                "sinkhorn": {"enabled": True, "weight": 0.1},
            },
        },
    }
    LossFunction(cfg)  # must NOT raise


def test_knn_graph_silent_silencing_guard_raises() -> None:
    """Same bug class as EDM: when knn_graph.spectral_layout is on
    and spectral_layout_gradient is off, position-space auxiliary
    losses get zero gradient. Guard must raise."""
    from metrics.loss_function import LossFunction
    cfg = {
        "model": {
            "knn_graph": {
                "enabled": True,
                "spectral_layout": True,
                "spectral_layout_gradient": False,
            },
            "loss": {
                "sinkhorn": {"enabled": True, "weight": 0.1},
            },
        },
    }
    try:
        LossFunction(cfg)
    except ValueError as e:
        msg = str(e)
        assert "Silent-silencing" in msg, f"wrong error: {msg}"
        assert "sinkhorn" in msg, f"sinkhorn not named in error: {msg}"
        assert "spectral_layout_gradient=true" in msg, (
            f"fix suggestion missing knn flag: {msg}"
        )
        return
    raise AssertionError(
        "LossFunction should raise on kNN-silencing combo but didn't"
    )


def test_knn_graph_silent_silencing_guard_passes_when_grad_on() -> None:
    """With spectral_layout_gradient=True, the combo is supported."""
    from metrics.loss_function import LossFunction
    cfg = {
        "model": {
            "knn_graph": {
                "enabled": True,
                "spectral_layout": True,
                "spectral_layout_gradient": True,
            },
            "loss": {
                "shape_matching": {"enabled": True, "weight": 0.1},
                "sinkhorn": {"enabled": True, "weight": 0.1},
            },
        },
    }
    LossFunction(cfg)  # must NOT raise


def test_knn_graph_silencing_guard_passes_with_spectral_off() -> None:
    """With spectral_layout=False, pred.positions falls through from
    the inner backbone (gradient-bearing), so no silencing — guard
    must not trip."""
    from metrics.loss_function import LossFunction
    cfg = {
        "model": {
            "knn_graph": {
                "enabled": True,
                "spectral_layout": False,
                "spectral_layout_gradient": False,
            },
            "loss": {
                "shape_matching": {"enabled": True, "weight": 0.1},
            },
        },
    }
    LossFunction(cfg)  # must NOT raise


def test_knn_graph_spectral_layout_gradient_flag_default() -> None:
    """Default behaviour (spectral_layout_gradient=False): the spectral
    layout replaces pred.positions with a gradient-free tensor.
    Mirror of test_edm_mds_gradient_flag_default_silences_pos_grad."""
    import torch.nn as nn
    from utils.data.dataholder import DataHolder
    from models.knn_graph_head import KNNGraphOutputWrapper

    class _Stub(nn.Module):
        def __init__(self):
            super().__init__()
            self.feat_proj = nn.Linear(8, 4)
            self.pos_proj = nn.Linear(8, 2)
        def forward(self, data, **_):
            f = self.feat_proj(data.node_features)
            p = self.pos_proj(data.node_features)
            return DataHolder(
                node_features=f, positions=p,
                diffusion_time=data.diffusion_time,
                cell_class=data.cell_class, cell_ID=data.cell_ID,
                t_int=data.t_int, t=data.t, node_mask=data.node_mask,
            )

    torch.manual_seed(0)
    stub = _Stub()
    wrap = KNNGraphOutputWrapper(
        inner_model=stub, inner_out_dim=4, embed_dim=8,
        spectral_layout=True, k_for_layout=4, temperature=0.1,
        spectral_layout_gradient=False,  # default
    )
    data = _build_data(B=1, N=24, gene_dim=8)
    pred = wrap(data)
    assert not pred.positions.requires_grad, (
        "pred.positions should be DETACHED under default "
        "spectral_layout_gradient=False"
    )


def test_knn_graph_spectral_layout_gradient_flag_true_restores() -> None:
    """With spectral_layout_gradient=True, the projector receives
    gradient from a pred.positions loss. Mirror of
    test_edm_mds_gradient_flag_true_restores_pos_grad."""
    import torch.nn as nn
    from utils.data.dataholder import DataHolder
    from models.knn_graph_head import KNNGraphOutputWrapper

    class _Stub(nn.Module):
        def __init__(self):
            super().__init__()
            self.feat_proj = nn.Linear(8, 4)
            self.pos_proj = nn.Linear(8, 2)
        def forward(self, data, **_):
            f = self.feat_proj(data.node_features)
            p = self.pos_proj(data.node_features)
            return DataHolder(
                node_features=f, positions=p,
                diffusion_time=data.diffusion_time,
                cell_class=data.cell_class, cell_ID=data.cell_ID,
                t_int=data.t_int, t=data.t, node_mask=data.node_mask,
            )

    torch.manual_seed(0)
    stub = _Stub()
    wrap = KNNGraphOutputWrapper(
        inner_model=stub, inner_out_dim=4, embed_dim=8,
        spectral_layout=True, k_for_layout=4, temperature=0.1,
        spectral_layout_gradient=True,
    )
    data = _build_data(B=1, N=24, gene_dim=8)
    pred = wrap(data)
    loss = (pred.positions ** 2).sum() + pred.knn_logits.sum() * 0.0
    wrap.zero_grad()
    loss.backward()
    proj_grad = sum(
        (p.grad.abs().sum().item() if p.grad is not None else 0.0)
        for p in wrap.projector.parameters()
    )
    assert proj_grad > 0.0, (
        f"projector should receive non-zero gradient from "
        f"pred.positions when spectral_layout_gradient=True; got "
        f"{proj_grad:.6e}"
    )


def test_procrustes_skip_when_degenerate() -> None:
    """When x_src is near-zero (training init), Procrustes' M is
    degenerate and SVD backward would NaN. The skip-when-degenerate
    path passes through x_src unchanged — math: when M=0, identity
    is one valid rotation, so x_src @ I = x_src is well-defined."""
    from metrics.loss_function import _procrustes_align_2d

    # Near-zero src — degenerate case.
    x_src = torch.zeros(20, 2, requires_grad=True)
    x_ref = torch.randn(20, 2)
    out = _procrustes_align_2d(x_src, x_ref)
    assert torch.equal(out, x_src), (
        "degenerate path should pass through x_src unchanged"
    )
    # Backward must NOT produce NaN even though SVD was skipped.
    out.sum().backward()
    assert x_src.grad is not None and torch.isfinite(x_src.grad).all(), (
        f"NaN in grad through degenerate Procrustes path; got "
        f"{x_src.grad}"
    )


def test_procrustes_normal_path_for_nondegenerate() -> None:
    """When x_src is non-degenerate, the SVD path runs normally."""
    from metrics.loss_function import _procrustes_align_2d

    torch.manual_seed(0)
    x_src = torch.randn(20, 2) * 0.5
    x_src = x_src - x_src.mean(0)
    # Rotate by 90° to make a non-trivial alignment target.
    theta = torch.tensor([[0.0, -1.0], [1.0, 0.0]])
    x_ref = x_src @ theta
    x_src_grad = x_src.clone().detach().requires_grad_(True)
    out = _procrustes_align_2d(x_src_grad, x_ref)
    # Aligned output should be close to x_ref.
    err = (out - x_ref).abs().max().item()
    assert err < 1e-4, (
        f"normal Procrustes path failed to align: max err {err:.3e}"
    )
    # Gradient must flow.
    out.sum().backward()
    assert x_src_grad.grad is not None, "no grad through normal path"


def test_shape_matching_eigvalsh_no_nan_grad_at_init() -> None:
    """At training init, pred positions are near-zero → cov_pred ≈ 0
    → eigvalsh has degenerate eigenvalues → backward would NaN
    without Tikhonov regularization. Verify finite gradient flows."""
    from utils.data.dataholder import DataHolder
    from metrics.loss_function import LossFunction

    cfg = {
        "model": {
            "loss": {
                "pairwise_distance_mse": {"enabled": False},
                "shape_matching": {
                    "enabled": True, "weight": 1.0, "variant": "eigvals",
                },
            },
        },
    }
    lf = LossFunction(cfg)
    # Construct a near-zero pred (init regime).
    pred_pos = torch.zeros(1, 24, 2, requires_grad=True)
    true_pos = torch.randn(1, 24, 2) * 0.5
    mask = torch.ones(1, 24, dtype=torch.bool)
    masked_pred = DataHolder(
        node_features=torch.zeros(1, 24, 4),
        positions=pred_pos,
        diffusion_time=torch.zeros(1, 1),
        cell_class=torch.zeros(1, 24, 1, dtype=torch.long),
        cell_ID=torch.arange(24).unsqueeze(0).unsqueeze(-1),
        t_int=torch.zeros(1, 1, dtype=torch.long),
        t=torch.zeros(1, 1),
        node_mask=mask,
    )
    masked_true = DataHolder(
        node_features=torch.zeros(1, 24, 4),
        positions=true_pos,
        diffusion_time=torch.zeros(1, 1),
        cell_class=torch.zeros(1, 24, 1, dtype=torch.long),
        cell_ID=torch.arange(24).unsqueeze(0).unsqueeze(-1),
        t_int=torch.zeros(1, 1, dtype=torch.long),
        t=torch.zeros(1, 1),
        node_mask=mask,
    )
    loss = lf._compute_shape_matching(masked_pred, masked_true)
    loss.backward()
    assert pred_pos.grad is not None, "no gradient through shape_matching"
    assert torch.isfinite(pred_pos.grad).all(), (
        f"shape_matching produced NaN/Inf gradient at init "
        f"(pred ≈ 0): max abs grad "
        f"{pred_pos.grad.abs().max().item():.3e}"
    )


def test_mds_eigh_no_nan_grad_with_tikhonov() -> None:
    """Classical MDS's eigh backward at degenerate eigenvalues was
    the NaN source for the EDM path under mds_align_gradient=True.
    Tikhonov regularization breaks the degeneracy. Verify a finite
    gradient flows on a near-zero D_sq (init regime)."""
    from models.edm_head import _classical_mds_2d

    # Near-zero D_sq — what the EDM head produces at init.
    n = 8
    D_sq = torch.zeros(n, n, requires_grad=True)
    pos = _classical_mds_2d(D_sq)
    pos.pow(2).sum().backward()
    assert D_sq.grad is not None, "no grad through MDS"
    assert torch.isfinite(D_sq.grad).all(), (
        f"MDS produced NaN/Inf gradient on near-zero D_sq: max abs "
        f"grad {D_sq.grad.abs().max().item():.3e}"
    )


def test_warmup_skips_autograd_graph_during_warmup() -> None:
    """During warmup, the loss VALUE is computed but the gradient
    contribution is exactly zero. Verifies that backward through a
    warmup'd component doesn't reach any model parameter."""
    import torch.nn as nn
    from utils.data.dataholder import DataHolder
    from metrics.loss_function import LossFunction

    cfg = {
        "model": {
            "loss": {
                # Only the warmup'd loss enabled. If warmup skips
                # backward correctly, no param should get gradient
                # through it.
                "pairwise_distance_mse": {
                    "enabled": True, "weight": 1.0, "warmup_steps": 5,
                },
            },
        },
    }
    lf = LossFunction(cfg)

    # Build a trivial gradient-bearing surrogate so we can ask
    # "did any parameter get gradient through the loss?".
    w = nn.Parameter(torch.randn(1, 1))
    pred_pos = torch.randn(1, 8, 2) * w  # gradient flows through w
    true_pos = torch.randn(1, 8, 2)
    mask = torch.ones(1, 8, dtype=torch.bool)
    pred = DataHolder(
        node_features=torch.zeros(1, 8, 4), positions=pred_pos,
        diffusion_time=torch.zeros(1, 1),
        cell_class=torch.zeros(1, 8, 1, dtype=torch.long),
        cell_ID=torch.arange(8).unsqueeze(0).unsqueeze(-1),
        t_int=torch.zeros(1, 1, dtype=torch.long),
        t=torch.zeros(1, 1), node_mask=mask,
    )
    true = DataHolder(
        node_features=torch.zeros(1, 8, 4), positions=true_pos,
        diffusion_time=torch.zeros(1, 1),
        cell_class=torch.zeros(1, 8, 1, dtype=torch.long),
        cell_ID=torch.arange(8).unsqueeze(0).unsqueeze(-1),
        t_int=torch.zeros(1, 1, dtype=torch.long),
        t=torch.zeros(1, 1), node_mask=mask,
    )

    # Step 0: in warmup. Total should be 0 (no graph contribution).
    loss, _ = lf.forward(pred, true, train_stage=True, log=False)
    per_comp = lf._last_per_component
    assert per_comp.get("pairwise_distance_mse_warmup_active") == 1.0, (
        "warmup flag should be 1.0 during warmup"
    )
    assert per_comp.get("pairwise_distance_mse") is not None, (
        "loss VALUE should still be computed for logging"
    )
    assert per_comp["pairwise_distance_mse_weighted"] == 0.0, (
        "weighted contribution should be 0 during warmup"
    )
    # The total IS zero-graph-attached. Backward gives zero grad on w.
    w.grad = None
    loss.backward()
    assert w.grad is None or w.grad.abs().sum().item() == 0.0, (
        f"warmup'd loss should not propagate gradient; got w.grad="
        f"{w.grad}"
    )


def test_warmup_unlocks_after_n_steps() -> None:
    """After warmup_steps training-stage forwards, the component
    re-engages and its gradient flows to parameters."""
    import torch.nn as nn
    from utils.data.dataholder import DataHolder
    from metrics.loss_function import LossFunction

    cfg = {
        "model": {
            "loss": {
                "pairwise_distance_mse": {
                    "enabled": True, "weight": 1.0, "warmup_steps": 3,
                },
            },
        },
    }
    lf = LossFunction(cfg)

    w = nn.Parameter(torch.randn(1, 1))
    # Advance the counter by 3 training-stage forwards.
    for _ in range(3):
        pred_pos = torch.randn(1, 8, 2) * w
        true_pos = torch.randn(1, 8, 2)
        mask = torch.ones(1, 8, dtype=torch.bool)
        pred = DataHolder(
            node_features=torch.zeros(1, 8, 4), positions=pred_pos,
            diffusion_time=torch.zeros(1, 1),
            cell_class=torch.zeros(1, 8, 1, dtype=torch.long),
            cell_ID=torch.arange(8).unsqueeze(0).unsqueeze(-1),
            t_int=torch.zeros(1, 1, dtype=torch.long),
            t=torch.zeros(1, 1), node_mask=mask,
        )
        true = DataHolder(
            node_features=torch.zeros(1, 8, 4), positions=true_pos,
            diffusion_time=torch.zeros(1, 1),
            cell_class=torch.zeros(1, 8, 1, dtype=torch.long),
            cell_ID=torch.arange(8).unsqueeze(0).unsqueeze(-1),
            t_int=torch.zeros(1, 1, dtype=torch.long),
            t=torch.zeros(1, 1), node_mask=mask,
        )
        lf.forward(pred, true, train_stage=True, log=False)
    # Counter is now 3 → warmup window (0..3 exclusive) ended. Next
    # forward should engage the loss.
    pred_pos = torch.randn(1, 8, 2) * w
    pred = DataHolder(
        node_features=torch.zeros(1, 8, 4), positions=pred_pos,
        diffusion_time=torch.zeros(1, 1),
        cell_class=torch.zeros(1, 8, 1, dtype=torch.long),
        cell_ID=torch.arange(8).unsqueeze(0).unsqueeze(-1),
        t_int=torch.zeros(1, 1, dtype=torch.long),
        t=torch.zeros(1, 1), node_mask=mask,
    )
    loss, _ = lf.forward(pred, true, train_stage=True, log=False)
    per_comp = lf._last_per_component
    assert per_comp.get("pairwise_distance_mse_warmup_active") == 0.0, (
        "warmup flag should be 0.0 after warmup ends"
    )
    assert per_comp["pairwise_distance_mse_weighted"] != 0.0, (
        f"weighted contribution should be nonzero post-warmup; got "
        f"{per_comp['pairwise_distance_mse_weighted']}"
    )
    w.grad = None
    loss.backward()
    assert w.grad is not None and w.grad.abs().sum().item() > 0.0, (
        f"post-warmup loss should propagate gradient; got w.grad="
        f"{w.grad}"
    )


def test_warmup_counter_doesnt_advance_on_val_stage() -> None:
    """Validation forwards must not advance the warmup counter —
    otherwise a long val pass could push past warmup before the
    training loop has caught up."""
    import torch.nn as nn
    from utils.data.dataholder import DataHolder
    from metrics.loss_function import LossFunction

    cfg = {
        "model": {
            "loss": {
                "pairwise_distance_mse": {
                    "enabled": True, "weight": 1.0, "warmup_steps": 3,
                },
            },
        },
    }
    lf = LossFunction(cfg)
    w = nn.Parameter(torch.randn(1, 1))
    mask = torch.ones(1, 8, dtype=torch.bool)

    def _step(train_stage):
        pred_pos = torch.randn(1, 8, 2) * w
        true_pos = torch.randn(1, 8, 2)
        pred = DataHolder(
            node_features=torch.zeros(1, 8, 4), positions=pred_pos,
            diffusion_time=torch.zeros(1, 1),
            cell_class=torch.zeros(1, 8, 1, dtype=torch.long),
            cell_ID=torch.arange(8).unsqueeze(0).unsqueeze(-1),
            t_int=torch.zeros(1, 1, dtype=torch.long),
            t=torch.zeros(1, 1), node_mask=mask,
        )
        true = DataHolder(
            node_features=torch.zeros(1, 8, 4), positions=true_pos,
            diffusion_time=torch.zeros(1, 1),
            cell_class=torch.zeros(1, 8, 1, dtype=torch.long),
            cell_ID=torch.arange(8).unsqueeze(0).unsqueeze(-1),
            t_int=torch.zeros(1, 1, dtype=torch.long),
            t=torch.zeros(1, 1), node_mask=mask,
        )
        return lf.forward(pred, true, train_stage=train_stage, log=False)

    # 10 val-stage forwards → counter should stay at 0.
    for _ in range(10):
        _step(train_stage=False)
    assert lf._step_count == 0, (
        f"counter should not advance on val stage; got "
        f"{lf._step_count}"
    )
    # 2 train-stage forwards → counter = 2 (still in warmup of 3).
    _step(train_stage=True)
    _step(train_stage=True)
    assert lf._step_count == 2
    _step(train_stage=False)  # val: shouldn't advance
    assert lf._step_count == 2, (
        f"val forward should not advance the counter; expected 2, "
        f"got {lf._step_count}"
    )


def test_graph_zero_bypasses_mds_path() -> None:
    """Regression test for the 0×Inf NaN trap.

    Setup: build a tiny EDM scenario where MDS's eigh backward is
    artificially set to produce Inf for any nonzero incoming
    gradient (we don't actually call MDS — we just verify the
    helper returns a zero whose autograd connection bypasses
    ``pred.positions``).

    Specifically: when ``masked_pred.edm_D`` exists, ``_graph_zero``
    must return a zero whose ``grad_fn`` traces through edm_D,
    NOT through positions. We check this by verifying that the
    returned tensor's autograd graph has edm_D in its inputs.
    """
    from utils.data.dataholder import DataHolder
    from metrics.loss_function import _graph_zero

    # Build a DataHolder where positions and edm_D are SEPARATE
    # gradient-bearing tensors. _graph_zero should pick edm_D.
    pos = torch.randn(1, 8, 2, requires_grad=True)
    edm_D = torch.randn(1, 8, 8, requires_grad=True)
    data = DataHolder(
        node_features=torch.zeros(1, 8, 4), positions=pos,
        diffusion_time=torch.zeros(1, 1),
        cell_class=torch.zeros(1, 8, 1, dtype=torch.long),
        cell_ID=torch.arange(8).unsqueeze(0).unsqueeze(-1),
        t_int=torch.zeros(1, 1, dtype=torch.long),
        t=torch.zeros(1, 1), node_mask=torch.ones(1, 8, dtype=torch.bool),
    )
    data.edm_D = edm_D  # stash, as EDM head does

    zero = _graph_zero(data)
    assert zero.item() == 0.0
    # Backward: gradient should land on edm_D, NOT on positions.
    # (Both have requires_grad=True; the helper should route
    # through whichever it prefers.)
    pos.grad = None
    edm_D.grad = None
    zero.backward()
    assert pos.grad is None or pos.grad.abs().sum().item() == 0.0, (
        "_graph_zero should not propagate gradient through "
        "pred.positions when edm_D is available — that's the 0×Inf "
        "NaN trap we're avoiding."
    )
    assert edm_D.grad is not None, (
        "_graph_zero should route gradient through edm_D"
    )


def test_graph_zero_falls_back_to_positions_without_edm_D() -> None:
    """When neither edm_D nor knn_logits is present (e.g.,
    pure-regression framework, no EDM head), the helper falls back
    to pred.positions. That's safe because in those configs
    pred.positions IS the inner backbone's direct output (no MDS
    overwrite), so no spectral op is in the backward."""
    from utils.data.dataholder import DataHolder
    from metrics.loss_function import _graph_zero

    pos = torch.randn(1, 8, 2, requires_grad=True)
    data = DataHolder(
        node_features=torch.zeros(1, 8, 4), positions=pos,
        diffusion_time=torch.zeros(1, 1),
        cell_class=torch.zeros(1, 8, 1, dtype=torch.long),
        cell_ID=torch.arange(8).unsqueeze(0).unsqueeze(-1),
        t_int=torch.zeros(1, 1, dtype=torch.long),
        t=torch.zeros(1, 1), node_mask=torch.ones(1, 8, dtype=torch.bool),
    )
    # No edm_D, no knn_logits stashed.
    zero = _graph_zero(data)
    zero.backward()
    assert pos.grad is not None, (
        "fallback should route gradient through pred.positions when "
        "no head-native stash is available"
    )


def test_b3_14_gene_recon_ldm_mutex() -> None:
    """The LightningModule's __init__ must raise when gene_recon AND
    latent_diffusion are both enabled. Text-inspection because
    importing diffusion_model pulls pytorch_lightning."""
    src_path = (
        Path(__file__).resolve().parent.parent / "src" / "diffusion_model.py"
    )
    src = src_path.read_text()
    assert 'gene_recon_enabled and framework == "latent_diffusion"' in src, (
        "regression: diffusion_model.py no longer enforces the "
        "gene_recon × latent_diffusion mutex. The two interact "
        "silently — encoder sees masked at train, unmasked at infer."
    )


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def main() -> int:
    tests = [
        ("B1.1 — DDPM sample_limit_dist has no manual_seed",
         test_b1_1_ddpm_sampler_does_not_reseed),
        ("B2.4 — sample_zs_from_zt returns (z_s, pred)",
         test_b2_4_sample_zs_from_zt_returns_pred),
        ("B2.5 — EDM-FM noise is mean-centred",
         test_b2_5_edm_fm_noise_mean_centered),
        ("B2.6 — max_diffusion_steps branches for LDM",
         test_b2_6_max_diffusion_steps_branches_for_ldm),
        ("B2.7 — _edm_h_t in forward propagation list",
         test_b2_7_edm_h_t_in_propagation_list),
        ("B2.8 — log_epoch_metrics reads stash (no recompute)",
         test_b2_8_log_epoch_metrics_uses_stash_not_recompute),
        ("B2.8 — forward stashes _last_per_component",
         test_b2_8_forward_stashes_per_component),
        ("B2.9 — PH dim=1 uses W2-correct diagonal slack",
         test_b2_9_ph_dim1_unpaired_penalty_formula),
        ("B3.13 — FM step preserves cell_class / cell_ID",
         test_b3_13_fm_sample_preserves_cell_class),
        ("EDM MDS gradient OFF (default): projector grad is zero",
         test_edm_mds_gradient_flag_default_silences_pos_grad),
        ("EDM MDS gradient ON: projector grad is non-zero",
         test_edm_mds_gradient_flag_true_restores_pos_grad),
        ("Silent-silencing guard raises on EDM-detached + aux loss",
         test_edm_silent_silencing_guard_raises),
        ("Silent-silencing guard passes when mds_align_gradient=True",
         test_edm_silent_silencing_guard_passes_when_mds_grad_on),
        ("Silent-silencing guard passes when EDM is off",
         test_edm_silent_silencing_guard_passes_without_edm),
        ("kNN-graph silent-silencing guard raises",
         test_knn_graph_silent_silencing_guard_raises),
        ("kNN-graph silent-silencing guard passes when grad ON",
         test_knn_graph_silent_silencing_guard_passes_when_grad_on),
        ("kNN-graph silent-silencing guard passes when spectral OFF",
         test_knn_graph_silencing_guard_passes_with_spectral_off),
        ("kNN-graph spectral_layout_gradient OFF: pos.grad is severed",
         test_knn_graph_spectral_layout_gradient_flag_default),
        ("kNN-graph spectral_layout_gradient ON: projector grad flows",
         test_knn_graph_spectral_layout_gradient_flag_true_restores),
        ("Procrustes skip-when-degenerate: no NaN grad on near-zero src",
         test_procrustes_skip_when_degenerate),
        ("Procrustes normal path still aligns + has gradient",
         test_procrustes_normal_path_for_nondegenerate),
        ("shape_matching: finite gradient at init (Tikhonov on cov)",
         test_shape_matching_eigvalsh_no_nan_grad_at_init),
        ("MDS eigh: finite gradient on near-zero D_sq (Tikhonov on B)",
         test_mds_eigh_no_nan_grad_with_tikhonov),
        ("Warmup: no autograd graph during warmup window",
         test_warmup_skips_autograd_graph_during_warmup),
        ("Warmup: gradient unlocks after warmup_steps training-steps",
         test_warmup_unlocks_after_n_steps),
        ("Warmup: val-stage forwards don't advance the counter",
         test_warmup_counter_doesnt_advance_on_val_stage),
        ("_graph_zero routes through edm_D, not positions (MDS bypass)",
         test_graph_zero_bypasses_mds_path),
        ("_graph_zero falls back to positions when no head stash",
         test_graph_zero_falls_back_to_positions_without_edm_D),
        ("B3.14 — gene_recon × LDM mutex raises",
         test_b3_14_gene_recon_ldm_mutex),
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
