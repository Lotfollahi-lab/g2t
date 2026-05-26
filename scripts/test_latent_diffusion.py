#!/usr/bin/env python
"""Smoke tests for the full latent-diffusion implementation.

Exercises every code path that runs during training and sampling
without requiring the full LUNA / scanpy / Lightning stack. Uses a
synthetic dataset and the actual LDM modules (encoder + denoiser +
decoder + noise model + loss components).

Run from the repo root::

    python scripts/test_latent_diffusion.py

What's covered:

  * LatentVAEEncoder: forward shapes, masking, KL formula sanity,
    grad flow.
  * LatentVAEDecoder: forward shapes, masking, mean-centring, grad flow.
  * Reparameterise + kl_normal_standard helpers: numerical sanity
    against analytic expectation.
  * LatentDiffusionWrapper: forward consumes z_t, decodes positions,
    propagates LDM stashes onto pred; padding mask respected.
  * LatentDiffusionModel (noise model): apply_noise round-trip,
    sample_limit_dist shapes, FM Euler step in latent space gives
    z_0_pred at s=0.
  * Loss components: latent_fm_mse returns zero when stashes missing,
    correct value when present, gradient flows back to z_0_pred.
    latent_kl returns zero when missing, correct value otherwise.
  * End-to-end training step simulation: encode → noise → forward
    → loss → backward; verify all sub-modules receive gradient.

Exits 0 on pass.
"""

from __future__ import annotations

import sys
import math
from pathlib import Path


def _setup_path() -> None:
    here = Path(__file__).resolve()
    scgg_src = here.parent.parent / "src"
    sys.path.insert(0, str(scgg_src))


_setup_path()

import torch  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

from utils.data.dataholder import DataHolder  # noqa: E402


# ---------------------------------------------------------------------------
# Test fixture builders
# ---------------------------------------------------------------------------


def _build_data(B: int = 2, N: int = 32, gene_dim: int = 16,
                pad_last: int = 0) -> DataHolder:
    """Build a DataHolder shaped like to_batch's output.

    pad_last lets you put N − pad_last real cells with the trailing
    pad_last as padding (mask=False). Defaults to no padding.
    """
    torch.manual_seed(0)
    nm = torch.ones(B, N, dtype=torch.bool)
    if pad_last > 0:
        nm[0, -pad_last:] = False
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


# ---------------------------------------------------------------------------
# Encoder / decoder / helpers
# ---------------------------------------------------------------------------


def test_encoder_shapes_and_mask() -> None:
    from models.latent_vae import LatentVAEEncoder

    B, N, gene_dim, k = 2, 32, 16, 8
    enc = LatentVAEEncoder(
        gene_dim=gene_dim, latent_dim=k,
        hidden_dim=32, n_layers=2, n_heads=4,
    )
    data = _build_data(B=B, N=N, gene_dim=gene_dim, pad_last=4)
    mu, logvar = enc(data.node_features, data.positions, data.node_mask)
    assert mu.shape == (B, N, k), f"mu shape {mu.shape}"
    assert logvar.shape == (B, N, k)
    # Padding cells must come out exactly zero.
    pad_max = mu[0, -4:].abs().max().item()
    assert pad_max < 1e-6, f"encoder mu not zero on padding (max={pad_max})"
    pad_max = logvar[0, -4:].abs().max().item()
    assert pad_max < 1e-6, f"encoder logvar not zero on padding"


def test_decoder_shapes_and_mask() -> None:
    from models.latent_vae import LatentVAEDecoder

    B, N, gene_dim, k = 2, 32, 16, 8
    dec = LatentVAEDecoder(
        gene_dim=gene_dim, latent_dim=k,
        hidden_dim=32, n_layers=2, n_heads=4,
    )
    data = _build_data(B=B, N=N, gene_dim=gene_dim, pad_last=4)
    z = torch.randn(B, N, k)
    z = z * data.node_mask.unsqueeze(-1).to(z.dtype)
    pos = dec(data.node_features, z, data.node_mask)
    assert pos.shape == (B, N, 2)
    pad_max = pos[0, -4:].abs().max().item()
    assert pad_max < 1e-6, f"decoder positions not zero on padding"
    # Mean-centred over real cells (per slice).
    real_mean_slice0 = pos[0, :-4].mean(dim=0).abs().max().item()
    real_mean_slice1 = pos[1].mean(dim=0).abs().max().item()
    assert real_mean_slice0 < 1e-5, f"slice 0 not centred ({real_mean_slice0})"
    assert real_mean_slice1 < 1e-5, f"slice 1 not centred ({real_mean_slice1})"


def test_reparam_and_kl_formula() -> None:
    from models.latent_vae import reparameterize, kl_normal_standard

    torch.manual_seed(42)
    B, N, k = 4, 16, 8
    # Reparam smoke: with logvar very negative, z ≈ mu.
    mu = torch.zeros(B, N, k)
    logvar = torch.full((B, N, k), -20.0)
    z = reparameterize(mu, logvar)
    assert (z - mu).abs().max().item() < 1e-3, (
        "reparam should be near-deterministic at logvar=-20"
    )

    # KL of N(0, I) vs N(0, I) is exactly 0.
    mu = torch.zeros(B, N, k)
    logvar = torch.zeros(B, N, k)  # var = exp(0) = 1
    mask = torch.ones(B, N, dtype=torch.bool)
    kl = kl_normal_standard(mu, logvar, mask)
    assert kl.abs().item() < 1e-6, f"KL(N(0,I)||N(0,I)) should be 0, got {kl.item()}"

    # KL of N(0, 4I) vs N(0, I) per dim: 0.5*(0 + 4 - 1 - log(4)) ≈
    # 0.5 * (3 - 1.386) = 0.807. Per cell: k * 0.807 = 6.456.
    logvar = torch.full((B, N, k), math.log(4.0))
    kl_expected_per_cell = 0.5 * k * (4.0 - 1.0 - math.log(4.0))
    kl_val = kl_normal_standard(mu, logvar, mask).item()
    assert abs(kl_val - kl_expected_per_cell) < 1e-4, (
        f"KL formula: expected {kl_expected_per_cell:.4f}, got {kl_val:.4f}"
    )


def test_encoder_gradient_flow() -> None:
    from models.latent_vae import LatentVAEEncoder

    enc = LatentVAEEncoder(gene_dim=8, latent_dim=4, hidden_dim=16, n_layers=1)
    data = _build_data(B=1, N=16, gene_dim=8)
    mu, logvar = enc(data.node_features, data.positions, data.node_mask)
    loss = (mu.pow(2).mean() + logvar.pow(2).mean())
    loss.backward()
    g = sum(
        p.grad.abs().sum().item()
        for p in enc.parameters() if p.grad is not None
    )
    assert g > 0, "no gradient reached encoder"


def test_decoder_gradient_flow() -> None:
    from models.latent_vae import LatentVAEDecoder

    dec = LatentVAEDecoder(gene_dim=8, latent_dim=4, hidden_dim=16, n_layers=1)
    data = _build_data(B=1, N=16, gene_dim=8)
    z = torch.randn(1, 16, 4, requires_grad=True)
    pos = dec(data.node_features, z, data.node_mask)
    pos.pow(2).mean().backward()
    assert z.grad is not None and z.grad.abs().sum().item() > 0
    g = sum(p.grad.abs().sum().item() for p in dec.parameters() if p.grad is not None)
    assert g > 0, "no gradient reached decoder"


# ---------------------------------------------------------------------------
# LatentDiffusionWrapper
# ---------------------------------------------------------------------------


def _make_wrapper(gene_dim: int = 16, latent_dim: int = 8):
    from models.latent_diffusion_wrapper import LatentDiffusionWrapper

    class _Cfg:
        def __init__(self, d):
            self.__dict__.update(d)
        def get(self, k, default=None):
            return getattr(self, k, default)

    input_dims = {
        "node_features_dimensions": gene_dim,
        "diffusion_time_dimensions": 1,
    }
    hidden_dims = {
        "dx": 32, "dy": 1, "num_heads": 4,
        "dim_ffX": 32, "dim_ffy": 32, "dd": 16,
        "output_features_to_pos_dims": 4,
    }
    return LatentDiffusionWrapper(
        input_dims=input_dims,
        n_layers=2,
        hidden_mlp_dims={"X": 32, "y": 32, "pos": 16},
        hidden_dims=hidden_dims,
        output_dims={"node_features_dimensions": gene_dim,
                     "diffusion_time_dimensions": 0},
        ldm_cfg=_Cfg(dict(
            latent_dim=latent_dim,
            vae_hidden_dim=32, vae_n_layers=2, vae_n_heads=4,
            denoiser_hidden_dim=32, denoiser_n_layers=2,
            denoiser_n_heads=4, denoiser_mlp_ratio=2,
            denoiser_time_embed_dim=32,
        )),
    )


def test_wrapper_forward_shapes() -> None:
    B, N, gene_dim, k = 2, 16, 16, 8
    wrap = _make_wrapper(gene_dim=gene_dim, latent_dim=k)
    data = _build_data(B=B, N=N, gene_dim=gene_dim, pad_last=2)
    # Need to set _ldm_z_t on data — that's what apply_noise does.
    data._ldm_z_t = torch.randn(B, N, k) * data.node_mask.unsqueeze(-1).float()
    pred = wrap(data)
    assert pred.positions.shape == (B, N, 2)
    assert hasattr(pred, "_ldm_z_0_pred")
    assert pred._ldm_z_0_pred.shape == (B, N, k)
    # Padding cells zero on positions and z_0_pred.
    assert pred.positions[0, -2:].abs().max().item() < 1e-5
    assert pred._ldm_z_0_pred[0, -2:].abs().max().item() < 1e-5


def test_wrapper_propagates_encoder_stashes() -> None:
    """When apply_noise has set _ldm_mu/_ldm_logvar/_ldm_z_0_target on
    z_t, the wrapper's forward must copy them onto pred so the loss
    components can read them."""
    B, N, k = 1, 16, 8
    wrap = _make_wrapper(gene_dim=16, latent_dim=k)
    data = _build_data(B=B, N=N, gene_dim=16)
    data._ldm_z_t = torch.randn(B, N, k)
    data._ldm_mu = torch.randn(B, N, k)
    data._ldm_logvar = torch.randn(B, N, k)
    data._ldm_z_0_target = torch.randn(B, N, k)
    pred = wrap(data)
    assert pred._ldm_mu is data._ldm_mu
    assert pred._ldm_logvar is data._ldm_logvar
    assert pred._ldm_z_0_target is data._ldm_z_0_target


def test_wrapper_inference_path_no_encoder_stashes() -> None:
    """At inference (no encoder run), only _ldm_z_t is set. The
    wrapper must still produce a valid pred (no mu/logvar/target on
    it, but z_0_pred + positions are still well-formed)."""
    B, N, k = 1, 16, 8
    wrap = _make_wrapper(gene_dim=16, latent_dim=k)
    data = _build_data(B=B, N=N, gene_dim=16)
    data._ldm_z_t = torch.randn(B, N, k)
    pred = wrap(data)
    assert pred.positions.shape == (B, N, 2)
    assert hasattr(pred, "_ldm_z_0_pred")
    assert not hasattr(pred, "_ldm_mu")
    assert not hasattr(pred, "_ldm_logvar")


# ---------------------------------------------------------------------------
# LatentDiffusionModel (noise model)
# ---------------------------------------------------------------------------


def _make_noise_model(latent_dim: int = 8, n_steps: int = 5):
    from utils.diffusion_model.diffusion.latent_diffusion_model import (
        LatentDiffusionModel,
    )

    cfg = OmegaConf.create({
        "model": {
            "latent_diffusion": {
                "latent_dim": latent_dim,
                "n_diffusion_steps": n_steps,
                "epsilon_t": 1e-3,
            },
        },
    })
    nm = LatentDiffusionModel(cfg)
    return nm


def test_noise_model_apply_noise() -> None:
    nm = _make_noise_model(latent_dim=8, n_steps=5)
    wrap = _make_wrapper(gene_dim=16, latent_dim=8)
    nm._ldm_wrapper = wrap
    data = _build_data(B=2, N=12, gene_dim=16)
    z_t = nm.apply_noise(data)
    assert z_t._ldm_z_t.shape == (2, 12, 8)
    assert z_t._ldm_z_0_target.shape == (2, 12, 8)
    assert z_t._ldm_mu.shape == (2, 12, 8)
    assert z_t._ldm_logvar.shape == (2, 12, 8)
    # positions field is a 2D visualization, well-formed.
    assert z_t.positions.shape == (2, 12, 2)
    assert torch.isfinite(z_t.positions).all()


def test_noise_model_sample_limit_dist() -> None:
    nm = _make_noise_model(latent_dim=8, n_steps=5)
    wrap = _make_wrapper(gene_dim=16, latent_dim=8)
    nm._ldm_wrapper = wrap
    B, N = 1, 12
    z = nm.sample_limit_dist(
        node_features=torch.randn(B, N, 16),
        node_mask=torch.ones(B, N, dtype=torch.bool),
        cell_ID=torch.arange(N).repeat(B, 1).unsqueeze(-1),
        cell_class=torch.zeros(B, N, 1, dtype=torch.long),
    )
    assert z._ldm_z_t.shape == (B, N, 8)
    assert z.positions.shape == (B, N, 2)
    # At inference, no encoder stashes.
    assert not hasattr(z, "_ldm_mu")


def test_noise_model_euler_at_s0_recovers_z0_pred() -> None:
    """Same property as the EDM-FM noise model: at s_int=0, the
    convex-combo Euler step returns exactly z_0_pred (modulo
    masking)."""
    nm = _make_noise_model(latent_dim=4, n_steps=5)
    wrap = _make_wrapper(gene_dim=8, latent_dim=4)
    nm._ldm_wrapper = wrap

    B, N, k = 1, 10, 4
    z_t = torch.randn(B, N, k)
    z_0_pred = torch.randn(B, N, k)
    z_t_holder = DataHolder(
        node_features=torch.randn(B, N, 8),
        positions=torch.randn(B, N, 2),
        diffusion_time=torch.tensor([[0.4]]),
        cell_class=torch.zeros(B, N, 1, dtype=torch.long),
        cell_ID=torch.arange(N).repeat(B, 1).unsqueeze(-1),
        t_int=torch.tensor([[400]]),
        t=torch.tensor([[0.4]]),
        node_mask=torch.ones(B, N, dtype=torch.bool),
    )
    z_t_holder._ldm_z_t = z_t.clone()
    pred = DataHolder(
        node_features=torch.randn(B, N, 8),
        positions=torch.randn(B, N, 2),
        diffusion_time=torch.tensor([[0.4]]),
        cell_class=torch.zeros(B, N, 1, dtype=torch.long),
        cell_ID=torch.arange(N).repeat(B, 1).unsqueeze(-1),
        t_int=torch.tensor([[400]]),
        t=torch.tensor([[0.4]]),
        node_mask=torch.ones(B, N, dtype=torch.bool),
    )
    pred._ldm_z_0_pred = z_0_pred.clone()

    z_s = nm.sample_zs_from_zt_and_pred(z_t_holder, pred, torch.tensor(0))
    err = (z_s._ldm_z_t - z_0_pred).abs().max().item()
    assert err < 1e-5, f"at s=0 Euler should give z_0_pred; err={err}"


# ---------------------------------------------------------------------------
# Loss components
# ---------------------------------------------------------------------------


def _make_loss_function():
    from metrics.loss_function import LossFunction
    cfg = OmegaConf.create({
        "model": {"loss": {
            "pairwise_distance_mse": {"enabled": False, "weight": 1.0},
            "latent_fm_mse": {"weight": 1.0},
            "latent_kl": {"weight": 0.1},
        }},
    })
    return LossFunction(cfg=cfg)


def test_loss_no_op_when_stashes_missing() -> None:
    """For a non-LDM pred (no stashes), both latent components return
    graph-attached zero."""
    loss_fn = _make_loss_function()
    B, N = 1, 16
    data = _build_data(B=B, N=N, gene_dim=8)
    pred = DataHolder(
        node_features=torch.randn(B, N, 1),
        positions=torch.randn(B, N, 2),
        diffusion_time=torch.zeros(B, 1),
        cell_class=torch.zeros(B, N, 1, dtype=torch.long),
        cell_ID=torch.arange(N).repeat(B, 1).unsqueeze(-1),
        t_int=torch.zeros(B, 1, dtype=torch.long),
        t=torch.zeros(B, 1),
        node_mask=torch.ones(B, N, dtype=torch.bool),
    )
    total, per_comp = loss_fn.compute_loss(pred, data)
    assert abs(per_comp.get("latent_fm_mse", 0.0)) < 1e-9
    assert abs(per_comp.get("latent_kl", 0.0)) < 1e-9


def test_latent_fm_mse_with_stashes() -> None:
    """When _ldm_z_0_pred and _ldm_z_0_target are present, the loss
    matches the manually computed masked MSE."""
    loss_fn = _make_loss_function()
    B, N, k = 1, 12, 4
    data = _build_data(B=B, N=N, gene_dim=8, pad_last=2)
    pred = DataHolder(
        node_features=torch.randn(B, N, 1),
        positions=torch.randn(B, N, 2),
        diffusion_time=torch.zeros(B, 1),
        cell_class=torch.zeros(B, N, 1, dtype=torch.long),
        cell_ID=torch.arange(N).repeat(B, 1).unsqueeze(-1),
        t_int=torch.zeros(B, 1, dtype=torch.long),
        t=torch.zeros(B, 1),
        node_mask=data.node_mask,
    )
    z0_pred = torch.randn(B, N, k, requires_grad=True)
    z0_target = torch.randn(B, N, k)
    pred._ldm_z_0_pred = z0_pred
    pred._ldm_z_0_target = z0_target
    _, per_comp = loss_fn.compute_loss(pred, data)
    # Reference: masked MSE over real cells × all k dims.
    mask = data.node_mask.unsqueeze(-1).float()
    sq = (z0_pred - z0_target).pow(2) * mask
    n_real_dims = mask.sum() * float(k)
    expected = (sq.sum() / n_real_dims).item()
    got = per_comp["latent_fm_mse"]
    assert abs(got - expected) < 1e-5, f"latent_fm_mse: got {got}, expected {expected}"


def test_latent_kl_with_stashes_and_grad() -> None:
    loss_fn = _make_loss_function()
    B, N, k = 1, 10, 4
    data = _build_data(B=B, N=N, gene_dim=8)
    pred = DataHolder(
        node_features=torch.randn(B, N, 1),
        positions=torch.randn(B, N, 2),
        diffusion_time=torch.zeros(B, 1),
        cell_class=torch.zeros(B, N, 1, dtype=torch.long),
        cell_ID=torch.arange(N).repeat(B, 1).unsqueeze(-1),
        t_int=torch.zeros(B, 1, dtype=torch.long),
        t=torch.zeros(B, 1),
        node_mask=data.node_mask,
    )
    mu = torch.zeros(B, N, k, requires_grad=True)
    logvar = torch.zeros(B, N, k, requires_grad=True)  # sigma = 1 → KL = 0
    pred._ldm_mu = mu
    pred._ldm_logvar = logvar
    total, per_comp = loss_fn.compute_loss(pred, data)
    assert abs(per_comp["latent_kl"]) < 1e-6, (
        f"KL of N(0,I) || N(0,I) should be 0; got {per_comp['latent_kl']}"
    )
    # Push mu away from zero and confirm KL goes up.
    mu = mu.detach() + 2.0
    mu = mu.requires_grad_(True)
    pred._ldm_mu = mu
    total, per_comp = loss_fn.compute_loss(pred, data)
    assert per_comp["latent_kl"] > 1.0, (
        f"KL with mu=2 should be >>0; got {per_comp['latent_kl']}"
    )
    # Gradient flows back to mu.
    total.backward()
    assert mu.grad is not None and mu.grad.abs().sum().item() > 0


# ---------------------------------------------------------------------------
# End-to-end joint-training step
# ---------------------------------------------------------------------------


def test_end_to_end_joint_training_step() -> None:
    """Simulate one optimizer step of joint LDM training:
       encode → noise → forward → loss → backward.

    Asserts that gradient reaches the encoder, denoiser, AND decoder
    submodules. This is the bug a real run could expose first: if
    one of the three doesn't see gradient, joint training silently
    fails for that piece.
    """
    from metrics.loss_function import LossFunction
    from utils.diffusion_model.diffusion.latent_diffusion_model import (
        LatentDiffusionModel,
    )

    B, N, gene_dim, k = 1, 12, 16, 8
    wrap = _make_wrapper(gene_dim=gene_dim, latent_dim=k)
    nm = _make_noise_model(latent_dim=k, n_steps=5)
    nm._ldm_wrapper = wrap

    cfg = OmegaConf.create({
        "model": {"loss": {
            # Include the recon loss too so the decoder gets gradient.
            "pairwise_distance_mse": {"enabled": True, "weight": 1.0},
            "latent_fm_mse": {"weight": 1.0},
            "latent_kl": {"weight": 0.001},
        }},
    })
    loss_fn = LossFunction(cfg=cfg)
    data = _build_data(B=B, N=N, gene_dim=gene_dim)

    z_t = nm.apply_noise(data)
    pred = wrap(z_t)
    total, per_comp = loss_fn.compute_loss(pred, data)
    assert torch.isfinite(total), f"total loss non-finite: {total.item()}"

    total.backward()
    enc_grad = sum(
        p.grad.abs().sum().item()
        for p in wrap.encoder.parameters() if p.grad is not None
    )
    dec_grad = sum(
        p.grad.abs().sum().item()
        for p in wrap.decoder.parameters() if p.grad is not None
    )
    # Denoiser params are the union of den_gene_embed, den_z_embed,
    # den_t_embed, den_blocks, den_final_norm, den_final_modulation,
    # den_proj_out, den_node_features_proj.
    denoiser_modules = [
        wrap.den_gene_embed, wrap.den_z_embed, wrap.den_t_embed,
        wrap.den_final_modulation, wrap.den_proj_out, wrap.den_node_features_proj,
    ] + list(wrap.den_blocks)
    den_grad = 0.0
    for m in denoiser_modules:
        for p in m.parameters():
            if p.grad is not None:
                den_grad += p.grad.abs().sum().item()

    assert enc_grad > 0, f"encoder received zero gradient (KL + FM should both touch it)"
    assert dec_grad > 0, f"decoder received zero gradient (recon should touch it)"
    assert den_grad > 0, f"denoiser received zero gradient (FM-on-z should touch it)"


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def main() -> int:
    tests = [
        ("VAE encoder forward shapes + mask",     test_encoder_shapes_and_mask),
        ("VAE decoder forward shapes + centring", test_decoder_shapes_and_mask),
        ("reparameterize + KL formula sanity",    test_reparam_and_kl_formula),
        ("encoder gradient flow",                 test_encoder_gradient_flow),
        ("decoder gradient flow",                 test_decoder_gradient_flow),
        ("wrapper forward shapes",                test_wrapper_forward_shapes),
        ("wrapper propagates encoder stashes",    test_wrapper_propagates_encoder_stashes),
        ("wrapper inference path (no encoder)",   test_wrapper_inference_path_no_encoder_stashes),
        ("noise model apply_noise shapes",        test_noise_model_apply_noise),
        ("noise model sample_limit_dist",         test_noise_model_sample_limit_dist),
        ("FM Euler at s=0 recovers z_0_pred",     test_noise_model_euler_at_s0_recovers_z0_pred),
        ("loss no-op when LDM stashes missing",   test_loss_no_op_when_stashes_missing),
        ("latent_fm_mse matches manual masked-MSE", test_latent_fm_mse_with_stashes),
        ("latent_kl correct + gradient flows",    test_latent_kl_with_stashes_and_grad),
        ("end-to-end joint training step grads enc/dec/denoiser",
                                                  test_end_to_end_joint_training_step),
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
