#!/usr/bin/env python
"""
Quick runtime sanity check for the flow-matching pipeline.

Build a freshly-initialised ScGG (LUNA architecture, flow_matching
objective) on a small synthetic batch and print the forward-pass
norms. The point is to catch "model output is stuck at zero" bugs
*before* committing to a multi-hour training run.

Expected output for a healthy initialisation:

    x_hat_0_norm       > 0.05          (model produces non-trivial output)
    u_norm             ~ 1.3           (per-cell target velocity norm)
    fm_pairwise_dist_mse  ~ 0.05 - 0.3 (cdist on points in [-0.5, 0.5]^2)
    fm_velocity_mse    irrelevant      (weight should be 0 with x_0 target)

Run it in the scGG env (not the LUNA env):

    python /nfs/team361/sb75/scgg/scripts/sanity_check_flow_matching.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(REPO_SRC))

import yaml
import torch

from scgg.model.scgg import ScGG


def main() -> int:
    cfg_path = REPO_SRC / "scgg" / "configs" / "default.yaml"
    cfg = yaml.safe_load(open(cfg_path))

    print(f"Using config: {cfg_path}")
    print(f"  velocity_net.type    = {cfg['model']['velocity_net']['type']}")
    print(f"  flow.pairwise_dist_weight = {cfg['model']['flow']['pairwise_dist_weight']}")
    print(f"  flow.velocity_mse_weight  = {cfg['model']['flow']['velocity_mse_weight']}")
    print(f"  data.normalize        = {cfg['data']['normalize']}")
    print(f"  data.scale            = {cfg['data']['scale']}")
    print(f"  data.coord_normalize  = {cfg['data']['coord_normalize']}")
    print()

    torch.manual_seed(0)
    n_genes = 254
    n_cells = 2000

    # Fresh ScGG.
    model = ScGG(n_genes=n_genes, config=cfg)
    model.eval()

    # Synthetic batch: raw-counts-like gene expression + per-section
    # min-max coords in [-0.5, 0.5]^2.
    gene_expr = torch.rand(n_cells, n_genes) * 5.0          # raw-ish counts
    coords = torch.rand(n_cells, 2) - 0.5                    # [-0.5, 0.5]

    with torch.no_grad():
        cell_embed, section_embed = model.encode(gene_expr)
    print("Encoder output (cell_embed):")
    print(f"  shape  = {tuple(cell_embed.shape)}")
    print(f"  mean   = {cell_embed.mean().item():.4f}")
    print(f"  std    = {cell_embed.std().item():.4f}")
    print(f"  norm/cell = {cell_embed.norm(dim=-1).mean().item():.4f}")
    print()

    k_target = torch.full((n_cells,), 10, dtype=torch.long)

    print("compute_loss() on a fresh model with synthetic data:")
    for trial in range(3):
        loss, metrics = model.flow.compute_loss(
            z_1=coords,
            cell_embed=cell_embed,
            section_embed=section_embed,
            k_target=k_target,
        )
        ms = ", ".join(
            f"{k}={v:.4g}"
            for k, v in sorted(metrics.items())
            if isinstance(v, (int, float))
        )
        print(f"  trial {trial}: total={loss.item():.4g}  |  {ms}")
    print()

    print("Direct LunaTransformerNet forward (no flow wrapping):")
    z_t = torch.randn(n_cells, 2)
    t = torch.rand(1).expand(n_cells)
    with torch.no_grad():
        net_out = model.velocity_net(z_t, t, cell_embed, section_embed, k_target)
    print(f"  output shape = {tuple(net_out.shape)}")
    print(f"  output mean  = {net_out.mean().item():.6f}")
    print(f"  output std   = {net_out.std().item():.6f}")
    print(f"  ||output||/cell = {net_out.norm(dim=-1).mean().item():.6f}")
    print()

    # Verdict
    x_norm = net_out.norm(dim=-1).mean().item()
    print("=" * 50)
    if x_norm < 0.005:
        print(f"FAIL: x_hat_0 norm is {x_norm:.6f} — model is stuck at zero.")
        print("      Something in the architecture is forcing output ≈ 0.")
        return 1
    elif x_norm < 0.1:
        print(f"WARN: x_hat_0 norm is small ({x_norm:.4f}). Should be > 0.05.")
        return 1
    else:
        print(f"OK: x_hat_0 norm = {x_norm:.4f} (looks healthy at init).")
        return 0


if __name__ == "__main__":
    sys.exit(main())
