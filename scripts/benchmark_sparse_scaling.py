#!/usr/bin/env python3
"""Empirical scaling benchmark for the de-quadratified sparse_local_distance
training step.

Measures peak GPU memory and steady-state (cache-warm) step time of the
loss forward+backward across a controlled sweep of N, and fits the log-log
exponent (≈1.0 = linear, ≈2.0 = quadratic). This isolates N as the single
variable — unlike comparing MMC vs CNS, where gene count and batch
composition are confounded.

Why this is the right test of the claim
---------------------------------------
The pieces I changed live in the LOSS: true-kNN build (kdtree → O(N log N))
+ local/random gather (O(N·(k+r))). The backbone is LUNA's
LinearAttentionTransformer (O(N) by construction) and the EDM head skips its
(N,N) D_sq at train time, so the loss is the part whose scaling needs
empirical proof. ``local_k + n_random`` is a CONSTANT, so the asymptotic
(large-N) regime should be linear; small N (where k+r ≈ N) is a transition.

Usage (one GPU):
  python scripts/benchmark_sparse_scaling.py --backend kdtree
  python scripts/benchmark_sparse_scaling.py --backend brute   # contrast
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

try:
    import torch
except ImportError:
    print("torch is required to run this benchmark."); sys.exit(1)
from omegaconf import OmegaConf

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from metrics.loss_function import LossFunction          # noqa: E402
from utils.data.dataholder import DataHolder            # noqa: E402


def build_loss(local_k, n_random, backend, balance):
    cfg = OmegaConf.create({"model": {"loss": {"sparse_local_distance": {
        "enabled": True, "weight": 1.0, "local_k": local_k,
        "n_random": n_random, "global_weight": 0.1, "weight_fn": "none",
        "sigma": 1.0, "knn_chunk": 1024, "cache_true": True,
        "knn_backend": backend, "n_sample": 0, "n_landmarks": 0,
        "landmark_weight": 0.1, "landmark_mode": "fps", "balance": balance,
        "warmup_steps": 0,
    }}}})
    return LossFunction(cfg)


def make_holders(N, kd, device, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    pos = (torch.randn(1, N, 2, generator=g)
           * torch.tensor([5.0, 1.0])).to(device)
    h = torch.randn(1, N, kd, generator=g).to(device).requires_grad_(True)
    mask = torch.ones(1, N, dtype=torch.bool, device=device)
    feats = torch.zeros(1, N, 4, device=device)
    pred = DataHolder(positions=pos.clone(), node_features=feats,
                      diffusion_time=0, node_mask=mask)
    true = DataHolder(positions=pos, node_features=feats,
                      diffusion_time=0, node_mask=mask)
    pred.edm_h = h
    pred.edm_D = None
    return pred, true, h


def bench_one(loss, N, kd, device, iters):
    pred, true, h = make_holders(N, kd, device)
    cuda = device.type == "cuda"
    # cold call (builds + caches the true kNN — the O(N log N) / O(N^2) step)
    if cuda:
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    loss._compute_sparse_local_distance(pred, true).backward()
    if cuda:
        torch.cuda.synchronize()
    cold = time.perf_counter() - t0
    # steady-state (cache warm): the per-step gradient path
    if cuda:
        torch.cuda.reset_peak_memory_stats()
    times = []
    for _ in range(iters):
        h.grad = None
        if cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        loss._compute_sparse_local_distance(pred, true).backward()
        if cuda:
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    peak = (torch.cuda.max_memory_allocated() / 1e9) if cuda else float("nan")
    del pred, true, h
    if cuda:
        torch.cuda.empty_cache()
    return cold, float(np.median(times)), peak


def _slope(xs, ys):
    return float(np.polyfit(np.log(xs), np.log(ys), 1)[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes",
                    default="2000,4000,8000,16000,32000,64000,128000,256000")
    ap.add_argument("--local_k", type=int, default=512)
    ap.add_argument("--n_random", type=int, default=512)
    ap.add_argument("--embed_dim", type=int, default=8)
    ap.add_argument("--backend", default="kdtree",
                    choices=["kdtree", "brute"])
    ap.add_argument("--balance", default="none", choices=["none", "equal"])
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--device",
                    default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    device = torch.device(args.device)
    sizes = [int(s) for s in args.sizes.split(",")]
    loss = build_loss(args.local_k, args.n_random, args.backend, args.balance)
    print(f"device={device}  backend={args.backend}  local_k={args.local_k}  "
          f"n_random={args.n_random}  embed_dim={args.embed_dim}")
    print(f"{'N':>10} {'cold_s':>10} {'step_s':>10} {'peak_GB':>10}")
    Ns, steps, mems = [], [], []
    for N in sizes:
        try:
            cold, step, peak = bench_one(
                loss, N, args.embed_dim, device, args.iters)
        except RuntimeError as e:
            print(f"{N:>10}  OOM/err: {str(e)[:70]}")
            break
        print(f"{N:>10} {cold:>10.4f} {step:>10.4f} {peak:>10.3f}")
        Ns.append(N); steps.append(step); mems.append(peak)

    if len(Ns) >= 3:
        # asymptotic = the upper half of N (where k+r << N → true regime)
        half = len(Ns) // 2
        print("\nfitted log-log exponents (≈1.0 linear, ≈2.0 quadratic):")
        print(f"  step time : all N -> N^{_slope(Ns, steps):.2f}   "
              f"large N -> N^{_slope(Ns[half:], steps[half:]):.2f}")
        if device.type == "cuda":
            print(f"  peak mem  : all N -> N^{_slope(Ns, mems):.2f}   "
                  f"large N -> N^{_slope(Ns[half:], mems[half:]):.2f}")
        print("note: small-N points sit in the k+r≈N transition; the "
              "large-N slope is the asymptotic complexity.")


if __name__ == "__main__":
    main()
