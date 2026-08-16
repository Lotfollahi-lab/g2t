#!/usr/bin/env python
"""Write PCA embeddings into a copy of a silver dir. ONE basis, fit on train.

Third input arm alongside make_hvg_subset.py / make_svg_subset.py, and the only
one of the reduced-input arms whose protocol is fully inductive.

WHY NOT precompute_embeddings.py --encoder pca
----------------------------------------------
That path is registered in _PER_SLICE_ENCODERS: it fits a separate TruncatedSVD
on every slice, so each section's embedding lives in its own basis with its own
arbitrary component order and sign. A model trained on the train sections'
bases would receive a test section expressed in an unrelated one, and would
transfer at chance -- indistinguishable from a genuine generalisation failure,
which is precisely the thing under investigation here. It also skips centering,
so it is not PCA but uncentered SVD, whose leading component is essentially the
library-size direction.

WHAT THIS DOES INSTEAD
----------------------
Fits ONE basis on the *_train.h5ad sections only, then projects every section --
train and test -- through it. The test sections contribute nothing to the mean
or the loadings; they are only transformed. That makes this arm genuinely
inductive, unlike the scVI arm, which fits its encoder and a per-section batch
embedding on all sections including the held-out ones. Read the two together
with that asymmetry in mind.

MEMORY
------
The covariance is accumulated in GENE space (G x G, ~196 MB at G=4948) one
section at a time, so cost is independent of cell count and the full 39-section
dataset is no harder than a 2-section split. Only one section is dense at a
time.

Three passes, because the mean is needed before centering and the basis before
projecting:
  A  train only, sparse   -> n, sum  =>  mean
  B  train only, dense    -> C += Xc^T Xc            (centered, so no
                                                      catastrophic cancellation)
  C  all sections, dense  -> Z = Xc @ V, written to obsm[field]

Ranking/fitting uses log1p, matching the HVG and SVG arms; genes are centered
but NOT scaled to unit variance (scanpy's default -- scaling would amplify
low-expression noise).

REPORTED DIAGNOSTIC
-------------------
For every section, the fraction of its own centered variance the train basis
captures. A test section far below the train sections quantifies the donor /
section shift directly, in the input space, before any model is trained.

USAGE
    python make_pca_embeddings.py \
        --in_dir  /nfs/team361/sb75/DATASETS/silver/xenium_bk20 \
        --out_dir /nfs/team361/sb75/DATASETS/silver/xenium_bk20_pca50 \
        --n_pcs 50
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def log1p_dense(a) -> np.ndarray:
    import scipy.sparse as sp

    X = a.X
    X = X.toarray() if sp.issparse(X) else np.asarray(X)
    return np.log1p(np.asarray(X, dtype=np.float32))


def log1p_colsum(a) -> tuple[np.ndarray, int]:
    """Column sums of log1p(X) without densifying (log1p keeps zeros zero)."""
    import scipy.sparse as sp

    X = a.X
    if sp.issparse(X):
        Xl = X.copy().astype(np.float64)
        Xl.data = np.log1p(Xl.data)
        return np.asarray(Xl.sum(axis=0)).ravel(), int(a.n_obs)
    Xl = np.log1p(np.asarray(X, dtype=np.float64))
    return Xl.sum(axis=0), int(a.n_obs)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--in_dir", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--n_pcs", type=int, default=50)
    p.add_argument("--field", default="",
                   help="obsm key to write (default: X_pca{n_pcs})")
    p.add_argument("--as_x", action="store_true",
                   help="ALSO replace .X with the PCs as n_pcs pseudo-genes "
                        "(PC1..PCn). Only the scgg pipeline understands "
                        "--embedding_field; LUNA and CeLEry read .X, so this "
                        "is how they get the same input. Equivalent for scgg "
                        "too: run_scgg_train.py substitutes obsm for the gene "
                        "block and renames the columns, nothing more.")
    p.add_argument("--dry_run", action="store_true",
                   help="fit and report, write nothing")
    args = p.parse_args()

    import anndata as ad
    import pandas as pd

    in_dir, out_dir = Path(args.in_dir), Path(args.out_dir)
    field = args.field or f"X_pca{args.n_pcs}"
    train = sorted(in_dir.glob("*_train.h5ad"))
    test = sorted(in_dir.glob("*_test.h5ad"))
    if not train or not test:
        raise SystemExit(f"need both *_train.h5ad and *_test.h5ad under {in_dir}")
    print(f"{len(train)} train / {len(test)} test sections in {in_dir.name}")
    print(f"fitting on the TRAIN sections only; writing obsm[{field!r}]")

    # ---- pass A: mean over train cells -------------------------------------
    var_names, s, n = None, None, 0
    for f in train:
        a = ad.read_h5ad(f)
        if var_names is None:
            var_names = list(map(str, a.var_names))
            s = np.zeros(a.n_vars, dtype=np.float64)
        elif list(map(str, a.var_names)) != var_names:
            raise SystemExit(f"{f.name}: gene panel differs from {train[0].name}")
        si, ni = log1p_colsum(a)
        s += si
        n += ni
        del a
    mu = (s / n).astype(np.float32)
    G = len(var_names)
    print(f"  mean over {n:,} train cells x {G} genes")

    if args.n_pcs > min(n - 1, G):
        raise SystemExit(f"--n_pcs {args.n_pcs} exceeds min(n_cells-1, n_genes)")

    # ---- pass B: centered covariance in gene space -------------------------
    C = np.zeros((G, G), dtype=np.float64)
    for f in train:
        a = ad.read_h5ad(f)
        Xc = log1p_dense(a) - mu
        C += (Xc.T @ Xc).astype(np.float64)
        print(f"  gram {f.name:38s} n={a.n_obs:6d}")
        del a, Xc
    C /= (n - 1)

    evals, evecs = np.linalg.eigh(C)             # ascending
    order = np.argsort(-evals)[:args.n_pcs]
    V = np.ascontiguousarray(evecs[:, order], dtype=np.float32)   # (G, n_pcs)
    lam = evals[order]
    total_var = float(np.trace(C))
    evr = lam / total_var
    print(f"\n  top {args.n_pcs} PCs explain {100*evr.sum():.1f}% of train variance"
          f"  (PC1 {100*evr[0]:.1f}%, PC{args.n_pcs} {100*evr[-1]:.2f}%)")
    if lam.min() <= 0:
        print("  WARNING: non-positive eigenvalue — n_pcs likely exceeds rank")

    # ---- pass C: project every section -------------------------------------
    out_dir.mkdir(parents=True, exist_ok=True) if not args.dry_run else None
    rows = []
    for f in train + test:
        a = ad.read_h5ad(f)
        Xc = log1p_dense(a) - mu
        Z = (Xc @ V).astype(np.float32)
        # How much of THIS section's own variance the train basis captures.
        tot = float((Xc ** 2).sum())
        cap = float((Z ** 2).sum())
        frac = cap / tot if tot > 0 else float("nan")
        split = "train" if f.name.endswith("_train.h5ad") else "test"
        rows.append({"section": f.name, "split": split, "n_cells": int(a.n_obs),
                     "frac_variance_captured": round(frac, 4)})
        print(f"  {split:5s} {f.name:38s} n={a.n_obs:6d}  "
              f"train-basis captures {100*frac:5.1f}% of its variance")
        if not args.dry_run:
            a.obsm[field] = Z
            if args.as_x:
                # Rebuild rather than assign: .X's second axis changes length.
                # obs / obsm / uns are carried over verbatim -- obsm['spatial']
                # above all, which is the ground truth every scorer reads.
                a = ad.AnnData(
                    X=Z.copy(),
                    obs=a.obs.copy(),
                    var=pd.DataFrame(index=[f"PC{i+1}" for i in range(Z.shape[1])]),
                    obsm={k: v.copy() for k, v in a.obsm.items()},
                    uns=dict(a.uns),
                )
                if "spatial" not in a.obsm:
                    raise SystemExit(f"{f.name}: obsm['spatial'] lost — refusing "
                                     "to write a file with no ground truth")
            a.write_h5ad(out_dir / f.name)
        del a, Xc, Z

    tr = [r["frac_variance_captured"] for r in rows if r["split"] == "train"]
    te = [r["frac_variance_captured"] for r in rows if r["split"] == "test"]
    print(f"\ncaptured variance: train {np.mean(tr):.3f} | test {np.mean(te):.3f}")
    if np.mean(te) < 0.8 * np.mean(tr):
        print("  the train basis describes the test sections markedly worse — "
              "that gap IS the section/donor shift, measured before any model")

    if args.dry_run:
        print("\ndry run: nothing written")
        return 0

    (out_dir / "pca_fit.json").write_text(json.dumps({
        "in_dir": str(in_dir.resolve()), "field": field, "n_pcs": args.n_pcs,
        "fit_on": "train_sections_only", "n_train_cells": n, "n_genes": G,
        "preprocessing": "log1p, gene-centered on train mean, not scaled",
        "explained_variance_ratio_sum": float(evr.sum()),
        "explained_variance_ratio": [float(x) for x in evr],
        "sections": rows,
    }, indent=2))
    np.save(out_dir / "pca_loadings.npy", V)

    print(f"\nwrote {len(train)+len(test)} files to {out_dir}")
    print(f"provenance: {out_dir/'pca_fit.json'}  loadings: pca_loadings.npy")
    print(f"--embedding_field {field}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
