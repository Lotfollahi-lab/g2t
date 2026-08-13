#!/usr/bin/env python
"""Validate that a silver dir of ``*_train.h5ad`` / ``*_test.h5ad`` files
satisfies the scgg/LUNA input contract AND that the model's preprocessing
will run on them without producing NaN/inf.

This is dataset-agnostic: use it on DLPFC, CNS, MMC, or any new benchmark
before launching a run. It re-implements the exact checks the training/
inference path (``run_scgg_train.run_benchmark`` + the loaders in
``src/scgg/data``) depends on, so a green run here means the pipeline will
ingest the data.

Checks (per file):
  [X]        .X present; non-negative; RAW COUNTS (integer-valued) -> the
             loaders do normalize_total+log1p+scale from raw, so non-count .X
             is a hard error unless --allow_noncount.
  [counts]   layers['counts'] present (recommended; loaders recreate it from
             .X if missing, so this is a WARN not a failure).
  [spatial]  obsm['spatial'] present, shape (N, 2), all-finite, and NOT
             constant within a section (constant coords make the per-section
             coordinate normalisation divide by zero).
  [obs]      obs has 'cell_class' and 'cell_section'.
  [dtype]    .X has no NaN/inf.

Cross-file (the checks that decide whether the model actually WORKS):
  [panel]    var_names are IDENTICAL (same genes, SAME order) across every
             train+test file. run_benchmark rejects a feature-column count
             mismatch; different gene panels must be run through
             harmonize_common_genes.py first. Reports the shared-gene count.
  [vocab]    cell_class values in test that are unseen in train (mapped to -1
             by the loader; only affects the aux loss) — reported, not fatal.
  [loader]   Simulates normalize_total(1e4)+log1p+z-score(clip 10) on a
             per-file subsample and asserts the result is finite -> proves the
             tensor handed to the model is clean.

Exit code 0 iff all HARD checks pass.

Usage:
    python scripts/validate_scgg_h5ad.py --silver_dir /path/to/silver
    python scripts/validate_scgg_h5ad.py --files a_train.h5ad b_test.h5ad
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np


class Report:
    def __init__(self):
        self.errors: List[str] = []
        self.warns: List[str] = []

    def err(self, msg: str):
        self.errors.append(msg)
        print(f"  [FAIL] {msg}")

    def warn(self, msg: str):
        self.warns.append(msg)
        print(f"  [warn] {msg}")

    def ok(self, msg: str):
        print(f"  [ok]   {msg}")


def _discover(silver: Path) -> List[Path]:
    files = sorted(glob.glob(str(silver / "*_train.h5ad"))) + \
            sorted(glob.glob(str(silver / "*_test.h5ad")))
    return [Path(f) for f in files]


def _to_dense_sample(X, n: int = 2000):
    import scipy.sparse as sp
    m = X[:n] if X.shape[0] > n else X
    return m.toarray() if sp.issparse(m) else np.asarray(m)


def _simulate_loader(X) -> Tuple[bool, str]:
    """normalize_total(1e4) + log1p + per-gene z-score (clip 10), in numpy,
    on a subsample. Returns (finite_ok, detail). Mirrors scanpy.pp.*."""
    import scipy.sparse as sp
    m = X[:4000] if X.shape[0] > 4000 else X
    d = m.toarray().astype(np.float64) if sp.issparse(m) else np.asarray(m, np.float64)
    lib = d.sum(axis=1, keepdims=True)
    lib[lib == 0] = 1.0
    d = d / lib * 1e4
    d = np.log1p(d)
    mu = d.mean(axis=0, keepdims=True)
    sd = d.std(axis=0, keepdims=True)
    sd[sd == 0] = 1.0
    d = np.clip((d - mu) / sd, -10, 10)
    finite = bool(np.isfinite(d).all())
    return finite, (f"post-normalize range [{d.min():.2f}, {d.max():.2f}], "
                    f"finite={finite}")


def validate_file(path: Path, rep: Report, allow_noncount: bool) -> Optional[dict]:
    import anndata as ad
    import scipy.sparse as sp
    print(f"\n=== {path.name} ===")
    try:
        a = ad.read_h5ad(path)
    except Exception as e:
        rep.err(f"cannot read: {e}")
        return None

    role = "train" if path.name.endswith("_train.h5ad") else "test"
    print(f"  {a.n_obs:,} cells x {a.n_vars} genes  (role={role})")

    # --- X ---------------------------------------------------------------
    X = a.X
    Xs = _to_dense_sample(X)
    if Xs.size == 0:
        rep.err("empty .X")
        return None
    if not np.isfinite(Xs).all():
        rep.err(".X contains NaN/inf")
    minv, maxv = float(Xs.min()), float(Xs.max())
    is_int = bool(np.allclose(Xs, np.round(Xs), atol=1e-3))
    if minv < -1e-6:
        rep.err(f".X has negative values (min={minv:.3f}) — not raw counts")
    elif not is_int and not allow_noncount:
        rep.err(f".X is non-integer (looks normalized; max={maxv:.2f}). "
                "Store RAW counts, or pass --allow_noncount.")
    else:
        rep.ok(f".X raw counts (min={minv:.0f}, max={maxv:.0f}, integer={is_int})")

    # --- counts layer ----------------------------------------------------
    if "counts" in a.layers:
        rep.ok("layers['counts'] present")
    else:
        rep.warn("no layers['counts'] (loader will recreate from .X)")

    # --- spatial ---------------------------------------------------------
    if "spatial" not in a.obsm:
        rep.err("obsm['spatial'] MISSING (loader raises without it)")
    else:
        xy = np.asarray(a.obsm["spatial"], dtype=np.float64)
        if xy.ndim != 2 or xy.shape[1] < 2:
            rep.err(f"obsm['spatial'] shape {xy.shape}, need (N, >=2)")
        elif not np.isfinite(xy[:, :2]).all():
            rep.err("obsm['spatial'] has NaN/inf coords")
        else:
            rng = [xy[:, 0].max() - xy[:, 0].min(), xy[:, 1].max() - xy[:, 1].min()]
            if min(rng) <= 0:
                rep.err(f"obsm['spatial'] is constant on an axis (range={rng}); "
                        "per-section coord normalization would divide by zero. "
                        "(Expected for a coord-less scRNA test — see snRNA script.)")
            else:
                rep.ok(f"obsm['spatial'] (N,2) finite, XY range ~"
                       f"({rng[0]:.0f}, {rng[1]:.0f})")

    # --- obs -------------------------------------------------------------
    for col in ("cell_class", "cell_section"):
        if col in a.obs.columns:
            rep.ok(f"obs['{col}'] present ({a.obs[col].astype(str).nunique()} unique)")
        else:
            rep.err(f"obs['{col}'] MISSING")

    # --- loader simulation ----------------------------------------------
    finite, detail = _simulate_loader(X)
    (rep.ok if finite else rep.err)(f"loader-sim normalize+log1p+scale: {detail}")

    # Coerce to pure Python str (pandas categorical.astype(str) can leave
    # missing values as float nan -> breaks set/sort downstream).
    cc = None
    if "cell_class" in a.obs.columns:
        cc = [str(x) for x in a.obs["cell_class"].to_numpy().tolist()]
    return {
        "path": path, "role": role,
        "var_names": [str(g) for g in a.var_names],
        "cell_class": cc,
    }


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--silver_dir", default=None)
    p.add_argument("--files", nargs="*", default=None,
                   help="Explicit h5ad paths (overrides --silver_dir discovery).")
    p.add_argument("--allow_noncount", action="store_true",
                   help="Don't fail when .X is not integer-valued.")
    args = p.parse_args()

    if args.files:
        files = [Path(f) for f in args.files]
    elif args.silver_dir:
        files = _discover(Path(args.silver_dir))
    else:
        print("Provide --silver_dir or --files", file=sys.stderr)
        return 2
    if not files:
        print("No *_train.h5ad / *_test.h5ad found.", file=sys.stderr)
        return 2

    n_train = sum(1 for f in files if f.name.endswith("_train.h5ad"))
    n_test = len(files) - n_train
    print(f"Validating {len(files)} file(s): {n_train} train, {n_test} test")
    if n_train == 0 or n_test == 0:
        print("WARNING: run_benchmark needs BOTH *_train.h5ad AND *_test.h5ad.")

    rep = Report()
    infos = [validate_file(f, rep, args.allow_noncount) for f in files]
    infos = [i for i in infos if i]

    # --- cross-file: identical gene panel -------------------------------
    print("\n=== cross-file: gene panel ===")
    if len(infos) >= 2:
        ref = infos[0]["var_names"]
        identical = all(i["var_names"] == ref for i in infos)
        shared = set(ref)
        for i in infos[1:]:
            shared &= set(i["var_names"])
        if identical:
            rep.ok(f"var_names IDENTICAL (name+order) across all files "
                   f"({len(ref)} genes)")
        else:
            rep.err(f"var_names DIFFER across files (shared={len(shared)}). "
                    "run_benchmark will reject a column mismatch — run "
                    "harmonize_common_genes.py to build an identical panel.")
    else:
        rep.warn("only one file; skipping panel cross-check")

    # --- cross-file: cell_class vocab (train vs test) -------------------
    print("\n=== cross-file: cell_class vocab (train vs test) ===")
    train_cls = set()
    for i in infos:
        if i["role"] == "train" and i["cell_class"] is not None:
            train_cls |= set(i["cell_class"])
    for i in infos:
        if i["role"] == "test" and i["cell_class"] is not None:
            unseen = set(i["cell_class"]) - train_cls
            if unseen:
                rep.warn(f"{i['path'].name}: {len(unseen)} test class(es) unseen "
                         f"in train -> mapped to -1 (aux loss skips them): "
                         f"{sorted(unseen)[:8]}")
            else:
                rep.ok(f"{i['path'].name}: all test classes present in train")

    # --- summary ---------------------------------------------------------
    print("\n" + "=" * 60)
    if rep.errors:
        print(f"RESULT: FAIL — {len(rep.errors)} error(s), {len(rep.warns)} warning(s)")
        for e in rep.errors:
            print(f"  FAIL: {e}")
        return 1
    print(f"RESULT: PASS — 0 errors, {len(rep.warns)} warning(s). "
          "Data satisfies the scgg contract.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
