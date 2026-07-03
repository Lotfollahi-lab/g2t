#!/usr/bin/env python
"""Restrict a mixed-panel silver dir to the genes COMMON to every slice, so
LUNA/scgg can train on it (they require identical gene columns across train
and test).

Motivation: in cns_luna_raw the *_train.h5ad are MERFISH ABC (~1122-gene
panel) and the *_test.h5ad are STARmap (~1022-gene panel). Harmony gave
both a shared 600-D latent; raw genes don't share a space. This script
computes the intersection of gene SYMBOLS across ALL slices (case-
insensitive), subsets + reorders every h5ad to that shared, canonically-
ordered panel, and writes to a new silver dir. Train and test then have
byte-identical var_names in the same order — a valid LUNA/scgg dataset.

X, layers['counts'], obsm['spatial'], obs and uns are preserved; only the
gene (var) axis is subset/reordered.

Usage:
    python scripts/harmonize_common_genes.py \
        --silver_dir /nfs/team361/sb75/DATASETS/silver/cns_luna_raw \
        --out_dir    /nfs/team361/sb75/DATASETS/silver/cns_luna_raw_common
"""
from __future__ import annotations

import argparse
import glob
import logging
import sys
from pathlib import Path
from typing import List

logger = logging.getLogger("harmonize_common_genes")


def _discover(silver_dir: Path) -> List[Path]:
    files = sorted(glob.glob(str(silver_dir / "*_train.h5ad"))) + \
        sorted(glob.glob(str(silver_dir / "*_test.h5ad")))
    return [Path(f) for f in files]


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--silver_dir", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--min_common", type=int, default=100,
                   help="Fail if fewer than this many common genes (guards "
                        "against a gene-ID-format mismatch, e.g. symbols vs "
                        "Ensembl).")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    import anndata as ad

    silver = Path(args.silver_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    files = _discover(silver)
    if not files:
        logger.error(f"No *_train.h5ad / *_test.h5ad under {silver}")
        return 1
    n_train = sum(1 for f in files if f.name.endswith("_train.h5ad"))
    logger.info(f"Found {len(files)} slices ({n_train} train, "
                f"{len(files) - n_train} test) under {silver}")

    # ---- pass 1: intersect gene symbols (case-insensitive) across ALL ----
    common_upper = None
    per_file_n = {}
    for f in files:
        a = ad.read_h5ad(f, backed="r")
        up = {str(g).upper() for g in a.var_names}
        per_file_n[f.name] = len(up)
        common_upper = up if common_upper is None else (common_upper & up)
    common = sorted(common_upper)
    logger.info(
        f"Per-slice gene counts: min={min(per_file_n.values())} "
        f"max={max(per_file_n.values())}"
    )
    logger.info(f"Common genes across ALL slices: {len(common)}")
    if len(common) < args.min_common:
        # Likely an ID-format mismatch (symbols vs Ensembl) — show samples.
        samples = {}
        for f in files[:1] + files[n_train:n_train + 1]:  # one train, one test
            a = ad.read_h5ad(f, backed="r")
            samples[f.name] = list(a.var_names[:5])
        logger.error(
            f"Only {len(common)} common genes (< {args.min_common}). Likely a "
            f"gene-ID-format mismatch. var_names samples: {samples}. If one "
            f"side is Ensembl and the other symbols, a mapping step is needed."
        )
        return 1

    # ---- pass 2: subset + reorder each slice to the canonical panel -------
    n_written = 0
    for f in files:
        out_path = out / f.name
        if out_path.exists() and not args.overwrite:
            logger.info(f"  exists, skipping: {f.name}")
            n_written += 1
            continue
        a = ad.read_h5ad(f)
        up = [str(g).upper() for g in a.var_names]
        # canonical upper-symbol -> first column position in this slice
        pos = {}
        for i, u in enumerate(up):
            if u in common_upper and u not in pos:
                pos[u] = i
        idx = [pos[g] for g in common]           # reorder to canonical list
        sub = a[:, idx].copy()
        sub.var_names = common                    # identical across all slices
        sub.var.index.name = "gene_symbol"
        sub.write_h5ad(out_path)
        logger.info(f"  -> {out_path.name} ({sub.n_obs:,} cells x {sub.n_vars} genes)")
        n_written += 1

    logger.info(f"Done. {n_written}/{len(files)} slices -> {out} "
                f"({len(common)} common genes)")
    return 0 if n_written else 1


if __name__ == "__main__":
    sys.exit(main())
