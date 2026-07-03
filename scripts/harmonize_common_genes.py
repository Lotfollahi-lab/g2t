#!/usr/bin/env python
"""Restrict a mixed-panel silver dir to the genes COMMON to every slice, so
LUNA/scgg can train on it (they require identical gene columns across train
and test).

In cns_luna_raw the *_train.h5ad are MERFISH ABC (~1122-gene panel, gene IDs
= **Ensembl** ENSMUSG...) and the *_test.h5ad are STARmap (~1022-gene panel,
gene IDs = **symbols** A2M, ...). So we (1) resolve every slice's genes to a
common SYMBOL space (Ensembl -> symbol via a var symbol column if present,
else via --gene_map, e.g. the ABC gene.csv), (2) intersect symbols across
ALL slices (case-insensitive), (3) subset + reorder each h5ad to that shared,
canonically-ordered panel and re-label var_names to the symbol. Train and
test then have byte-identical var_names -> a valid LUNA/scgg dataset.

X, layers['counts'], obsm['spatial'], obs and uns are preserved; only the
gene (var) axis is subset/reordered/relabelled.

Usage (ABC gene.csv supplies the Ensembl->symbol map):
    python scripts/harmonize_common_genes.py \
        --silver_dir /nfs/team361/sb75/DATASETS/silver/cns_luna_raw \
        --out_dir    /nfs/team361/sb75/DATASETS/silver/cns_luna_raw_common \
        --gene_map   /nfs/team361/sb75/DATASETS/bronze/abc_luna_raw/gene.csv
"""
from __future__ import annotations

import argparse
import glob
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

logger = logging.getLogger("harmonize_common_genes")

_SYM_COL_CANDIDATES = [
    "gene_symbol", "symbol", "gene_name", "Gene", "feature_name",
    "gene_short_name", "GeneSymbol",
]
_ID_COL_CANDIDATES = [
    "gene_identifier", "gene_id", "ensembl_id", "ensembl", "gene_identifier_label",
    "ensembl_gene_id", "Accession",
]


def _discover(silver_dir: Path) -> List[Path]:
    return [Path(f) for f in (
        sorted(glob.glob(str(silver_dir / "*_train.h5ad")))
        + sorted(glob.glob(str(silver_dir / "*_test.h5ad")))
    )]


def _looks_ensembl(names: List[str]) -> bool:
    if not names:
        return False
    hits = sum(1 for n in names if str(n).upper().startswith("ENS"))
    return hits > 0.5 * len(names)


def _load_gene_map(path: Path, id_col: Optional[str],
                   sym_col: Optional[str]) -> Dict[str, str]:
    """{ENSEMBL_UPPER -> SYMBOL_UPPER} from a mapping CSV (e.g. ABC gene.csv)."""
    df = pd.read_csv(path, dtype=str)
    ic = id_col or next((c for c in _ID_COL_CANDIDATES if c in df.columns), None)
    sc = sym_col or next((c for c in _SYM_COL_CANDIDATES if c in df.columns), None)
    if ic is None or sc is None:
        raise ValueError(
            f"--gene_map {path.name}: could not find id/symbol columns "
            f"(cols={list(df.columns)[:12]}). Pass --gene_map_id_col / "
            f"--gene_map_symbol_col."
        )
    logger.info(f"  gene_map: {path.name} using id={ic!r} symbol={sc!r}")
    m: Dict[str, str] = {}
    for e, s in zip(df[ic].astype(str), df[sc].astype(str)):
        if e and s and s.lower() != "nan":
            m[e.upper()] = s.upper()
    logger.info(f"  gene_map: {len(m)} Ensembl->symbol entries")
    return m


def _symbols_for(adata, gene_map: Optional[Dict[str, str]]) -> List[str]:
    """Canonical UPPERCASED gene symbol per var, resolving Ensembl if needed.
    Unresolved Ensembl IDs pass through (upper) and simply won't intersect."""
    names = [str(g) for g in adata.var_names]
    if not _looks_ensembl(names):
        return [n.upper() for n in names]
    for c in _SYM_COL_CANDIDATES:              # symbol column already in var?
        if c in adata.var.columns:
            return [str(s).upper() for s in adata.var[c].tolist()]
    if gene_map:
        return [gene_map.get(n.upper(), n.upper()) for n in names]
    raise ValueError(
        "var_names look like Ensembl but there's no symbol column in .var and "
        "no --gene_map was given. Pass --gene_map (e.g. the ABC gene.csv)."
    )


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--silver_dir", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--gene_map", default=None,
                   help="CSV mapping Ensembl->symbol (e.g. ABC gene.csv). Only "
                        "needed if the Ensembl-ID slices lack a symbol column.")
    p.add_argument("--gene_map_id_col", default=None)
    p.add_argument("--gene_map_symbol_col", default=None)
    p.add_argument("--min_common", type=int, default=100)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    import anndata as ad

    silver, out = Path(args.silver_dir), Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    files = _discover(silver)
    if not files:
        logger.error(f"No *_train.h5ad / *_test.h5ad under {silver}")
        return 1
    n_train = sum(1 for f in files if f.name.endswith("_train.h5ad"))
    logger.info(f"Found {len(files)} slices ({n_train} train, "
                f"{len(files) - n_train} test)")

    gene_map = (_load_gene_map(Path(args.gene_map), args.gene_map_id_col,
                               args.gene_map_symbol_col)
                if args.gene_map else None)

    # ---- pass 1: resolve to symbols + intersect across ALL slices --------
    common_upper = None
    for f in files:
        a = ad.read_h5ad(f, backed="r")
        syms = set(_symbols_for(a, gene_map))
        common_upper = syms if common_upper is None else (common_upper & syms)
    common = sorted(common_upper)
    logger.info(f"Common gene SYMBOLS across ALL slices: {len(common)}")
    if len(common) < args.min_common:
        logger.error(
            f"Only {len(common)} common genes (< {args.min_common}). If the "
            f"Ensembl side still isn't mapping, check --gene_map columns."
        )
        return 1

    # ---- pass 2: subset + reorder + relabel to the canonical symbol panel -
    n_written = 0
    for f in files:
        out_path = out / f.name
        if out_path.exists() and not args.overwrite:
            logger.info(f"  exists, skipping: {f.name}")
            n_written += 1
            continue
        a = ad.read_h5ad(f)
        syms = _symbols_for(a, gene_map)
        pos: Dict[str, int] = {}
        for i, s in enumerate(syms):
            if s in common_upper and s not in pos:
                pos[s] = i
        idx = [pos[g] for g in common]         # canonical order, identical everywhere
        sub = a[:, idx].copy()
        sub.var_names = common
        sub.var.index.name = "gene_symbol"
        sub.write_h5ad(out_path)
        logger.info(f"  -> {out_path.name} ({sub.n_obs:,} cells x {sub.n_vars} genes)")
        n_written += 1

    logger.info(f"Done. {n_written}/{len(files)} slices -> {out} "
                f"({len(common)} common genes)")
    return 0 if n_written else 1


if __name__ == "__main__":
    sys.exit(main())
