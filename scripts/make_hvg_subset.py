#!/usr/bin/env python
"""Write a gene-subset copy of a silver dir, keeping only the top-N HVGs.

WHY
---
On the Xenium benchmark G2T memorises a single section perfectly (train=test
Spearman 0.9997, predicted anisotropy 0.609 vs true 0.594) yet transfers at
~0.10 to held-out sections. With 4,948 genes every cell carries a near-unique
expression fingerprint, so the network can learn a fingerprint -> position
lookup without acquiring any spatial rule. Cutting the input to the most
informative genes removes that shortcut and forces it to use structure that
generalises. Cortex, where the method works, has 254 genes.

NO LEAKAGE
----------
Genes are selected from the TRAIN sections of the given silver dir only
(*_train.h5ad), never the test sections, and the selection is redone per split
-- so xenium_donorsplit and xenium_xhs1000 get their own gene lists. Passing a
--gene_list makes the choice explicit and reusable across splits when you want
the input held fixed.

SELECTION
---------
scanpy highly_variable_genes with batch_key='cell_section', so a gene must be
variable in many sections rather than in one outlier section. Silver .X is
per-cell-normalised LINEAR counts, and the seurat/cell_ranger flavours expect
log data, so a log1p'd COPY is used for ranking only. The written matrices keep
the original linear values -- this changes which genes are present, nothing else.

USAGE
    python make_hvg_subset.py \
        --in_dir  /nfs/team361/sb75/DATASETS/silver/xenium_donorsplit \
        --out_dir /nfs/team361/sb75/DATASETS/silver/xenium_donorsplit_hvg1000 \
        --n_top 1000
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--in_dir", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--n_top", type=int, default=1000)
    p.add_argument("--gene_list", default="",
                   help="newline-separated gene names to use INSTEAD of selecting "
                        "(keeps the input identical across splits)")
    p.add_argument("--dry_run", action="store_true")
    args = p.parse_args()

    import anndata as ad
    import scanpy as sc

    in_dir, out_dir = Path(args.in_dir), Path(args.out_dir)
    train = sorted(in_dir.glob("*_train.h5ad"))
    test = sorted(in_dir.glob("*_test.h5ad"))
    if not train or not test:
        raise SystemExit(f"need both *_train.h5ad and *_test.h5ad under {in_dir}")
    print(f"{len(train)} train / {len(test)} test sections in {in_dir.name}")

    if args.gene_list:
        genes = [g for g in Path(args.gene_list).read_text().split("\n") if g.strip()]
        print(f"using {len(genes)} genes from {args.gene_list}")
    else:
        print(f"selecting top {args.n_top} HVGs from the TRAIN sections only ...")
        parts = [ad.read_h5ad(f) for f in train]
        cat = ad.concat(parts, join="outer", index_unique=None)
        n_cells = cat.n_obs
        del parts
        tmp = cat.copy()
        sc.pp.log1p(tmp)                       # ranking only; not written out
        sc.pp.highly_variable_genes(
            tmp, n_top_genes=args.n_top, batch_key="cell_section")
        genes = list(tmp.var_names[tmp.var["highly_variable"].to_numpy()])
        nb = tmp.var.loc[genes, "highly_variable_nbatches"] \
            if "highly_variable_nbatches" in tmp.var else None
        print(f"  selected {len(genes)} genes from {n_cells:,} train cells")
        if nb is not None:
            print(f"  variable in >=half the sections: "
                  f"{int((nb >= len(train)/2).sum())}/{len(genes)}")
        del tmp, cat

    if args.dry_run:
        print("dry run: nothing written")
        print("\n".join(genes[:10]) + "\n  ...")
        return 0

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "hvg_genes.txt").write_text("\n".join(genes) + "\n")
    kept = set(genes)
    for f in train + test:
        a = ad.read_h5ad(f)
        missing = kept - set(map(str, a.var_names))
        if missing:
            raise SystemExit(f"{f.name}: {len(missing)} selected gene(s) absent "
                             f"(e.g. {sorted(missing)[:3]}) — panels differ")
        # Preserve the SELECTION order so every file has identical columns.
        sub = a[:, genes].copy()
        sub.write_h5ad(out_dir / f.name)
        print(f"  {f.name:34s} {sub.n_obs:6d} cells x {sub.n_vars} genes")

    print(f"\nwrote {len(train)+len(test)} files to {out_dir}")
    print(f"gene list: {out_dir/'hvg_genes.txt'}")
    print(f"n_genes for the pipeline: {len(genes)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
