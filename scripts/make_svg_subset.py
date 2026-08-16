#!/usr/bin/env python
"""Write a gene-subset copy of a silver dir, keeping only the top-N SVGs.

Sibling of make_hvg_subset.py. Same inputs, same outputs, same file layout --
the ONLY difference is the ranking criterion: spatial autocorrelation (Moran's
I) instead of expression variance. Run both and the comparison isolates "does
picking genes by spatial structure beat picking them by variance", with nothing
else moving.

WHY SVGs
--------
On the Xenium benchmark G2T memorises a single section perfectly (train=test
Spearman 0.9997) yet transfers at ~0.10 to held-out sections: with 4,948 genes
every cell carries a near-unique fingerprint and the network learns a
fingerprint -> position lookup instead of a spatial rule. HVG selection cuts
the input down but ranks genes by variance, which rewards genes that separate
cell types whether or not they vary across SPACE. Moran's I ranks a gene by
whether nearby cells share its expression level -- which is the signal a
positional model actually needs.

NO LEAKAGE -- READ THIS
-----------------------
Moran's I is a function of the coordinates. Computing it on a test section
would select input features using the very coordinates the model is asked to
predict, which is leakage of the sharpest possible kind. Selection therefore
runs on *_train.h5ad ONLY, and the script refuses to read a test file during
ranking. Training coordinates are already supervision, so using them to choose
genes leaks nothing about the held-out sections.

This is still supervised feature selection, and should be described as such:
"genes ranked by Moran's I on the training sections". It is not a claim that
the genes were chosen blind. The HVG arm uses train-only information too, so
the two are on equal footing.

PER SECTION, THEN AGGREGATED
----------------------------
A spatial graph only means something within one section -- sections do not
share a coordinate frame, and the pipeline min-max normalises each one
separately. So Moran's I is computed per train section and then aggregated
across sections. The default is the MEDIAN, mirroring how scanpy's
highly_variable_genes handles batch_key: a gene must be spatially structured in
most sections, not in one outlier. The report prints, for each selected gene,
how many sections it ranks in the top-N of -- the auditable version of
highly_variable_nbatches.

ESTIMATOR
---------
Identical to ablations/plot_svg_comparison.py::morans_i -- kNN graph,
row-normalised weights (so S0 = N), I = sum_i z_i * mean_j(z_j) / sum_i z_i^2.
Implemented here as a sparse matmul (W @ z) rather than fancy-indexing a
(N, k, G) block, which would peak at ~4 GB on the largest section.

Ranking uses a log1p'd COPY (Moran's I is per-gene scale-invariant, but log
compresses count outliers that would otherwise dominate a single gene's
neighbour covariance). Written matrices keep the original linear values -- this
changes which genes are present, nothing else. Same convention as the HVG arm.

USAGE
    python make_svg_subset.py \
        --in_dir  /nfs/team361/sb75/DATASETS/silver/xenium_donorsplit \
        --out_dir /nfs/team361/sb75/DATASETS/silver/xenium_donorsplit_svg500 \
        --n_top 500
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def section_coords(a) -> np.ndarray:
    """(N, 2) raw coordinates. obsm['spatial'] is the silver convention."""
    if "spatial" in a.obsm:
        return np.asarray(a.obsm["spatial"], dtype=np.float64)[:, :2]
    if {"coord_X", "coord_Y"} <= set(a.obs.columns):
        return a.obs[["coord_X", "coord_Y"]].to_numpy(dtype=np.float64)
    raise SystemExit("no coordinates: need obsm['spatial'] or obs coord_X/coord_Y")


def morans_i(coords: np.ndarray, expr: np.ndarray, k: int = 6) -> np.ndarray:
    """Per-gene Moran's I on a kNN spatial graph. Returns (G,).

    Same estimator as ablations/plot_svg_comparison.py, written as a sparse
    matmul so memory is O(N*G) rather than O(N*k*G).
    """
    import scipy.sparse as sp
    from sklearn.neighbors import NearestNeighbors

    n = coords.shape[0]
    k = min(k, n - 1)
    if k < 1:
        raise SystemExit(f"section has {n} cell(s); cannot build a kNN graph")
    nn = NearestNeighbors(n_neighbors=k + 1).fit(coords)
    _, idx = nn.kneighbors(coords)
    idx = idx[:, 1:]                                     # drop self

    # Row-normalised weights => every row sums to 1 => S0 = N.
    rows = np.repeat(np.arange(n), k)
    W = sp.csr_matrix((np.full(n * k, 1.0 / k), (rows, idx.ravel())), shape=(n, n))

    z = expr - expr.mean(axis=0, keepdims=True)          # (N, G) centred
    denom = (z ** 2).sum(axis=0) + 1e-12                 # (G,)
    num = (z * (W @ z)).sum(axis=0)                      # (G,)
    return np.asarray(num / denom, dtype=np.float64)


def dense_log1p(a) -> np.ndarray:
    """Dense log1p copy of .X, for ranking only."""
    import scipy.sparse as sp

    X = a.X
    X = X.toarray() if sp.issparse(X) else np.asarray(X)
    return np.log1p(np.asarray(X, dtype=np.float32))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--in_dir", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--n_top", type=int, default=500)
    p.add_argument("--knn", type=int, default=6, help="k for the Moran's I graph")
    p.add_argument("--aggregate", choices=["median", "mean"], default="median",
                   help="how to combine per-section Moran's I (default median: "
                        "a gene must be spatial in most sections)")
    p.add_argument("--gene_list", default="",
                   help="newline-separated gene names to use INSTEAD of ranking "
                        "(keeps the input identical across splits)")
    p.add_argument("--dry_run", action="store_true",
                   help="rank and report, write nothing")
    args = p.parse_args()

    import anndata as ad
    import pandas as pd

    in_dir, out_dir = Path(args.in_dir), Path(args.out_dir)
    train = sorted(in_dir.glob("*_train.h5ad"))
    test = sorted(in_dir.glob("*_test.h5ad"))
    if not train or not test:
        raise SystemExit(f"need both *_train.h5ad and *_test.h5ad under {in_dir}")
    print(f"{len(train)} train / {len(test)} test sections in {in_dir.name}")

    if args.gene_list:
        genes = [g for g in Path(args.gene_list).read_text().split("\n") if g.strip()]
        print(f"using {len(genes)} genes from {args.gene_list}")
        rank_table = None
    else:
        print(f"ranking by Moran's I (k={args.knn}) on the TRAIN sections only ...")
        per_section, var_names, n_cells = [], None, 0
        for f in train:
            a = ad.read_h5ad(f)
            if var_names is None:
                var_names = list(map(str, a.var_names))
            elif list(map(str, a.var_names)) != var_names:
                raise SystemExit(f"{f.name}: gene panel differs from {train[0].name}")
            mi = morans_i(section_coords(a), dense_log1p(a), k=args.knn)
            per_section.append(mi)
            n_cells += a.n_obs
            print(f"  {f.name:38s} n={a.n_obs:6d}  "
                  f"median I={np.median(mi):+.4f}  max I={mi.max():+.4f}")
            del a

        M = np.vstack(per_section)                       # (n_sections, G)
        score = np.median(M, axis=0) if args.aggregate == "median" else M.mean(axis=0)
        order = np.argsort(-score)[:args.n_top]
        genes = [var_names[i] for i in order]

        # How consistent is each pick? Count sections where it makes the top-N.
        topn_per_section = [set(np.argsort(-row)[:args.n_top]) for row in M]
        nsec = np.array([sum(i in s for s in topn_per_section) for i in order])
        rank_table = pd.DataFrame({
            "gene": genes,
            f"morans_i_{args.aggregate}": score[order],
            "morans_i_min": M[:, order].min(axis=0),
            "morans_i_max": M[:, order].max(axis=0),
            "n_sections_in_top": nsec,
        })
        print(f"\n  ranked {len(var_names)} genes over {n_cells:,} train cells")
        print(f"  selected top {len(genes)} by {args.aggregate} Moran's I: "
              f"{score[order].min():+.4f} .. {score[order].max():+.4f}")
        print(f"  in top-{args.n_top} of >=half the sections: "
              f"{int((nsec >= len(train) / 2).sum())}/{len(genes)}")
        print(f"  discarded genes' {args.aggregate} I: "
              f"max {score[np.argsort(-score)[args.n_top]]:+.4f}")
        print("\n  top 10:")
        print(rank_table.head(10).to_string(index=False))

    if args.dry_run:
        print("\ndry run: nothing written")
        return 0

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "svg_genes.txt").write_text("\n".join(genes) + "\n")
    if rank_table is not None:
        rank_table.to_csv(out_dir / "svg_ranking.csv", index=False)
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
        print(f"  {f.name:38s} {sub.n_obs:6d} cells x {sub.n_vars} genes")

    print(f"\nwrote {len(train)+len(test)} files to {out_dir}")
    print(f"gene list: {out_dir/'svg_genes.txt'}")
    print(f"n_genes for the pipeline: {len(genes)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
