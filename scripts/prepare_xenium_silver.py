#!/usr/bin/env python
"""Convert the Xenium xhs1000-39b_1p batches into a silver dir the pipeline reads.

Produces ``{out_dir}/{section}_{train,test}.h5ad`` with the same layout as the
existing silver dirs (mmc_luna, cns_luna), so run_scgg/luna/celery pipelines
discover it by the usual ``*_{train,test}.h5ad`` glob.

DECISIONS MADE HERE, EACH DELIBERATE
------------------------------------
cell_class   <- obs['annotation_new']
cell_section <- obs['Sample ID']   (NOT the batch file: a file may hold several
                samples, and the pipeline needs one section per file because it
                falls back to the filename stem whenever cell_section is not
                uniform within a file.)

CELL IDS. The source obs_names are NOT unique across batches (anndata warns on
load). The LUNA CSV is keyed on a numeric cell id and the scorers assume a
unique index, so a collision would silently merge cells. We therefore assign a
globally unique integer cell_id across the whole dataset and keep the originals
in obs['cell_id_src'] / obs['obs_name_src'] so nothing is lost.

EXPRESSION. This is the one place the two benchmarks could silently diverge.
Cortex silver .X is LUNA's published CSV "as-is (non-integer per-cell-normalized
counts)" (build_h5ad_from_luna_csv.py header) -- linear space, per-cell
normalised, NOT log, NOT raw integers. Xenium .X here is raw integer counts
(csr, int64, max ~30). Handing raw counts to a pipeline whose other benchmark is
per-cell normalised would confound "second dataset" with "different
preprocessing", so we apply the SAME transform family: scale each cell to a
common total, in linear space, no log.

The common total is the dataset's own MEDIAN total counts per cell. That is what
makes the two datasets comparable in kind rather than in absolute units: each is
expressed in its own native count magnitude, with between-cell depth variation
removed. Forcing Xenium onto the cortex panel's numeric range would be worse --
it would rescale a different assay to another assay's arbitrary units.

Nothing is log-transformed. scgg/src/utils/data/load.py::log2_norm has zero call
sites in either repo, and neither LUNA nor G2T nor novosparc logs its input.

USAGE
    python prepare_xenium_silver.py \
        --in_dir  /nfs/team361/sb75/DATASETS/silver/xhs1000-39b_1p \
        --out_dir /nfs/team361/sb75/DATASETS/silver/xenium_xhs1000 \
        --test_batches 4,6,11,12,18,19,32
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np

CLASS_COL = "annotation_new"
SECTION_COL = "Sample ID"


def sanitize(label: str) -> str:
    """Filesystem-safe section label (same rule as build_h5ad_from_luna_csv)."""
    return re.sub(r"[/\x00-\x1f\s]+", "_", str(label)).strip("_")


def batch_num(path: Path) -> int:
    m = re.search(r"batch(\d+)", path.stem)
    if not m:
        raise SystemExit(f"cannot parse a batch number from {path.name}")
    return int(m.group(1))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--in_dir", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--test_batches", required=True,
                   help="comma-separated batch numbers held out for test")
    p.add_argument("--dry_run", action="store_true",
                   help="report the plan and every guard, write nothing")
    args = p.parse_args()

    import anndata as ad
    import scipy.sparse as sp

    in_dir, out_dir = Path(args.in_dir), Path(args.out_dir)
    test_b = {int(s) for s in args.test_batches.split(",") if s.strip()}
    files = sorted(in_dir.glob("*.h5ad"), key=batch_num)
    if not files:
        raise SystemExit(f"no h5ad under {in_dir}")
    unknown = test_b - {batch_num(f) for f in files}
    if unknown:
        raise SystemExit(f"--test_batches names absent batches: {sorted(unknown)}")

    # ---- pass 1: section map, guards, and the global median total ----------
    print(f"pass 1/2: scanning {len(files)} files")
    sec_batches: dict[str, set] = {}
    totals: list[np.ndarray] = []
    n_cells = 0
    for f in files:
        a = ad.read_h5ad(f)
        b = batch_num(f)
        for col in (CLASS_COL, SECTION_COL):
            if col not in a.obs:
                raise SystemExit(f"{f.name}: obs[{col!r}] missing")
        if "spatial" not in a.obsm:
            raise SystemExit(f"{f.name}: obsm['spatial'] missing")
        for s in a.obs[SECTION_COL].astype(str).unique():
            sec_batches.setdefault(s, set()).add(b)
        X = a.X
        totals.append(np.asarray(X.sum(axis=1)).ravel() if sp.issparse(X)
                      else np.asarray(X).sum(axis=1))
        n_cells += a.n_obs

    # A section spanning a train AND a test batch would put the same section's
    # cells on both sides of the split. Refuse rather than leak.
    straddle = {s: sorted(bs) for s, bs in sec_batches.items()
                if (bs & test_b) and (bs - test_b)}
    if straddle:
        raise SystemExit(
            "these sections span both train and test batches, which would leak:\n"
            + "\n".join(f"  {s}: batches {b}" for s, b in straddle.items()))

    all_tot = np.concatenate(totals)
    if not np.all(all_tot > 0):
        raise SystemExit(f"{int((all_tot <= 0).sum())} cells have zero counts")
    target = float(np.median(all_tot))
    n_test_sec = sum(1 for s, bs in sec_batches.items() if bs & test_b)
    print(f"  {n_cells:,} cells, {len(sec_batches)} sections "
          f"({len(sec_batches)-n_test_sec} train / {n_test_sec} test)")
    print(f"  median total counts/cell = {target:.1f}  (per-cell scaling target)")
    if args.dry_run:
        print("dry run: guards passed, nothing written")
        return 0

    # ---- pass 2: normalise, relabel, split, write --------------------------
    out_dir.mkdir(parents=True, exist_ok=True)
    next_id, written = 0, []
    print(f"pass 2/2: writing to {out_dir}")
    for f in files:
        a = ad.read_h5ad(f)
        b = batch_num(f)
        split = "test" if b in test_b else "train"
        secs = a.obs[SECTION_COL].astype(str)
        for s in sorted(secs.unique()):
            sub = a[(secs == s).to_numpy()].copy()
            X = sub.X.astype(np.float32)
            tot = (np.asarray(X.sum(axis=1)).ravel() if sp.issparse(X)
                   else np.asarray(X).sum(axis=1))
            scale = (target / tot).astype(np.float32)
            # Per-cell scaling, linear space, no log -- matches cortex silver X.
            sub.X = (sp.diags(scale) @ X) if sp.issparse(X) else X * scale[:, None]

            sub.obs["cell_class"] = sub.obs[CLASS_COL].astype(str)
            sub.obs["cell_section"] = s
            sub.obs["obs_name_src"] = sub.obs_names.astype(str)
            if "cell_id" in sub.obs:
                sub.obs["cell_id_src"] = sub.obs["cell_id"].astype(str)
            ids = np.arange(next_id, next_id + sub.n_obs, dtype=np.int64)
            next_id += sub.n_obs
            sub.obs["cell_id"] = ids
            sub.obs_names = [str(i) for i in ids]   # globally unique

            path = out_dir / f"{sanitize(s)}_{split}.h5ad"
            if path.exists():
                raise SystemExit(f"{path.name} already exists — section label "
                                 f"collision after sanitising {s!r}")
            sub.write_h5ad(path)
            written.append((path.name, sub.n_obs, split,
                            int(sub.obs['cell_class'].nunique())))
            print(f"  {path.name:44s} n={sub.n_obs:6d} {split:5s} "
                  f"classes={sub.obs['cell_class'].nunique()}")

    n_tr = sum(1 for _, _, s, _ in written if s == "train")
    print(f"\nwrote {len(written)} files: {n_tr} train / {len(written)-n_tr} test, "
          f"{next_id:,} cells, unique cell ids 0..{next_id-1}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
