"""Run LUNA on the CNS cross-modality split (Fig 4).

Thin wrapper around ``run_luna_on_mmc.run_benchmark`` with the CNS
defaults baked in:

  * train CSV  = /nfs/team361/sb75/DATASETS/bronze/cns_luna/ABCA_harmonized_train.csv
                 (spatial MERFISH cells from the ABCA mouse brain atlas)
  * test CSV   = /nfs/team361/sb75/DATASETS/bronze/cns_luna/scRNA_harmonized_test.csv
                 (Shi 2023 scRNA cells; LUNA infers their 2-D positions)

Both CSVs are already in LUNA's expected format (gene columns, then
``coord_X``, ``coord_Y``, ``cell_section``, ``cell_class``), so no
h5ad → CSV conversion happens here — we just pass them straight
through the existing pre-built CSV path.

Outputs go to the same artifacts layout the cortex benchmark uses:
``/nfs/team361/sb75/scgg-reproducibility/artifacts/cns_luna/luna_model/<TS>/``
unless you pass ``--output_dir``.

Example::

    python scripts/run_luna_on_cns.py \\
        --wandb_run_name luna_cns_fig4

The vendored LUNA at ``scgg/src/`` is used by default — same Python env,
identical Hydra config tree as the cortex run.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Reuse the full pre-built-CSV pipeline from run_luna_on_mmc.py — CSV
# loading, n_genes inference, Hydra override composition, LUNA
# subprocess invocation, post-train evaluation. No duplication.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_luna_on_mmc  # noqa: E402  (local-script import, intentional)

logger = logging.getLogger("luna_cns")


_DEFAULT_TRAIN_CSV = "/nfs/team361/sb75/DATASETS/bronze/cns_luna/ABCA_harmonized_train.csv"
_DEFAULT_TEST_CSV = "/nfs/team361/sb75/DATASETS/bronze/cns_luna/scRNA_harmonized_test.csv"


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--train_csv", default=_DEFAULT_TRAIN_CSV,
        help=f"Pre-built LUNA-format train CSV. Default: {_DEFAULT_TRAIN_CSV}",
    )
    p.add_argument(
        "--test_csv", default=_DEFAULT_TEST_CSV,
        help=f"Pre-built LUNA-format test CSV. Default: {_DEFAULT_TEST_CSV}",
    )
    p.add_argument(
        "--output_dir", default=None,
        help="Where to write the LUNA run dir + checkpoints + metrics. "
             "Default: {ARTIFACTS_ROOT}/<train_csv_parent_dir_name>/"
             "luna_model/<YYYYMMDD_HHMMSS>/.",
    )
    p.add_argument("--epochs", type=int, default=1000,
                   help="LUNA train.n_epochs (default 1000).")
    p.add_argument("--batch_size", type=int, default=6,
                   help="LUNA train.batch_size (sections per step; default 6).")
    p.add_argument("--lr", type=float, default=None,
                   help="LUNA train.lr (default: LUNA's own default, 5e-4).")
    p.add_argument("--seed", type=int, default=0,
                   help="LUNA general.seed (default 0).")
    p.add_argument("--n_genes", type=int, default=None,
                   help="Explicit gene-column count. Default: inferred from "
                        "the CSV header by locating the gene/metadata boundary.")
    p.add_argument("--wandb_run_name", default="MERFISH_CNS_harmonized",
                   help="LUNA general.name — drives the wandb run name and "
                        "the LUNA run-dir basename.")
    p.add_argument("--wandb_mode", default="disabled",
                   choices=("disabled", "online", "offline"),
                   help="LUNA general.wandb (default 'disabled').")
    p.add_argument(
        "--luna_repo", default=str(run_luna_on_mmc._DEFAULT_LUNA_REPO),
        help=f"Path to the LUNA source tree (must contain main.py + "
             f"configs/). Default: vendored LUNA at "
             f"{run_luna_on_mmc._DEFAULT_LUNA_REPO}",
    )
    p.add_argument(
        "--extra_override", action="append", default=None, metavar="KEY=VALUE",
        help="Extra Hydra override(s) passed through to LUNA. Repeat the "
             "flag for multiple.",
    )
    args = p.parse_args()

    run_luna_on_mmc.run_benchmark(
        data_dir=None,
        train_csv=args.train_csv,
        test_csv=args.test_csv,
        n_genes=args.n_genes,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        luna_repo=args.luna_repo,
        run_name=args.wandb_run_name,
        wandb_mode=args.wandb_mode,
        extra_overrides=args.extra_override,
    )


if __name__ == "__main__":
    sys.exit(main())
