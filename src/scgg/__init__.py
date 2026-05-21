"""
scGG entry-point package.

The actual model, training loop, loss, and sampling now live in LUNA's
vendored source under ``scgg/src/{models,utils,metrics,datasets,configs}``
(see https://github.com/mlbio-epfl/LUNA), invoked via
``scgg/src/main.py`` from ``scripts/run_scgg.py``.

This package retains only the pieces that wrap LUNA from the outside:

  * ``scgg.evaluation`` — Spearman / contact F1 / Kabsch RSSD on LUNA's
    predictions.
  * ``scgg.data`` — h5ad-side loaders for the cortex / ABC / CNS
    pipelines (LUNA itself reads CSVs).

To run the LUNA reproduction with scGG's CLI::

    python scripts/run_scgg.py \\
        --data_dir /nfs/team361/sb75/DATASETS/silver/mmc_luna \\
        --wandb_run_name my_run

To call LUNA directly with Hydra overrides, use ``scgg/src/main.py``
(it's just LUNA's ``main.py``).
"""

__version__ = "0.2.0"  # bumped: LUNA-vendored release
