# G2T: Gene-to-Tissue

G2T reconstructs the spatial layout of a tissue from single-cell gene expression
alone. A Transformer conditioned on the cells' expression is trained by invariant
distance denoising along a rectified-flow (flow-matching) path, through a
Euclidean-distance-matrix (EDM) head: it predicts a *K*-dimensional embedding per
cell, and the pairwise distances between these embeddings are trained to match
the true pairwise cell distances. At
inference, classical multidimensional scaling (MDS) of that matrix (equivalently,
the top-2 principal components of the centred embeddings) turns the prediction
into 2-D coordinates.

This repository holds the model, training and inference code for the MLCB 2026
paper (see [Citation](#citation)). The code that reproduces the paper's
experiments is in
[g2t-reproducibility](https://github.com/Lotfollahi-lab/g2t-reproducibility).

> **Naming.** The Python package, scripts and output folders still use the
> project's development name, `scgg` (`import scgg`,
> `scripts/run_scgg_train.py`, `scgg_model/`, ...).

G2T builds on [LUNA](https://github.com/mlbio-epfl/LUNA) (Yu et al., 2025). The
training engine under `src/` (`main.py`, `diffusion_model.py`, `models/`,
`utils/`, `metrics/`, `datasets/`, `configs/`) started as a copy of LUNA's code
and was extended with, among other things, the EDM head
(`src/models/edm_head.py`) and flow matching.

## Installation

You need Python 3.10 or newer. The recommended workflow uses
[uv](https://github.com/astral-sh/uv), which knows about per-package indexes
and so handles the PyTorch CUDA wheel selection automatically.

### Recommended: uv

```bash
# 1. Install uv (one-time, ~10 MB)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Clone, then create .venv and install everything from the repo root
git clone https://github.com/Lotfollahi-lab/g2t.git
cd g2t
uv sync                  # or `uv pip install -e .` to install into a pre-existing venv
source .venv/bin/activate
```

`pyproject.toml` already pins torch to the **CUDA 12.4 wheel index**
(`[tool.uv.sources]` block). If your cluster has a different CUDA, change the
URL in that section, e.g. `https://download.pytorch.org/whl/cu121` for CUDA
12.1 or `https://download.pytorch.org/whl/cpu` for CPU-only. The available
PyTorch CUDA builds are: cu121, cu124, cu126, cu128, cpu.

### Fallback: pip + install.sh

If you cannot use uv, the included `install.sh` detects your CUDA driver via
`nvidia-smi` and installs the matching PyTorch wheel, then the package in
editable mode:

```bash
bash install.sh                      # auto-detect
bash install.sh --cuda 12.4          # force a specific wheel
bash install.sh --cpu                # CPU only
bash install.sh --skip-torch         # torch is already installed
```

### Fallback: pure pip

```bash
# Pick ONE of these depending on your CUDA driver:
pip install torch --index-url https://download.pytorch.org/whl/cu124  # CUDA 12.4
pip install torch --index-url https://download.pytorch.org/whl/cu121  # CUDA 12.1
pip install torch --index-url https://download.pytorch.org/whl/cpu    # CPU-only

# Then, from the repo root:
pip install -e .
```

If `pip install -e .` fails with
`build backend is missing the 'build_editable' hook`, upgrade build tooling
first: `pip install --upgrade pip setuptools wheel`.

Installing also puts the engine's top-level modules (`main`,
`diffusion_model`, `models`, `utils`, `metrics`, `datasets`, `configs`) on the
Python path. Use a dedicated environment, as these names can clash with other
packages (e.g. Hugging Face `datasets`).

## Input data

The scripts read a directory of per-section `.h5ad` files whose suffix sets the
split:

```text
data/my_dataset/
├── <section>_train.h5ad    # training sections (expression + coordinates)
└── <section>_test.h5ad     # held-out sections to reconstruct and score
```

Each file holds one section, with

- `.X`: expression (cells × genes), used as is (no log transform by default);
  all sections must share the same genes in the same order;
- `.obsm["spatial"]`: 2-D coordinates (or `obs["coord_X"]` and
  `obs["coord_Y"]`). Test files need them too, because the scripts score the
  predictions against them;
- `obs["cell_class"]` (optional): cell-type labels, used by the per-class
  metrics;
- `obs["cell_section"]` (optional): section label; the file name is used
  otherwise.

`scripts/build_h5ad_from_luna_csv.py` converts LUNA-format CSVs into this
layout (example in [Reproducing the paper](#reproducing-the-paper)).
Alternatively, pass LUNA-format CSVs directly with `--train_csv` and
`--test_csv` instead of `--data_dir` in the commands below.

## Usage

The paper's models were trained on a single NVIDIA H200 GPU; a CUDA GPU is
strongly recommended.

By default the scripts write to a cluster path. Set `SCGG_ARTIFACTS_ROOT` (used
by all three scripts) or pass `--output_dir` (train and inference scripts).
Training logs to Weights & Biases by default; add `--wandb_mode disabled` if you
are not logged in.

### Train and evaluate in one command

```bash
export SCGG_ARTIFACTS_ROOT=$PWD/artifacts

python scripts/run_scgg_pipeline.py \
    --data_dir data/my_dataset \
    --epochs 1000 --seed 0 \
    --wandb_mode disabled
```

This trains on the `*_train.h5ad` sections, then reconstructs and scores the
`*_test.h5ad` sections. Both steps share one timestamp:

```text
$SCGG_ARTIFACTS_ROOT/my_dataset/scgg_model/<YYYYMMDD_HHMMSS>/       # training
$SCGG_ARTIFACTS_ROOT/my_dataset/scgg_inference/<YYYYMMDD_HHMMSS>/   # inference
```

Use `--skip_inference` to train only, or `--skip_training --checkpoint <ckpt>`
to evaluate an existing model.

### Train and run inference separately

```bash
python scripts/run_scgg_train.py \
    --data_dir data/my_dataset \
    --output_dir runs/my_dataset/train \
    --epochs 1000 --seed 0 \
    --wandb_mode disabled

python scripts/run_scgg_inference.py \
    --data_dir data/my_dataset \
    --checkpoint runs/my_dataset/train/best_model.ckpt \
    --output_dir runs/my_dataset/inference
```

### Changing the configuration

The engine is configured with [Hydra](https://hydra.cc) (`src/configs/`). Pass
any config key with `--override`, which takes one or more `key=value` tokens and
can be repeated:

```bash
python scripts/run_scgg_pipeline.py --data_dir data/my_dataset --wandb_mode disabled \
    --override model.edm.embed_dim=16 train.lr=1e-4 \
    --override dataset.num_workers=4
```

Model settings are fixed at training time. At inference the model configuration
is restored from the training run's snapshot (`luna_run/.hydra/config.yaml`), so
`model.*` overrides passed at inference have no effect.

Checkpoints are written every `validation.save_model_every_n_epochs` epochs (250
in the default experiment config). For a short test run, lower it as well, e.g.
`--epochs 10 --override validation.save_model_every_n_epochs=10`.

Other flags (`--embedding_field`, `--n_inference_samples`,
`--exclude_test_files`, ...) are documented in each script's `--help`.

### Outputs

Training directory:

- `best_model.ckpt`: link to the last checkpoint in `luna_run/checkpoints/`;
- `luna_run/.hydra/config.yaml`: full Hydra configuration of the run (read back
  at inference);
- `config.yaml`, `train.log`, `luna_stdout.log`, `runtime.csv`.

Inference directory:

- `luna_run/test_results/.../<section>_<i>/metadata_pred.csv`: predicted
  coordinates per test section (min–max normalised to [-0.5, 0.5] per section),
  next to the ground truth in `metadata_true.csv`;
- `per_slice_metrics.csv` and `aggregate_metrics.json`: per-cell Spearman
  correlation of pairwise-distance ranks per section, and its mean over
  sections of the per-section medians (`spearman_mean_of_medians`);
- `plots/`: true-vs-predicted plots per section (skip with `--no_plots`, or
  `--no_inference_plots` in the pipeline).

Contact F1 and Sum RSSD as reported in the paper are computed by
`analysis/benchmarking/plots/compute_extended_metrics.py` in the
reproducibility repository.

## Key model options

Defaults are in `src/configs/model/default.yaml` (keys under `model.`) and
`src/configs/train/default.yaml` (keys under `train.`);
`src/configs/experiment/MERFISH_mouse_cortex.yaml`, loaded by default, sets the
dataset and run settings (e.g. `dataset.num_workers`, checkpoint interval).

| Key | Default | Meaning |
|---|---|---|
| `model.framework` | `flow_matching` | Generative path: `flow_matching` (rectified flow, x₀-prediction), `diffusion` (LUNA-style DDPM) or `regression` |
| `model.flow_matching.n_sampling_steps` | `50` | Euler steps of the reverse ODE at inference |
| `model.edm.enabled` | `true` | EDM head: predict per-cell embeddings whose pairwise squared distances form the output, instead of 2-D coordinates |
| `model.edm.embed_dim` | `8` | Embedding width *K* |
| `model.edm.mds_solver` | `svd` | MDS read-out: `svd`, `eigh` or `lobpcg` (see below) |
| `model.edm.mds_dtype` | `fp64` | Precision of the read-out: `fp64` or `fp32` |
| `model.edm.mds_align_train` | `true` | Also run the read-out during training; `false` skips it there (inference always runs it) |
| `model.hidden_dims.num_heads` | `16` | Attention heads (the paper uses 32) |
| `train.lr` / `train.ema_decay` | `5e-4` / `0.95` | Learning rate and weight-EMA decay |

### MDS read-out (`model.edm.mds_solver`)

Because the predicted distances are induced by the embeddings **H**, the
double-centred Gram matrix is **B** = (**JH**)(**JH**)ᵀ, so the classical-MDS
coordinates are the top-2 principal components of the centred embeddings.

- `svd` (default): computes them directly from the SVD of the centred
  embeddings; the read-out takes O(NK²) time and O(NK) memory and needs no
  N × N eigendecomposition.
- `eigh`: full eigendecomposition of **B** (O(N³)).
- `lobpcg`: iterative top-2 eigensolver on **B**; this is the route used in all
  runs reported in the paper (with `mds_dtype=fp32`).

The three routes are algebraically equivalent; numerically they agree up to
floating-point precision, solver tolerance and the small Tikhonov term
(`model.edm.mds_tikhonov_eps`, default `1e-6`) that `eigh` and `lobpcg` add to
stabilise the eigendecomposition. The coordinates are then Procrustes-aligned
to the network's own 2-D position estimate (which carries the frame of the
current iterate), removing the rotation/reflection ambiguity.
`tests/test_mds_svd_readout.py` checks that `svd` matches the exact
eigendecomposition with the Tikhonov term set to zero. The reported runs used
`model.edm.mds_solver=lobpcg model.edm.mds_dtype=fp32`; see
[Reproducing the paper](#reproducing-the-paper).

## Reproducing the paper

The headline cortex and CNS G2T models were trained with the model code of
commit
[`9dbe98f`](https://github.com/Lotfollahi-lab/g2t/tree/9dbe98f0fabd7c50feac1886c2b22c830f492b1a)
(30 May 2026). To rerun them, check out
[`2ea2a9e`](https://github.com/Lotfollahi-lab/g2t/tree/2ea2a9eb81f2fdc339a94188cdd95ea1b277a710)
from the same day: its model code (`src/`) is identical, and its scripts add
the `--exclude_test_files` flag that the CNS evaluation needs.

Data: LUNA's preprocessed CSVs. Download them with
`analysis/benchmarking/download_mmc_luna_csvs.sh` and
`download_cns_luna_csvs.sh` from the reproducibility repository (set `DEST_DIR`
to choose the target folder), then convert them (CSV paths shortened):

```bash
python scripts/build_h5ad_from_luna_csv.py --out_dir data/mmc_luna \
    --train_csv MERFISH_mouse_cortex_train.csv --test_csv MERFISH_mouse_cortex_test.csv
python scripts/build_h5ad_from_luna_csv.py --out_dir data/cns_luna \
    --train_csv ABCA_harmonized_train.csv --test_csv scRNA_harmonized_test.csv
```

Runs: these overrides on top of the defaults, seeds 0–4 (repeat each command
with `--seed 1` to `--seed 4`).

```bash
git checkout 2ea2a9e

# Cortex: 1,000 epochs, all 31 test sections
python scripts/run_scgg_pipeline.py --data_dir data/mmc_luna \
    --epochs 1000 --seed 0 --wandb_mode disabled \
    --override model.hidden_dims.num_heads=32 model.edm.mds_align_train=false \
               model.edm.mds_dtype=fp32 model.edm.mds_solver=lobpcg dataset.num_workers=0

# CNS: 3,500 epochs; 14 of the 18 test sections are scored
python scripts/run_scgg_pipeline.py --data_dir data/cns_luna \
    --epochs 3500 --seed 0 --wandb_mode disabled \
    --override train.batch_size=1 model.hidden_dims.num_heads=32 model.edm.mds_align_train=false \
               model.edm.mds_dtype=fp32 model.edm.mds_solver=lobpcg dataset.num_workers=0 \
    --exclude_test_files sagittal1_test.h5ad,sagittal2_test.h5ad,sagittal3_test.h5ad,spinalcord_test.h5ad
```

Paper values for G2T (mean over seeds 0–4), as scored by
`analysis/benchmarking/plots/compute_extended_metrics.py` in the
reproducibility repository; the pipeline itself reports only Spearman:

| Benchmark | Spearman | Contact F1 | Sum RSSD |
|---|---|---|---|
| Cortex | 0.4722 | 0.0625 | 76.10 |
| CNS | 0.1797 | 0.04100 | 622.0 |

The [reproducibility repository](https://github.com/Lotfollahi-lab/g2t-reproducibility)
has the LSF launch scripts used for the paper
(`analysis/benchmarking/lsf/submit_pipeline.sh`, and
`analysis/benchmarking/ablations/run_ablations.sh` for the ablations; both
default to the authors' cluster paths), the baseline environments and the
metric and figure scripts. The ablations (Table 1, seeds 0–9) were run later,
from July 2026, on newer commits: some variants (e.g. the coordinate-L2
control) use options that `2ea2a9e` does not have.

Trained checkpoints are not distributed, and the Xenium human-skin data are not
publicly available.

Baseline runners with the same interface are in `scripts/`
(`run_luna_pipeline.py`, `run_celery_pipeline.py`,
`run_novosparc_pipeline.py`); they need their own environments, which the
reproducibility repository sets up.

## Tests

```bash
uv sync --extra dev          # or: pip install -e ".[dev]"
pytest tests/
```

Run `pytest tests/` rather than a bare `pytest`: `scripts/` also contains
`test_*.py` files, which are standalone development scripts.

## Repository layout

| Path | Contents |
|---|---|
| `src/models/`, `src/diffusion_model.py`, `src/utils/`, `src/metrics/`, `src/datasets/`, `src/main.py` | Model, training loop, loss and sampling (adapted from LUNA); the EDM head and MDS read-outs are in `src/models/edm_head.py` |
| `src/configs/` | Hydra configuration |
| `src/scgg/` | Evaluation and data-loading helpers (`scgg.evaluation`, `scgg.data`) |
| `scripts/` | Command-line entry points, data preparation and baseline runners |
| `tests/` | Unit tests |
| `experiments/` | Run recipes for `scripts/launch_experiment.py` (LSF) from later development runs |

## Citation

If you use G2T, please cite:

```bibtex
@inproceedings{birk2026g2t,
  title     = {{G2T}: Tissue Reconstruction from Gene Expression via Embedding-Distance Flow Matching},
  author    = {Birk, Sebastian and Theis, Fabian J. and Lotfollahi, Mohammad},
  booktitle = {Proceedings of the Machine Learning in Computational Biology (MLCB) 2026},
  series    = {Proceedings of Machine Learning Research},
  publisher = {PMLR},
  year      = {2026},
  note      = {Volume and pages to be assigned}
}
```

Please also cite LUNA, on which G2T builds:

```bibtex
@article{luna2025,
  title   = {Tissue Reassembly with Generative {AI}},
  author  = {Yu, Tingyang and Ekbote, Chanakya and Morozov, Nikita and Herrera, Antonio and Fan, Jiashuo and Dominguez Mantes, Albert and Novello, Salvatore and Lashuel, Hilal and Frossard, Pascal and d'Ascoli, St{\'e}phane and La Manno, Gioele and Brbi{\'c}, Maria},
  journal = {bioRxiv},
  year    = {2025},
  doi     = {10.1101/2025.02.13.638045}
}
```

## Contact

For questions and bug reports, please use the
[issue tracker](https://github.com/Lotfollahi-lab/g2t/issues).

## Licence

BSD 3-Clause; see [LICENSE](LICENSE).
