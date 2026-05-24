#!/usr/bin/env python
"""Precompute per-cell embeddings into the h5ad's ``obsm`` slot.

The result of running this script is that each silver h5ad gets a new
``adata.obsm[<field>]`` matrix of shape ``(n_cells, embedding_dim)``.
Downstream, ``run_scgg_train.py`` (when called with
``--embedding_field <field>``) reads from that obsm field instead of
``adata.X``, so the inner backbone's gene encoder sees a pretrained
representation rather than raw counts.

Why use ``obsm`` instead of replacing ``X`` outright
----------------------------------------------------
Keeping the raw counts in ``X`` means the same h5ad supports BOTH the
raw-gene baseline AND the pretrained-encoder ablation. Switching
between them is a flag at training time, not a re-run of the
embedding pipeline. We can also stack multiple obsm fields on the
same h5ad (PCA, scVI, scGPT) and pick one per run for ablation.

Currently supported encoders
----------------------------
* ``pca``        — sklearn TruncatedSVD on the cell × gene matrix.
                   Lightweight, deterministic, no external deps
                   beyond sklearn. Per-slice basis. Good baseline /
                   sanity test of the encoder pipeline.
* ``autoencoder`` — simple 3-layer MLP autoencoder trained on the
                   cell × gene matrix. Outputs the bottleneck. Per-
                   slice. ~1 minute on cortex-scale data.
* ``scvi``       — scVI (Lopez et al., 2018) from scvi-tools. Trained
                   ONCE GLOBALLY over the union of slices, with each
                   slice's filename used as the ``batch_key`` so
                   scVI's batch-correction machinery models cross-
                   slice variation. Tutorial defaults: ``n_layers=2``,
                   ``gene_likelihood='zinb'``, ``dispersion='gene'``,
                   ``max_epochs=auto`` (early stopping). The
                   ``embedding_dim`` flag controls ``n_latent``.
                   The result is then split back per-slice and
                   written to each h5ad's ``obsm[<field>]``.

Pluggable encoders (TODO — uniform ``obsm``-based interface)
------------------------------------------------------------
The script structure is designed so adding a new encoder is just
implementing one ``_compute_<name>`` function (per-slice) OR a
``_run_<name>_global`` function (global). Candidates:

* ``scgpt``      — scGPT foundation-model embeddings. Requires the
                   scgpt repo + checkpoint download.
* ``geneformer`` — Geneformer foundation-model embeddings. Same caveat.

For MERFISH panels (~500 genes) these foundation models need careful
handling because they were trained on full transcriptomes — usually
either gene-overlap restriction or zero-padding the missing genes.
The pattern is identical to PCA/AE once you have the resulting
matrix: ``adata.obsm[field] = embedding_matrix``.

Usage
-----
::

    # 1. PCA embeddings into adata.obsm['pca_64'] for every silver h5ad
    python scgg/scripts/precompute_embeddings.py \\
        --silver_dir /nfs/team361/sb75/DATASETS/silver/mmc_luna \\
        --encoder pca --embedding_dim 64 --field pca_64

    # 2. Autoencoder embeddings into adata.obsm['ae_128']
    python scgg/scripts/precompute_embeddings.py \\
        --silver_dir /nfs/team361/sb75/DATASETS/silver/mmc_luna \\
        --encoder autoencoder --embedding_dim 128 --field ae_128 \\
        --epochs 30

    # 3. scVI embeddings (global; per-slice batch correction)
    #    n_latent=10 matches the scvi-tools tutorial default.
    python scgg/scripts/precompute_embeddings.py \\
        --silver_dir /nfs/team361/sb75/DATASETS/silver/mmc_luna \\
        --encoder scvi --embedding_dim 10 --field scvi_10

The script writes back to the SAME h5ad files (in place). Set
``--out_dir`` to write to a parallel tree instead of overwriting.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np


logger = logging.getLogger("precompute_embeddings")


def _silver_files(silver_dir: Path) -> List[Path]:
    """Discover *_train.h5ad and *_test.h5ad files, matching the
    silver layout the training pipeline expects."""
    train = sorted(silver_dir.glob("*_train.h5ad"))
    test = sorted(silver_dir.glob("*_test.h5ad"))
    if not train and not test:
        # Fall back to all h5ads if the silver suffix convention isn't used.
        return sorted(silver_dir.glob("*.h5ad"))
    return train + test


def _load_gene_matrix(adata) -> np.ndarray:
    """Return adata.X as a dense float32 numpy array."""
    import scipy.sparse as sp
    X = adata.X
    if sp.issparse(X):
        X = X.toarray()
    return np.asarray(X, dtype=np.float32)


# ---------------------------------------------------------------------------
# Encoder implementations
# ---------------------------------------------------------------------------


def _compute_pca(
    gene_matrix: np.ndarray, embedding_dim: int,
) -> np.ndarray:
    """Truncated-SVD PCA. Fits on the SAME matrix it transforms, so
    every slice's embedding lives in its own basis. For cross-slice
    consistency you'd fit on the whole dataset once and apply per
    slice; we do per-slice here for simplicity, and document it. The
    embeddings are still consistent for the model since each slice
    is independently normalized to its own coordinate system anyway.
    """
    from sklearn.decomposition import TruncatedSVD
    # Cap dim by min(n_cells, n_genes) - 1 (SVD requirement).
    n_cells, n_genes = gene_matrix.shape
    eff_dim = min(embedding_dim, n_cells - 1, n_genes - 1)
    if eff_dim < embedding_dim:
        logger.warning(
            f"Reducing embedding_dim from {embedding_dim} to {eff_dim} "
            f"because the input matrix is too small "
            f"(n_cells={n_cells}, n_genes={n_genes})."
        )
    svd = TruncatedSVD(n_components=eff_dim, random_state=0)
    return svd.fit_transform(gene_matrix).astype(np.float32)


def _compute_autoencoder(
    gene_matrix: np.ndarray,
    embedding_dim: int,
    epochs: int = 30,
    hidden_dim: int = 256,
) -> np.ndarray:
    """Train a simple AE on the cell × gene matrix, return the
    bottleneck embeddings. Reasonable default for testing whether a
    richer encoder helps.

    Single-slice training — embeddings are slice-local. If you want
    a globally-consistent encoder, train on a concatenated dataset
    once and apply per slice (small refactor).
    """
    import torch
    import torch.nn as nn
    import torch.optim as optim

    n_cells, n_genes = gene_matrix.shape
    device = "cuda" if torch.cuda.is_available() else "cpu"
    X = torch.from_numpy(gene_matrix).to(device)

    class AE(nn.Module):
        def __init__(self):
            super().__init__()
            self.enc = nn.Sequential(
                nn.Linear(n_genes, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, embedding_dim),
            )
            self.dec = nn.Sequential(
                nn.Linear(embedding_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, n_genes),
            )

        def forward(self, x):
            z = self.enc(x)
            return self.dec(z), z

    ae = AE().to(device)
    opt = optim.AdamW(ae.parameters(), lr=1e-3)
    batch_size = 256

    ae.train()
    for epoch in range(epochs):
        # Random shuffle for SGD over cells.
        perm = torch.randperm(n_cells, device=device)
        losses = []
        for i in range(0, n_cells, batch_size):
            idx = perm[i:i + batch_size]
            x_batch = X[idx]
            x_hat, _ = ae(x_batch)
            loss = ((x_hat - x_batch) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        if (epoch + 1) % 5 == 0 or epoch == 0:
            logger.info(
                f"  AE epoch {epoch + 1}/{epochs}: loss = {np.mean(losses):.4f}"
            )

    ae.eval()
    with torch.no_grad():
        _, Z = ae(X)
    return Z.cpu().numpy().astype(np.float32)


_PER_SLICE_ENCODERS = {
    "pca": _compute_pca,
    "autoencoder": _compute_autoencoder,
}

# Global encoders are trained ONCE on the union of slices and produce
# embeddings for every cell jointly. Each entry in this set has a
# matching ``_run_<name>_global`` function below; the main loop
# dispatches accordingly.
_GLOBAL_ENCODERS = {"scvi"}

_ALL_ENCODERS = sorted(set(_PER_SLICE_ENCODERS.keys()) | _GLOBAL_ENCODERS)


# ---------------------------------------------------------------------------
# Global-encoder implementations
# ---------------------------------------------------------------------------


def _run_scvi_global(
    files: List[Path],
    field: str,
    embedding_dim: int,
    out_dir: Optional[Path],
    overwrite_field: bool,
    round_to_int: bool = True,
    max_epochs: Optional[int] = None,
    n_layers: int = 2,
) -> None:
    """Train scVI ONCE on the concatenated h5ads, then split the
    latent representation back per slice.

    Why global
    ----------
    scVI's strength is modelling cross-batch variation explicitly.
    Per-slice training would throw that away (each slice's encoder
    would learn a different latent basis, defeating the purpose of
    a "pretrained" encoder). Global training with per-slice
    ``batch_key`` lets scVI correct for cross-section drift while
    producing comparable embeddings.

    Tutorial defaults
    -----------------
    Following the scvi-tools introduction tutorial
    (https://docs.scvi-tools.org/en/1.3.3/tutorials/index.html):
    * ``setup_anndata(layer='counts', batch_key=<filename>)`` — slice
      filename becomes the batch covariate.
    * ``SCVI(n_layers=2, n_latent=embedding_dim, gene_likelihood='zinb')``.
    * ``train()`` with default ``max_epochs`` (auto-determined with
      early stopping) unless overridden.
    * ``get_latent_representation()`` to extract the per-cell latent.

    Raw counts caveat
    -----------------
    scVI expects integer raw counts. LUNA's published MMC CSVs are
    per-cell-normalised (non-integer, max ~250). By default we round
    to int before scVI setup (close enough to satisfy ZINB) and
    document the choice. Set ``round_to_int=False`` if your h5ad
    already has integer counts in the layer.
    """
    try:
        import anndata as ad
        import scvi  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "encoder='scvi' requires scvi-tools. Install via "
            "`pip install scvi-tools`. Underlying error: " + str(e)
        )

    import anndata as ad
    import scvi

    logger.info(f"scVI: loading {len(files)} slices...")
    adatas = []
    n_per_file: List[int] = []
    for path in files:
        ad_i = ad.read_h5ad(path)
        # Slice-level batch label — scVI's batch_key.
        ad_i.obs["_scvi_batch"] = path.stem
        adatas.append(ad_i)
        n_per_file.append(ad_i.n_obs)

    # Concatenate; require shared gene panel (join='inner') so the
    # resulting AnnData has a clean X matrix without NaNs that
    # scVI would choke on.
    big = ad.concat(
        adatas, axis=0, join="inner", merge="unique",
        label="_scvi_origin",
    )
    logger.info(
        f"scVI: concatenated {big.n_obs:,} cells × {big.n_vars} genes "
        f"across {len(files)} slices."
    )

    # scVI expects raw counts. Put them in a 'counts' layer.
    X = big.X
    import scipy.sparse as sp
    if sp.issparse(X):
        X = X.toarray()
    X = np.asarray(X, dtype=np.float32)
    if round_to_int:
        X = np.round(X).astype(np.float32)
    big.layers["counts"] = X

    scvi.model.SCVI.setup_anndata(
        big, layer="counts", batch_key="_scvi_batch",
    )
    model = scvi.model.SCVI(
        big, n_latent=embedding_dim, n_layers=n_layers,
    )
    logger.info(
        f"scVI: training (n_latent={embedding_dim}, n_layers={n_layers}, "
        f"max_epochs={'auto' if max_epochs is None else max_epochs})..."
    )
    train_kwargs = {}
    if max_epochs is not None:
        train_kwargs["max_epochs"] = max_epochs
    model.train(**train_kwargs)

    latent = model.get_latent_representation()           # (total_cells, n_latent)
    logger.info(f"scVI: latent shape = {latent.shape}")

    # Split per slice. The concat order is files[0], files[1], ...
    # so cumulative-sum slicing recovers each h5ad's cells.
    cumulative = 0
    for path, n_i in zip(files, n_per_file):
        ad_i = ad.read_h5ad(path)
        if field in ad_i.obsm and not overwrite_field:
            logger.info(f"  {path.name}: obsm[{field!r}] exists, skipping write")
            cumulative += n_i
            continue
        ad_i.obsm[field] = latent[cumulative:cumulative + n_i].astype(np.float32)
        cumulative += n_i
        out_path = (out_dir / path.name) if out_dir is not None else path
        ad_i.write_h5ad(out_path)
        logger.info(
            f"  → wrote {out_path} (obsm[{field!r}] shape="
            f"{ad_i.obsm[field].shape})"
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--silver_dir", required=True,
        help="Directory containing silver *_train.h5ad / *_test.h5ad files. "
             "Embeddings are added to each h5ad's obsm in place.",
    )
    p.add_argument(
        "--encoder", required=True,
        choices=_ALL_ENCODERS,
        help="Embedding method. 'pca' is the cheap sanity-check baseline; "
             "'autoencoder' trains a small AE per slice (~1 min); "
             "'scvi' trains ONCE globally over all slices with per-slice "
             "batch correction (tutorial defaults; ~10-30 min).",
    )
    p.add_argument(
        "--embedding_dim", type=int, default=64,
        help="Output dimensionality of the embedding. 64 is a good "
             "starting point — comparable to LUNA's hidden_mlp_dims.X "
             "and small enough to be tractable.",
    )
    p.add_argument(
        "--field", default=None,
        help="adata.obsm key to write to. Defaults to "
             "'<encoder>_<embedding_dim>' (e.g., 'pca_64').",
    )
    p.add_argument(
        "--epochs", type=int, default=30,
        help="Number of training epochs (autoencoder only).",
    )
    p.add_argument(
        "--scvi_max_epochs", type=int, default=None,
        help="Max epochs for scVI training. Default: scvi-tools' auto "
             "(early stopping based on validation loss; usually 100-400).",
    )
    p.add_argument(
        "--scvi_n_layers", type=int, default=2,
        help="scVI encoder/decoder depth. Tutorial default = 2.",
    )
    p.add_argument(
        "--scvi_no_round", action="store_true",
        help="Skip the rounding-to-int step before scVI setup. Use "
             "this if your h5ad's adata.X already contains integer raw "
             "counts. Default: round (LUNA's MMC values are non-integer "
             "normalised counts, scVI's ZINB likelihood needs ints).",
    )
    p.add_argument(
        "--out_dir", default=None,
        help="Optional: write modified h5ads to this directory instead "
             "of overwriting the inputs. Default: in-place.",
    )
    p.add_argument(
        "--overwrite_field", action="store_true",
        help="If the obsm field already exists, overwrite it. Default: "
             "skip files where it's already present.",
    )
    args = p.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    import anndata as ad

    silver_dir = Path(args.silver_dir).resolve()
    files = _silver_files(silver_dir)
    if not files:
        logger.error(f"No h5ad files found under {silver_dir}")
        return 1
    logger.info(f"Discovered {len(files)} h5ad files under {silver_dir}")

    field = args.field or f"{args.encoder}_{args.embedding_dim}"
    logger.info(f"Will write embeddings to adata.obsm[{field!r}]")

    out_dir = Path(args.out_dir).resolve() if args.out_dir else None
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)

    # Dispatch: global vs per-slice training pathway.
    if args.encoder in _GLOBAL_ENCODERS:
        if args.encoder == "scvi":
            _run_scvi_global(
                files=files,
                field=field,
                embedding_dim=args.embedding_dim,
                out_dir=out_dir,
                overwrite_field=args.overwrite_field,
                round_to_int=not args.scvi_no_round,
                max_epochs=args.scvi_max_epochs,
                n_layers=args.scvi_n_layers,
            )
        else:
            raise ValueError(
                f"Encoder {args.encoder!r} is in _GLOBAL_ENCODERS but has "
                f"no _run_<name>_global dispatch wired up. Add one."
            )
    else:
        encoder_fn = _PER_SLICE_ENCODERS[args.encoder]
        for path in files:
            adata = ad.read_h5ad(path)
            if field in adata.obsm and not args.overwrite_field:
                logger.info(f"  {path.name}: obsm[{field!r}] exists, skipping")
                continue

            gene_matrix = _load_gene_matrix(adata)
            n_cells, n_genes = gene_matrix.shape
            logger.info(
                f"  {path.name}: computing {args.encoder} embeddings "
                f"({n_cells} cells × {n_genes} genes → dim={args.embedding_dim})"
            )

            if args.encoder == "autoencoder":
                emb = encoder_fn(gene_matrix, args.embedding_dim, epochs=args.epochs)
            else:
                emb = encoder_fn(gene_matrix, args.embedding_dim)

            adata.obsm[field] = emb

            out_path = (out_dir / path.name) if out_dir else path
            adata.write_h5ad(out_path)
            logger.info(f"  → wrote {out_path} (obsm[{field!r}] shape={emb.shape})")

    logger.info("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
