#!/usr/bin/env python
"""
Run scGG inference on the Shi 2023 / STARmap-integrated CNS scRNA-seq atlas
using a model that was trained on the ABC MERFISH atlas (Animal 1). This
reproduces the inference half of LUNA Figure 4.

Inputs
------
  --checkpoint   Path to the scGG checkpoint produced by
                 scripts/train_scgg_on_abc.py (default:
                 ./results/abc_animal1/checkpoints/best_model.pt).
  --silver_dir   Per-well CNS scRNA silver h5ads from
                 scripts/prepare_cns_silver.py
                 (default: /nfs/team361/sb75/DATASETS/silver/cns_luna).
  --sections     One or more well identifiers to run on, e.g.
                 `--sections well06`. Default: `well06`. Pass `all` to
                 run on every well in --silver_dir.
  --color        adata.obs column for plot coloring (default: cell_class).

Outputs (under --output_dir, default ./scgg-reproducibility/artifacts/cns_luna)
-----------------------------------------------------------------------------
  <well>_predicted.h5ad        Copy of the input h5ad with
                               obsm['spatial_pred'] populated.
  <well>_comparison.svg        Side-by-side scatter: ground truth (left)
                               vs scGG prediction (right), colored by
                               `--color`. SVG fonts are kept as text
                               (editable in Illustrator / Inkscape).
  inference_metadata.json      Per-well summary (cell counts, gene-panel
                               coverage, runtime, etc.).

Gene-panel alignment
--------------------
The ABC model was trained on a specific gene panel (typically the 1,122
MERFISH genes). The scRNA-seq atlas has ~11K genes — we subset / pad to
match the ABC order. The expected gene list is read from
<checkpoint>/../data_summary.json (saved by train_scgg_on_abc.py). If
that file is missing (older training), pass --gene_panel_h5ad pointing
at any silver h5ad from the ABC training set.

Coordinate projection
---------------------
scGG (contrastive mode) outputs a d-dim metric embedding. We project
to 2-D for plotting using PCA on the embedding. PCA preserves the
dominant variance directions and is fast; for a tighter distance match
you can use MDS via --projection mds. If the model was trained in
flow_matching mode it already outputs 2-D coords and projection is a
no-op.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import yaml

logger = logging.getLogger("scgg.infer_cns")


# ---------------------------------------------------------------------------
# Loading checkpoint + gene panel
# ---------------------------------------------------------------------------


def _load_checkpoint(checkpoint_path: Path, device: str):
    """Returns (model, cfg, gene_names, gene_symbols, data_summary)."""
    import torch
    from scgg.model.scgg import ScGG

    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = state["config"]

    # Look for gene_names + data_summary.json next to the checkpoint dir.
    summary_path = checkpoint_path.parent.parent / "data_summary.json"
    data_summary = None
    gene_names: List[str] = []
    gene_symbols: List[str] = []
    if summary_path.exists():
        with open(summary_path) as f:
            data_summary = json.load(f)
        gene_names = list(data_summary.get("gene_names") or [])
        gene_symbols = list(data_summary.get("gene_symbols") or [])
        if gene_names:
            logger.info(
                f"  loaded gene panel from {summary_path}: "
                f"{len(gene_names)} Ensembl IDs"
                + (f" + {len(gene_symbols)} symbols" if gene_symbols else " (no symbols)")
            )
        else:
            logger.warning(
                f"  {summary_path} exists but does not contain gene_names; "
                "you may need --gene_panel_h5ad"
            )
    else:
        logger.warning(
            f"  no data_summary.json next to checkpoint dir ({summary_path}); "
            "you may need --gene_panel_h5ad to recover the trained gene panel"
        )

    n_genes_expected = (
        data_summary.get("n_genes") if data_summary else None
    )
    if not gene_names and n_genes_expected is None:
        # Fall back: infer n_genes from the encoder's first linear weight.
        first_layer_key = next(
            (k for k in state["model_state_dict"].keys() if "encoder.backbone.0.0.weight" in k),
            None,
        )
        if first_layer_key is None:
            raise RuntimeError(
                "Could not determine n_genes from the checkpoint. "
                "Pass --gene_panel_h5ad to specify it."
            )
        n_genes_expected = state["model_state_dict"][first_layer_key].shape[1]
        logger.info(
            f"  inferred n_genes={n_genes_expected} from the encoder's first layer"
        )

    n_genes = len(gene_names) if gene_names else n_genes_expected
    model = ScGG(n_genes=n_genes, config=cfg)
    model.load_state_dict(state["model_state_dict"])
    model = model.to(device)
    model.eval()

    return model, cfg, gene_names, gene_symbols, data_summary


def _gene_panel_from_h5ad(path: Path) -> Tuple[List[str], List[str]]:
    """Return (var_names, gene_symbols) from an ABC silver h5ad."""
    import anndata as ad
    a = ad.read_h5ad(path)
    names = list(a.var_names)
    symbols = []
    for col in ("gene_symbol", "gene_name", "symbol"):
        if col in a.var.columns:
            symbols = a.var[col].astype(str).tolist()
            break
    return names, symbols


# ---------------------------------------------------------------------------
# Per-well: load + preprocess + align genes + run inference
# ---------------------------------------------------------------------------


def _looks_like_ensembl(names: List[str]) -> bool:
    """Heuristic: do these strings look like mouse/human Ensembl gene IDs?"""
    if not names:
        return False
    return sum(n.startswith(("ENSMUSG", "ENSG")) for n in names[:100]) > 50


def _harmonize_query_genes(
    adata_var_names: List[str],
    target_ensembl: List[str],
    target_symbols: List[str],
) -> Tuple[List[Optional[str]], Dict[str, int]]:
    """Translate adata.var_names into the target Ensembl namespace.

    Returns:
      mapped:     list[Optional[str]] of length = len(adata_var_names).
                  Each entry is the matching Ensembl ID (string in
                  target_ensembl) or None if no match.
      diagnostics: counters describing the matching attempt.
    """
    target_set = set(target_ensembl)

    query_looks_ensembl = _looks_like_ensembl(adata_var_names)
    if query_looks_ensembl:
        # Direct match on Ensembl IDs (case-sensitive — Ensembl IDs are
        # never lowercased in practice).
        mapped = [n if n in target_set else None for n in adata_var_names]
        n_matched = sum(m is not None for m in mapped)
        return mapped, {
            "matching_mode": "ensembl_direct",
            "n_query_genes": len(adata_var_names),
            "n_matched": n_matched,
        }

    # The query uses symbols. We need a symbol -> Ensembl lookup. Build it
    # from the target gene panel; case-insensitive matching because the
    # scRNA-seq atlas reports symbols in UPPERCASE while the ABC panel uses
    # mouse capitalization (Cbln2 vs CBLN2).
    if not target_symbols or len(target_symbols) != len(target_ensembl):
        return [None] * len(adata_var_names), {
            "matching_mode": "symbol_unavailable",
            "n_query_genes": len(adata_var_names),
            "n_matched": 0,
            "note": (
                "Query gene namespace looks like gene symbols but the "
                "checkpoint did not save gene_symbols. Re-train with the "
                "updated training script, or pass --gene_panel_h5ad."
            ),
        }

    sym_to_ens: Dict[str, str] = {}
    n_dupe = 0
    for sym, ens in zip(target_symbols, target_ensembl):
        k = str(sym).strip().lower()
        if not k or k == "nan":
            continue
        if k in sym_to_ens and sym_to_ens[k] != ens:
            n_dupe += 1
        sym_to_ens[k] = ens  # last one wins; rare collisions

    mapped: List[Optional[str]] = []
    for name in adata_var_names:
        k = str(name).strip().lower()
        ens = sym_to_ens.get(k)
        mapped.append(ens if (ens is not None and ens in target_set) else None)
    n_matched = sum(m is not None for m in mapped)
    return mapped, {
        "matching_mode": "symbol_to_ensembl_case_insensitive",
        "n_query_genes": len(adata_var_names),
        "n_matched": n_matched,
        "n_duplicate_symbols": n_dupe,
    }


def _preprocess_and_align(
    adata,
    target_ensembl: List[str],
    target_symbols: List[str],
    normalize: bool,
    scale: bool,
) -> Tuple[np.ndarray, dict]:
    """Bring the query expression into the trained ABC gene-panel order.

    Handles the case where the query (scRNA-seq) uses gene symbols while
    the trained model uses Ensembl IDs. Genes that don't map are filled
    with zeros.
    """
    import scanpy as sc
    import scipy.sparse as sp

    query_var_names = list(adata.var_names)
    mapped, diag = _harmonize_query_genes(query_var_names, target_ensembl, target_symbols)
    coverage = diag["n_matched"] / max(1, len(target_ensembl))
    logger.info(
        f"  gene panel match: mode={diag['matching_mode']}, "
        f"{diag['n_matched']}/{len(target_ensembl)} target genes mapped "
        f"({coverage:.1%}), {diag['n_query_genes']} query genes considered"
    )
    if diag["n_matched"] == 0:
        logger.error(
            "  no genes matched between the query and the trained panel. "
            "Inference will use a zero matrix and is meaningless. Check "
            "the gene namespace of your scRNA-seq h5ad."
        )

    # Build a subset AnnData of the matched query genes, in the order they
    # appear in the query (so scanpy preprocessing operates on real data).
    matched_idx = [i for i, m in enumerate(mapped) if m is not None]
    if matched_idx:
        sub = adata[:, matched_idx].copy()
        if "counts" not in sub.layers:
            sub.layers["counts"] = sub.X.copy()
        if normalize:
            sc.pp.normalize_total(sub, target_sum=1e4)
            sc.pp.log1p(sub)
        if scale:
            sc.pp.scale(sub, max_value=10)
        Xsub = sub.X
        if sp.issparse(Xsub):
            Xsub = Xsub.toarray()
        Xsub = np.asarray(Xsub, dtype=np.float32)
    else:
        Xsub = np.zeros((adata.n_obs, 0), dtype=np.float32)

    # Scatter the matched-gene expression into the full ABC-ordered matrix.
    target_idx = {ens: i for i, ens in enumerate(target_ensembl)}
    out = np.zeros((adata.n_obs, len(target_ensembl)), dtype=np.float32)
    for sub_col, query_col in enumerate(matched_idx):
        ens = mapped[query_col]
        j = target_idx[ens]
        out[:, j] = Xsub[:, sub_col]
    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)

    return out, {
        "n_present": diag["n_matched"],
        "n_missing": len(target_ensembl) - diag["n_matched"],
        "coverage": coverage,
        "matching_mode": diag["matching_mode"],
        "n_query_genes": diag["n_query_genes"],
    }


def _predict_2d(
    model,
    gene_expr_np: np.ndarray,
    projection: str,
    device: str,
) -> np.ndarray:
    """Run scGG and project the result down to 2-D for plotting.

    For flow_matching mode the model already returns 2-D coords; for
    contrastive mode we apply PCA (or MDS) on the metric embedding.
    """
    import torch
    from scgg.evaluation.luna_metrics import embedding_to_2d

    ge = torch.from_numpy(gene_expr_np).float().to(device)
    with torch.no_grad():
        if model.objective == "contrastive":
            emb = model.embed_batched(ge)
        else:
            emb = model.generate_embeddings(ge)
    emb_np = emb.detach().cpu().numpy()
    if emb_np.shape[1] == 2:
        return emb_np
    return embedding_to_2d(emb_np, method=projection)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def _plot_comparison(
    adata,
    color_col: str,
    out_svg: Path,
    title_prefix: str = "",
) -> None:
    """Side-by-side scatter: GT (obsm['spatial']) vs predicted (obsm['spatial_pred']).

    Both panels share a single legend placed below the figure, so each
    panel gets the full plotting width. SVG fonts are kept as <text>
    elements (editable in Illustrator / Inkscape).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    import scanpy as sc

    matplotlib.rcParams["svg.fonttype"] = "none"
    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"] = 42

    has_gt = "spatial" in adata.obsm and adata.obsm["spatial"].shape[1] >= 2
    has_pred = "spatial_pred" in adata.obsm
    spot_size = max(5.0, min(20.0, 600.0 / np.sqrt(max(adata.n_obs, 1))))

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    # Plot WITHOUT scanpy's per-panel legend so we can attach one shared
    # legend below the figure.
    common_kw = dict(
        color=color_col, show=False, size=spot_size,
        legend_loc=None, frameon=True,
    )
    if has_gt:
        sc.pl.embedding(
            adata, basis="spatial", ax=axes[0],
            title=f"{title_prefix}Ground truth", **common_kw,
        )
    else:
        axes[0].set_title(f"{title_prefix}Ground truth (none)")
        axes[0].set_axis_off()

    if has_pred:
        sc.pl.embedding(
            adata, basis="spatial_pred", ax=axes[1],
            title=f"{title_prefix}scGG prediction", **common_kw,
        )
    else:
        axes[1].set_title(f"{title_prefix}scGG prediction (none)")
        axes[1].set_axis_off()

    # Build a shared legend at the bottom. scanpy will have populated
    # adata.uns[f"{color}_colors"] during the first sc.pl call; if the
    # column is non-categorical (numeric), skip the legend.
    color_series = adata.obs[color_col]
    is_categorical = (
        color_series.dtype.name == "category"
        or color_series.dtype == object
    )
    if is_categorical:
        cats = color_series.astype("category").cat.categories.tolist()
        colors_key = f"{color_col}_colors"
        palette = adata.uns.get(colors_key)
        # Fall back to a tab20 cycle if scanpy didn't set the palette.
        if palette is None or len(palette) < len(cats):
            cmap = plt.get_cmap("tab20", max(len(cats), 1))
            palette = [cmap(i) for i in range(len(cats))]
        patches = [Patch(facecolor=c, label=str(cat)) for c, cat in zip(palette, cats)]
        # Pick a column count that keeps the legend readable.
        # Aim for ~4 rows max.
        n_cats = len(cats)
        ncol = min(max(1, (n_cats + 3) // 4), 6)
        fig.legend(
            handles=patches,
            labels=[str(c) for c in cats],
            loc="lower center",
            bbox_to_anchor=(0.5, 0.0),
            ncol=ncol,
            frameon=False,
            fontsize="small",
        )
        # Leave room at the bottom for the legend (number of rows-dependent).
        n_rows = (n_cats + ncol - 1) // ncol
        bottom = min(0.30, 0.05 + 0.04 * n_rows)
        fig.tight_layout(rect=(0, bottom, 1, 1))
    else:
        fig.tight_layout()

    out_svg.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_svg, format="svg", bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  saved plot: {out_svg}")


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def _resolve_section_files(silver_dir: Path, sections_arg: List[str]) -> List[Path]:
    """Map --sections values (well IDs or filenames) to silver h5ad paths."""
    if sections_arg == ["all"]:
        files = sorted(silver_dir.glob("cns_scrna_*.h5ad"))
        if not files:
            raise FileNotFoundError(f"No cns_scrna_*.h5ad files under {silver_dir}")
        return files
    out = []
    for s in sections_arg:
        # Accept either bare well-id ('well06') or filename ('cns_scrna_well06.h5ad')
        candidates = [
            silver_dir / f"cns_scrna_{s}.h5ad",
            silver_dir / s,
            silver_dir / f"{s}.h5ad",
        ]
        for c in candidates:
            if c.exists():
                out.append(c)
                break
        else:
            raise FileNotFoundError(
                f"No silver h5ad matches section spec {s!r}. "
                f"Tried: {[str(c) for c in candidates]}"
            )
    return out


def run_inference(
    checkpoint: str,
    silver_dir: str,
    sections: List[str],
    output_dir: str,
    color_col: str,
    projection: str,
    device: Optional[str],
    gene_panel_h5ad: Optional[str],
    no_normalize: bool,
    no_scale: bool,
) -> None:
    import anndata as ad
    import torch

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    ckpt_path = Path(checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Loading checkpoint: {ckpt_path}")
    logger.info(f"Device: {dev}")
    model, cfg, gene_names, gene_symbols, data_summary = _load_checkpoint(ckpt_path, dev)
    logger.info(
        f"Model: objective={model.objective}, "
        f"params={sum(p.numel() for p in model.parameters()):,}"
    )

    if (not gene_names or not gene_symbols) and gene_panel_h5ad:
        # Always re-load both names + symbols from the fallback h5ad so the
        # inference script has them even if the older training run only
        # saved gene_names.
        gn, gs = _gene_panel_from_h5ad(Path(gene_panel_h5ad))
        if not gene_names:
            gene_names = gn
        if not gene_symbols and gs:
            gene_symbols = gs
        logger.info(
            f"  --gene_panel_h5ad supplied: {len(gn)} genes, "
            f"{len(gs)} symbols from {gene_panel_h5ad}"
        )
    if not gene_names:
        raise RuntimeError(
            "Could not recover the trained gene panel. Re-train with the "
            "updated train_scgg_on_abc.py (saves gene_names), or pass "
            "--gene_panel_h5ad pointing at any silver h5ad from the ABC "
            "training set."
        )
    if not gene_symbols:
        logger.warning(
            "  no gene_symbols available — query namespace can only be "
            "matched if it already uses the trained Ensembl IDs."
        )

    # Preprocessing flags: prefer values stored in data_summary, then config,
    # then CLI overrides (--no_normalize / --no_scale).
    normalize = (
        bool(data_summary.get("normalize", cfg.get("data", {}).get("normalize", True)))
        if data_summary
        else cfg.get("data", {}).get("normalize", True)
    )
    scale = (
        bool(data_summary.get("scale", cfg.get("data", {}).get("scale", True)))
        if data_summary
        else cfg.get("data", {}).get("scale", True)
    )
    if no_normalize:
        normalize = False
    if no_scale:
        scale = False
    logger.info(f"  preprocessing: normalize={normalize}, scale={scale}")

    section_paths = _resolve_section_files(Path(silver_dir), sections)
    logger.info(f"Inference on {len(section_paths)} sections: "
                 f"{[p.name for p in section_paths]}")

    summary = []
    for path in section_paths:
        well_id = path.stem.replace("cns_scrna_", "")
        logger.info(f"[{well_id}] {path}")
        adata = ad.read_h5ad(path)
        logger.info(f"  shape: {adata.shape}")

        if color_col not in adata.obs.columns:
            logger.warning(
                f"  color column {color_col!r} missing from obs; falling back to first categorical column"
            )
            cat_cols = [c for c in adata.obs.columns if adata.obs[c].dtype == "object"
                         or adata.obs[c].dtype.name == "category"]
            color_col_eff = cat_cols[0] if cat_cols else None
        else:
            color_col_eff = color_col

        t0 = time.time()
        X_aligned, panel_stats = _preprocess_and_align(
            adata, gene_names, gene_symbols,
            normalize=normalize, scale=scale,
        )
        coords_pred = _predict_2d(model, X_aligned, projection, dev)
        elapsed = time.time() - t0
        logger.info(
            f"  inference done in {elapsed:.1f}s; pred shape={coords_pred.shape}"
        )

        adata.obsm["spatial_pred"] = coords_pred.astype(np.float32)
        out_h5ad = out_dir / f"{well_id}_predicted.h5ad"
        adata.write(out_h5ad)
        logger.info(f"  wrote h5ad with spatial_pred: {out_h5ad}")

        if color_col_eff is not None:
            out_svg = out_dir / f"{well_id}_comparison.svg"
            _plot_comparison(adata, color_col_eff, out_svg, title_prefix=f"[{well_id}] ")
        else:
            logger.warning("  no usable color column; skipping plot")

        summary.append({
            "well_id": well_id,
            "input_path": str(path),
            "output_h5ad": str(out_h5ad),
            "n_cells": int(adata.n_obs),
            **panel_stats,
            "inference_seconds": elapsed,
            "color_col": color_col_eff,
            "projection": projection,
        })

    with open(out_dir / "inference_metadata.json", "w") as f:
        json.dump(
            {
                "checkpoint": str(ckpt_path),
                "gene_panel_source": (
                    "checkpoint.data_summary.json"
                    if data_summary and data_summary.get("gene_names")
                    else (gene_panel_h5ad or "unknown")
                ),
                "n_gene_panel": len(gene_names),
                "sections": summary,
            },
            f, indent=2, default=str,
        )
    logger.info(f"Wrote inference_metadata.json to {out_dir}")


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--checkpoint",
        default="./results/abc_animal1/checkpoints/best_model.pt",
    )
    p.add_argument(
        "--silver_dir",
        default="/nfs/team361/sb75/DATASETS/silver/cns_luna",
    )
    p.add_argument(
        "--sections", nargs="+", default=["well06"],
        help="Well IDs to run on. Pass 'all' to run on every well "
             "in --silver_dir. Default: well06.",
    )
    p.add_argument(
        "--output_dir",
        default="../scgg-reproducibility/artifacts/cns_luna",
        help="Where to write predicted h5ads and comparison SVGs. Default "
             "points at the sibling scgg-reproducibility repository.",
    )
    p.add_argument("--color", default="cell_class",
                   help="adata.obs column to color plots by (default: cell_class).")
    p.add_argument(
        "--projection", default="pca", choices=("pca", "mds"),
        help="2-D projection of the metric embedding (contrastive mode only). "
             "Default: pca.",
    )
    p.add_argument("--device", default=None, help="cuda|cpu (auto if omitted)")
    p.add_argument(
        "--gene_panel_h5ad", default=None,
        help="Optional fallback: path to any silver h5ad from the ABC "
             "training set, used to recover gene order if the checkpoint's "
             "data_summary.json lacks gene_names.",
    )
    p.add_argument("--no_normalize", action="store_true",
                   help="Skip normalize_total + log1p (use only if the input "
                        "h5ads are already log-normalized).")
    p.add_argument("--no_scale", action="store_true",
                   help="Skip per-gene z-score scaling.")
    args = p.parse_args()

    run_inference(
        checkpoint=args.checkpoint,
        silver_dir=args.silver_dir,
        sections=args.sections,
        output_dir=args.output_dir,
        color_col=args.color,
        projection=args.projection,
        device=args.device,
        gene_panel_h5ad=args.gene_panel_h5ad,
        no_normalize=args.no_normalize,
        no_scale=args.no_scale,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
