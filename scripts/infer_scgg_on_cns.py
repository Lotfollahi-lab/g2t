#!/usr/bin/env python
"""
Run scGG inference on a held-out spatial dataset.

Two intended uses:

1. **scRNA-seq de novo reconstruction (LUNA Figure 4).** Apply an
   ABC-trained checkpoint to the Shi 2023 / STARmap-integrated CNS
   scRNA-seq atlas. Bridges gene-symbol queries to Ensembl-ID targets
   via mygene with an offline cache.

2. **MERFISH cross-animal held-out test (LUNA Figure 3).** Apply a
   cortex-trained checkpoint (Mouse 1) to Mouse 2 sections from
   `mmc_luna` (legacy: `merfish_mouse_cortex_luna`). Same gene panel
   on both sides — direct case-insensitive symbol match.

Inputs
------
  --checkpoint   Path to the scGG checkpoint (default:
                 ./results/abc_animal1/checkpoints/best_model.pt).
  --silver_dir   Directory with per-section silver h5ads. The script
                 supports the silver naming conventions:
                   cns_scrna_<well>.h5ad
                   mmc_mouse{1,2}_slice{N}.h5ad  (preferred)
                   merfish_mouse_cortex_mouse{1,2}_slice{N}.h5ad (legacy)
                   abc_zhuang_abca1_<section>.h5ad
  --sections     Section IDs, filenames, or glob patterns. Accepts the
                 special values 'all' and 'all_test' (= every mouse2_*
                 slice for the cortex layout, both prefix schemes).
                 Default is data-driven (see CLI help).
  --color        adata.obs column for plot coloring (default: cell_class).
  --mygene_cache JSON cache for symbol->Ensembl translations (used only
                 when the target panel is Ensembl-namespaced).
  --species      mygene species name (default: mouse).

Outputs (under --output_dir, default ../scgg-reproducibility/artifacts/<silver_dir_name>/)
------------------------------------------------------------------------------------------
  <section>_predicted.h5ad     Copy of the input h5ad with
                               obsm['spatial_pred'] populated.
  <section>_comparison.svg     Side-by-side scatter: ground truth (left)
                               vs scGG prediction (right), colored by
                               `--color`. SVG fonts are kept as text
                               (editable in Illustrator / Inkscape).
  inference_metadata.json      Per-section summary (cell counts,
                               gene-panel coverage, runtime, etc.).

Gene-panel alignment
--------------------
Handles all four target/query namespace combinations:
  target Ensembl + query Ensembl : direct match
  target Ensembl + query Symbol  : target's symbol map + mygene fallback
  target Symbol  + query Symbol  : case-insensitive direct match (cortex)
  target Symbol  + query Ensembl : not supported (convert query upstream)

The expected gene list comes from (in order of preference):
  1. <checkpoint>/../data_summary.json  (saved by train_scgg_on_abc.py)
  2. --gene_panel_h5ad fallback
  3. Auto-discovery of a representative file in --silver_dir
     (e.g. mmc_mouse1_slice1.h5ad for cortex models).

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
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import yaml

logger = logging.getLogger("scgg.infer_cns")


# ---------------------------------------------------------------------------
# mygene: offline-cached symbol -> Ensembl translation
# ---------------------------------------------------------------------------


class MygeneCache:
    """JSON-backed offline cache for mygene.info symbol -> Ensembl lookups.

    Translating gene symbols to Ensembl IDs via mygene is needed to recover
    matches lost to symbol aliases (e.g. 'Marchf1' vs 'March1'). The first
    call hits the network; the cache file persists so re-runs are offline
    and reproducible.
    """

    def __init__(
        self,
        cache_path: Optional[Path] = None,
        species: str = "mouse",
    ) -> None:
        self.cache_path = Path(cache_path) if cache_path is not None else None
        self.species = species
        # Maps lowercased query symbol -> Ensembl ID (str), or "" for
        # negative-cached lookups (so we don't re-query missing symbols).
        self._cache: Dict[str, str] = {}
        self._dirty = False
        self._load()

    def _load(self) -> None:
        if self.cache_path is None or not self.cache_path.exists():
            return
        try:
            with open(self.cache_path) as f:
                blob = json.load(f)
            if isinstance(blob, dict):
                stored = blob.get(self.species, {})
                if isinstance(stored, dict):
                    self._cache = {str(k).lower(): str(v) for k, v in stored.items()}
                    logger.info(
                        f"  mygene cache: loaded {len(self._cache)} entries "
                        f"from {self.cache_path}"
                    )
        except (OSError, json.JSONDecodeError) as e:
            logger.warning(f"  mygene cache: failed to load {self.cache_path}: {e}")

    def save(self) -> None:
        if self.cache_path is None or not self._dirty:
            return
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        existing: Dict[str, Dict[str, str]] = {}
        if self.cache_path.exists():
            try:
                with open(self.cache_path) as f:
                    blob = json.load(f)
                if isinstance(blob, dict):
                    existing = {
                        str(k): dict(v) for k, v in blob.items() if isinstance(v, dict)
                    }
            except (OSError, json.JSONDecodeError):
                existing = {}
        existing[self.species] = dict(self._cache)
        tmp = self.cache_path.with_suffix(self.cache_path.suffix + ".tmp")
        with open(tmp, "w") as f:
            json.dump(existing, f, indent=2)
        os.replace(tmp, self.cache_path)
        logger.info(
            f"  mygene cache: saved {len(self._cache)} entries to {self.cache_path}"
        )
        self._dirty = False

    def lookup(self, symbols: List[str]) -> Dict[str, Optional[str]]:
        """Return {input_symbol: ensembl_or_None} for the given symbols.

        Uses cache when possible; hits mygene.info for misses. Symbols that
        the API cannot map are negatively cached as "" so subsequent runs
        skip them.
        """
        out: Dict[str, Optional[str]] = {}
        misses: List[str] = []
        for sym in symbols:
            key = str(sym).strip().lower()
            if not key or key == "nan":
                out[sym] = None
                continue
            if key in self._cache:
                ens = self._cache[key] or None
                out[sym] = ens
            else:
                misses.append(sym)
        if not misses:
            return out

        try:
            import mygene  # type: ignore
        except ImportError:
            logger.warning(
                "  mygene not installed — skipping symbol-to-Ensembl fallback "
                "for %d symbols. Install with: pip install mygene",
                len(misses),
            )
            for sym in misses:
                out[sym] = None
            return out

        logger.info(
            f"  mygene: querying {len(misses)} uncached symbols "
            f"(species={self.species})..."
        )
        mg = mygene.MyGeneInfo()
        try:
            results = mg.querymany(
                misses,
                scopes="symbol,alias",
                fields="ensembl.gene",
                species=self.species,
                returnall=False,
            )
        except Exception as e:  # network/API failure should not abort inference
            logger.warning(f"  mygene query failed: {e}")
            for sym in misses:
                out[sym] = None
            return out

        for r in results:
            query = r.get("query")
            if query is None:
                continue
            ens = None
            if not r.get("notfound", False):
                eg = r.get("ensembl")
                if isinstance(eg, list):
                    eg = eg[0] if eg else None
                if isinstance(eg, dict):
                    ens = eg.get("gene")
            self._cache[str(query).lower()] = ens or ""
            self._dirty = True
            out[query] = ens

        # Anything that came back with no entry — negative cache it.
        for sym in misses:
            if sym not in out:
                self._cache[str(sym).lower()] = ""
                self._dirty = True
                out[sym] = None

        n_hit = sum(1 for sym in misses if out.get(sym))
        logger.info(
            f"  mygene: resolved {n_hit}/{len(misses)} new symbols "
            f"({n_hit / max(1, len(misses)):.1%})"
        )
        return out


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
    """Return (var_names, gene_symbols) from a silver h5ad."""
    import anndata as ad
    a = ad.read_h5ad(path)
    names = list(a.var_names)
    symbols = []
    for col in ("gene_symbol", "gene_name", "symbol"):
        if col in a.var.columns:
            symbols = a.var[col].astype(str).tolist()
            break
    return names, symbols


def _autodiscover_gene_panel(
    silver_dir: Path, expected_n_genes: Optional[int] = None,
) -> Optional[Tuple[Path, List[str], List[str]]]:
    """Find a representative silver h5ad in `silver_dir` and read its panel.

    Used when the checkpoint has no `data_summary.json` (e.g. the cortex
    benchmark script never wrote one). For cortex, the train set is
    ``mmc_mouse1_*.h5ad`` (or legacy ``merfish_mouse_cortex_mouse1_*.h5ad``);
    we prefer those.

    If `expected_n_genes` is given, we accept the first file whose gene
    count matches — this guards against picking up a file with a different
    panel by accident.
    """
    if not silver_dir.exists():
        return None

    # Prefer training-side files (mouse1 for cortex); otherwise just take
    # the first silver h5ad with a known prefix.
    candidates: List[Path] = []
    for g in _CORTEX_MOUSE_GLOBS:
        candidates.extend(sorted(silver_dir.glob(g.format(m=1))))
    if not candidates:
        candidates = _all_silver_files(silver_dir)
    for path in candidates:
        try:
            names, symbols = _gene_panel_from_h5ad(path)
        except Exception as e:
            logger.warning(f"  could not read {path.name} for gene panel: {e}")
            continue
        if expected_n_genes is not None and len(names) != expected_n_genes:
            continue
        return path, names, symbols
    return None


# ---------------------------------------------------------------------------
# Per-well: load + preprocess + align genes + run inference
# ---------------------------------------------------------------------------


def _looks_like_ensembl(names: List[str]) -> bool:
    """Heuristic: do these strings look like mouse/human Ensembl gene IDs?"""
    if not names:
        return False
    head = names[:100]
    n_hits = sum(str(n).startswith(("ENSMUSG", "ENSG")) for n in head)
    return n_hits >= 0.5 * len(head)


def _looks_like_integer_index(names: List[str]) -> bool:
    """Are these strings actually integer position indices, not gene names?

    Some h5ad files store gene names in a `var` column ('Gene', 'Symbol',
    etc.) and leave `var.index` as a default integer index. In that case
    `var_names` is ['0', '1', '2', ...] which is useless for gene matching.
    """
    if not names:
        return False
    n_int = 0
    for n in names[:50]:
        s = str(n).strip()
        if s and (s.isdigit() or (s.startswith("-") and s[1:].isdigit())):
            n_int += 1
    return n_int > 40


def _resolve_query_gene_names(adata) -> Tuple[List[str], str]:
    """Get the *real* gene names from a query AnnData.

    Prefers adata.var_names, but falls back to a column in adata.var when
    var_names is just an integer index. Returns (gene_names, source) where
    source is either 'var_names' or 'var.{column_name}'.
    """
    var_names = list(adata.var_names)
    if not _looks_like_integer_index(var_names):
        return var_names, "var_names"

    # var_names is bogus — look for a column carrying the real gene symbols.
    candidate_cols = (
        "Gene", "gene", "gene_symbol", "gene_name",
        "Symbol", "symbol", "feature_name", "gene_id",
    )
    for col in candidate_cols:
        if col in adata.var.columns:
            vals = adata.var[col].astype(str).tolist()
            # Reject if the column is mostly NaN/empty
            n_valid = sum(1 for v in vals if v and v.lower() != "nan")
            if n_valid > 0.5 * len(vals):
                return vals, f"var.{col}"
    # No usable column — return the (useless) integer-index names and let
    # the caller surface a clear error.
    return var_names, "var_names_integer_no_fallback"


def _harmonize_query_genes(
    adata_var_names: List[str],
    target_names: List[str],
    target_symbols: Optional[List[str]] = None,
    mygene_cache: Optional["MygeneCache"] = None,
) -> Tuple[List[Optional[str]], Dict[str, object]]:
    """Map query gene names into the target gene panel.

    Handles all four combinations of {Ensembl, Symbol} on the target side
    and the query side:

      target Ensembl / query Ensembl  -> direct match
      target Ensembl / query Symbol   -> target's symbol map, then mygene
      target Symbol  / query Symbol   -> case-insensitive match
      target Symbol  / query Ensembl  -> reverse mygene lookup
                                          (rare; falls back to None)

    Returns:
      mapped:      list[Optional[str]] of length = len(adata_var_names).
                   Each entry is the matching *target* identifier or None.
      diagnostics: counters describing the matching attempt.
    """
    target_is_ensembl = _looks_like_ensembl(target_names)
    query_is_ensembl = _looks_like_ensembl(adata_var_names)

    # Case 1: both Ensembl -> direct
    if target_is_ensembl and query_is_ensembl:
        target_set = set(target_names)
        mapped = [n if n in target_set else None for n in adata_var_names]
        return mapped, {
            "matching_mode": "ensembl_direct",
            "target_namespace": "ensembl",
            "query_namespace": "ensembl",
            "n_query_genes": len(adata_var_names),
            "n_matched": sum(m is not None for m in mapped),
            "n_matched_via_mygene": 0,
        }

    # Case 2: both Symbol -> case-insensitive match against target_names
    if not target_is_ensembl and not query_is_ensembl:
        target_lower_to_orig: Dict[str, str] = {}
        for n in target_names:
            k = str(n).strip().lower()
            if k and k != "nan":
                target_lower_to_orig[k] = n
        mapped = [
            target_lower_to_orig.get(str(n).strip().lower())
            for n in adata_var_names
        ]
        return mapped, {
            "matching_mode": "symbol_case_insensitive",
            "target_namespace": "symbol",
            "query_namespace": "symbol",
            "n_query_genes": len(adata_var_names),
            "n_matched": sum(m is not None for m in mapped),
            "n_matched_via_mygene": 0,
        }

    # Case 3: target Ensembl, query Symbol.
    # First try the target's bundled symbol->Ensembl map (fast, offline),
    # then fall back to mygene for whatever didn't resolve.
    if target_is_ensembl and not query_is_ensembl:
        target_set = set(target_names)
        sym_to_ens: Dict[str, str] = {}
        n_dupe = 0
        if target_symbols and len(target_symbols) == len(target_names):
            for sym, ens in zip(target_symbols, target_names):
                k = str(sym).strip().lower()
                if not k or k == "nan":
                    continue
                if k in sym_to_ens and sym_to_ens[k] != ens:
                    n_dupe += 1
                sym_to_ens[k] = ens

        mapped: List[Optional[str]] = []
        unresolved_idx: List[int] = []
        for i, name in enumerate(adata_var_names):
            k = str(name).strip().lower()
            ens = sym_to_ens.get(k)
            if ens is not None and ens in target_set:
                mapped.append(ens)
            else:
                mapped.append(None)
                unresolved_idx.append(i)
        n_after_direct = sum(m is not None for m in mapped)

        # mygene fallback for unresolved symbols
        n_via_mygene = 0
        if mygene_cache is not None and unresolved_idx:
            misses = [adata_var_names[i] for i in unresolved_idx]
            sym_to_ens_mg = mygene_cache.lookup(misses)
            for i, sym in zip(unresolved_idx, misses):
                ens = sym_to_ens_mg.get(sym)
                if ens is not None and ens in target_set:
                    mapped[i] = ens
                    n_via_mygene += 1

        return mapped, {
            "matching_mode": (
                "symbol_to_ensembl_with_mygene"
                if (mygene_cache is not None and unresolved_idx)
                else "symbol_to_ensembl_case_insensitive"
            ),
            "target_namespace": "ensembl",
            "query_namespace": "symbol",
            "n_query_genes": len(adata_var_names),
            "n_matched": sum(m is not None for m in mapped),
            "n_matched_via_direct": n_after_direct,
            "n_matched_via_mygene": n_via_mygene,
            "n_duplicate_symbols": n_dupe,
        }

    # Case 4: target Symbol, query Ensembl. Rare. mygene reverse-lookup
    # would be needed; skip for now and return all-None with a clear note.
    return [None] * len(adata_var_names), {
        "matching_mode": "ensembl_to_symbol_unsupported",
        "target_namespace": "symbol",
        "query_namespace": "ensembl",
        "n_query_genes": len(adata_var_names),
        "n_matched": 0,
        "n_matched_via_mygene": 0,
        "note": (
            "Query uses Ensembl IDs but the trained panel is in symbols; "
            "reverse Ensembl->symbol mapping is not implemented. Convert "
            "the query to symbols beforehand."
        ),
    }


def _preprocess_and_align(
    adata,
    target_names: List[str],
    target_symbols: Optional[List[str]],
    normalize: bool,
    scale: bool,
    mygene_cache: Optional["MygeneCache"] = None,
) -> Tuple[np.ndarray, dict]:
    """Bring the query expression into the trained gene-panel order.

    Handles all four combinations of {Ensembl, Symbol} on the target/query
    side. Also handles the case where var_names is a useless integer index
    and the actual gene names live in a `var` column. Unmapped genes are
    filled with zeros.
    """
    import scanpy as sc
    import scipy.sparse as sp

    query_var_names, name_source = _resolve_query_gene_names(adata)
    logger.info(
        f"  query gene-name source: {name_source} "
        f"(e.g., {query_var_names[:3]})"
    )
    mapped, diag = _harmonize_query_genes(
        query_var_names, target_names, target_symbols, mygene_cache=mygene_cache,
    )
    diag["name_source"] = name_source
    coverage = diag["n_matched"] / max(1, len(target_names))
    extra = ""
    if diag.get("n_matched_via_mygene"):
        extra = f" (+{diag['n_matched_via_mygene']} via mygene)"
    logger.info(
        f"  gene panel match: mode={diag['matching_mode']}, "
        f"target={diag['target_namespace']}, query={diag['query_namespace']}, "
        f"{diag['n_matched']}/{len(target_names)} target genes mapped "
        f"({coverage:.1%}){extra}, "
        f"{diag['n_query_genes']} query genes considered"
    )
    if diag["n_matched"] == 0:
        msg = (
            "No genes matched between the query and the trained panel "
            "(coverage 0%). Inference cannot proceed — the model would "
            "produce a single constant output for every cell.\n"
            f"  Query gene-name source: {name_source}\n"
            f"  Matching mode attempted: {diag['matching_mode']}\n"
            f"  First 5 query names: {list(query_var_names[:5])}\n"
            f"  First 5 target names:  {target_names[:5]}\n"
            f"  First 5 target symbols: {(target_symbols or [])[:5]}\n"
            "Likely fixes:\n"
            "  1. Make sure you have the latest scripts/infer_scgg_on_cns.py\n"
            "     (pull from git and re-run).\n"
            "  2. If the checkpoint has no gene_symbols (older training), "
            "pass --gene_panel_h5ad pointing at any training-set silver h5ad.\n"
            "  3. If the query stores symbols in a non-standard var column, "
            "pass --query_gene_col <colname>."
        )
        raise RuntimeError(msg)
    if coverage < 0.10:
        logger.warning(
            f"  VERY LOW coverage ({coverage:.1%}). Predictions will be "
            "dominated by the zero-padded missing genes; expect degraded "
            "results. Investigate the gene-name mismatch before trusting "
            "this run."
        )

    # Build a subset AnnData of the matched query genes, in the order they
    # appear in the query (so scanpy preprocessing operates on real data).
    matched_idx = [i for i, m in enumerate(mapped) if m is not None]
    if matched_idx:
        sub = adata[:, matched_idx].copy()
        if "counts" not in sub.layers:
            sub.layers["counts"] = sub.X.copy()
        if normalize:
            # Count cells with zero total counts in the matched panel — these
            # will become NaN after normalize_total. Track them so we can
            # report and zero-pad them safely.
            sums = sub.X.sum(axis=1)
            if sp.issparse(sub.X):
                sums = np.asarray(sums).ravel()
            n_zero = int((sums == 0).sum())
            if n_zero > 0:
                logger.warning(
                    f"  {n_zero:,} / {sub.n_obs:,} cells have zero counts in "
                    "the matched gene panel (they don't express any of the "
                    "trained genes). These cells will be zero-padded."
                )
            sc.pp.normalize_total(sub, target_sum=1e4)
            sc.pp.log1p(sub)
            # CRITICAL: replace NaN / Inf from zero-count cells BEFORE
            # scaling. Otherwise sc.pp.scale uses np.mean / np.std which
            # propagate NaN, and a single zero-count cell ends up turning
            # the entire matrix into zeros after the downstream nan_to_num.
            cleaned = sub.X
            if sp.issparse(cleaned):
                cleaned = cleaned.toarray()
            n_nan_before = int(np.isnan(cleaned).sum()) + int(np.isinf(cleaned).sum())
            cleaned = np.nan_to_num(cleaned, nan=0.0, posinf=0.0, neginf=0.0)
            if n_nan_before > 0:
                logger.info(
                    f"  cleaned {n_nan_before:,} non-finite values after "
                    "normalize_total + log1p (these came from the "
                    f"{n_zero:,} zero-count cells)."
                )
            sub.X = cleaned
        if scale:
            sc.pp.scale(sub, max_value=10)
        Xsub = sub.X
        if sp.issparse(Xsub):
            Xsub = Xsub.toarray()
        Xsub = np.asarray(Xsub, dtype=np.float32)
    else:
        Xsub = np.zeros((adata.n_obs, 0), dtype=np.float32)

    # Scatter the matched-gene expression into the full target-ordered matrix.
    target_idx = {n: i for i, n in enumerate(target_names)}
    out = np.zeros((adata.n_obs, len(target_names)), dtype=np.float32)
    for sub_col, query_col in enumerate(matched_idx):
        tgt = mapped[query_col]
        j = target_idx[tgt]
        out[:, j] = Xsub[:, sub_col]
    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)

    return out, {
        "n_present": diag["n_matched"],
        "n_missing": len(target_names) - diag["n_matched"],
        "coverage": coverage,
        "matching_mode": diag["matching_mode"],
        "target_namespace": diag.get("target_namespace"),
        "query_namespace": diag.get("query_namespace"),
        "n_query_genes": diag["n_query_genes"],
        "n_matched_via_mygene": diag.get("n_matched_via_mygene", 0),
    }


def _predict_embedding(
    model,
    gene_expr_np: np.ndarray,
    device: str,
) -> np.ndarray:
    """Run scGG and return the raw d-dim metric embedding (or 2-D coords
    for flow_matching mode). Caller decides whether to project to 2-D."""
    import torch

    ge = torch.from_numpy(gene_expr_np).float().to(device)
    with torch.no_grad():
        if model.objective == "contrastive":
            emb = model.embed_batched(ge)
        else:
            emb = model.generate_embeddings(ge)
    return emb.detach().cpu().numpy()


def _project_2d(emb_np: np.ndarray, method: str) -> np.ndarray:
    """Project a (N, d) embedding to (N, 2) for plotting / RSSD."""
    from scgg.evaluation.luna_metrics import embedding_to_2d
    if emb_np.shape[1] == 2:
        return emb_np
    return embedding_to_2d(emb_np, method=method)


# ---------------------------------------------------------------------------
# Diagnostics: per-class Spearman, UMAP on the raw embedding
# ---------------------------------------------------------------------------


def _per_class_spearman(
    coords_true: np.ndarray,
    coords_pred: np.ndarray,
    cell_class: Optional[np.ndarray],
    min_cells_per_class: int = 50,
) -> Dict[str, object]:
    """Median per-cell Spearman, restricted to within-class pairwise distances.

    Confirms whether good global Spearman is being carried by between-class
    structure (cells of different types are far apart) while within-class
    spatial geometry is missing. If `per_class_median` ≪ `global_median`,
    the embedding has collapsed to a cell-type classifier.
    """
    from scgg.evaluation.luna_metrics import compute_spearman_correlation

    global_res = compute_spearman_correlation(coords_true, coords_pred)
    out: Dict[str, object] = {
        "global_median": global_res["median"],
        "global_mean": global_res["mean"],
        "n_cells": int(coords_true.shape[0]),
        "min_cells_per_class": int(min_cells_per_class),
        "per_class": {},
    }
    if cell_class is None:
        out["note"] = "no cell_class provided; only global Spearman computed"
        return out

    cell_class = np.asarray(cell_class)
    per_class: Dict[str, Dict[str, float]] = {}
    medians: List[float] = []
    for cls in sorted({str(c) for c in cell_class}):
        mask = cell_class == cls
        n = int(mask.sum())
        if n < min_cells_per_class:
            continue
        res = compute_spearman_correlation(coords_true[mask], coords_pred[mask])
        per_class[cls] = {
            "median": res["median"],
            "mean": res["mean"],
            "n_cells": n,
        }
        if not np.isnan(res["median"]):
            medians.append(res["median"])

    out["per_class"] = per_class
    out["mean_of_per_class_medians"] = (
        float(np.mean(medians)) if medians else float("nan")
    )
    out["median_of_per_class_medians"] = (
        float(np.median(medians)) if medians else float("nan")
    )
    return out


def _log_per_class_spearman(
    diag: Dict[str, object], section_id: str, top_n: int = 10,
) -> None:
    """Pretty-log the per-class Spearman summary."""
    g = diag["global_median"]
    pcm = diag.get("mean_of_per_class_medians", float("nan"))
    logger.info(
        f"  [{section_id}] Spearman diagnostic: "
        f"global_median={g:.4f}, mean(per_class_medians)={pcm:.4f}"
    )
    if (
        not np.isnan(g)
        and not np.isnan(pcm)
        and (g - pcm) > 0.10
    ):
        logger.warning(
            f"  [{section_id}] GLOBAL Spearman is {g - pcm:+.3f} higher than "
            f"the mean of per-class medians ({g:.3f} vs {pcm:.3f}). "
            "This is the signature of a cell-type-collapsed embedding: "
            "between-class structure is what drives the headline metric, "
            "but within-class spatial geometry is weak."
        )
    pc = diag.get("per_class", {}) or {}
    if pc:
        sorted_items = sorted(
            pc.items(), key=lambda kv: -kv[1]["n_cells"]
        )[:top_n]
        logger.info(f"  [{section_id}] per-class medians (top {len(sorted_items)} by cell count):")
        for cls, stats in sorted_items:
            logger.info(
                f"    {cls:>20s}  n={stats['n_cells']:>6d}  "
                f"median={stats['median']:+.4f}  mean={stats['mean']:+.4f}"
            )


def _plot_umap_diagnostic(
    embedding: np.ndarray,
    cell_class: Optional[np.ndarray],
    coords_true: Optional[np.ndarray],
    out_svg: Path,
    title_prefix: str = "",
    spot_size: Optional[float] = None,
    n_neighbors: int = 30,
    min_dist: float = 0.1,
    random_state: int = 42,
) -> None:
    """UMAP the raw embedding; show it colored by cell class and by GT y-axis.

    If the model is a cell-type classifier in disguise, the by-class panel
    shows tight, well-separated clusters and the by-GT-y panel shows no
    smooth gradient — same-color cells (=close in y) are spread across the
    UMAP, while within-class cells (=same UMAP cluster) span the full y
    range.
    """
    try:
        import umap  # type: ignore
    except ImportError:
        logger.warning(
            "  umap-learn not installed; skipping UMAP diagnostic. "
            "Install with: pip install umap-learn"
        )
        return

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    matplotlib.rcParams["svg.fonttype"] = "none"
    matplotlib.rcParams["pdf.fonttype"] = 42

    n = embedding.shape[0]
    if n < n_neighbors + 1:
        logger.warning(
            f"  UMAP needs at least n_neighbors+1={n_neighbors + 1} cells; "
            f"got {n}. Skipping."
        )
        return

    logger.info(
        f"  UMAP: fitting on (N={n}, d={embedding.shape[1]}) embedding..."
    )
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=random_state,
    )
    emb_2d = reducer.fit_transform(embedding)

    if spot_size is None:
        spot_size = max(8.0, min(40.0, 1800.0 / np.sqrt(max(n, 1))))

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    # Panel 1: colored by cell class (discrete)
    ax = axes[0]
    if cell_class is not None:
        cell_class = np.asarray(cell_class)
        cats = sorted({str(c) for c in cell_class})
        cmap = plt.get_cmap("tab20", max(len(cats), 1))
        cat_to_color = {c: cmap(i) for i, c in enumerate(cats)}
        colors = [cat_to_color[str(c)] for c in cell_class]
        ax.scatter(emb_2d[:, 0], emb_2d[:, 1], c=colors, s=spot_size, linewidths=0)
        ax.set_title(f"{title_prefix}UMAP of raw embedding\n(colored by cell class)")
        patches = [Patch(facecolor=cat_to_color[c], label=c) for c in cats]
        ncol = min(max(1, (len(cats) + 3) // 4), 6)
        ax.legend(
            handles=patches, labels=cats, loc="center left",
            bbox_to_anchor=(1.0, 0.5), ncol=1, fontsize="xx-small",
            frameon=False,
        )
    else:
        ax.scatter(emb_2d[:, 0], emb_2d[:, 1], s=spot_size, linewidths=0)
        ax.set_title(f"{title_prefix}UMAP of raw embedding")
    ax.set_xlabel("UMAP1"); ax.set_ylabel("UMAP2")

    # Panel 2: colored by GT y-coordinate (continuous gradient).
    # The signal we're looking for: if the embedding encodes spatial
    # position, the color should vary smoothly across each UMAP cluster.
    # If clusters are uniformly colored / spatially scrambled, the
    # embedding has lost within-class spatial structure.
    ax = axes[1]
    if coords_true is not None and coords_true.shape[1] >= 2:
        c_vals = coords_true[:, 1].astype(np.float32)
        sc = ax.scatter(
            emb_2d[:, 0], emb_2d[:, 1], c=c_vals,
            s=spot_size, linewidths=0, cmap="viridis",
        )
        fig.colorbar(sc, ax=ax, label="GT y-coordinate")
        ax.set_title(
            f"{title_prefix}UMAP of raw embedding\n"
            "(colored by GT y; smooth gradient → spatial signal preserved)"
        )
    else:
        ax.scatter(emb_2d[:, 0], emb_2d[:, 1], s=spot_size, linewidths=0)
        ax.set_title(f"{title_prefix}UMAP of raw embedding (no GT)")
    ax.set_xlabel("UMAP1"); ax.set_ylabel("UMAP2")

    fig.tight_layout()
    out_svg.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_svg, format="svg", bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  saved UMAP diagnostic: {out_svg}")


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def _umeyama_align(
    src: np.ndarray, dst: np.ndarray, allow_reflection: bool = True,
) -> np.ndarray:
    """Best similarity transform (rotate + scale + translate, optional
    reflection) mapping `src` onto `dst`. Returns the transformed src.

    Implements Umeyama 1991. NaN/Inf rows are filtered out by a joint mask
    to preserve row correspondence, then the same transform is applied to
    every row of the original `src` (NaN rows pass through unchanged).
    """
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    finite = np.isfinite(src).all(axis=1) & np.isfinite(dst).all(axis=1)
    if finite.sum() < 3:
        return src.astype(np.float32)

    a = src[finite]
    b = dst[finite]
    mu_a, mu_b = a.mean(axis=0), b.mean(axis=0)
    ac, bc = a - mu_a, b - mu_b
    var_a = (ac ** 2).sum() / a.shape[0]
    if var_a < 1e-12:
        return src.astype(np.float32)

    cov = (bc.T @ ac) / a.shape[0]
    U, S, Vt = np.linalg.svd(cov)
    d = np.eye(cov.shape[0])
    if not allow_reflection and np.linalg.det(U @ Vt) < 0:
        d[-1, -1] = -1
    R = U @ d @ Vt
    s = (S * np.diag(d)).sum() / var_a
    t = mu_b - s * R @ mu_a

    out = (s * (src @ R.T)) + t
    return out.astype(np.float32)


def _plot_comparison(
    adata,
    color_col: str,
    out_svg: Path,
    title_prefix: str = "",
    spot_size: Optional[float] = None,
    align_for_plot: bool = True,
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

    # Similarity-align predicted coords to GT for visualization. The
    # contrastive embedding lives in an arbitrary frame (PCA picks max-
    # variance axes), so without this the plot is rotated / flipped /
    # rescaled relative to GT, even when the per-cell metric is correct.
    # We only modify the plot view, not the saved obsm['spatial_pred'].
    pred_plot_key = "spatial_pred"
    if align_for_plot and has_gt and has_pred:
        aligned = _umeyama_align(
            adata.obsm["spatial_pred"][:, :2],
            adata.obsm["spatial"][:, :2],
            allow_reflection=True,
        )
        adata.obsm["spatial_pred_aligned"] = aligned
        pred_plot_key = "spatial_pred_aligned"

    # Default spot size: scale ~ figure_area / sqrt(n_cells), with a higher
    # floor / ceiling than before so individual cells are still visible at
    # 5k+ cells per slice.
    if spot_size is None:
        spot_size = max(10.0, min(60.0, 2400.0 / np.sqrt(max(adata.n_obs, 1))))

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
        pred_title = (
            f"{title_prefix}scGG prediction (aligned)"
            if pred_plot_key == "spatial_pred_aligned"
            else f"{title_prefix}scGG prediction"
        )
        sc.pl.embedding(
            adata, basis=pred_plot_key, ax=axes[1],
            title=pred_title, **common_kw,
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


_KNOWN_SILVER_PREFIXES = (
    "cns_scrna_",
    "mmc_",                  # new name for the LUNA cortex silver
    "merfish_mouse_cortex_", # legacy name, still supported
    "abc_zhuang_abca1_",
)

# Glob patterns that match either the new or legacy prefix for the LUNA
# cortex dataset, parameterized by mouse id (e.g. "1" for train, "2" for
# held-out test). Kept as a tuple so callers can iterate both schemes.
_CORTEX_MOUSE_GLOBS = ("mmc_mouse{m}_*.h5ad", "merfish_mouse_cortex_mouse{m}_*.h5ad")


def _strip_known_prefix(stem: str) -> str:
    """Strip the silver-file prefix to recover a bare section id."""
    for pref in _KNOWN_SILVER_PREFIXES:
        if stem.startswith(pref):
            return stem[len(pref):]
    return stem


def _all_silver_files(silver_dir: Path) -> List[Path]:
    """Return every silver h5ad in `silver_dir` that matches a known prefix."""
    out: List[Path] = []
    for p in sorted(silver_dir.glob("*.h5ad")):
        if any(p.name.startswith(pref) for pref in _KNOWN_SILVER_PREFIXES):
            out.append(p)
    return out


def _resolve_section_files(silver_dir: Path, sections_arg: List[str]) -> List[Path]:
    """Resolve --sections values to silver h5ad paths.

    Accepted forms (any combination):
      - ``all``            -> every silver h5ad with a known prefix
      - ``all_test``       -> for the LUNA cortex layout, every mouse2 slice
      - a bare section id  (``well06``, ``mouse2_slice1``,
                            ``Zhuang-ABCA-1-001``)
      - a filename         (``cns_scrna_well06.h5ad``)
      - a glob             (``mouse2_*`` or ``cns_scrna_well0*``)
    """
    if not silver_dir.exists():
        raise FileNotFoundError(f"silver_dir does not exist: {silver_dir}")

    if sections_arg == ["all"]:
        files = _all_silver_files(silver_dir)
        if not files:
            raise FileNotFoundError(
                f"No silver h5ad files under {silver_dir} matching any of "
                f"the known prefixes: {_KNOWN_SILVER_PREFIXES}"
            )
        return files

    if sections_arg == ["all_test"]:
        # LUNA cortex convention: Mouse 2 is the held-out test set.
        files: List[Path] = []
        seen: set = set()
        for g in _CORTEX_MOUSE_GLOBS:
            for p in sorted(silver_dir.glob(g.format(m=2))):
                if p not in seen:
                    seen.add(p)
                    files.append(p)
        if not files:
            raise FileNotFoundError(
                f"No held-out test files (mmc_mouse2_*.h5ad or legacy "
                f"merfish_mouse_cortex_mouse2_*.h5ad) under {silver_dir}"
            )
        return files

    out: List[Path] = []
    for s in sections_arg:
        # Glob pattern: contains a wildcard somewhere.
        if any(ch in s for ch in "*?["):
            matched: List[Path] = []
            # Try the pattern raw, then with each known prefix prepended.
            patterns = [s, s + ".h5ad"]
            for pref in _KNOWN_SILVER_PREFIXES:
                patterns.append(f"{pref}{s}.h5ad")
                patterns.append(f"{pref}{s}")
            for pat in patterns:
                matched.extend(sorted(silver_dir.glob(pat)))
            # Deduplicate while preserving order.
            seen = set()
            uniq = []
            for p in matched:
                if p in seen:
                    continue
                seen.add(p)
                if p.suffix == ".h5ad":
                    uniq.append(p)
            if not uniq:
                raise FileNotFoundError(
                    f"Glob {s!r} matched no silver h5ads under {silver_dir}"
                )
            out.extend(uniq)
            continue

        # Literal section spec — try the known naming conventions.
        candidates = [silver_dir / s, silver_dir / f"{s}.h5ad"]
        for pref in _KNOWN_SILVER_PREFIXES:
            candidates.append(silver_dir / f"{pref}{s}.h5ad")
        for c in candidates:
            if c.exists() and c.suffix == ".h5ad":
                out.append(c)
                break
        else:
            raise FileNotFoundError(
                f"No silver h5ad matches section spec {s!r}. "
                f"Tried: {[str(c) for c in candidates]}"
            )

    # Deduplicate but preserve user order.
    seen = set()
    uniq_out: List[Path] = []
    for p in out:
        if p in seen:
            continue
        seen.add(p)
        uniq_out.append(p)
    return uniq_out


def run_inference(
    checkpoint: str,
    silver_dir: str,
    sections: List[str],
    output_dir: Optional[str],
    color_col: str,
    projection: str,
    device: Optional[str],
    gene_panel_h5ad: Optional[str],
    no_normalize: bool,
    no_scale: bool,
    query_gene_col: Optional[str] = None,
    mygene_cache_path: Optional[str] = None,
    species: str = "mouse",
    no_mygene: bool = False,
    spot_size: Optional[float] = None,
    no_align_plot: bool = False,
    per_class_spearman: bool = False,
    umap_diagnostic: bool = False,
    diagnostic_class_col: Optional[str] = None,
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

    silver_path = Path(silver_dir)
    # Default output_dir derived from silver_dir.name so cortex inference
    # lands in `../scgg-reproducibility/artifacts/mmc_luna/`
    # and CNS inference in `.../cns_luna/` etc.
    if output_dir is None:
        out_dir = Path("../scgg-reproducibility/artifacts") / silver_path.name
    else:
        out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Loading checkpoint: {ckpt_path}")
    logger.info(f"Device: {dev}")
    logger.info(f"Output dir: {out_dir}")
    model, cfg, gene_names, gene_symbols, data_summary = _load_checkpoint(ckpt_path, dev)
    logger.info(
        f"Model: objective={model.objective}, "
        f"params={sum(p.numel() for p in model.parameters()):,}"
    )

    expected_n_genes = None
    if data_summary and data_summary.get("n_genes"):
        expected_n_genes = int(data_summary["n_genes"])
    else:
        first_layer_key = next(
            (k for k in model.state_dict().keys()
             if "encoder.backbone.0.0.weight" in k),
            None,
        )
        if first_layer_key is not None:
            expected_n_genes = int(model.state_dict()[first_layer_key].shape[1])

    gene_panel_source = "checkpoint.data_summary.json" if (
        data_summary and data_summary.get("gene_names")
    ) else None

    if (not gene_names or not gene_symbols) and gene_panel_h5ad:
        # Always re-load both names + symbols from the fallback h5ad so the
        # inference script has them even if the older training run only
        # saved gene_names.
        gn, gs = _gene_panel_from_h5ad(Path(gene_panel_h5ad))
        if not gene_names:
            gene_names = gn
            gene_panel_source = f"--gene_panel_h5ad={gene_panel_h5ad}"
        if not gene_symbols and gs:
            gene_symbols = gs
        logger.info(
            f"  --gene_panel_h5ad supplied: {len(gn)} genes, "
            f"{len(gs)} symbols from {gene_panel_h5ad}"
        )

    # Last-resort fallback: auto-discover a representative training section
    # in silver_dir. Needed for cortex-style checkpoints that never wrote a
    # data_summary.json.
    if not gene_names:
        disco = _autodiscover_gene_panel(silver_path, expected_n_genes=expected_n_genes)
        if disco is not None:
            disco_path, gn, gs = disco
            gene_names = gn
            if not gene_symbols and gs:
                gene_symbols = gs
            gene_panel_source = f"autodiscover:{disco_path}"
            logger.info(
                f"  auto-discovered gene panel from {disco_path.name}: "
                f"{len(gn)} genes"
                + (f", {len(gs)} symbols" if gs else " (no symbol column)")
            )
    if not gene_names:
        raise RuntimeError(
            "Could not recover the trained gene panel. Either:\n"
            "  1. Re-train with the updated train_scgg_on_abc.py (saves "
            "gene_names in data_summary.json), or\n"
            "  2. Pass --gene_panel_h5ad pointing at any silver h5ad from "
            "the training set, or\n"
            "  3. Put a representative training h5ad in --silver_dir so "
            "the script can auto-discover the panel."
        )
    if not gene_symbols:
        logger.info(
            "  no gene_symbols available — symbol-namespace queries will "
            "rely on direct case-insensitive matching against target names."
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

    # Set up the mygene cache only when we actually need symbol->Ensembl
    # bridging (target is Ensembl). For symbol-symbol matching (cortex)
    # it's not used.
    mygene_cache: Optional[MygeneCache] = None
    if not no_mygene and _looks_like_ensembl(gene_names):
        cache_p = (
            Path(mygene_cache_path)
            if mygene_cache_path
            else Path.home() / ".cache" / "scgg" / f"mygene_{species}.json"
        )
        mygene_cache = MygeneCache(cache_path=cache_p, species=species)
        logger.info(
            f"  mygene cache: enabled (path={cache_p}, species={species})"
        )

    section_paths = _resolve_section_files(silver_path, sections)
    logger.info(f"Inference on {len(section_paths)} sections: "
                 f"{[p.name for p in section_paths]}")

    summary = []
    for path in section_paths:
        section_id = _strip_known_prefix(path.stem)
        logger.info(f"[{section_id}] {path}")
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

        # Manual override for the gene-name column (e.g. --query_gene_col Gene
        # when the scRNA h5ad stores symbols in a var column and has a useless
        # integer index in var_names).
        if query_gene_col is not None:
            if query_gene_col not in adata.var.columns:
                raise ValueError(
                    f"--query_gene_col {query_gene_col!r} not found in "
                    f"adata.var.columns ({list(adata.var.columns)})"
                )
            override_names = adata.var[query_gene_col].astype(str).tolist()
            adata.var_names = override_names
            adata.var_names_make_unique()
            logger.info(
                f"  applied --query_gene_col override: var_names <- "
                f"adata.var['{query_gene_col}'] ({override_names[:3]}...)"
            )

        t0 = time.time()
        X_aligned, panel_stats = _preprocess_and_align(
            adata, gene_names, gene_symbols,
            normalize=normalize, scale=scale,
            mygene_cache=mygene_cache,
        )
        # Defensive: catch all-zero inputs early.
        in_max = float(np.abs(X_aligned).max()) if X_aligned.size else 0.0
        in_std = float(X_aligned.std()) if X_aligned.size else 0.0
        logger.info(
            f"  aligned input stats: shape={X_aligned.shape}, "
            f"max|x|={in_max:.4f}, std={in_std:.4f}"
        )
        if in_max == 0.0:
            raise RuntimeError(
                "Aligned input matrix is all zeros — inference cannot run. "
                "Check the gene-panel matching log above."
            )

        embedding = _predict_embedding(model, X_aligned, dev)
        coords_pred = _project_2d(embedding, projection)
        elapsed = time.time() - t0
        logger.info(
            f"  inference done in {elapsed:.1f}s; "
            f"emb shape={embedding.shape}, pred 2D shape={coords_pred.shape}"
        )
        # Detect a fully-collapsed prediction (every cell at the same point)
        # and warn loudly — usually means OOD shift was too large or the
        # model didn't converge well.
        coord_std = coords_pred.std(axis=0)
        coord_range = coords_pred.max(axis=0) - coords_pred.min(axis=0)
        logger.info(
            f"  pred stats: std={coord_std.tolist()}  range={coord_range.tolist()}"
        )
        if float(coord_std.max()) < 1e-3:
            logger.warning(
                "  PREDICTED COORDS ARE ESSENTIALLY CONSTANT across cells. "
                "The model has collapsed to a single point in metric space. "
                "Common causes: (a) the input matrix is mostly zero-padded "
                "(check coverage above), (b) the checkpoint did not converge, "
                "(c) the ABC model never saw input distributions like this. "
                "If coverage is high but you still see this, the model needs "
                "more training or a domain-adaptation step (e.g. Harmony)."
            )

        adata.obsm["spatial_pred"] = coords_pred.astype(np.float32)

        # Diagnostics: per-class Spearman + UMAP of the raw embedding.
        # Confirm whether good headline Spearman is being carried by
        # between-class structure while within-class spatial geometry is
        # weak (i.e. cell-type-collapsed embedding).
        diag_class_col = diagnostic_class_col or color_col_eff
        cell_class_arr = None
        if diag_class_col and diag_class_col in adata.obs.columns:
            cell_class_arr = adata.obs[diag_class_col].astype(str).to_numpy()

        spearman_diag: Optional[Dict[str, object]] = None
        coords_true_2d = (
            adata.obsm["spatial"][:, :2]
            if "spatial" in adata.obsm and adata.obsm["spatial"].shape[1] >= 2
            else None
        )
        if per_class_spearman and coords_true_2d is not None:
            # Use the raw embedding (Spearman is dimension-agnostic — it
            # operates on ranked pairwise distances, so passing the full
            # d-dim embedding is exactly what LUNA does).
            spearman_diag = _per_class_spearman(
                coords_true_2d, embedding, cell_class_arr,
            )
            _log_per_class_spearman(spearman_diag, section_id)
        elif per_class_spearman and coords_true_2d is None:
            logger.warning(
                "  --per_class_spearman requested but no obsm['spatial'] "
                "ground truth in this h5ad; skipping diagnostic."
            )

        out_h5ad = out_dir / f"{section_id}_predicted.h5ad"
        adata.write(out_h5ad)
        logger.info(f"  wrote h5ad with spatial_pred: {out_h5ad}")

        if color_col_eff is not None:
            out_svg = out_dir / f"{section_id}_comparison.svg"
            _plot_comparison(
                adata, color_col_eff, out_svg,
                title_prefix=f"[{section_id}] ",
                spot_size=spot_size,
                align_for_plot=not no_align_plot,
            )
        else:
            logger.warning("  no usable color column; skipping plot")

        if umap_diagnostic:
            out_umap_svg = out_dir / f"{section_id}_umap_diagnostic.svg"
            _plot_umap_diagnostic(
                embedding=embedding,
                cell_class=cell_class_arr,
                coords_true=coords_true_2d,
                out_svg=out_umap_svg,
                title_prefix=f"[{section_id}] ",
                spot_size=spot_size,
            )

        section_record = {
            "section_id": section_id,
            "input_path": str(path),
            "output_h5ad": str(out_h5ad),
            "n_cells": int(adata.n_obs),
            **panel_stats,
            "inference_seconds": elapsed,
            "color_col": color_col_eff,
            "projection": projection,
        }
        if spearman_diag is not None:
            section_record["spearman_diagnostic"] = spearman_diag
        summary.append(section_record)

    # Persist any new mygene resolutions for offline reproducibility.
    if mygene_cache is not None:
        mygene_cache.save()

    with open(out_dir / "inference_metadata.json", "w") as f:
        json.dump(
            {
                "checkpoint": str(ckpt_path),
                "silver_dir": str(silver_path),
                "gene_panel_source": gene_panel_source or "unknown",
                "n_gene_panel": len(gene_names),
                "target_namespace": (
                    "ensembl" if _looks_like_ensembl(gene_names) else "symbol"
                ),
                "species": species,
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
        "--sections", nargs="+", default=None,
        help=(
            "Section ids / filenames / glob patterns to run on. Accepts: "
            "bare ids ('well06', 'mouse2_slice1'), filenames "
            "('cns_scrna_well06.h5ad'), globs ('mouse2_*'), or the special "
            "values 'all' (every silver h5ad) and 'all_test' (mouse2_* — "
            "the LUNA cortex held-out split). Default depends on "
            "--silver_dir: 'all_test' for mmc_* / merfish_mouse_cortex_*, "
            "'well06' for cns_*, 'all' otherwise."
        ),
    )
    p.add_argument(
        "--output_dir",
        default=None,
        help="Where to write predicted h5ads and comparison SVGs. Default "
             "derives from --silver_dir's basename and points at "
             "../scgg-reproducibility/artifacts/<silver_dir_name>/.",
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
    p.add_argument(
        "--query_gene_col", default=None,
        help="Manually override which column of the query h5ad's `var` "
             "contains the real gene names (e.g. --query_gene_col Gene). "
             "Use when var_names is a useless integer index. The script "
             "auto-detects this case, but the flag lets you force it.",
    )
    p.add_argument(
        "--mygene_cache", default=None,
        help="JSON file for offline-cached symbol->Ensembl translations. "
             "Default: ~/.cache/scgg/mygene_<species>.json. Only used when "
             "the trained panel is in Ensembl IDs and the query is in "
             "symbols (gene-alias recovery).",
    )
    p.add_argument(
        "--species", default="mouse",
        help="mygene species name when bridging symbols -> Ensembl "
             "(default: mouse).",
    )
    p.add_argument(
        "--no_mygene", action="store_true",
        help="Disable the mygene fallback. Only direct (target-panel) "
             "symbol matching will be attempted.",
    )
    p.add_argument(
        "--spot_size", type=float, default=None,
        help="Marker size in the comparison plot. Default auto-scales with "
             "cell count (~10-60). Try 30-100 for sparser slices, 5-15 for "
             "very dense slices.",
    )
    p.add_argument(
        "--no_align_plot", action="store_true",
        help="Don't similarity-align predicted coords to GT before plotting. "
             "The model's metric embedding lives in an arbitrary frame "
             "(rotation/scale/reflection are free), so by default we run "
             "a Umeyama alignment for visualization only — the raw "
             "obsm['spatial_pred'] is always preserved unchanged.",
    )
    p.add_argument(
        "--per_class_spearman", action="store_true",
        help="Compute median per-cell Spearman restricted to within-class "
             "pairs (and the global value). If global ≫ mean(per_class), "
             "the embedding is acting as a cell-type classifier and the "
             "headline metric is being carried by between-class structure "
             "rather than within-class spatial geometry. Logs per-class "
             "values and saves them to inference_metadata.json.",
    )
    p.add_argument(
        "--umap_diagnostic", action="store_true",
        help="Generate <section>_umap_diagnostic.svg: UMAP of the raw "
             "d-dim embedding, colored (left) by cell class and (right) "
             "by GT y-coordinate. A tightly clustered by-class panel + a "
             "scrambled by-GT-y panel is the visual signature of a "
             "cell-type-collapsed embedding. Requires umap-learn.",
    )
    p.add_argument(
        "--diagnostic_class_col", default=None,
        help="Override which obs column is used for per-class Spearman "
             "and UMAP coloring (default: --color value, falling back to "
             "the first categorical column).",
    )
    args = p.parse_args()

    # Pick a sensible default for --sections depending on the dataset.
    sections = args.sections
    if sections is None:
        silver_name = Path(args.silver_dir).name.lower()
        if (
            "merfish_mouse_cortex" in silver_name
            or silver_name == "mmc_luna"
            or silver_name.startswith("mmc_")
        ):
            sections = ["all_test"]
        elif silver_name.startswith("cns_"):
            sections = ["well06"]
        else:
            sections = ["all"]

    run_inference(
        checkpoint=args.checkpoint,
        silver_dir=args.silver_dir,
        sections=sections,
        output_dir=args.output_dir,
        color_col=args.color,
        projection=args.projection,
        device=args.device,
        gene_panel_h5ad=args.gene_panel_h5ad,
        no_normalize=args.no_normalize,
        no_scale=args.no_scale,
        query_gene_col=args.query_gene_col,
        mygene_cache_path=args.mygene_cache,
        species=args.species,
        no_mygene=args.no_mygene,
        spot_size=args.spot_size,
        no_align_plot=args.no_align_plot,
        per_class_spearman=args.per_class_spearman,
        umap_diagnostic=args.umap_diagnostic,
        diagnostic_class_col=args.diagnostic_class_col,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
