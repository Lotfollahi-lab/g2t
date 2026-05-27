"""
Nicheformer cell-level embedding pre-processor.

NOTE — this file is VENDORED from the squint-reproducibility tree
(``squint-reproducibility/analysis/benchmarking/cell_type_identification/
_nicheformer_embedding.py``). Vendoring rather than importing keeps
scgg self-contained — no cross-repo path dependencies, no fragile
``sys.path`` mucking. If the upstream squint helper changes (new
gene-mapping routes, new technology tokens, etc.) re-copy from
there. Standard-library + numpy/anndata/pandas only — no other
in-repo imports — so the copy is safe.

Mirrors the official tokenization notebook
(``notebooks/tokenization/merfish_mouse_brain.ipynb`` in the
[theislab/nicheformer](https://github.com/theislab/nicheformer)
repository) and wraps it so it can be used as a drop-in replacement
for :func:`compute_scvi_embedding`. Writes the resulting per-cell
embeddings to ``adata.obsm["emb"]`` for the cell-embedding tokenizer.

Why this is non-trivial
-----------------------
Nicheformer's reference vocabulary is **human Ensembl IDs**
(``ENSG...``). For mouse data the published pipeline maps mouse
Ensembl IDs to their human orthologs via a Biomart export
(``mart_export.txt`` with columns ``Gene stable ID`` and
``Human gene stable ID``). This wrapper supports two routes:

- **Recommended:** pass ``gene_mapper_path=<mart_export.csv>`` for
  the canonical recipe (deterministic, reproducible).
- **Fallback:** pass nothing and call
  :func:`add_human_ortholog_ensembl_ids` beforehand to populate
  ``adata.var['human_ensembl_id']`` via mygene + Ensembl REST, then
  call this function with ``gene_col='human_ensembl_id'``.

Pipeline (matches the notebook step-for-step)
---------------------------------------------
1. Resolve a per-row "alignment ID": for each row of ``adata.var``,
   produce a human Ensembl ID (or fall back to the mouse Ensembl ID).
2. ``ad.concat([model.h5ad, adata], join='outer', axis=0)`` and drop
   the placeholder reference observation.
3. Subset/reorder to ``model.var_names`` so the gene order / count
   matches the technology-mean vector exactly (20,310 genes).
4. Process ``technology_mean``: ``nan_to_num`` -> round-to-int ->
   replace zeros with one. (Verified against the official .npy file:
   visible values are in the 50-100 range, and ~71% of entries are
   NaN, indicating genes whose tech-specific mean falls back to 1
   after this processing.)
5. Tokenize via ``sf_normalize`` (target_sum=10_000) -> divide by the
   processed mean -> ``_sub_tokenize_data`` (numba) which ranks each
   row's non-zero genes, takes the top ``max_seq_len`` (default 1500
   = the model's actual context length), and adds ``aux_tokens=30``.
6. Load Nicheformer from the ``.ckpt`` and call
   ``model.get_embeddings(batch, layer=-1)`` per batch, collecting
   per-cell vectors into ``adata.obsm[obsm_key]``.
7. Optional: PCA-project to ``n_latent`` dims (default None = keep
   all 512).

Token-ID dictionaries are taken verbatim from the notebook (cell 4)::

    modality: dissociated=3, spatial=4
    specie:   human=5, mouse=6
    assay:    merfish=7, cosmx=8, visium=9, '10x 5\\' v2'=10, ...

Source-verified facts (from theislab/nicheformer DeepWiki + paper)
------------------------------------------------------------------
- Model architecture: ``dim_model=512``, ``nlayers=12``, ``nheads=16``,
  ``context_length=1500``, ``n_tokens=20340`` (20,310 genes + 30 aux).
- Technology mean has shape (20,310,), float64, ~71% NaN.
- ``get_embeddings(batch, layer=-1)`` is the canonical inference call.
"""

from __future__ import annotations

import logging
import os
from typing import Dict, Optional

import anndata as ad
import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)


# ===========================================================================
# Notebook-verified token vocabularies (cell 4 of merfish_mouse_brain.ipynb)
# ===========================================================================

MODALITY_TOKENS: Dict[str, int] = {
    "dissociated": 3,
    "spatial": 4,
}

SPECIES_TOKENS: Dict[str, int] = {
    "human": 5,
    "Homo sapiens": 5,
    "Mus musculus": 6,
    "mouse": 6,
}

TECHNOLOGY_TOKENS: Dict[str, int] = {
    "merfish": 7,
    "MERFISH": 7,
    "cosmx": 8,
    "CosMx": 8,
    "visium": 9,
    "Visium": 9,
    "10x 5' v2": 10,
    "10x 3' v3": 11,
    "10x 3' v2": 12,
    "10x 5' v1": 13,
    "10x 3' v1": 14,
    "10x 3' transcription profiling": 15,
    "10x transcription profiling": 15,
    "10x 5' transcription profiling": 16,
    "CITE-seq": 17,
    "Smart-seq v4": 18,
}


# Model-verified constant: pretrained Nicheformer has context_length=1500.
# Sequences longer than this will fail at the positional-embedding lookup.
NICHEFORMER_CONTEXT_LENGTH = 1500


# ===========================================================================
# Ensembl REST helper: mouse Ensembl -> human Ensembl
# ===========================================================================

def _ensembl_homology_lookup(
    mouse_ensembl_ids: list,
    verbose: bool = True,
    max_workers: int = 8,
    timeout: float = 15.0,
) -> Dict[str, str]:
    """
    Query the Ensembl REST homology endpoint for each mouse Ensembl
    gene ID and return a ``{mouse_id: human_id}`` dict.

    Uses ``ThreadPoolExecutor`` with a small number of concurrent
    requests; Ensembl REST allows ~15 req/s sustained.

    NOTE: For reproducible benchmarking against the published
    Nicheformer paper, prefer ``gene_mapper_path=<mart_export.txt>``
    (a static Biomart download) over this REST-based lookup.
    """
    try:
        import requests  # noqa: F401
    except Exception as e:
        raise ImportError(
            "_ensembl_homology_lookup requires `requests`. "
            f"Original error: {e!s}")
    import requests
    from concurrent.futures import ThreadPoolExecutor

    base = "https://rest.ensembl.org/homology/id/mus_musculus"
    headers = {"Accept": "application/json"}
    params_template = {
        "target_species": "homo_sapiens",
        "type": "orthologues",
        "format": "condensed",
    }

    def _query(mens_id: str) -> tuple:
        try:
            r = requests.get(
                f"{base}/{mens_id}",
                params=params_template,
                headers=headers,
                timeout=timeout,
            )
            if r.status_code != 200:
                return mens_id, None
            data = r.json()
            for entry in data.get("data", []):
                for h in entry.get("homologies", []):
                    target_id = h.get("id")
                    if target_id and str(target_id).startswith("ENSG"):
                        return mens_id, str(target_id)
            return mens_id, None
        except Exception:
            return mens_id, None

    out: Dict[str, str] = {}
    n = len(mouse_ensembl_ids)
    if verbose and n > 0:
        print(
            f"  ...calling Ensembl REST for {n} mouse Ensembl IDs "
            f"(this is the slow step; expect ~{max(5, n // 15)}s).",
            flush=True)
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for mens_id, hens_id in ex.map(_query, mouse_ensembl_ids):
            if hens_id:
                out[str(mens_id)] = hens_id
    return out


# ===========================================================================
# Public helper: map mouse symbols -> human Ensembl IDs via mygene
# ===========================================================================

def add_human_ortholog_ensembl_ids(
    adata: ad.AnnData,
    species: str = "mouse",
    gene_col_in: Optional[str] = None,
    gene_col_out: str = "human_ensembl_id",
    fallback_to_input: bool = True,
    verbose: bool = True,
    inplace: bool = True,
) -> ad.AnnData:
    """
    Populate ``adata.var[gene_col_out]`` with human Ensembl IDs derived
    from mouse gene symbols (or mouse Ensembl IDs).

    For each row in ``adata.var``:
      - Look up the gene symbol (or Ensembl ID) in mygene.info.
      - Find its mouse Ensembl ID.
      - Map mouse Ensembl -> human Ensembl via Ensembl REST homology.
      - If no mapping exists and ``fallback_to_input=True``, keep the
        original input string so it can be picked up by the outer-join
        with ``model.h5ad`` (mouse-only genes get dropped at the
        ``model.var_names`` subsetting step downstream).
    """
    if species.lower() not in ("mouse", "mus musculus"):
        # For non-mouse species (typically `human`), there's no ortholog
        # mapping to do — but we may still need to convert SYMBOLS to
        # Ensembl IDs so downstream code that expects ENSG in var
        # (Nicheformer, Geneformer) gets the right format. Decision tree:
        #
        #   - If the input column already holds Ensembl IDs (≥50% start
        #     with `ENSG`), just copy verbatim — nothing to do.
        #   - Otherwise treat the input as HGNC symbols and run a
        #     species-specific mygene lookup (e.g. CD3D -> ENSG00000167286
        #     when species=human).
        #
        # This makes the helper "smart" for human input too: every
        # downstream method that consumes `human_ensembl_id` gets a
        # proper Ensembl column regardless of whether the silver var_names
        # were symbols or already ENSG.
        if not inplace:
            adata = adata.copy()
        if gene_col_in is None:
            raw_ids = adata.var_names.astype(str).to_numpy()
        else:
            if gene_col_in not in adata.var.columns:
                raise KeyError(
                    f"gene_col_in={gene_col_in!r} not in adata.var.")
            raw_ids = adata.var[gene_col_in].astype(str).to_numpy()

        is_ensembl = np.array(
            [v.startswith("ENSG") for v in raw_ids], dtype=bool,
        )
        if is_ensembl.mean() > 0.5:
            if verbose:
                logger.info(
                    "add_human_ortholog_ensembl_ids: species=%r and "
                    "input looks like Ensembl IDs (%d/%d start with "
                    "ENSG); using verbatim.",
                    species, int(is_ensembl.sum()), len(raw_ids))
            adata.var[gene_col_out] = raw_ids
            return adata

        # Symbols → ENSG via mygene's species-specific DB.
        try:
            import mygene
        except Exception as e:
            raise ImportError(
                "add_human_ortholog_ensembl_ids requires the 'mygene' "
                "package for human-symbol → ENSG mapping. Install via "
                f"`pip install mygene`. Original error: {e!s}")
        if verbose:
            print(
                f"[add_human_ortholog_ensembl_ids] species={species!r}: "
                f"mapping {len(raw_ids)} symbols → Ensembl via mygene "
                f"({species}-DB lookup, no cross-species step).")
        mg = mygene.MyGeneInfo()
        # Strip species name for mygene (it accepts "human", "mouse", etc.).
        mg_species = species.lower().split()[0]
        df = mg.querymany(
            list(raw_ids),
            scopes=["symbol", "alias", "ensembl.gene"],
            species=mg_species,
            fields="ensembl.gene",
            returnall=False, as_dataframe=True,
        )
        df = df[~df.index.duplicated(keep="first")]

        def _first_ens(x):
            if isinstance(x, list) and x and isinstance(x[0], dict):
                return x[0].get("gene")
            if isinstance(x, dict):
                return x.get("gene")
            return x if isinstance(x, str) else None

        col = "ensembl.gene" if "ensembl.gene" in df.columns else "ensembl"
        sym_to_ens = df[col].apply(_first_ens) if col in df.columns else pd.Series(dtype=str)
        sym_to_ens = sym_to_ens[~sym_to_ens.index.duplicated(keep="first")]
        if fallback_to_input:
            mapped = np.array(
                [str(sym_to_ens.get(s, s)) for s in raw_ids], dtype=object,
            )
        else:
            mapped = np.array(
                [str(sym_to_ens.get(s, "")) for s in raw_ids], dtype=object,
            )
        adata.var[gene_col_out] = mapped
        n_mapped = int(np.sum(
            [str(v).startswith("ENSG") for v in mapped]
        ))
        if verbose:
            print(
                f"[add_human_ortholog_ensembl_ids] {n_mapped}/{len(raw_ids)} "
                f"symbols mapped to ENSG (species={species!r}).")
        return adata

    try:
        import mygene
    except Exception as e:
        raise ImportError(
            "add_human_ortholog_ensembl_ids requires the 'mygene' "
            "package. Install via `pip install mygene`. "
            f"Original error: {e!s}")

    if not inplace:
        adata = adata.copy()

    if gene_col_in is None:
        symbols = adata.var_names.astype(str).to_numpy()
    else:
        if gene_col_in not in adata.var.columns:
            raise KeyError(
                f"gene_col_in={gene_col_in!r} not in adata.var.")
        symbols = adata.var[gene_col_in].astype(str).to_numpy()

    if verbose:
        print(
            f"[add_human_ortholog_ensembl_ids] Mapping {len(symbols)} "
            "mouse gene IDs -> human Ensembl via mygene + Ensembl REST.")

    mg = mygene.MyGeneInfo()

    # ---- Step 1: input -> mouse Ensembl gene ID -------------------
    step1 = mg.querymany(
        symbols.tolist(),
        scopes=["symbol", "alias", "ensembl.gene"],
        species="mouse",
        fields="ensembl.gene",
        returnall=False, as_dataframe=True,
    )

    def _first_ens(x) -> Optional[str]:
        # Extract a single Ensembl gene ID from whatever shape mygene
        # returned for the `ensembl.gene` field. Supports BOTH human
        # (ENSG...) and mouse (ENSMUSG...) IDs because Step 1 queries
        # the mouse DB (returns ENSMUSG) and Step 3 queries the human
        # DB (returns ENSG) — using the same helper for both.
        if isinstance(x, list) and x and isinstance(x[0], dict):
            return x[0].get("gene")
        if isinstance(x, dict):
            return x.get("gene")
        if isinstance(x, str) and (
            x.startswith("ENSG") or x.startswith("ENSMUSG")
        ):
            return x
        return None

    if "ensembl.gene" in step1.columns:
        sym_to_mens = step1["ensembl.gene"].apply(_first_ens).dropna()
    elif "ensembl" in step1.columns:
        sym_to_mens = step1["ensembl"].apply(_first_ens).dropna()
    else:
        sym_to_mens = pd.Series(dtype=str)
    sym_to_mens = sym_to_mens[
        ~sym_to_mens.index.duplicated(keep="first")].astype(str)

    if verbose:
        print(
            f"  Step 1 (symbol -> mouse Ensembl):    "
            f"{len(sym_to_mens)}/{len(symbols)} mapped"
            + (f"   sample={list(sym_to_mens.items())[:3]}"
               if len(sym_to_mens) else ""))

    # ---- Step 2: mouse Ensembl -> human Ensembl via Ensembl REST --
    mens_to_hens: Dict[str, str] = {}
    if len(sym_to_mens):
        mens_to_hens = _ensembl_homology_lookup(
            list(sym_to_mens.unique()), verbose=verbose)
    if verbose:
        print(
            f"  Step 2 (mouse Ensembl -> human Ensembl via Ensembl "
            f"REST): {len(mens_to_hens)}/"
            f"{len(sym_to_mens.unique()) if len(sym_to_mens) else 0} "
            f"mapped"
            + (f"   sample={list(mens_to_hens.items())[:3]}"
               if mens_to_hens else ""))

    # ---- Compose Step 1+2: input symbol -> human Ensembl ----------
    sym_to_hens: Dict[str, str] = {}
    for sym, mens in sym_to_mens.items():
        h = mens_to_hens.get(mens)
        if h:
            sym_to_hens[str(sym)] = str(h)
    n_after_step2 = len(sym_to_hens)

    # ---- Step 3 (FALLBACK): direct mouse-symbol -> human Ensembl --
    # For genes Ensembl REST didn't cover, query mygene against the
    # HUMAN DB with the original (mouse-cased) symbol. Many mammalian
    # genes share their symbol spelling across species (modulo case),
    # and mygene's symbol/alias scopes are case-insensitive — so this
    # picks up the orthologs that have a directly-matching human gene
    # but for which Ensembl's curated homology table happens to be
    # incomplete. Cheap (one batched mygene call) and usually adds
    # 10-20 percentage points of coverage on mouse panels.
    unmapped_after_2 = [s for s in symbols if s not in sym_to_hens]
    if unmapped_after_2:
        try:
            step3 = mg.querymany(
                unmapped_after_2,
                scopes=["symbol", "alias"],
                species="human",
                fields="ensembl.gene",
                returnall=False, as_dataframe=True,
            )
            col = ("ensembl.gene" if "ensembl.gene" in step3.columns
                   else ("ensembl" if "ensembl" in step3.columns else None))
            if col is not None:
                step3_map = step3[col].apply(_first_ens).dropna()
                step3_map = step3_map[
                    step3_map.astype(str).str.startswith("ENSG")]
                step3_map = step3_map[
                    ~step3_map.index.duplicated(keep="first")]
                added = 0
                for sym, ensg in step3_map.items():
                    sym = str(sym)
                    if sym not in sym_to_hens:
                        sym_to_hens[sym] = str(ensg)
                        added += 1
                if verbose:
                    print(
                        f"  Step 3 (mouse symbol -> human Ensembl direct): "
                        f"+{added} new mappings ("
                        f"{n_after_step2}->{len(sym_to_hens)} total).")
        except Exception as exc:
            if verbose:
                print(f"  Step 3 (direct human-DB lookup) failed: {exc}")
    n_after_step3 = len(sym_to_hens)

    # ---- Step 4 (FALLBACK): NCBI HomoloGene via mygene -------------
    # Final fallback: for genes that even direct human-DB lookup
    # missed (mouse-specific symbols, or symbol-renamed orthologs),
    # query mygene's `homologene` field on the MOUSE Ensembl ID. The
    # `homologene.genes` field is a list of [taxid, gene_id] pairs;
    # taxid 9606 == human. We get back NCBI Entrez gene IDs, which we
    # then convert to Ensembl in a second mygene call.
    unmapped_after_3 = [s for s in symbols if s not in sym_to_hens]
    if unmapped_after_3 and len(sym_to_mens):
        try:
            mens_for_step4 = [
                str(sym_to_mens[s]) for s in unmapped_after_3
                if s in sym_to_mens.index
            ]
            if mens_for_step4:
                step4a = mg.querymany(
                    mens_for_step4,
                    scopes="ensembl.gene",
                    species="mouse",
                    fields="homologene",
                    returnall=False, as_dataframe=True,
                )
                # Extract human Entrez IDs (taxid 9606) from each row.
                human_entrez_per_mens: Dict[str, str] = {}
                if "homologene.genes" in step4a.columns:
                    for mens, val in step4a["homologene.genes"].items():
                        if not isinstance(val, list):
                            continue
                        for pair in val:
                            if (isinstance(pair, list) and len(pair) >= 2
                                    and int(pair[0]) == 9606):
                                human_entrez_per_mens[str(mens)] = str(pair[1])
                                break
                # Convert Entrez -> Ensembl in one batched call.
                if human_entrez_per_mens:
                    step4b = mg.querymany(
                        list(human_entrez_per_mens.values()),
                        scopes="entrezgene",
                        species="human",
                        fields="ensembl.gene",
                        returnall=False, as_dataframe=True,
                    )
                    col4 = ("ensembl.gene" if "ensembl.gene" in step4b.columns
                            else ("ensembl" if "ensembl" in step4b.columns
                                  else None))
                    if col4 is not None:
                        entrez_to_ensg = step4b[col4].apply(_first_ens).dropna()
                        entrez_to_ensg = entrez_to_ensg[
                            entrez_to_ensg.astype(str).str.startswith("ENSG")]
                        entrez_to_ensg = entrez_to_ensg[
                            ~entrez_to_ensg.index.duplicated(keep="first")]
                        entrez_to_ensg = entrez_to_ensg.astype(str).to_dict()
                        added = 0
                        for sym, mens in sym_to_mens.items():
                            sym = str(sym)
                            if sym in sym_to_hens:
                                continue
                            entrez = human_entrez_per_mens.get(str(mens))
                            if entrez and entrez in entrez_to_ensg:
                                sym_to_hens[sym] = entrez_to_ensg[entrez]
                                added += 1
                        if verbose:
                            print(
                                f"  Step 4 (NCBI HomoloGene via mygene): "
                                f"+{added} new mappings ("
                                f"{n_after_step3}->{len(sym_to_hens)} total).")
        except Exception as exc:
            if verbose:
                print(f"  Step 4 (HomoloGene fallback) failed: {exc}")

    # ---- Compose final ensembl_id column --------------------------
    n_mapped = len(sym_to_hens)
    out = []
    for s in symbols:
        h = sym_to_hens.get(s)
        if h is None:
            out.append(str(s) if fallback_to_input else "")
        else:
            out.append(str(h))
    adata.var[gene_col_out] = np.asarray(out, dtype=object)

    if verbose:
        print(
            f"[add_human_ortholog_ensembl_ids] FINAL: "
            f"{n_mapped}/{len(symbols)} input IDs mapped to human "
            "Ensembl. The rest "
            + ("fall back to the input string."
               if fallback_to_input else "are empty strings.")
            + f"\nSample of mapped pairs: "
            f"{list(sym_to_hens.items())[:5]}")
    return adata


# ===========================================================================
# Numba-JIT'd inner tokenizer (verbatim from the notebook)
# ===========================================================================

def _sub_tokenize_data(
    x: np.ndarray, max_seq_len: int = NICHEFORMER_CONTEXT_LENGTH,
    aux_tokens: int = 30,
) -> np.ndarray:
    """
    Per cell: argsort genes by (descending) value, take the top
    ``max_seq_len`` non-zero indices, add ``aux_tokens`` so they sit
    above the reserved special tokens. Padding with zeros.
    """
    try:
        import numba

        @numba.jit(nopython=True, nogil=True)
        def _impl(x, max_seq_len, aux_tokens):
            scores_final = np.empty(
                (x.shape[0], max_seq_len if max_seq_len > 0 else x.shape[1]))
            for i in range(x.shape[0]):
                cell = x[i]
                nonzero_mask = np.nonzero(cell)[0]
                sorted_indices = nonzero_mask[
                    np.argsort(-cell[nonzero_mask])][:max_seq_len]
                sorted_indices = sorted_indices + aux_tokens
                if max_seq_len:
                    scores = np.zeros(max_seq_len, dtype=np.int32)
                else:
                    scores = np.zeros_like(cell, dtype=np.int32)
                scores[: sorted_indices.size] = sorted_indices.astype(np.int32)
                scores_final[i, :] = scores
            return scores_final

        return _impl(x, max_seq_len, aux_tokens)
    except ImportError:
        scores_final = np.zeros(
            (x.shape[0], max_seq_len), dtype=np.int32)
        for i in range(x.shape[0]):
            cell = x[i]
            nonzero_mask = np.nonzero(cell)[0]
            sorted_indices = nonzero_mask[
                np.argsort(-cell[nonzero_mask])][:max_seq_len]
            sorted_indices = sorted_indices + aux_tokens
            scores_final[i, : sorted_indices.size] = (
                sorted_indices.astype(np.int32))
        return scores_final


def _sf_normalize(X: np.ndarray) -> np.ndarray:
    """Size-factor normalize to 10_000 counts per cell (notebook's
    ``sf_normalize``). Operates on a dense numpy array."""
    counts = np.asarray(X.sum(axis=1), dtype=np.float64).reshape(-1)
    counts[counts == 0] = 1.0
    scaling = (10_000.0 / counts).astype(np.float32)
    return (X.astype(np.float32) * scaling[:, None]).astype(np.float32)


def _process_tech_mean(tech_mean: np.ndarray) -> np.ndarray:
    """
    Process the technology-mean vector before division.
    Operations (in order, matching the canonical notebook):
      1. NaN -> 0  (the ~71% NaN entries in the official .npy mean).
      2. Round to nearest int (values are in the 50-100 range -- see
         actual model file inspection -- so rounding loss is <1%).
      3. Zero -> 1  (avoids div-by-zero for genes whose mean was NaN).
    """
    tm = np.asarray(tech_mean, dtype=np.float64)
    n_nan = int(np.isnan(tm).sum())
    if n_nan > 0:
        logger.info(
            "technology_mean: %d/%d entries are NaN (%.1f%%); "
            "these gene columns will use mean=1 after processing.",
            n_nan, tm.size, 100.0 * n_nan / tm.size)
    tm = np.nan_to_num(tm)
    tm = np.where((tm % 1) >= 0.5, np.ceil(tm), np.floor(tm))
    tm = np.where(tm == 0, 1.0, tm)
    return tm.astype(np.float32)


# ===========================================================================
# Gene mapping (mart_export.txt path)
# ===========================================================================

def _load_gene_mapper(
    gene_mapper_path: str,
    mouse_id_col: str = "Gene stable ID",
    human_id_col: str = "Human gene stable ID",
    sep: str = ",",
) -> pd.Series:
    """
    Load a Biomart export with mouse -> human ortholog Ensembl IDs.

    Returns a ``pd.Series`` indexed by mouse Ensembl ID with values
    being the human ortholog Ensembl ID (or ``NaN``).
    """
    if not os.path.exists(gene_mapper_path):
        raise FileNotFoundError(
            f"gene_mapper_path={gene_mapper_path!r} not found.")
    df = pd.read_csv(gene_mapper_path, sep=sep)
    if mouse_id_col not in df.columns or human_id_col not in df.columns:
        raise KeyError(
            f"gene_mapper file must contain columns "
            f"{mouse_id_col!r} and {human_id_col!r}; got "
            f"{list(df.columns)}.")
    df = df.drop_duplicates(mouse_id_col).set_index(mouse_id_col)
    return df[human_id_col]


# ===========================================================================
# Diagnostics -- emitted on zero-overlap before tokenization
# ===========================================================================

def _diagnose_zero_overlap(
    ref: ad.AnnData,
    adata: ad.AnnData,
    gene_col: Optional[str],
) -> str:
    ref_names = ref.var_names.astype(str).to_numpy()
    user_names = adata.var_names.astype(str).to_numpy()

    def _overlap(arr):
        if arr.size == 0:
            return 0
        return int(np.isin(arr, ref_names).sum())

    msg = [
        "Nicheformer gene-alignment FAILED: 0 genes overlap between "
        "your AnnData and the reference at model.h5ad.",
        f"Reference n_genes: {ref.n_vars}",
        f"Reference var_names sample: {ref_names[:5].tolist()}",
        f"User n_genes:      {adata.n_vars}",
        f"User var_names sample:      {user_names[:5].tolist()}",
        "",
        f"User var_names overlap with reference: {_overlap(user_names)}",
    ]
    suggestions = []
    for col in adata.var.columns:
        vals = adata.var[col].astype(str).to_numpy()
        if vals.size == 0:
            continue
        ov = _overlap(vals)
        if ov >= 0.05 * ref.n_vars or ov >= 100:
            suggestions.append((col, ov, vals[:3].tolist()))
    if suggestions:
        suggestions.sort(key=lambda t: -t[1])
        msg += ["", "Candidate adata.var columns with non-zero overlap:"]
        for col, ov, sample in suggestions[:5]:
            msg.append(
                f"  - gene_col={col!r}: {ov} matches; sample={sample}")
        msg.append(
            f"\nTry re-calling with gene_col={suggestions[0][0]!r}.")
    else:
        msg += [
            "",
            "No adata.var column matches the reference well.",
            "For mouse data with gene symbols, run:",
            "  from _nicheformer_embedding import "
            "add_human_ortholog_ensembl_ids",
            "  adata = add_human_ortholog_ensembl_ids(adata, "
            "species='mouse')",
            "  adata = compute_nicheformer_embedding("
            "..., gene_col='human_ensembl_id')",
        ]
    return "\n".join(msg)


# ===========================================================================
# PCA helper (mirrors other embedders)
# ===========================================================================

def _project_with_pca(emb: np.ndarray, n_latent: int | None) -> np.ndarray:
    """PCA-project to n_latent dims, preserving NaN rows. None -> passthrough."""
    if n_latent is None or n_latent >= emb.shape[1]:
        return emb.astype(np.float32)
    from sklearn.decomposition import PCA
    valid = ~np.isnan(emb).any(axis=1)
    if valid.sum() == 0:
        raise RuntimeError("All embeddings are NaN -- cannot fit PCA.")
    pca = PCA(n_components=n_latent, random_state=0)
    proj = np.full((emb.shape[0], n_latent), np.nan, dtype=np.float32)
    proj[valid] = pca.fit_transform(emb[valid]).astype(np.float32)
    return proj


# ===========================================================================
# Public API
# ===========================================================================

def compute_nicheformer_embedding(
    adata: ad.AnnData,
    pretrained_model_path: str,
    model_h5ad_path: str,
    technology_mean_path: str,
    technology: str = "merfish",
    species: str = "mouse",
    modality: str = "spatial",
    gene_col: Optional[str] = None,
    gene_mapper_path: Optional[str] = None,
    obsm_key: str = "emb",
    batch_size: int = 32,
    num_workers: int = 0,
    device: Optional[str] = None,
    max_seq_len: int = NICHEFORMER_CONTEXT_LENGTH,
    aux_tokens: int = 30,
    embedding_layer: int = -1,
    technology_token: Optional[int] = None,
    species_token: Optional[int] = None,
    modality_token: Optional[int] = None,
    n_latent: Optional[int] = None,
    # API parity stubs (silently ignored).
    batch_key: Optional[str] = None,
    layer: Optional[str] = None,
    n_hidden: Optional[int] = None,
    n_layers: Optional[int] = None,
    dropout_rate: Optional[float] = None,
    max_epochs: Optional[int] = None,
    early_stopping: Optional[bool] = None,
    save_model_path: Optional[str] = None,
    accelerator: Optional[str] = None,
) -> ad.AnnData:
    """
    Compute Nicheformer cell embeddings and store them in
    ``adata.obsm[obsm_key]`` (default ``"emb"``).

    Notes
    -----
    The user supplies one of two routes for gene-id alignment:

      A. ``gene_mapper_path=<mart_export.csv>``: a Biomart export with
         columns ``Gene stable ID`` (mouse Ensembl) and
         ``Human gene stable ID``. ``adata.var_names`` (or
         ``adata.var[gene_col]`` if given) must hold mouse Ensembl IDs.
         **Recommended for reproducibility against the published paper.**

      B. ``gene_col=<column with human Ensembl IDs already>``: skip
         ortholog mapping. Use :func:`add_human_ortholog_ensembl_ids`
         beforehand to populate this column from mouse symbols.

    Parameters
    ----------
    max_seq_len
        Token sequence length per cell. **Must be <= 1500** -- the
        pretrained Nicheformer has ``context_length=1500`` learned
        positional embeddings; longer sequences will index out of
        bounds. Default is 1500 (the model's actual context length).
    n_latent
        If set, PCA-project the 512-dim embedding to ``n_latent``
        dims. Default ``None`` (keep all 512). Only enable for direct
        comparison against fixed-dim baselines like scVI.
    """
    # --- Imports --------------------------------------------------
    try:
        import torch
    except Exception as e:
        raise ImportError(
            "compute_nicheformer_embedding requires PyTorch. "
            f"Original error: {e!s}")
    try:
        from nicheformer.models import Nicheformer
    except Exception as e:
        raise ImportError(
            "Nicheformer is not importable. Install from "
            "https://github.com/theislab/nicheformer. "
            f"Original error: {e!s}")
    from scipy import sparse

    # --- Validate inputs -----------------------------------------
    for p, name in (
        (pretrained_model_path, "pretrained_model_path"),
        (model_h5ad_path, "model_h5ad_path"),
        (technology_mean_path, "technology_mean_path"),
    ):
        if not os.path.exists(p):
            raise FileNotFoundError(f"{name}={p!r} not found.")

    if max_seq_len > NICHEFORMER_CONTEXT_LENGTH:
        raise ValueError(
            f"max_seq_len={max_seq_len} exceeds Nicheformer's "
            f"context_length={NICHEFORMER_CONTEXT_LENGTH}. The pretrained "
            "model has only 1500 positional embeddings; longer sequences "
            "will fail at the positional-embedding lookup. Use "
            f"max_seq_len <= {NICHEFORMER_CONTEXT_LENGTH}.")

    species_l = species.lower().strip() if species else ""
    technology_l = technology.lower().strip() if technology else ""
    modality_l = modality.lower().strip() if modality else ""

    sp_tok = (species_token if species_token is not None
              else SPECIES_TOKENS.get(species, SPECIES_TOKENS.get(species_l)))
    tech_tok = (technology_token if technology_token is not None
                else TECHNOLOGY_TOKENS.get(
                    technology, TECHNOLOGY_TOKENS.get(technology_l)))
    mod_tok = (modality_token if modality_token is not None
               else MODALITY_TOKENS.get(
                   modality, MODALITY_TOKENS.get(modality_l)))

    if sp_tok is None:
        raise ValueError(
            f"species={species!r} not in vocab "
            f"{list(SPECIES_TOKENS)}. Pass species_token=<int> to override.")
    if tech_tok is None:
        raise ValueError(
            f"technology={technology!r} not in vocab "
            f"{list(TECHNOLOGY_TOKENS)}. Pass technology_token=<int> "
            "to override.")
    if mod_tok is None:
        raise ValueError(
            f"modality={modality!r} not in vocab "
            f"{list(MODALITY_TOKENS)}. Pass modality_token=<int> "
            "to override.")

    if save_model_path is not None:
        logger.info("save_model_path is unused; ignoring %s",
                    save_model_path)
    if batch_key is not None:
        logger.info(
            "batch_key=%r ignored (Nicheformer does batch correction "
            "via its assay/specie/modality tokens).", batch_key)

    if device is None:
        device_t = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
    else:
        device_t = torch.device(device)

    logger.info(
        "Nicheformer zero-shot inference: n_cells=%d, n_genes=%d, "
        "species=%r (tok=%d), technology=%r (tok=%d), modality=%r "
        "(tok=%d), batch_size=%d, max_seq_len=%d, device=%s",
        adata.n_obs, adata.n_vars, species, sp_tok, technology, tech_tok,
        modality, mod_tok, batch_size, max_seq_len, device_t)

    # --- Step 1: align gene IDs (notebook cell 54-58 + subset) ---
    aligned = _align_to_reference(
        adata=adata,
        model_h5ad_path=model_h5ad_path,
        gene_col=gene_col,
        gene_mapper_path=gene_mapper_path,
    )
    logger.info(
        "Gene-aligned to reference: %d cells x %d canonical genes.",
        aligned.n_obs, aligned.n_vars)

    # --- Step 2: tokenize via the notebook's exact pipeline -----
    tech_mean = np.load(technology_mean_path)
    tech_mean = _process_tech_mean(tech_mean)
    if tech_mean.size != aligned.n_vars:
        raise RuntimeError(
            f"technology_mean has length {tech_mean.size} but "
            f"aligned matrix has {aligned.n_vars} columns. Verify that "
            f"model_h5ad_path corresponds to this technology_mean (both "
            "should describe the same 20,310-gene reference).")

    X = aligned.X
    if sparse.issparse(X):
        X = X.toarray()
    X = np.nan_to_num(np.asarray(X, dtype=np.float32))

    X = _sf_normalize(X)
    X = X / tech_mean.reshape((1, -1))

    tokens = _sub_tokenize_data(
        X, max_seq_len=int(max_seq_len), aux_tokens=int(aux_tokens),
    ).astype(np.int64)
    del X

    # --- Step 3: load model + run inference ----------------------
    logger.info("Loading Nicheformer from %s ...", pretrained_model_path)
    model = Nicheformer.load_from_checkpoint(
        pretrained_model_path, map_location=device_t, strict=False)
    model.to(device_t).eval()
    for p in model.parameters():
        p.requires_grad = False

    raw_emb = _run_nicheformer_inference(
        model=model,
        tokens=tokens,
        species_token=int(sp_tok),
        technology_token=int(tech_tok),
        modality_token=int(mod_tok),
        batch_size=int(batch_size),
        device=device_t,
        embedding_layer=int(embedding_layer),
    )

    if raw_emb.ndim != 2 or raw_emb.shape[0] != adata.n_obs:
        raise RuntimeError(
            f"Nicheformer returned embedding with shape "
            f"{raw_emb.shape}; expected (n_obs={adata.n_obs}, D).")

    logger.info("Raw embedding shape: %s", raw_emb.shape)

    # --- Step 4: optional PCA projection -------------------------
    adata.obsm[obsm_key] = _project_with_pca(raw_emb, n_latent)
    logger.info(
        "Wrote adata.obsm[%r] with shape %s.", obsm_key,
        adata.obsm[obsm_key].shape)
    return adata


# ===========================================================================
# Pipeline steps
# ===========================================================================

def _align_to_reference(
    adata: ad.AnnData,
    model_h5ad_path: str,
    gene_col: Optional[str],
    gene_mapper_path: Optional[str],
) -> ad.AnnData:
    """
    Mirrors notebook cells 54-58 + subset:

    1. Read ``model.h5ad`` (the gene reference; 20,310 canonical genes).
    2. Build an "alignment ID" per row of ``adata.var``:
        - if ``gene_col`` is given, use that column,
        - else if ``gene_mapper_path`` is given, treat
          ``adata.var_names`` as mouse Ensembl IDs and look up the
          human ortholog (fallback: keep mouse ID),
        - else use ``adata.var_names`` as-is.
    3. ``ad.concat([model, adata_aligned], join='outer', axis=0)`` and
       drop the placeholder reference observation with ``[1:]``.
    4. Reorder columns to ``model.var_names`` exactly so ``X`` matches
       the technology mean's gene order / count.
    """
    ref = ad.read_h5ad(model_h5ad_path)
    if ref.n_obs == 0:
        raise RuntimeError(
            f"{model_h5ad_path} has 0 observations; expected at least "
            "one placeholder row.")

    aligned = adata.copy()
    if gene_col is not None:
        if gene_col not in aligned.var.columns:
            raise KeyError(
                f"gene_col={gene_col!r} not in adata.var; available "
                f"columns: {list(aligned.var.columns)[:10]}.")
        new_index = aligned.var[gene_col].astype(str).to_numpy()
    elif gene_mapper_path is not None:
        mapper = _load_gene_mapper(gene_mapper_path)
        old_index = aligned.var_names.astype(str).to_numpy()
        mapped = pd.Series(old_index).map(mapper)
        new_index = np.where(
            mapped.isna() | (mapped.astype(str) == "nan"),
            old_index, mapped.astype(str).to_numpy())
    else:
        new_index = aligned.var_names.astype(str).to_numpy()

    aligned.var_names = pd.Index(new_index, dtype=object)
    aligned.var_names_make_unique()

    # Outer concat as in the notebook, then drop the model placeholder.
    merged = ad.concat([ref, aligned], join="outer", axis=0)
    merged = merged[ref.n_obs:].copy()

    # Subset/reorder columns to match ref.var_names exactly.
    keep = ref.var_names.astype(str).to_numpy()
    keep_set = set(keep)
    mask = np.array(
        [v in keep_set for v in merged.var_names.astype(str)],
        dtype=bool)
    n_kept = int(mask.sum())
    if n_kept == 0:
        raise RuntimeError(_diagnose_zero_overlap(
            ref=ref, adata=aligned, gene_col=gene_col))

    name_to_pos = {n: i for i, n in enumerate(merged.var_names.astype(str))}
    order = [name_to_pos[n] for n in keep if n in name_to_pos]
    if len(order) != ref.n_vars:
        raise RuntimeError(
            f"Could only reorder to {len(order)} of {ref.n_vars} "
            "reference genes -- outer-join did not include the full "
            "reference gene set. Check that model_h5ad_path is the "
            "right one for this technology mean.")
    return merged[:, order].copy()


def _run_nicheformer_inference(
    model,
    tokens: np.ndarray,
    species_token: int,
    technology_token: int,
    modality_token: int,
    batch_size: int,
    device,
    embedding_layer: int,
) -> np.ndarray:
    """Run model.get_embeddings over the tokenized data in batches."""
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    n = tokens.shape[0]
    tok_t = torch.from_numpy(tokens).long()
    sp_t = torch.full((n,), int(species_token), dtype=torch.long)
    as_t = torch.full((n,), int(technology_token), dtype=torch.long)
    mo_t = torch.full((n,), int(modality_token), dtype=torch.long)
    idx_t = torch.arange(n, dtype=torch.long)

    ds = TensorDataset(tok_t, sp_t, as_t, mo_t, idx_t)
    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=False, num_workers=0)

    if not hasattr(model, "get_embeddings"):
        raise RuntimeError(
            "Loaded Nicheformer instance has no get_embeddings method.")

    parts: list = []
    with torch.no_grad():
        for tok, sp, asy, mo, idx in loader:
            tok = tok.to(device, non_blocking=True)
            sp = sp.to(device, non_blocking=True)
            asy = asy.to(device, non_blocking=True)
            mo = mo.to(device, non_blocking=True)
            idx = idx.to(device, non_blocking=True)

            # Canonical batch format (matches src/nicheformer/_embeddings.py
            # label_keys: ['assay', 'specie', 'modality', 'X', 'idx']).
            batch = {
                "X": tok,
                "specie": sp,
                "assay": asy,
                "modality": mo,
                "idx": idx,
            }
            try:
                emb = model.get_embeddings(batch, layer=embedding_layer)
            except TypeError:
                # Older signature: positional args
                try:
                    emb = model.get_embeddings(
                        tok, sp, asy, mo, layer=embedding_layer)
                except TypeError:
                    # Alternative key naming used by some forks
                    batch_alt = {
                        "input_ids": tok, "species": sp,
                        "technology": asy, "modality": mo, "idx": idx,
                    }
                    emb = model.get_embeddings(
                        batch_alt, layer=embedding_layer)

            if hasattr(emb, "detach"):
                emb = emb.detach().float().cpu().numpy()
            else:
                emb = np.asarray(emb, dtype=np.float32)
            if emb.ndim == 3:
                # Per-token embeddings returned; mean-pool over sequence.
                emb = emb.mean(axis=1)
            parts.append(emb)

    return np.concatenate(parts, axis=0).astype(np.float32)
