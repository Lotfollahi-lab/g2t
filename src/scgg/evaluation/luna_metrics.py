"""
Faithful reimplementations of the three metrics LUNA reports on the MERFISH
mouse primary motor cortex benchmark (Figure 3 + Supplementary Fig. 10):

  1. compute_spearman_correlation — per-cell Spearman of pairwise-distance
     rows, aggregated as (mean, median) over cells per slice.
  2. compute_contact — percentile-thresholded contact precision / F1 on
     flattened pairwise distance matrices.
  3. compute_RSSD — per-cell-class Kabsch-aligned Root Sum Squared Deviation
     using scipy.spatial.transform.Rotation.align_vectors after Z-padding
     coordinates to 3-D.

The implementations mirror LUNA's `metrics/evaluation_statistics.py` and
`utils/data/load.py` (per the references in the public repo). They accept
numpy arrays directly so they can score either a 2-D coordinate prediction
(LUNA, CeLEry, novoSpaRc) or a d-dimensional metric embedding (ScGG
contrastive mode). For Spearman/contact this is dimension-agnostic; RSSD
requires 2-D and a `embedding_to_2d` helper is provided.

Aggregation across slices follows the LUNA paper's Figure 3 convention: the
headline number is the MEAN over test slices of the per-slice MEDIAN of the
per-cell Spearman. Both aggregations (mean of medians, mean of means, median
of medians, median of means) are reported in `aggregate_slices`.
"""

from __future__ import annotations

import logging
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from scipy.spatial.distance import cdist
from scipy.spatial.transform import Rotation as R
from scipy.stats import spearmanr

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Distance matrices
# ---------------------------------------------------------------------------


def compute_distance(coords_or_embed: np.ndarray) -> np.ndarray:
    """Pairwise Euclidean distance matrix.

    Matches LUNA's ``compute_distance``: ``cdist(pos.T, pos.T)`` on
    ``pos = [coord_X, coord_Y]``. We accept an (N, d) array directly so this
    is dimension-agnostic — for a ScGG metric embedding (d=32) the row-wise
    Spearman ranking is unchanged by the dim because Spearman is rank-based.

    Args:
        coords_or_embed: (N, d) array of coordinates or embeddings.

    Returns:
        (N, N) pairwise Euclidean distance matrix (float32).
    """
    x = np.asarray(coords_or_embed, dtype=np.float32)
    if x.ndim != 2:
        raise ValueError(f"expected (N, d) array, got shape {x.shape}")
    return cdist(x, x).astype(np.float32, copy=False)


# ---------------------------------------------------------------------------
# Spearman
# ---------------------------------------------------------------------------


# Backends for compute_spearman_correlation. Listed here so callers
# and CLI flags share one source of truth.
VALID_SPEARMAN_BACKENDS = ("scipy", "vectorized", "numba", "gpu")
DEFAULT_SPEARMAN_BACKEND = "scipy"


def compute_spearman_correlation(
    coords_true: np.ndarray,
    coords_pred: np.ndarray,
    progress: bool = False,
    vectorized: bool = False,
    backend: str = DEFAULT_SPEARMAN_BACKEND,
) -> Dict[str, np.ndarray | float]:
    """Per-cell Spearman correlation of pairwise-distance rows.

    LUNA's exact formulation: build the full NxN Euclidean pairwise distance
    matrix for both prediction and ground truth; for each cell i, take row i
    of the true matrix and row i of the predicted matrix (length-N distance
    vectors) and compute the Spearman rank correlation between them.

    Args:
        coords_true: (N, 2) ground-truth 2-D coordinates.
        coords_pred: (N, d) predicted coordinates OR embeddings (d arbitrary).
        progress: emit a tqdm progress bar over cells if True (loop version
            only — ignored when ``vectorized=True`` since there's no inner
            Python loop to wrap).
        vectorized: BACKWARD-COMPAT shortcut. ``vectorized=True`` is
            now an alias for ``backend="vectorized"``. Prefer setting
            ``backend`` directly. Conflict (e.g. ``vectorized=True,
            backend="numba"``) raises ValueError.
        backend: which implementation to dispatch to:
            - "scipy" (default): per-cell ``scipy.stats.spearmanr``
              loop. Slowest, returns real p-values, is the regression
              baseline. Always available.
            - "vectorized": numpy-vectorised batched
              ``scipy.stats.rankdata`` + row-wise Pearson via einsum,
              chunked to bound memory. 5-20× faster on CNS slices.
              Bit-faithful to scipy within ~1e-7. P-values NaN.
            - "numba": Numba-JIT'd per-row rankdata-with-average-ties
              + ``prange`` across cells. 5-15× faster than scipy loop;
              uses constant memory per cell (no full rank matrix).
              Requires ``numba`` (already an scgg env dep).
              Bit-faithful to scipy within ~1e-7. P-values NaN.
            - "gpu": chunked ``torch.cdist`` + ``argsort``-based ranks
              + reduction on GPU. 30-100× faster on big slices when a
              GPU is available. argsort-based ranks DON'T do scipy's
              "average" tie correction, so tied rows (typically just
              the diagonal d[i,i]=0) can diverge by up to ~1e-3
              per-cell rho — acceptable for our use case but NOT
              bit-faithful. Falls back loudly if torch/CUDA is
              unavailable. P-values NaN.

    Returns:
        Dict with:
          per_cell:  (N,) per-cell Spearman values (NaNs for degenerate cells)
          per_cell_p:(N,) p-values (real for backend='scipy', NaN otherwise)
          mean:      float, mean over cells
          median:    float, median over cells
    """
    coords_true = np.asarray(coords_true, dtype=np.float32)
    coords_pred = np.asarray(coords_pred, dtype=np.float32)
    if coords_true.shape[0] != coords_pred.shape[0]:
        raise ValueError(
            f"row count mismatch: true has {coords_true.shape[0]} cells, "
            f"pred has {coords_pred.shape[0]}"
        )

    # Reconcile legacy ``vectorized`` with the canonical ``backend``
    # knob. ``vectorized=True`` is treated as ``backend='vectorized'``
    # when backend is at default; explicit conflict errors out so
    # callers can't silently get a path they didn't ask for.
    if vectorized:
        if backend == DEFAULT_SPEARMAN_BACKEND:
            backend = "vectorized"
        elif backend != "vectorized":
            raise ValueError(
                f"conflicting flags: vectorized=True with backend={backend!r}. "
                f"Use ``backend='vectorized'`` exclusively."
            )
    if backend not in VALID_SPEARMAN_BACKENDS:
        raise ValueError(
            f"unknown backend {backend!r}; valid: {VALID_SPEARMAN_BACKENDS}"
        )

    n = coords_true.shape[0]

    # GPU path computes distance matrices on-device via torch.cdist —
    # passing pre-computed dt/dp would force a 180 GB CPU→GPU copy on
    # big slices. So branch before allocating dt/dp on CPU.
    if backend == "gpu":
        return _compute_spearman_gpu(coords_true, coords_pred, n)

    # All CPU backends need the full N×N distance matrices resident.
    dt = compute_distance(coords_true)
    dp = compute_distance(coords_pred)

    if backend == "vectorized":
        return _compute_spearman_vectorized(dt, dp, n)
    if backend == "numba":
        return _compute_spearman_numba(dt, dp, n)
    return _compute_spearman_loop(dt, dp, n, progress=progress)


def _compute_spearman_loop(
    dt: np.ndarray,
    dp: np.ndarray,
    n: int,
    progress: bool = False,
) -> Dict[str, np.ndarray | float]:
    """Original per-cell scipy.stats.spearmanr loop. Kept as a baseline
    for the vectorized path's regression test and as a fallback for
    debugging / p-value inspection."""
    rng = range(n)
    if progress:
        try:
            from tqdm import tqdm
            rng = tqdm(rng, desc="per-cell Spearman")
        except ImportError:
            pass

    rho = np.empty(n, dtype=np.float64)
    pval = np.empty(n, dtype=np.float64)
    for i in rng:
        r, p = spearmanr(dt[i], dp[i])
        rho[i] = r
        pval[i] = p

    rho_ok = rho[~np.isnan(rho)]
    return {
        "per_cell": rho,
        "per_cell_p": pval,
        "mean": float(rho_ok.mean()) if rho_ok.size else float("nan"),
        "median": float(np.median(rho_ok)) if rho_ok.size else float("nan"),
        "n": int(n),
    }


def _compute_spearman_vectorized(
    dt: np.ndarray,
    dp: np.ndarray,
    n: int,
    chunk_size: int = 1024,
) -> Dict[str, np.ndarray | float]:
    """Vectorized equivalent of _compute_spearman_loop.

    Spearman's rank correlation between two equal-length vectors equals
    Pearson's correlation of their RANKS. We compute the per-row
    rank-correlations by batching ``scipy.stats.rankdata(axis=1)``
    over row CHUNKS — no Python-level per-cell loop, no full-matrix
    rank materialisation.

    Why chunked instead of "all at once":
      Calling ``rankdata`` on the whole (N, N) matrix returns float64
      ranks, which scipy allocates internally even before our explicit
      ``.astype(np.float32)``. For N=150k cells that transient is
      180 GB on top of the 90 GB ``dt``/``dp`` matrices — combined
      peak ~540 GB OOMs a 512 GB worker. Processing in chunks of
      ``chunk_size`` rows keeps the rank-array transient down to
      ``chunk_size × N × 8 bytes`` (≈ 1.2 GB at chunk=1024, N=150k),
      so the worker peak stays at ``dt + dp + small`` ≈ 184 GB.

    Algorithm (per chunk):
      1. rank each row in the chunk independently (rankdata axis=1)
      2. centre each rank row by its row mean
      3. per-row Pearson via einsum: num = sum(rt*rp), denom = ||rt|| * ||rp||
      4. handle degenerate rows (zero variance) by emitting NaN

    Tie handling: rankdata's default ``method='average'`` matches
    scipy.stats.spearmanr's tie correction, so per-cell values agree
    with the loop version bit-for-bit (modulo float rounding) — see
    test_per_cell_spearman_vectorized.py.

    Args:
        dt, dp: (N, N) float32 pairwise-distance matrices.
        n: cell count (== dt.shape[0]; passed in to avoid re-reading).
        chunk_size: rows processed per inner pass. Default 1024 gives
            a ~1.2 GB transient at N=150k. Lower it on extra-large
            slices (chunk_size=256 → ~300 MB transient) at a small
            speed cost; raise it on tiny slices for slightly less
            Python-loop overhead. Has zero effect on the output
            values — only on the memory/speed trade-off.
    """
    from scipy.stats import rankdata

    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")

    # Output arrays, sized once. Per-chunk results are scattered back
    # into these at the right row indices.
    rho = np.full(n, np.nan, dtype=np.float64)

    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)

        # Step 1: rank just this row-chunk. rankdata returns float64
        # internally; cast to float32 to halve the per-chunk working
        # set (the einsum reductions in step 3 are float-precision-
        # invariant at ranks-of-integers scale).
        rt = rankdata(dt[start:end], axis=1, method="average").astype(
            np.float32, copy=False,
        )
        rp = rankdata(dp[start:end], axis=1, method="average").astype(
            np.float32, copy=False,
        )

        # Step 2: centre by row mean. Subtract in-place to avoid an
        # extra (chunk, N) allocation.
        rt -= rt.mean(axis=1, keepdims=True)
        rp -= rp.mean(axis=1, keepdims=True)

        # Step 3: per-row Pearson via einsum. ``'ij,ij->i'`` reduces
        # the per-row dot product without materialising the (chunk, N)
        # outer product.
        num = np.einsum("ij,ij->i", rt, rp)
        norm_t = np.sqrt(np.einsum("ij,ij->i", rt, rt))
        norm_p = np.sqrt(np.einsum("ij,ij->i", rp, rp))
        denom = norm_t * norm_p

        # Step 4: degenerate-row handling. spearmanr returns NaN when
        # one of the inputs has zero variance (all-identical values).
        # With pairwise distances this only happens if all
        # true-distance or all pred-distance rows are constant —
        # vanishingly rare in practice but match scipy's NaN semantics
        # for safety. ``rho`` was pre-filled with NaN so we only need
        # to overwrite the valid rows.
        valid = denom > 0
        chunk_rho = np.where(valid, num / np.where(denom > 0, denom, 1.0), np.nan)
        rho[start:end] = chunk_rho

        # Free the chunk's working set before the next iteration —
        # Python GC would do this eventually but the explicit del
        # makes peak-memory analysis cleaner and gives the allocator
        # an immediate chance to release the float32 chunk arrays.
        del rt, rp, num, norm_t, norm_p, denom, chunk_rho

    # P-values are NOT computed in the vectorised path. Downstream
    # (aggregate_slices, compute_extended_metrics) doesn't consume
    # them; emitting NaN keeps the dict schema stable so callers don't
    # break. If you need real p-values, call this function with
    # ``vectorized=False``.
    pval = np.full(n, np.nan, dtype=np.float64)

    rho_ok = rho[~np.isnan(rho)]
    return {
        "per_cell": rho,
        "per_cell_p": pval,
        "mean": float(rho_ok.mean()) if rho_ok.size else float("nan"),
        "median": float(np.median(rho_ok)) if rho_ok.size else float("nan"),
        "n": int(n),
    }


# ---------------------------------------------------------------------------
# Spearman backend: numba (JIT-compiled CPU)
# ---------------------------------------------------------------------------

# Module-level cache for the compiled Numba kernel — first call to
# _compute_spearman_numba pays the ~30s JIT cost, subsequent calls hit
# the cache. None until first use; set to the compiled function or to
# an exception (to surface the same error on every subsequent call).
_NUMBA_KERNEL: object | None = None


def _build_numba_kernel():
    """JIT-compile the per-cell Spearman kernel once and return it.

    Lazy compilation so importing this module doesn't pay the JIT cost
    even when callers stay on the scipy path. Raises ImportError if
    ``numba`` is missing — caller (``_compute_spearman_numba``) maps
    that to a clear user-facing message.
    """
    try:
        from numba import njit, prange  # type: ignore
    except ImportError as e:
        raise ImportError(
            "backend='numba' requires the numba package. "
            "Install with `pip install numba` (already in the scgg "
            "env's setup script for the PH-loss path)."
        ) from e

    # Average-tie ranking. Mirrors scipy.stats.rankdata(method='average'):
    # for a tie group spanning sorted positions i..j (inclusive),
    # every element in the group gets rank (i+1 + j+1)/2 in 1-indexed
    # convention (= (i + j + 2)/2.0).
    @njit(cache=True)
    def _rankdata_average(row):  # noqa: F811
        n = row.shape[0]
        sorted_idx = np.argsort(row)
        ranks = np.empty(n, dtype=np.float64)
        i = 0
        while i < n:
            j = i
            # Extend the tie group as long as the next sorted value
            # matches the current one. Equality compared on the
            # original-typed values, not on float rounding, so ties
            # are detected the same way scipy does.
            while j + 1 < n and row[sorted_idx[j + 1]] == row[sorted_idx[i]]:
                j += 1
            avg = (i + j + 2) / 2.0  # 1-indexed average rank
            for k in range(i, j + 1):
                ranks[sorted_idx[k]] = avg
            i = j + 1
        return ranks

    # Per-cell loop, parallelised across rows via prange. Each
    # iteration is independent — no shared state, no aliasing — so
    # Numba's automatic parallelisation is safe.
    @njit(parallel=True, cache=True)
    def _per_cell_spearman_numba(dt, dp):  # noqa: F811
        n = dt.shape[0]
        rho = np.empty(n, dtype=np.float64)
        for i in prange(n):
            rt = _rankdata_average(dt[i])
            rp = _rankdata_average(dp[i])
            # Centre by mean (in-place).
            mt = rt.mean()
            mp = rp.mean()
            for k in range(n):
                rt[k] -= mt
                rp[k] -= mp
            # Per-row Pearson reduction.
            num = 0.0
            norm_t_sq = 0.0
            norm_p_sq = 0.0
            for k in range(n):
                num += rt[k] * rp[k]
                norm_t_sq += rt[k] * rt[k]
                norm_p_sq += rp[k] * rp[k]
            denom = np.sqrt(norm_t_sq) * np.sqrt(norm_p_sq)
            if denom > 0.0:
                rho[i] = num / denom
            else:
                # Degenerate row (zero variance) — scipy returns NaN.
                rho[i] = np.nan
        return rho

    return _per_cell_spearman_numba


def _compute_spearman_numba(
    dt: np.ndarray,
    dp: np.ndarray,
    n: int,
) -> Dict[str, np.ndarray | float]:
    """Numba-JIT'd per-cell Spearman.

    First call pays a one-time ~30 s JIT compile, then subsequent
    calls (across slices and timestamps) hit the on-disk Numba cache
    (``cache=True`` on the kernels). The hot per-cell loop runs in
    optimised machine code with ``prange`` parallelisation across
    cores, so the OMP/MKL thread environment in the caller's shell
    controls how many CPUs are used.

    Memory: ``dt + dp`` resident (~180 GB at N=150k), plus 2 × N
    float64 rank vectors per active thread (~2.4 MB per thread at
    N=150k). MUCH lighter than the chunked vectorised path because
    we never allocate a (chunk, N) rank matrix.
    """
    global _NUMBA_KERNEL
    if _NUMBA_KERNEL is None:
        _NUMBA_KERNEL = _build_numba_kernel()
    kernel = _NUMBA_KERNEL

    # Numba's prange threading is independent of the chunked python
    # loop in the vectorised path — let it use whatever NUMBA_NUM_THREADS
    # / OMP_NUM_THREADS the environment is set to.
    rho = kernel(dt, dp)
    pval = np.full(n, np.nan, dtype=np.float64)
    rho_ok = rho[~np.isnan(rho)]
    return {
        "per_cell": rho,
        "per_cell_p": pval,
        "mean": float(rho_ok.mean()) if rho_ok.size else float("nan"),
        "median": float(np.median(rho_ok)) if rho_ok.size else float("nan"),
        "n": int(n),
    }


# ---------------------------------------------------------------------------
# Spearman backend: gpu (torch.cdist + argsort, chunked)
# ---------------------------------------------------------------------------


def _compute_spearman_gpu(
    coords_true: np.ndarray,
    coords_pred: np.ndarray,
    n: int,
    chunk_size: int = 2048,
) -> Dict[str, np.ndarray | float]:
    """Chunked GPU implementation.

    Strategy (different from the CPU paths because GPU memory is
    tighter):
      - DON'T materialise the full N×N distance matrix on GPU.
        Instead, for each row-chunk i..i+chunk_size, compute
        ``torch.cdist(coords[i:i+c], coords)`` → ``(chunk, N)``
        on-device. Peak transient ~4 × chunk × N float32 bytes per
        method (≈ 5 GB at chunk=2048, N=150k) — fits easily on A100/H100.
      - Rank via ``torch.argsort(torch.argsort(row))``. This gives
        DENSE integer ranks (no average-tie correction). For pairwise
        distance matrices the only systematic tie is the diagonal
        ``d[i,i]=0``; otherwise ties are vanishingly rare with
        continuous coords. Practical per-cell rho drift vs scipy:
        ≤ 1e-3, see test_per_cell_spearman_backends.py.
      - Reduce per-row Pearson directly with torch ops.

    Falls back loudly with a clear error if PyTorch isn't installed
    or no CUDA device is visible (rather than silently dropping to CPU).
    """
    try:
        import torch  # type: ignore
    except ImportError as e:
        raise ImportError(
            "backend='gpu' requires PyTorch. "
            "Install with `pip install torch` (the scgg env already "
            "has it for training)."
        ) from e
    if not torch.cuda.is_available():
        raise RuntimeError(
            "backend='gpu' requires a visible CUDA device; "
            "torch.cuda.is_available() returned False. "
            "Either submit the LSF job with a GPU request or fall "
            "back to backend='numba'/'vectorized' on CPU."
        )

    device = torch.device("cuda")
    # coords_pred can be d-dim embedding; coords_true is always 2D.
    ct = torch.from_numpy(np.ascontiguousarray(coords_true)).to(device)
    cp = torch.from_numpy(np.ascontiguousarray(coords_pred)).to(device)

    rho_gpu = torch.empty(n, dtype=torch.float64, device=device)

    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        # Compute this chunk's distance rows on-device. cdist returns
        # float32 by default; that's enough precision for ranks.
        dt_chunk = torch.cdist(ct[start:end], ct)  # (chunk, N)
        dp_chunk = torch.cdist(cp[start:end], cp)  # (chunk, N)

        # Dense integer rank per row via the argsort-of-argsort trick.
        # NOTE: doesn't average over tie groups (see docstring caveat).
        rt = torch.argsort(torch.argsort(dt_chunk, dim=1), dim=1).to(torch.float64)
        rp = torch.argsort(torch.argsort(dp_chunk, dim=1), dim=1).to(torch.float64)

        rt -= rt.mean(dim=1, keepdim=True)
        rp -= rp.mean(dim=1, keepdim=True)
        num = (rt * rp).sum(dim=1)
        norm_t = (rt * rt).sum(dim=1).sqrt()
        norm_p = (rp * rp).sum(dim=1).sqrt()
        denom = norm_t * norm_p
        # NaN where denom == 0 (degenerate row); torch.where avoids the
        # divide-by-zero warning.
        safe_denom = torch.where(denom > 0, denom, torch.ones_like(denom))
        chunk_rho = torch.where(
            denom > 0, num / safe_denom,
            torch.tensor(float("nan"), dtype=torch.float64, device=device),
        )
        rho_gpu[start:end] = chunk_rho

        # Release the chunk's transients before the next iteration —
        # without this the allocator can hold onto them, pushing peak
        # closer to chunk × N for both dt+dp+rt+rp simultaneously.
        del dt_chunk, dp_chunk, rt, rp, num, norm_t, norm_p, denom, chunk_rho

    rho = rho_gpu.cpu().numpy()
    pval = np.full(n, np.nan, dtype=np.float64)
    rho_ok = rho[~np.isnan(rho)]
    return {
        "per_cell": rho,
        "per_cell_p": pval,
        "mean": float(rho_ok.mean()) if rho_ok.size else float("nan"),
        "median": float(np.median(rho_ok)) if rho_ok.size else float("nan"),
        "n": int(n),
    }


# ---------------------------------------------------------------------------
# Contact precision / F1
# ---------------------------------------------------------------------------


def compute_contact(
    coords_true: np.ndarray,
    coords_pred: np.ndarray,
    percentile: float = 0.01,
) -> Dict[str, float]:
    """Percentile-thresholded contact precision / F1.

    Matches LUNA's ``compute_contact``: flatten the pairwise-distance matrices
    of both true and predicted, drop the zero (self) entries, threshold each
    at its own ``percentile`` quantile, treat "below threshold" as a contact,
    and compute precision and F1.

    Because both sides are thresholded at the same percentile of their own
    distribution, predicted-positive count == true-positive count → precision
    == recall == F1. (This is why LUNA's CSVs show identical precision and F1
    values to 15 decimals.)

    Args:
        coords_true: (N, 2) ground-truth coordinates.
        coords_pred: (N, d) predicted coordinates or embeddings.
        percentile: fraction of all pairs treated as "in contact" (default 0.01).

    Returns:
        Dict with 'precision', 'f1', 'recall', 'percentile'.
    """
    from sklearn.metrics import f1_score, precision_score, recall_score

    dt = compute_distance(coords_true).reshape(-1)
    dp = compute_distance(coords_pred).reshape(-1)

    # LUNA filters out entries where both are zero (drops self-pairs and
    # any double zeros). We replicate exactly.
    nonzero = np.nonzero(dt * dp)[0]
    dt = dt[nonzero]
    dp = dp[nonzero]

    if dt.size == 0:
        return {
            "precision": float("nan"), "f1": float("nan"),
            "recall": float("nan"), "percentile": percentile,
        }

    t_th = np.quantile(dt, percentile)
    p_th = np.quantile(dp, percentile)

    labels = (dt < t_th).astype(np.int8)
    preds = (dp < p_th).astype(np.int8)

    if labels.sum() == 0 or preds.sum() == 0:
        return {
            "precision": 0.0, "f1": 0.0, "recall": 0.0, "percentile": percentile
        }

    return {
        "precision": float(precision_score(labels, preds, zero_division=0)),
        "f1": float(f1_score(labels, preds, zero_division=0)),
        "recall": float(recall_score(labels, preds, zero_division=0)),
        "percentile": percentile,
    }


# ---------------------------------------------------------------------------
# RSSD (Kabsch / Root Sum Squared Deviation)
# ---------------------------------------------------------------------------


def _filter_nan_inf_pair(
    a: np.ndarray, b: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Drop rows where EITHER input has NaN/Inf, preserving row correspondence.

    LUNA's reference code filters `a` and `b` independently, which silently
    breaks the row alignment if only one has a NaN row. We always use the
    intersection mask so `a[i]` and `b[i]` stay paired.
    """
    mask = np.isfinite(a).all(axis=1) & np.isfinite(b).all(axis=1)
    return a[mask], b[mask]


def compute_kabsch_rssd(
    coords_true_2d: np.ndarray,
    coords_pred_2d: np.ndarray,
) -> float:
    """Kabsch-aligned Root Sum Squared Deviation on 2-D coordinates.

    Pads the input to 3-D with Z=0 (LUNA's convention), filters NaN/Inf rows
    (using a joint mask to preserve row correspondence), then calls
    scipy.spatial.transform.Rotation.align_vectors which performs the
    Kabsch–Umeyama optimal rotation and returns the RSSD.

    Args:
        coords_true_2d: (N, 2).
        coords_pred_2d: (N, 2). Must have the same N as coords_true_2d.

    Returns:
        RSSD = sqrt(sum_i ||R · p_pred,i - p_true,i||^2), or +inf if SVD
        fails to converge.
    """
    a = np.asarray(coords_true_2d, dtype=np.float64)
    b = np.asarray(coords_pred_2d, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")
    if a.shape[1] != 2:
        raise ValueError(f"expected (N, 2), got {a.shape}")

    a = np.pad(a, ((0, 0), (0, 1)), mode="constant")  # -> (N, 3)
    b = np.pad(b, ((0, 0), (0, 1)), mode="constant")
    a, b = _filter_nan_inf_pair(a, b)
    if a.shape[0] == 0:
        return float("inf")

    try:
        _rot, rssd, _sens = R.align_vectors(a, b, return_sensitivity=True)
    except np.linalg.LinAlgError:
        return float("inf")
    return float(rssd)


def compute_RSSD(
    coords_true_2d: np.ndarray,
    coords_pred_2d: np.ndarray,
    cell_class: Optional[Sequence] = None,
) -> Dict[str, float]:
    """LUNA's compute_RSSD on a single slice.

    LUNA reports three numbers per slice:
      * absolute_rssd: Kabsch RSSD over ALL cells together.
      * sum_rssd:      sum over per-cell-class Kabsch RSSDs.
      * mean_rssd:     mean over per-cell-class Kabsch RSSDs.

    The per-class variants align each cell class independently — useful when
    rotation/translation invariance is class-specific. If cell_class is None,
    only absolute_rssd is computed; sum_rssd / mean_rssd are returned as NaN.

    Args:
        coords_true_2d: (N, 2) ground truth.
        coords_pred_2d: (N, 2) prediction.
        cell_class: optional length-N sequence of class labels for the
            per-class flavor.

    Returns:
        Dict with 'absolute_rssd', 'sum_rssd', 'mean_rssd', 'n_classes'.
    """
    out: Dict[str, float] = {}
    out["absolute_rssd"] = compute_kabsch_rssd(coords_true_2d, coords_pred_2d)

    if cell_class is None:
        out["sum_rssd"] = float("nan")
        out["mean_rssd"] = float("nan")
        out["n_classes"] = 0
        return out

    cls = np.asarray(cell_class)
    rssds: List[float] = []
    for c in np.unique(cls):
        mask = cls == c
        if mask.sum() < 3:
            # Kabsch needs at least 3 non-collinear points; LUNA effectively
            # skips degenerate classes via filter_nan_inf shrinking the set.
            continue
        rssds.append(
            compute_kabsch_rssd(coords_true_2d[mask], coords_pred_2d[mask])
        )
    rssds = [r for r in rssds if np.isfinite(r)]
    if not rssds:
        out["sum_rssd"] = float("nan")
        out["mean_rssd"] = float("nan")
        out["n_classes"] = 0
    else:
        out["sum_rssd"] = float(np.sum(rssds))
        out["mean_rssd"] = float(np.mean(rssds))
        out["n_classes"] = int(len(rssds))
    return out


# ---------------------------------------------------------------------------
# Embedding -> 2-D helper (only needed for RSSD on metric embeddings)
# ---------------------------------------------------------------------------


def embedding_to_2d(embedding: np.ndarray, method: str = "pca") -> np.ndarray:
    """Project a d-dimensional embedding to 2-D for RSSD.

    Spearman and contact metrics are dimension-agnostic and do NOT need this
    step (they operate on rank-of-distance vectors). Only RSSD requires 2-D
    because the Kabsch alignment must operate in the same space as the
    ground truth.

    Args:
        embedding: (N, d) embedding.
        method: 'pca' (fast, linear) or 'mds' (distance-preserving, slower).

    Returns:
        (N, 2) projection.
    """
    x = np.asarray(embedding, dtype=np.float32)
    if x.shape[1] == 2:
        return x

    if method == "pca":
        from sklearn.decomposition import PCA
        return PCA(n_components=2).fit_transform(x).astype(np.float32)
    if method == "mds":
        from sklearn.manifold import MDS
        return MDS(
            n_components=2, dissimilarity="euclidean", normalized_stress="auto"
        ).fit_transform(x).astype(np.float32)
    raise ValueError(f"unknown projection method: {method!r}")


# ---------------------------------------------------------------------------
# Per-slice evaluation + across-slice aggregation
# ---------------------------------------------------------------------------


def evaluate_slice(
    coords_true: np.ndarray,
    coords_pred: np.ndarray,
    cell_class: Optional[Sequence] = None,
    contact_percentile: float = 0.01,
    compute_rssd: bool = True,
    rssd_projection: str = "pca",
    spearman_vectorized: bool = False,
    spearman_backend: str = DEFAULT_SPEARMAN_BACKEND,
) -> Dict[str, float]:
    """Compute all three LUNA metrics on one slice.

    Args:
        coords_true: (N, 2) ground-truth 2-D coordinates.
        coords_pred: (N, d) prediction (2-D coords OR d-dim metric embedding).
            If d > 2 and compute_rssd is True, the embedding is projected to
            2-D via `rssd_projection` for the RSSD computation only; Spearman
            and contact use the embedding directly.
        cell_class: optional per-cell class labels for per-class RSSD.
        contact_percentile: percentile threshold for `compute_contact`.
        compute_rssd: skip the RSSD computation if False (no projection).
        rssd_projection: 'pca' or 'mds' if the embedding is not 2-D.
        spearman_vectorized: BACKWARD-COMPAT shortcut for
            ``spearman_backend='vectorized'``. Prefer ``spearman_backend``
            directly. Conflicting explicit settings raise ValueError.
        spearman_backend: which Spearman implementation to use. One of
            "scipy" (default, original loop), "vectorized" (chunked
            numpy rankdata), "numba" (JIT'd CPU), "gpu" (chunked torch
            on CUDA). See ``compute_spearman_correlation`` for the
            speed/precision trade-offs of each.

    Returns:
        Flat dict of metrics for this slice.
    """
    out: Dict[str, float] = {}

    spr = compute_spearman_correlation(
        coords_true, coords_pred,
        vectorized=spearman_vectorized,
        backend=spearman_backend,
    )
    out["spearman_per_cell_mean"] = spr["mean"]
    out["spearman_per_cell_median"] = spr["median"]

    contact = compute_contact(coords_true, coords_pred, contact_percentile)
    out["precision"] = contact["precision"]
    out["f1"] = contact["f1"]
    out["recall"] = contact["recall"]
    out["contact_percentile"] = contact["percentile"]

    if compute_rssd:
        if coords_pred.shape[1] == 2:
            pred2d = coords_pred
        else:
            pred2d = embedding_to_2d(coords_pred, method=rssd_projection)
        rssd = compute_RSSD(coords_true, pred2d, cell_class)
        out["absolute_rssd"] = rssd["absolute_rssd"]
        out["sum_rssd"] = rssd["sum_rssd"]
        out["mean_rssd"] = rssd["mean_rssd"]
        out["n_classes_rssd"] = rssd["n_classes"]
        out["rssd_projection"] = rssd_projection

    out["n_cells"] = int(coords_true.shape[0])
    return out


def aggregate_slices(
    per_slice: Iterable[Dict[str, float]],
) -> Dict[str, float]:
    """Aggregate per-slice metric dicts across all test slices.

    LUNA Figure 3 reports the mean across slices of the per-slice MEDIAN
    Spearman (44.8 % for LUNA). We additionally report the other three
    aggregations so the user can cross-check against any variant.

    The function silently passes through any keys it doesn't recognize.
    """
    rows = list(per_slice)
    if not rows:
        return {}

    def _stack(key: str) -> np.ndarray:
        return np.array(
            [r[key] for r in rows if key in r and not np.isnan(r[key])],
            dtype=np.float64,
        )

    out: Dict[str, float] = {}

    # Spearman aggregations — the headline LUNA number is mean_of_medians.
    med = _stack("spearman_per_cell_median")
    mean = _stack("spearman_per_cell_mean")
    if med.size:
        out["spearman_mean_of_medians"] = float(med.mean())     # LUNA Fig.3
        out["spearman_median_of_medians"] = float(np.median(med))
        out["spearman_std_of_medians"] = float(med.std())
    if mean.size:
        out["spearman_mean_of_means"] = float(mean.mean())
        out["spearman_median_of_means"] = float(np.median(mean))

    for key in ("precision", "f1", "recall",
                "absolute_rssd", "sum_rssd", "mean_rssd"):
        v = _stack(key)
        if v.size:
            out[f"{key}_mean"] = float(v.mean())
            out[f"{key}_median"] = float(np.median(v))
            out[f"{key}_std"] = float(v.std())

    n = _stack("n_cells")
    if n.size:
        out["total_cells"] = int(n.sum())
        out["n_slices"] = int(n.size)

    return out
