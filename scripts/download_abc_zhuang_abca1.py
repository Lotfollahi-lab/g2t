#!/usr/bin/env python
"""
Download the ABC Brain Cell Atlas — Zhuang-ABCA-1 (Animal 1) — to the bronze
data directory. This is the dataset LUNA uses to train the model for the
scRNA-seq de novo reconstruction task (Section 3.3 of the paper).

Default destination: /nfs/team361/sb75/DATASETS/bronze/abc_luna

What gets downloaded (from `s3://allen-brain-cell-atlas/`, public bucket,
no AWS credentials needed):

  1. Cell metadata (~tens of MB):
       Zhuang-ABCA-1 cell_metadata.csv         (cells, sections, basic info)
       Zhuang-ABCA-1-CCF cell_metadata_with_parcellation_annotation.csv
                                               (CCF-aligned coordinates +
                                                parcellation labels)
  2. Cluster annotation tables (~MB):
       WMB-taxonomy: cluster_to_cluster_annotation_membership.csv etc.
       (joins cluster IDs -> class / subclass / supertype labels)
  3. Expression matrices (the big part, ~10 GB):
       Zhuang-ABCA-1-raw.h5ad   (cell × gene, raw counts, sparse)

Approximate total size on disk: ~10–12 GB. Allow 1–3 h for the first
download depending on bandwidth.

Usage:
    pip install abc_atlas_access  # or: uv add abc_atlas_access
    python scripts/download_abc_zhuang_abca1.py \\
        --bronze_dir /nfs/team361/sb75/DATASETS/bronze/abc_luna

If `abc_atlas_access` is unavailable, falls back to plain S3 over HTTPS via
`requests`. The S3 paths in the official package are version-gated; if the
fallback runs, you'll need to set --release_version manually (see Allen
docs: https://alleninstitute.github.io/abc_atlas_access/).
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

logger = logging.getLogger("scgg.download_abc")


# Allen ABC public S3 bucket: no credentials required.
S3_BUCKET_HTTPS = "https://allen-brain-cell-atlas.s3.us-west-2.amazonaws.com"


# ---------------------------------------------------------------------------
# Path resolution (abc_atlas_access)
# ---------------------------------------------------------------------------


def _download_via_abc_atlas_access(bronze_dir: Path) -> None:
    """Use the official library — handles release versioning + cache logic.

    The library exposes:
      cache.list_metadata_files / cache.list_data_files
      cache.get_metadata_path(directory, file_name)
      cache.get_data_path(directory, file_name)
    The "directory" arg is the dataset id (e.g. 'Zhuang-ABCA-1') and
    the "file_name" arg is the asset name (e.g. 'cell_metadata').
    """
    from abc_atlas_access.abc_atlas_cache.abc_project_cache import AbcProjectCache

    cache = AbcProjectCache.from_s3_cache(bronze_dir)
    logger.info(f"abc_atlas_access cache rooted at {bronze_dir}")

    # 1. Cell metadata (the canonical join key).
    logger.info("[1/4] Cell metadata (Zhuang-ABCA-1)")
    for f in ("cell_metadata",):
        try:
            p = cache.get_metadata_path("Zhuang-ABCA-1", f)
            logger.info(f"    -> {f}: {p}")
        except Exception as e:  # noqa: BLE001
            logger.warning(f"    {f} unavailable: {e}")

    # 2. CCF-aligned coordinates + parcellation labels.
    logger.info("[2/4] CCF-aligned coordinates (Zhuang-ABCA-1-CCF)")
    for f in (
        "cell_metadata_with_parcellation_annotation",
        "cell_metadata_with_reconstructed_coordinates",
    ):
        try:
            p = cache.get_metadata_path("Zhuang-ABCA-1-CCF", f)
            logger.info(f"    -> {f}: {p}")
        except Exception as e:  # noqa: BLE001
            logger.warning(f"    {f} unavailable: {e}")

    # 3. Taxonomy: maps cluster ids to class / subclass / supertype names.
    logger.info("[3/4] WMB taxonomy (cluster -> class / subclass / supertype)")
    for f in (
        "cluster_to_cluster_annotation_membership",
        "cluster_annotation_term",
        "cluster_annotation_term_set",
        "cluster",
    ):
        for d in ("WMB-taxonomy",):
            try:
                p = cache.get_metadata_path(d, f)
                logger.info(f"    -> {d}/{f}: {p}")
                break
            except Exception:
                continue
        else:
            logger.warning(f"    {f} unavailable across WMB-taxonomy")

    # 4. Expression matrix (the big one — sparse raw counts).
    logger.info("[4/4] Expression matrix (Zhuang-ABCA-1, raw counts; ~10 GB)")
    found_expr = False
    for asset in ("Zhuang-ABCA-1-raw", "Zhuang-ABCA-1-log2"):
        try:
            p = cache.get_data_path("Zhuang-ABCA-1", asset)
            logger.info(f"    -> {asset}: {p}")
            found_expr = True
        except Exception as e:  # noqa: BLE001
            logger.debug(f"    {asset} unavailable: {e}")
    if not found_expr:
        logger.error(
            "    No expression matrix could be downloaded. List available "
            "data files with `cache.list_data_files` to discover the exact "
            "asset name on this release."
        )
        raise RuntimeError("expression matrix unavailable")

    logger.info("Download complete.")


# ---------------------------------------------------------------------------
# Path resolution (boto3 / requests fallback)
# ---------------------------------------------------------------------------


def _download_via_https_fallback(
    bronze_dir: Path,
    release_version: str,
) -> None:
    """Plain HTTPS fallback when abc_atlas_access isn't installed.

    The Allen ABC bucket exposes a directory listing via S3 ListObjectsV2.
    We download an enumerated list of expected files; if any are missing
    the user should bump --release_version or inspect the bucket directly:
        aws s3 ls s3://allen-brain-cell-atlas/ --no-sign-request
    """
    import requests

    if release_version == "latest":
        # The release directory layout is stable enough that bumping to
        # the most recent version we know about works as a default. The
        # user can override --release_version if Allen ships a newer one.
        release_version = "20231215"

    files = [
        # (subpath relative to bucket root, local destination)
        (f"metadata/Zhuang-ABCA-1/{release_version}/views/cell_metadata.csv",
         f"metadata/Zhuang-ABCA-1/{release_version}/views/cell_metadata.csv"),
        (f"metadata/Zhuang-ABCA-1-CCF/{release_version}/views/"
         f"cell_metadata_with_parcellation_annotation.csv",
         f"metadata/Zhuang-ABCA-1-CCF/{release_version}/views/"
         f"cell_metadata_with_parcellation_annotation.csv"),
        (f"metadata/WMB-taxonomy/{release_version}/views/cluster.csv",
         f"metadata/WMB-taxonomy/{release_version}/views/cluster.csv"),
        (f"metadata/WMB-taxonomy/{release_version}/views/"
         f"cluster_annotation_term.csv",
         f"metadata/WMB-taxonomy/{release_version}/views/"
         f"cluster_annotation_term.csv"),
        (f"metadata/WMB-taxonomy/{release_version}/views/"
         f"cluster_to_cluster_annotation_membership.csv",
         f"metadata/WMB-taxonomy/{release_version}/views/"
         f"cluster_to_cluster_annotation_membership.csv"),
        (f"expression_matrices/Zhuang-ABCA-1/{release_version}/"
         f"Zhuang-ABCA-1-raw.h5ad",
         f"expression_matrices/Zhuang-ABCA-1/{release_version}/"
         f"Zhuang-ABCA-1-raw.h5ad"),
    ]
    for remote, local in files:
        url = f"{S3_BUCKET_HTTPS}/{remote}"
        out = bronze_dir / local
        out.parent.mkdir(parents=True, exist_ok=True)
        if out.exists() and out.stat().st_size > 0:
            logger.info(f"  exists, skipping: {out}")
            continue
        logger.info(f"  GET {url} -> {out}")
        _stream_download(url, out)


def _stream_download(url: str, dest: Path, chunk: int = 8 << 20) -> None:
    """Stream a remote file in chunks with simple progress feedback."""
    import requests

    with requests.get(url, stream=True) as r:
        if r.status_code != 200:
            logger.error(f"    HTTP {r.status_code} for {url}")
            r.raise_for_status()
        total = int(r.headers.get("Content-Length", "0"))
        seen = 0
        last_pct = -1
        tmp = dest.with_suffix(dest.suffix + ".partial")
        with open(tmp, "wb") as f:
            for c in r.iter_content(chunk_size=chunk):
                if not c:
                    continue
                f.write(c)
                seen += len(c)
                if total:
                    pct = int(100 * seen / total)
                    if pct != last_pct and pct % 5 == 0:
                        logger.info(f"      {pct}%  ({seen/1e9:.2f} / {total/1e9:.2f} GB)")
                        last_pct = pct
        tmp.rename(dest)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--bronze_dir",
        default="/nfs/team361/sb75/DATASETS/bronze/abc_luna",
        help="Directory to cache the raw download.",
    )
    p.add_argument(
        "--release_version", default="latest",
        help="ABC release version (HTTPS fallback only). "
             "Default: latest known to this script.",
    )
    p.add_argument(
        "--force_https_fallback", action="store_true",
        help="Skip abc_atlas_access and use the HTTPS fallback directly.",
    )
    args = p.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    bronze = Path(args.bronze_dir)
    bronze.mkdir(parents=True, exist_ok=True)
    logger.info(f"Bronze directory: {bronze.resolve()}")

    if args.force_https_fallback:
        _download_via_https_fallback(bronze, args.release_version)
        return 0

    try:
        _download_via_abc_atlas_access(bronze)
        return 0
    except ImportError:
        logger.warning(
            "abc_atlas_access not installed (`pip install abc_atlas_access` "
            "or `uv add abc_atlas_access`). Falling back to plain HTTPS."
        )
    except Exception as e:  # noqa: BLE001
        logger.exception(f"abc_atlas_access path failed: {e}. Trying HTTPS fallback.")

    _download_via_https_fallback(bronze, args.release_version)
    return 0


if __name__ == "__main__":
    sys.exit(main())
