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


def _list_s3_subdirs(prefix: str) -> list[str]:
    """List the common-prefix subdirectories under an S3 prefix.

    The Allen bucket is anonymous-readable. We use the unsigned REST API:
    GET https://{bucket}/?list-type=2&prefix={prefix}&delimiter=/

    Returns a list of subdirectory names (without the trailing slash).
    """
    import xml.etree.ElementTree as ET
    import requests

    url = f"{S3_BUCKET_HTTPS}/?list-type=2&prefix={prefix}&delimiter=/"
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    ns = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}
    root = ET.fromstring(r.content)
    out: list[str] = []
    for cp in root.findall("s3:CommonPrefixes", ns):
        p = cp.find("s3:Prefix", ns)
        if p is None or not p.text:
            continue
        name = p.text[len(prefix):].rstrip("/")
        if name:
            out.append(name)
    return out


def _discover_release_version() -> str:
    """Find the latest Zhuang-ABCA-1 release present in the bucket.

    We list `metadata/Zhuang-ABCA-1/` and pick the lexicographically latest
    date-stamped subdirectory. The Allen team uses YYYYMMDD names.
    """
    versions = _list_s3_subdirs("metadata/Zhuang-ABCA-1/")
    if not versions:
        raise RuntimeError(
            "Could not list metadata/Zhuang-ABCA-1/ on the ABC bucket. "
            "Check network access to https://allen-brain-cell-atlas.s3.us-west-2.amazonaws.com"
        )
    # Sort lexicographically — works for YYYYMMDD-style names.
    versions.sort()
    return versions[-1]


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

    Dynamically discovers the latest release version per dataset rather
    than hard-coding a date (Allen rolls these forward periodically). Each
    dataset (`Zhuang-ABCA-1`, `Zhuang-ABCA-1-CCF`, `WMB-taxonomy`) has its
    own release directory, and the latest version may differ between them.

    If `--release_version` is passed (not "latest"), uses that for all
    datasets.
    """
    pinned = release_version if release_version != "latest" else None

    if pinned is None:
        zh_ver = _discover_release_version()
        logger.info(f"  discovered Zhuang-ABCA-1 release version: {zh_ver}")
    else:
        zh_ver = pinned
        logger.info(f"  using pinned release version: {zh_ver}")

    # The CCF and taxonomy datasets have their own release dirs; discover
    # them independently with a one-shot listing so we don't 404 if Allen
    # versions them out of sync.
    if pinned is None:
        try:
            ccf_versions = _list_s3_subdirs("metadata/Zhuang-ABCA-1-CCF/")
            ccf_ver = sorted(ccf_versions)[-1] if ccf_versions else zh_ver
        except Exception as e:  # noqa: BLE001
            logger.warning(f"  could not list Zhuang-ABCA-1-CCF/: {e}")
            ccf_ver = zh_ver
        try:
            tax_versions = _list_s3_subdirs("metadata/WMB-taxonomy/")
            tax_ver = sorted(tax_versions)[-1] if tax_versions else zh_ver
        except Exception as e:  # noqa: BLE001
            logger.warning(f"  could not list WMB-taxonomy/: {e}")
            tax_ver = zh_ver
        try:
            expr_versions = _list_s3_subdirs("expression_matrices/Zhuang-ABCA-1/")
            expr_ver = sorted(expr_versions)[-1] if expr_versions else zh_ver
        except Exception as e:  # noqa: BLE001
            logger.warning(f"  could not list expression_matrices/Zhuang-ABCA-1/: {e}")
            expr_ver = zh_ver
    else:
        ccf_ver = tax_ver = expr_ver = pinned

    logger.info(
        f"  versions: Zhuang-ABCA-1={zh_ver} CCF={ccf_ver} "
        f"WMB-taxonomy={tax_ver} expression={expr_ver}"
    )

    # File layout discovered from the bucket listing (Oct 2025 release):
    #   metadata/Zhuang-ABCA-1/{ver}/
    #       cell_metadata.csv         (~661 MB; raw per-cell metadata, no labels)
    #       gene.csv                  (~80 KB; gene panel)
    #       views/cell_metadata_with_cluster_annotation.csv
    #                                 (~871 MB; cell_metadata pre-joined with
    #                                  cluster/class/subclass/supertype names)
    #   metadata/Zhuang-ABCA-1-CCF/{ver}/
    #       ccf_coordinates.csv       (~221 MB; CCF 3D coords + parcellation)
    #   expression_matrices/Zhuang-ABCA-1/{ver}/
    #       Zhuang-ABCA-1-raw.h5ad    (~1.2 GB; sparse raw counts)
    #
    # We prefer the pre-joined view (so prepare_abc_silver.py doesn't need to
    # join WMB-taxonomy itself). WMB-taxonomy files are still useful for
    # gene-set-level analyses, so we fetch them as optional.
    files = [
        # (s3 path, optional?)
        (f"metadata/Zhuang-ABCA-1/{zh_ver}/cell_metadata.csv", False),
        (f"metadata/Zhuang-ABCA-1/{zh_ver}/gene.csv", False),
        (f"metadata/Zhuang-ABCA-1/{zh_ver}/views/"
         f"cell_metadata_with_cluster_annotation.csv", False),
        (f"metadata/Zhuang-ABCA-1-CCF/{ccf_ver}/ccf_coordinates.csv", False),
        (f"metadata/WMB-taxonomy/{tax_ver}/cluster.csv", True),
        (f"metadata/WMB-taxonomy/{tax_ver}/cluster_annotation_term.csv", True),
        (f"metadata/WMB-taxonomy/{tax_ver}/"
         f"cluster_to_cluster_annotation_membership.csv", True),
        (f"expression_matrices/Zhuang-ABCA-1/{expr_ver}/Zhuang-ABCA-1-raw.h5ad", False),
    ]

    n_ok = 0
    n_missing = 0
    for remote, optional in files:
        out = bronze_dir / remote
        url = f"{S3_BUCKET_HTTPS}/{remote}"
        out.parent.mkdir(parents=True, exist_ok=True)
        if out.exists() and out.stat().st_size > 0:
            logger.info(f"  exists, skipping: {out}")
            n_ok += 1
            continue
        logger.info(f"  GET {url} -> {out}")
        try:
            _stream_download(url, out)
            n_ok += 1
        except Exception as e:  # noqa: BLE001
            if optional:
                logger.warning(f"    optional file unavailable, continuing: {e}")
                n_missing += 1
                continue
            logger.error(
                f"    REQUIRED file missing at {url}\n"
                f"    Inspect the available files at: "
                f"{S3_BUCKET_HTTPS}/?prefix={remote.rsplit('/', 1)[0]}/&delimiter=/"
            )
            raise

    logger.info(f"  downloaded {n_ok} files, {n_missing} optional missing")


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
        logger.info("Forcing HTTPS fallback (--force_https_fallback set).")
        _download_via_https_fallback(bronze, args.release_version)
        return 0

    try:
        _download_via_abc_atlas_access(bronze)
        return 0
    except ImportError:
        logger.warning("=" * 70)
        logger.warning("abc_atlas_access library NOT INSTALLED.")
        logger.warning("This is the SUPPORTED way to download Allen ABC data.")
        logger.warning("Install with:")
        logger.warning("    uv add abc_atlas_access")
        logger.warning("    # or: pip install abc_atlas_access")
        logger.warning("Falling back to plain HTTPS (less robust).")
        logger.warning("=" * 70)
    except Exception as e:  # noqa: BLE001
        logger.exception(
            f"abc_atlas_access path failed: {e}. Trying HTTPS fallback."
        )

    _download_via_https_fallback(bronze, args.release_version)
    return 0


if __name__ == "__main__":
    sys.exit(main())
