#!/usr/bin/env python
"""Build a train/test split of an existing silver dir by re-labelling sections.

A silver dir encodes its split ONLY in the filename suffix
(``{section}_train.h5ad`` / ``{section}_test.h5ad``); the cells themselves carry
no split flag. So a new split is just a new directory pointing at the same
files under different names -- symlinks by default, so an alternative split
costs bytes rather than gigabytes.

WHY A SCRIPT AND NOT A SHELL LOOP
---------------------------------
The donor column on this dataset has already been misread once: the ``BKxx``
prefix of a Sample ID is a block, and the real donor is the Sanger patient ID,
which does not have to agree with it. A split whose whole point is "this donor
is unseen" is worthless if the donor labels are wrong, and a symlink loop
cannot check. This script reads the donor column out of obs and REFUSES to
write when a test donor also appears in train, unless you say --allow_seen_donor
because that overlap is the design.

It also reports section and cell counts per split, because shrinking the
training set is a confound: if a model trained on one donor does worse, that
can be the donor restriction or simply less data, and you cannot tell those
apart after the fact without the counts.

USAGE
    python make_donor_split.py \
        --src_dir  /nfs/team361/sb75/DATASETS/silver/xenium_xhs1000 \
        --out_dir  /nfs/team361/sb75/DATASETS/silver/xenium_bk20 \
        --test_sections BK20-SKI-27-FO-2-S6_0,BK18-SKI-27-FO-2-S6_0 \
        --train_prefix BK20 \
        --dry_run

NOTE ON SYMLINKS
----------------
The output holds symlinks into --src_dir. Anything that writes into an h5ad in
place -- precompute_embeddings.py without --out_dir, most obviously -- will
write straight through them and modify the shared originals every other split
reads. Always give such tools their own --out_dir.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

DONOR_HINTS = ("sanger patient id", "patient id", "patient", "donor")


def section_of(path: Path) -> tuple[str, str]:
    """('BK20-..._train.h5ad') -> ('BK20-...', 'train')."""
    stem = path.name[: -len(".h5ad")]
    for split in ("train", "test"):
        if stem.endswith("_" + split):
            return stem[: -(len(split) + 1)], split
    raise SystemExit(f"{path.name}: not a *_train.h5ad / *_test.h5ad file")


def pick_donor_col(obs_cols) -> str | None:
    lower = {str(c).lower(): str(c) for c in obs_cols}
    for hint in DONOR_HINTS:
        for low, orig in lower.items():
            if low == hint:
                return orig
    for hint in DONOR_HINTS:
        for low, orig in lower.items():
            if hint in low:
                return orig
    return None


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src_dir", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--test_sections", required=True,
                   help="comma-separated section names for the test split")
    p.add_argument("--train_prefix", default="",
                   help="keep train sections whose name starts with this "
                        "(e.g. BK20). Omit to take every non-test section.")
    p.add_argument("--train_sections", default="",
                   help="explicit comma-separated train sections; overrides "
                        "--train_prefix")
    p.add_argument("--donor_col", default="",
                   help="obs column holding the donor (default: auto-detect)")
    p.add_argument("--allow_seen_donor", action="store_true",
                   help="permit a test donor that also appears in train "
                        "(deliberate for a seen-donor/unseen-section arm)")
    p.add_argument("--allow_train_test_overlap", action="store_true",
                   help="permit the SAME section in both splits. This is a "
                        "deliberately leaky memorisation control (train==test) "
                        "for measuring capacity, never a generalisation "
                        "result. Implies --allow_seen_donor.")
    p.add_argument("--copy", action="store_true",
                   help="copy files instead of symlinking")
    p.add_argument("--dry_run", action="store_true")
    args = p.parse_args()

    import anndata as ad

    src, out = Path(args.src_dir).resolve(), Path(args.out_dir)
    files = sorted(src.glob("*.h5ad"))
    if not files:
        raise SystemExit(f"no h5ad under {src}")

    by_section: dict[str, Path] = {}
    for f in files:
        sec, _ = section_of(f)
        if sec in by_section:
            raise SystemExit(f"section {sec!r} appears twice in {src}")
        by_section[sec] = f
    print(f"{len(by_section)} sections in {src.name}")

    want_test = [s.strip() for s in args.test_sections.split(",") if s.strip()]
    unknown = [s for s in want_test if s not in by_section]
    if unknown:
        for u in unknown:
            near = [s for s in by_section if u.split("-")[0] in s][:5]
            print(f"  unknown test section {u!r}; sections starting alike: {near}")
        raise SystemExit("--test_sections names sections that do not exist")

    if args.train_sections:
        want_train = [s.strip() for s in args.train_sections.split(",") if s.strip()]
        unknown = [s for s in want_train if s not in by_section]
        if unknown:
            raise SystemExit(f"--train_sections unknown: {unknown}")
    else:
        want_train = [s for s in sorted(by_section)
                      if s not in want_test
                      and s.startswith(args.train_prefix)]
    overlap = sorted(set(want_train) & set(want_test))
    if overlap and not args.allow_train_test_overlap:
        raise SystemExit(f"section(s) in both splits: {overlap}\n"
                         "Pass --allow_train_test_overlap if this is a "
                         "memorisation control.")
    if overlap:
        args.allow_seen_donor = True     # implied: same section, same donor
        print(f"\n*** TRAIN == TEST for {len(overlap)} section(s): {overlap}")
        print("*** Memorisation control. Any score here measures capacity to "
              "fit seen data,\n*** NOT generalisation. Do not report it as a "
              "benchmark result.")
    if not want_train:
        raise SystemExit("no train sections selected")

    # ---- donor audit: the whole point of the split ------------------------
    donor_col, rows, n_cells = args.donor_col or None, [], {"train": 0, "test": 0}
    for split, secs in (("train", want_train), ("test", want_test)):
        for s in secs:
            a = ad.read_h5ad(by_section[s], backed="r")
            if donor_col is None:
                donor_col = pick_donor_col(a.obs.columns)
                if donor_col is None:
                    raise SystemExit(
                        "cannot find a donor column in obs; pass --donor_col. "
                        f"Available: {list(map(str, a.obs.columns))}")
                print(f"donor column: {donor_col!r} (auto-detected)")
            if donor_col not in a.obs:
                raise SystemExit(f"{s}: obs[{donor_col!r}] missing")
            donors = sorted(set(a.obs[donor_col].astype(str)))
            if len(donors) > 1:
                raise SystemExit(f"{s}: spans {len(donors)} donors {donors} — "
                                 "a section must belong to one donor")
            rows.append({"section": s, "split": split, "donor": donors[0],
                         "n_cells": int(a.n_obs)})
            n_cells[split] += int(a.n_obs)
            a.file.close()

    train_donors = {r["donor"] for r in rows if r["split"] == "train"}
    print(f"\n{'section':40s} {'split':6s} {'donor':16s} cells")
    for r in rows:
        seen = " (donor seen in train)" if (
            r["split"] == "test" and r["donor"] in train_donors) else ""
        print(f"{r['section']:40s} {r['split']:6s} {r['donor']:16s} "
              f"{r['n_cells']:7d}{seen}")

    print(f"\ntrain: {len(want_train)} sections, {n_cells['train']:,} cells, "
          f"donor(s) {sorted(train_donors)}")
    print(f"test : {len(want_test)} sections, {n_cells['test']:,} cells")

    seen_donor = [r for r in rows
                  if r["split"] == "test" and r["donor"] in train_donors]
    if seen_donor and not args.allow_seen_donor:
        raise SystemExit(
            "these test sections share a donor with train:\n"
            + "\n".join(f"  {r['section']} (donor {r['donor']})" for r in seen_donor)
            + "\nPass --allow_seen_donor if that is the design.")
    if seen_donor:
        print(f"\nseen-donor test section(s) allowed: "
              f"{[r['section'] for r in seen_donor]}")

    if args.dry_run:
        print("\ndry run: nothing written")
        return 0

    # ---- write -------------------------------------------------------------
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} exists and is not empty — refusing to mix splits")
    out.mkdir(parents=True, exist_ok=True)
    import shutil
    for split, secs in (("train", want_train), ("test", want_test)):
        for s in secs:
            dst = out / f"{s}_{split}.h5ad"
            if args.copy:
                shutil.copy2(by_section[s], dst)
            else:
                dst.symlink_to(by_section[s].resolve())
    (out / "split.json").write_text(json.dumps({
        "src_dir": str(src), "donor_col": donor_col,
        "mode": "copy" if args.copy else "symlink",
        "train_prefix": args.train_prefix or None,
        "train_test_overlap": overlap,
        "n_train_sections": len(want_train), "n_test_sections": len(want_test),
        "n_train_cells": n_cells["train"], "n_test_cells": n_cells["test"],
        "train_donors": sorted(train_donors),
        "sections": rows,
    }, indent=2))

    print(f"\nwrote {len(want_train)+len(want_test)} "
          f"{'copies' if args.copy else 'symlinks'} to {out}")
    print(f"provenance: {out/'split.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
