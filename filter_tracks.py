#!/usr/bin/env python
"""Drop short-lived spurious tracks from a run's tracks.csv and masks.npz.

Chunked tracking occasionally invents an id: a weak detection appears in one
chunk, never matches anything in the next, and is exported as its own object.
Those artefacts are easy to tell from real subjects by lifetime alone -- a fly
is present for essentially the whole recording, while an artefact cannot outlive
the chunk that produced it (chunk - overlap = 210 frames of a 12.5k-frame run).
The two populations differ by three orders of magnitude, so the exact cutoff
does not matter; anything between one chunk and half the run selects the same
ids.

Both artefact stores are filtered together. Leaving masks.npz unfiltered would
let a re-render paint blobs the CSV no longer knows about.

Kept CSV rows are copied through as text rather than reparsed, so measurements
survive byte for byte -- a float round-trip through pandas rewrites the last
digit of most rows for no gain.

Example:
    python filter_tracks.py runs/batch/30_7_26/CS_T5_30_7_26 --dry-run
    python filter_tracks.py runs/batch/*/*/ --backup runs/superseded/prefilter
"""

from __future__ import annotations

import argparse
import os
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np

import config as cfgmod

# Per-frame keys written by track_long.py, all indexed by object within a frame.
OBJECT_KEYS = ("m", "c", "i", "b", "p")
FRAME_COL, ID_COL = 0, 3


def read_lifetimes(csv_path: Path) -> tuple[list[str], dict[str, set[str]]]:
    """-> (all lines including header, frames each obj_id appears in)."""
    lines = csv_path.read_text().splitlines()
    seen: dict[str, set[str]] = defaultdict(set)
    for line in lines[1:]:
        parts = line.split(",")
        seen[parts[ID_COL]].add(parts[FRAME_COL])
    return lines, seen


def select_drop(seen: dict[str, set[str]], n_frames: int, expect_n: int | None,
                min_fraction: float, min_frames: int | None) -> tuple[set[str], str]:
    """Choose which obj_ids to discard. -> (ids to drop, one-line reason).

    Two modes. Without --expect-n, an id is an artefact if it is present in
    fewer than `cutoff` frames -- the original rule, and the right one when the
    subject count is unknown. With --expect-n, the N longest-lived ids are the
    subjects by definition and everything else goes; the cutoff then only
    reports, because an id that survives the top-N cut while sitting below the
    cutoff means a real animal was lost, not that an artefact was found.
    """
    cutoff = min_frames or int(n_frames * min_fraction)
    if expect_n is None:
        drop = {i for i, frames in seen.items() if len(frames) < cutoff}
        return drop, f"under {cutoff} frames"

    # Longest-lived first; ties broken by id so a rerun drops the same ones.
    ranked = sorted(seen, key=lambda i: (-len(seen[i]), int(i)))
    keep, drop = set(ranked[:expect_n]), set(ranked[expect_n:])
    if len(ranked) < expect_n:
        return drop, (f"expected {expect_n}, found only {len(ranked)} ids "
                      f"-- nothing to drop")
    shortest_kept = min(len(seen[i]) for i in keep)
    if shortest_kept < cutoff:
        return drop, (f"top {expect_n} by lifetime, but the shortest kept id has "
                      f"{shortest_kept} frames, below the {cutoff}-frame cutoff "
                      f"-- a subject was probably lost")
    return drop, f"top {expect_n} by lifetime"


def backup_dest(root: Path, run: Path, runs: list[Path]) -> Path:
    """Where `run`'s originals are copied under `root`.

    Mirrors each run's path below the runs' common ancestor. Using the leaf
    name alone (as this did originally) makes 30_7_26/CS_T5 and 31_7_26/CS_T5
    write to the same place -- the second silently destroys the first's backup,
    after the originals have already been rewritten in place.
    """
    if len(runs) == 1:
        return root / run.name
    base = Path(os.path.commonpath([str(r.resolve()) for r in runs]))
    rel = run.resolve().relative_to(base)
    return root / (rel if rel != Path(".") else Path(run.name))


def filter_masks(npz_path: Path, drop: set[int]) -> tuple[int, int]:
    """Rewrite masks.npz without `drop`. -> (frames touched, objects removed)."""
    z = np.load(npz_path)
    out = {k: z[k] for k in ("height", "width", "format", "frames") if k in z}
    dropped = np.array(sorted(drop))
    touched = removed = 0

    for f in z["frames"].tolist():
        keep = ~np.isin(z[f"i{f}"], dropped)
        if not keep.all():
            touched += 1
            removed += int((~keep).sum())
        # Tiles stay padded to the frame's old maximum box. That is still
        # correct -- each object's true extent is its own row in c{f} -- and
        # re-cropping would mean unpacking and repacking every frame.
        for k in OBJECT_KEYS:
            out[f"{k}{f}"] = z[f"{k}{f}"][keep]

    np.savez_compressed(npz_path, **out)
    return touched, removed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path, nargs="+",
                    help="runs/<name> dirs holding tracks.csv and masks.npz")
    ap.add_argument("--min-fraction", type=float, default=None,
                    help="drop ids present in fewer than this share of frames "
                         "(default: 0.5)")
    ap.add_argument("--min-frames", type=int, default=None,
                    help="absolute cutoff; overrides --min-fraction")
    ap.add_argument("--expect-n", type=int, default=None,
                    help="keep the N longest-lived ids and drop the rest; "
                         "--min-fraction then only warns")
    ap.add_argument("--backup", type=Path, default=None,
                    help="copy the originals under here before overwriting")
    ap.add_argument("--no-masks", action="store_true",
                    help="filter tracks.csv only, leaving masks.npz stale")
    ap.add_argument("--dry-run", action="store_true",
                    help="report what would be dropped, write nothing")
    cfgmod.add_config_arg(ap)
    cfgmod.apply_to(ap, cfgmod.load_config(cfgmod.preparse_config()), "filter", "subjects")
    args = ap.parse_args()

    for run in args.run_dir:
        csv_path = run / "tracks.csv"
        if not csv_path.exists():
            print(f"[skip] {run.name}: no tracks.csv")
            continue

        lines, seen = read_lifetimes(csv_path)
        n_frames = len({l.split(",")[FRAME_COL] for l in lines[1:]})
        drop, reason = select_drop(seen, n_frames, args.expect_n,
                                   args.min_fraction if args.min_fraction is not None
                                   else 0.5,
                                   args.min_frames)
        if not drop:
            print(f"[ok]   {run.name}: {len(seen)} ids, none dropped ({reason})")
            continue

        kept = [lines[0]] + [l for l in lines[1:] if l.split(",")[ID_COL] not in drop]
        detail = ", ".join(f"{i} ({len(seen[i])}f)"
                           for i in sorted(drop, key=int))
        print(f"[drop] {run.name}: ids {detail} -> {len(lines) - 1} to "
              f"{len(kept) - 1} rows, {len(seen)} to {len(seen) - len(drop)} ids "
              f"({reason})")
        if args.dry_run:
            continue

        npz_path = run / "masks.npz"
        if args.backup:
            dest = backup_dest(args.backup, run, args.run_dir)
            dest.mkdir(parents=True, exist_ok=True)
            shutil.copy2(csv_path, dest / "tracks.csv")
            if npz_path.exists() and not args.no_masks:
                shutil.copy2(npz_path, dest / "masks.npz")
            print(f"       originals -> {dest}")

        csv_path.write_text("\n".join(kept) + "\n")
        if not args.no_masks and npz_path.exists():
            touched, removed = filter_masks(npz_path, {int(i) for i in drop})
            print(f"       masks.npz: {removed} objects off {touched} frames "
                  f"({npz_path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
