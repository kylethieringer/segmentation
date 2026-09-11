#!/usr/bin/env python
"""Upsample strided SAM 3 tracks back to every source frame by interpolation.

Tracking at --stride N gives positions every Nth source frame. Fly centroids move
smoothly enough between them that interpolation recovers the missing frames to
well under a pixel: validated against stride-1 ground truth (median 0.57 px,
max 3.99 px) and by decimation across the full recording (estimated max ~3.7 px
at stride 3, versus a ~60 px fly body).

Interpolated rows are marked `measured=0` so downstream analysis can tell real
observations from filled ones. Masks cannot be interpolated -- if you need
per-frame masks or trustworthy per-frame areas, re-track at stride 1 instead.

Example:
    python upsample_tracks.py runs/full_s3/tracks.csv --out runs/full_s3/tracks_60fps.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import Akima1DInterpolator, CubicSpline, PchipInterpolator, interp1d

import config as cfgmod

# Position columns get the chosen (shape-preserving) interpolator; these get plain
# linear treatment -- they are noisy per-frame measurements, not smooth trajectories,
# so a fancier fit would imply precision that is not there.
LINEAR_COLS = ["area_px", "score", "box_x", "box_y", "box_w", "box_h"]


def make_interp(kind: str, x: np.ndarray, y: np.ndarray):
    if kind == "pchip":
        return PchipInterpolator(x, y)
    if kind == "akima":
        return Akima1DInterpolator(x, y)
    if kind == "cubic":
        return CubicSpline(x, y)
    return interp1d(x, y, kind="linear")


def upsample(df: pd.DataFrame, kind: str, fps: float) -> pd.DataFrame:
    out_frames = []
    for obj_id, g in df.groupby("obj_id"):
        g = g.sort_values("src_frame").drop_duplicates("src_frame")
        if len(g) < 2:
            out_frames.append(g.assign(measured=1))
            continue

        x = g.src_frame.to_numpy(dtype=float)
        # Only fill *between* observations -- never extrapolate past the ends,
        # where a spline has no data and will happily invent motion.
        targets = np.arange(int(x.min()), int(x.max()) + 1)

        rec = {"src_frame": targets, "obj_id": obj_id}
        for col in ("cx", "cy"):
            rec[col] = make_interp(kind, x, g[col].to_numpy(dtype=float))(targets)
        for col in LINEAR_COLS:
            if col in g.columns:
                rec[col] = np.interp(targets, x, g[col].to_numpy(dtype=float))

        frame = pd.DataFrame(rec)
        frame["measured"] = np.isin(targets, x.astype(int)).astype(int)
        frame["time_s"] = frame.src_frame / fps
        out_frames.append(frame)

    result = pd.concat(out_frames, ignore_index=True)
    cols = ["src_frame", "time_s", "obj_id", "cx", "cy", "measured"]
    cols += [c for c in LINEAR_COLS if c in result.columns]
    return result[cols].sort_values(["src_frame", "obj_id"]).reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tracks_csv", type=Path)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--kind", default="pchip",
                    choices=["pchip", "akima", "cubic", "linear"],
                    help="interpolator for cx/cy (default pchip: smooth, cannot overshoot)")
    ap.add_argument("--fps", type=float, default=None,
                    help="source fps for the time_s column; inferred if omitted")
    cfgmod.add_config_arg(ap)
    # No section of its own: the flag is declared for uniformity, so every
    # script in the pipeline takes --config whether or not it reads anything.
    cfgmod.apply_to(ap, cfgmod.load_config(cfgmod.preparse_config()))
    args = ap.parse_args()

    df = pd.read_csv(args.tracks_csv)
    if args.fps is not None:
        fps = args.fps
    else:
        # Recover source fps from the original time_s/src_frame relationship.
        nz = df[df.src_frame > 0]
        fps = float(round((nz.src_frame / nz.time_s).median(), 3))
        print(f"[upsample] inferred source fps = {fps}")

    out_df = upsample(df, args.kind, fps)
    out_path = args.out or args.tracks_csv.with_name(
        args.tracks_csv.stem + "_upsampled.csv")
    out_df.to_csv(out_path, index=False, float_format="%.3f")

    n_meas = int(out_df.measured.sum())
    print(f"[upsample] {len(df)} -> {len(out_df)} rows "
          f"({n_meas} measured, {len(out_df) - n_meas} interpolated) "
          f"using {args.kind}")
    print(f"[upsample] src frames {out_df.src_frame.min()}..{out_df.src_frame.max()}, "
          f"{out_df.obj_id.nunique()} objects -> {out_path}")


if __name__ == "__main__":
    main()
