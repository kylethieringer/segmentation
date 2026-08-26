#!/usr/bin/env python
"""Generate figures for the SAM 3 fly-tracking results, in light and dark variants.

Identity is carried by panel titles and axis labels, never by colour alone: the
categorical palette only validates all-pairs to three slots and there are seven
flies, so every figure facets instead of cycling hues.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

ARENA = dict(cx=569.3, cy=566.8, r=530.0)   # least-squares fit to outer 3% of positions
SRC_FPS = 60.0
STRIDE = 3

THEMES = {
    "light": dict(surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", grid="#e3e2dd",
                  series="#2a78d6", ramp=["#fcfcfb", "#9dc3ee", "#2a78d6", "#123a68"]),
    "dark":  dict(surface="#1a1a19", ink="#ffffff", ink2="#c3c2b7", grid="#39382f",
                  series="#3987e5", ramp=["#1a1a19", "#1f4a7d", "#3987e5", "#a9cdf5"]),
}


def style(t: dict) -> None:
    plt.rcParams.update({
        "figure.facecolor": t["surface"], "axes.facecolor": t["surface"],
        "savefig.facecolor": t["surface"], "text.color": t["ink"],
        "axes.labelcolor": t["ink2"], "xtick.color": t["ink2"], "ytick.color": t["ink2"],
        "axes.edgecolor": t["grid"], "grid.color": t["grid"],
        "axes.titlecolor": t["ink"], "font.size": 11,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.linewidth": 0.6, "grid.alpha": 0.7,
        "figure.dpi": 130, "axes.axisbelow": True,
    })


def recessive(ax, t):
    ax.tick_params(length=3, width=0.6)
    for s in ax.spines.values():
        s.set_linewidth(0.8)


def load(run: Path):
    d = pd.read_csv(run / "tracks.csv").sort_values(["obj_id", "frame"])
    d["r_norm"] = np.hypot(d.cx - ARENA["cx"], d.cy - ARENA["cy"]) / ARENA["r"]
    dt = STRIDE / SRC_FPS
    parts = []
    for _, g in d.groupby("obj_id"):
        g = g.sort_values("frame").copy()
        g["speed"] = np.hypot(g.cx.diff(), g.cy.diff()) / dt
        parts.append(g)
    return pd.concat(parts, ignore_index=True)


# ---------------------------------------------------------------- figures

def fig_trajectories(d, t, out: Path):
    ids = sorted(d.obj_id.unique())
    fig, axes = plt.subplots(2, 4, figsize=(13, 7.4))
    for ax, oid in zip(axes.ravel(), ids):
        g = d[d.obj_id == oid]
        ax.add_patch(plt.Circle((ARENA["cx"], ARENA["cy"]), ARENA["r"], fill=False,
                                color=t["grid"], lw=1.4))
        ax.plot(g.cx, g.cy, lw=0.5, color=t["series"], alpha=0.85, solid_capstyle="round")
        ax.plot(g.cx.iloc[0], g.cy.iloc[0], "o", ms=5, color=t["ink"], zorder=3)
        dist = np.hypot(g.cx.diff(), g.cy.diff()).sum()
        ax.set_title(f"Fly {oid}", fontsize=11, pad=4)
        # Caption inside the panel: below the axes it collides with the next row's title.
        ax.text(0.02, 0.02, f"{dist/1000:.1f}k px", transform=ax.transAxes,
                ha="left", va="bottom", fontsize=9.5, color=t["ink2"])
        ax.set_box_aspect(1); ax.set_aspect("equal"); ax.axis("off")
        ax.set_xlim(0, 1120); ax.set_ylim(1120, 0)
    axes.ravel()[-1].axis("off")
    axes.ravel()[-1].text(0.5, 0.5, "Dot marks\nstart position", ha="center", va="center",
                          fontsize=10, color=t["ink2"], transform=axes.ravel()[-1].transAxes)
    fig.suptitle("Individual walking paths over 3 minutes", fontsize=14, y=0.985)
    fig.text(0.5, 0.015, "Distance travelled shown per panel  ·  arena outline fitted to "
             "outermost tracked positions (r = 530 px)",
             ha="center", fontsize=9, color=t["ink2"])
    fig.subplots_adjust(top=0.90, bottom=0.06, hspace=0.22, wspace=0.05)
    fig.savefig(out); plt.close(fig)


def fig_speed(d, t, out: Path):
    ids = sorted(d.obj_id.unique())
    frames = sorted(d.frame.unique())
    time_s = np.array(frames) * STRIDE / SRC_FPS
    mat = np.full((len(ids), len(frames)), np.nan)
    idx = {f: i for i, f in enumerate(frames)}
    for r, oid in enumerate(ids):
        g = d[d.obj_id == oid]
        for f, sp in zip(g.frame, g.speed):
            if not np.isnan(sp):
                mat[r, idx[f]] = sp
    # Smooth over ~1 s so the plot shows activity bouts, not per-frame jitter.
    win = int(round(SRC_FPS / STRIDE))
    smooth = pd.DataFrame(mat).T.rolling(win, min_periods=1, center=True).mean().T.to_numpy()

    cmap = LinearSegmentedColormap.from_list("seq", t["ramp"])
    fig, ax = plt.subplots(figsize=(13, 3.6))
    vmax = np.nanpercentile(smooth, 99)
    im = ax.imshow(smooth, aspect="auto", cmap=cmap, vmin=0, vmax=vmax,
                   extent=[0, time_s[-1], len(ids) - 0.5, -0.5], interpolation="nearest")
    ax.set_yticks(range(len(ids))); ax.set_yticklabels([f"Fly {i}" for i in ids])
    ax.set_xlabel("Time (seconds)"); ax.grid(False)
    cb = fig.colorbar(im, ax=ax, pad=0.012, fraction=0.03)
    cb.set_label("Speed (px/s, 1 s mean)", color=t["ink2"], fontsize=10)
    cb.ax.tick_params(colors=t["ink2"], length=3); cb.outline.set_edgecolor(t["grid"])
    ax.set_title("Activity over time — darker means faster", fontsize=13, pad=8)
    recessive(ax, t)
    fig.tight_layout(); fig.savefig(out, bbox_inches="tight"); plt.close(fig)


def fig_radial(d, t, out: Path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.2),
                                   gridspec_kw={"width_ratios": [1.15, 1]})
    bins = np.linspace(0, 1.05, 44)
    ax1.hist(d.r_norm, bins=bins, color=t["series"], alpha=0.9, edgecolor=t["surface"], lw=0.5)
    ax1.axvspan(0.75, 1.05, color=t["ink2"], alpha=0.10, lw=0)
    frac = (d.r_norm > 0.75).mean()
    ax1.text(0.875, ax1.get_ylim()[1] * 0.92, f"outer third\n{frac:.0%} of time",
             ha="center", va="top", fontsize=10, color=t["ink2"])
    ax1.set_xlabel("Distance from arena centre  (r / R)")
    ax1.set_ylabel("Observations")
    ax1.set_title("Flies hug the wall (thigmotaxis)", fontsize=13, pad=8)
    recessive(ax1, t)

    ids = sorted(d.obj_id.unique())
    per = [(d[d.obj_id == i].r_norm > 0.75).mean() for i in ids]
    order = np.argsort(per)
    ax2.barh([f"Fly {ids[i]}" for i in order], [per[i] for i in order],
             color=t["series"], height=0.62)
    for y, i in enumerate(order):
        ax2.text(per[i] - 0.02, y, f"{per[i]:.0%}", va="center", ha="right",
                 color=t["surface"], fontsize=10, fontweight="bold")
    ax2.set_xlim(0, 1); ax2.set_xlabel("Fraction of time in outer third")
    ax2.set_title("Wall preference varies widely by individual", fontsize=13, pad=8)
    ax2.grid(axis="y", visible=False)
    recessive(ax2, t)
    fig.tight_layout(); fig.savefig(out, bbox_inches="tight"); plt.close(fig)


def fig_validation(t, out: Path):
    truth = pd.read_csv("runs/smoke_v3/tracks.csv")
    up = pd.read_csv("runs/full_s3/tracks_60fps.csv")
    u0 = up[up.src_frame == 0].set_index("obj_id")
    t0 = truth[truth.src_frame == 0].set_index("obj_id")
    pair = {u: (np.hypot(t0.cx - r.cx, t0.cy - r.cy)).idxmin() for u, r in u0.iterrows()}
    err = []
    for uid, tid in pair.items():
        a = up[(up.obj_id == uid) & (up.src_frame <= 119)].set_index("src_frame")
        b = truth[truth.obj_id == tid].set_index("src_frame")
        j = a.join(b[["cx", "cy"]], rsuffix="_t", how="inner")
        err.extend(np.hypot(j.cx - j.cx_t, j.cy - j.cy_t).tolist())
    err = np.sort(np.array(err))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.0))
    ax1.plot(err, np.arange(1, len(err) + 1) / len(err) * 100, lw=2, color=t["series"])
    for q, lab in ((50, "median"), (95, "p95")):
        v = np.percentile(err, q)
        ax1.plot([v, v], [0, q], lw=1, ls=":", color=t["ink2"])
        ax1.text(v + 0.06, q - 8, f"{lab} {v:.2f} px", fontsize=9, color=t["ink2"])
    ax1.set_xlabel("Position error vs stride-1 ground truth (px)")
    ax1.set_ylabel("Percent of observations")
    ax1.set_title("Interpolation error stays sub-pixel", fontsize=13, pad=8)
    recessive(ax1, t)

    # measured overlap IoU reported by track_long.py at each of the 8 chunk seams
    iou = [0.92, 0.84, 0.95, 0.77, 0.93, 0.84, 0.83, 0.80]
    ax2.bar(range(1, 9), iou, color=t["series"], width=0.6)
    ax2.axhline(0.3, color=t["ink2"], ls="--", lw=1.2)
    ax2.text(8.45, 0.36, "match threshold 0.30", ha="right", va="bottom",
             fontsize=9, color=t["ink2"])
    ax2.set_ylim(0, 1.0); ax2.set_xlabel("Chunk boundary"); ax2.set_ylabel("Overlap IoU")
    ax2.set_title("All 8 seams matched 7/7 tracks", fontsize=13, pad=8)
    ax2.set_xticks(range(1, 9))
    recessive(ax2, t)
    fig.tight_layout(); fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    return err


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, default=Path("runs/full_s3"))
    ap.add_argument("--outdir", type=Path, default=Path("figures"))
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    d = load(args.run)
    for mode, t in THEMES.items():
        style(t)
        fig_trajectories(d, t, args.outdir / f"trajectories_{mode}.png")
        fig_speed(d, t, args.outdir / f"speed_{mode}.png")
        fig_radial(d, t, args.outdir / f"radial_{mode}.png")
        err = fig_validation(t, args.outdir / f"validation_{mode}.png")
        print(f"[figures] {mode} done")

    summary = []
    for oid, g in d.groupby("obj_id"):
        summary.append(dict(fly=oid,
                            dist_px=float(np.hypot(g.cx.diff(), g.cy.diff()).sum()),
                            mean_speed=float(g.speed.mean()),
                            p95_speed=float(g.speed.quantile(.95)),
                            outer_third=float((g.r_norm > 0.75).mean()),
                            med_area=float(g.area_px.median())))
    pd.DataFrame(summary).to_csv(args.outdir / "summary.csv", index=False)
    print(pd.DataFrame(summary).to_string(index=False))


if __name__ == "__main__":
    main()
