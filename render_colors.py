#!/usr/bin/env python
"""Render tracked masks as a video, one distinct colour per object id.

Reads the masks.npz written by `track_video.py --save-masks`, so re-colouring
never requires re-running the tracker.

Example:
    python render_colors.py runs/smoke_v3 --trails --out runs/smoke_v3/colored.mp4
"""

from __future__ import annotations

import argparse
import subprocess
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

# ColorBrewer Set1 + extensions: chosen to stay distinguishable from each other
# AND from a grey backlit arena. Indexed by object id, so a fly keeps its colour
# for the whole video.
PALETTE = [
    (228, 26, 28), (55, 126, 184), (77, 175, 74), (255, 127, 0),
    (152, 78, 163), (0, 206, 209), (255, 225, 25), (247, 129, 191),
    (166, 86, 40), (128, 128, 0), (0, 128, 128), (240, 50, 230),
    (60, 180, 75), (145, 30, 180), (255, 250, 200), (170, 110, 40),
]


def load_masks(npz_path: Path):
    """Unpack masks.npz -> {frame: (ids, masks[bool], boxes, probs)}."""
    z = np.load(npz_path)
    width = int(z["width"])
    frames = z["frames"].tolist()
    out = {}
    for f in frames:
        packed = z[f"m{f}"]
        # packbits was applied along the last axis; unpack and trim the padding.
        masks = np.unpackbits(packed, axis=-1).astype(bool)[..., :width]
        out[f] = (z[f"i{f}"], masks, z[f"b{f}"], z[f"p{f}"])
    return out, int(z["height"]), width


def open_frame_source(run_dir: Path, video: Path | None):
    """Return frame_idx -> BGR image. Prefers extracted JPEGs, falls back to the video.

    The frames/ dir is a bulky intermediate people delete; masks.npz alone cannot
    reconstruct the picture, so fall back to decoding the source video using the
    src_frame column in tracks.csv to map back to the original frame numbers.
    """
    frames_dir = run_dir / "frames"
    if frames_dir.is_dir() and any(frames_dir.glob("*.jpg")):
        def from_jpg(fidx: int):
            img = cv2.imread(str(frames_dir / f"{fidx}.jpg"))
            if img is None:
                raise SystemExit(f"missing frame {frames_dir / f'{fidx}.jpg'}")
            return img
        return from_jpg

    if video is None:
        raise SystemExit(
            f"{frames_dir} is empty and no --video given. Re-extract the frames, or "
            "pass --video to decode them from the source.")

    import csv as _csv
    import decord
    mapping: dict[int, int] = {}
    with (run_dir / "tracks.csv").open() as fh:
        for row in _csv.DictReader(fh):
            mapping[int(row["frame"])] = int(row["src_frame"])
    reader = decord.VideoReader(str(video))

    def from_video(fidx: int):
        src = mapping.get(fidx)
        if src is None:
            raise SystemExit(f"frame {fidx} not in tracks.csv; cannot map to source")
        rgb = reader[src].asnumpy()
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    return from_video


def render(run_dir: Path, out_path: Path, alpha: float, fps: float,
           trails: bool, labels: bool, outline: int, trail_len: int,
           video: Path | None) -> None:
    get_frame = open_frame_source(run_dir, video)
    data, height, width = load_masks(run_dir / "masks.npz")

    trail_pts: dict[int, list[tuple[int, int]]] = defaultdict(list)
    # NOT in /tmp: ffmpeg here is a snap, and snaps get a private /tmp namespace,
    # so a file written to the real /tmp is invisible to it. Stage next to the output.
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.parent / f".{out_path.stem}.raw.mp4"
    writer = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"),
                             fps, (width, height))
    if not writer.isOpened():
        raise SystemExit("cv2.VideoWriter failed to open")

    for fidx in sorted(data):
        img = get_frame(fidx)

        obj_ids, masks, _boxes, probs = data[fidx]
        overlay = img.copy()

        for i, obj_id in enumerate(np.asarray(obj_ids).tolist()):
            mask = masks[i]
            if not mask.any():
                continue
            r, g, b = PALETTE[int(obj_id) % len(PALETTE)]
            bgr = np.array([b, g, r], dtype=np.float32)  # cv2 is BGR

            # Tint the mask interior.
            overlay[mask] = (overlay[mask] * (1 - alpha) + bgr * alpha).astype(np.uint8)

            # A crisp full-saturation outline keeps small subjects legible even
            # where the translucent fill washes out against a bright arena.
            if outline > 0:
                contours, _ = cv2.findContours(mask.astype(np.uint8),
                                               cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(overlay, contours, -1, (int(b), int(g), int(r)), outline)

            ys, xs = np.nonzero(mask)
            cx, cy = int(xs.mean()), int(ys.mean())
            if trails:
                pts = trail_pts[int(obj_id)]
                pts.append((cx, cy))
                # Cap the tail: over thousands of frames an unbounded trail is both
                # slow to draw and unreadable, since it fills the whole arena.
                if trail_len > 0 and len(pts) > trail_len:
                    del pts[:-trail_len]
            if labels:
                cv2.putText(overlay, str(int(obj_id)), (cx + 14, cy - 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (int(b), int(g), int(r)), 2, cv2.LINE_AA)

        if trails:
            for obj_id, pts in trail_pts.items():
                r, g, b = PALETTE[obj_id % len(PALETTE)]
                if len(pts) > 1:
                    cv2.polylines(overlay, [np.array(pts, dtype=np.int32)], False,
                                  (int(b), int(g), int(r)), 2, cv2.LINE_AA)

        cv2.putText(overlay, f"frame {fidx}", (16, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (255, 255, 255), 2, cv2.LINE_AA)
        writer.write(overlay)

    writer.release()

    # mp4v is poorly supported by browsers/players; re-encode to H.264.
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(tmp),
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", str(out_path)],
                   check=True)
    tmp.unlink(missing_ok=True)
    print(f"[colored] {len(data)} frames -> {out_path} "
          f"({out_path.stat().st_size / 1e6:.1f} MB)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", type=Path, help="a runs/<name> dir containing masks.npz")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--video", type=Path, default=None,
                    help="source video, used when frames/ has been deleted")
    ap.add_argument("--alpha", type=float, default=0.65, help="mask fill opacity")
    ap.add_argument("--fps", type=float, default=30.0, help="playback frame rate")
    ap.add_argument("--trails", action="store_true", help="draw centroid motion trails")
    ap.add_argument("--trail-len", type=int, default=60,
                    help="trail length in frames; 0 for unbounded")
    ap.add_argument("--no-labels", action="store_true")
    ap.add_argument("--outline", type=int, default=2, help="contour thickness, 0 to disable")
    args = ap.parse_args()

    out = args.out or (args.run_dir / "colored.mp4")
    render(args.run_dir, out, args.alpha, args.fps, args.trails,
           not args.no_labels, args.outline, args.trail_len, args.video)


if __name__ == "__main__":
    main()
