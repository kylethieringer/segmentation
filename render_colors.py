#!/usr/bin/env python
"""Render tracked masks as a video, one distinct colour per object id.

Reads the masks.npz written by `track_video.py --save-masks`, so re-colouring
never requires re-running the tracker.

Example:
    python render_colors.py runs/smoke_v3 --trails --out runs/smoke_v3/smoke_v3_tracked.mp4
"""

from __future__ import annotations

import argparse
import subprocess
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

import config as cfgmod
from track_video import MASK_FORMAT

# ColorBrewer Set1 + extensions: chosen to stay distinguishable from each other
# AND from a grey backlit arena. Indexed by object id, so a fly keeps its colour
# for the whole video.
PALETTE = [
    (228, 26, 28), (55, 126, 184), (77, 175, 74), (255, 127, 0),
    (152, 78, 163), (0, 206, 209), (255, 225, 25), (247, 129, 191),
    (166, 86, 40), (128, 128, 0), (0, 128, 128), (240, 50, 230),
    (60, 180, 75), (145, 30, 180), (255, 250, 200), (170, 110, 40),
]


class MaskStore:
    """Lazy per-frame accessor for masks.npz.

    Unpacked masks run ~9 MB a frame (7 objects at 1120x1120), so holding a
    whole run in memory needs tens of GB and OOMs on anything untrided. Keep
    the npz open and decompress one frame at a time instead.
    """

    def __init__(self, npz_path: Path):
        self.z = np.load(npz_path)
        self.height = int(self.z["height"])
        self.width = int(self.z["width"])
        self.frames = sorted(self.z["frames"].tolist())
        # Runs tracked before cropped masks landed carry no format key.
        self.format = int(self.z["format"]) if "format" in self.z else 1
        if self.format > MASK_FORMAT:
            raise SystemExit(f"{npz_path} is format {self.format}; this build reads "
                             f"up to {MASK_FORMAT}. Update render_colors.py.")

    def __len__(self) -> int:
        return len(self.frames)

    def __getitem__(self, f: int):
        """-> (ids, masks[bool], boxes, probs) for one frame."""
        packed = self.z[f"m{f}"]
        if self.format == 1:
            # Whole frames, packbits along the last axis: unpack and trim the padding.
            masks = np.unpackbits(packed, axis=-1).astype(bool)[..., :self.width]
        else:
            masks = self._uncrop(packed, self.z[f"c{f}"])
        return self.z[f"i{f}"], masks, self.z[f"b{f}"], self.z[f"p{f}"]

    def _uncrop(self, packed: np.ndarray, crops: np.ndarray) -> np.ndarray:
        """Paste format-2 bounding-box tiles back onto full frames.

        Objects were padded to the largest box in the frame, so the tile width
        is recoverable from `crops` alone. An object with an empty mask has an
        all-zero crop row and contributes nothing.
        """
        masks = np.zeros((len(crops), self.height, self.width), dtype=bool)
        if not len(crops):
            return masks
        tiles = np.unpackbits(packed, axis=-1).astype(bool)[..., :int(crops[:, 3].max())]
        for i, (y0, x0, h, w) in enumerate(crops.tolist()):
            if h:
                masks[i, y0:y0 + h, x0:x0 + w] = tiles[i, :h, :w]
        return masks


def playback_fps(run_dir: Path) -> float | None:
    """Real-time playback rate for the mask frames, from tracks.csv.

    Deliberately frame/time_s and not src_frame/time_s: one rendered frame is
    one *mask* frame, so a strided run plays back slower than its source. For
    a 60 fps source this gives 60.0 unstrided and 20.0 at --stride 3; using
    src_frame instead would run a strided render at 3x speed.
    """
    import csv as _csv
    best = None
    try:
        with (run_dir / "tracks.csv").open() as fh:
            for i, row in enumerate(_csv.DictReader(fh)):
                if i >= 20000:
                    break
                # time_s is written to 6dp, so the ratio is only precise for a
                # large frame index; keep the furthest row seen.
                frame, t = int(row["frame"]), float(row["time_s"])
                if t > 0 and (best is None or frame > best[0]):
                    best = (frame, t)
    except (OSError, KeyError, ValueError):
        return None
    return round(best[0] / best[1], 3) if best else None


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


def render(run_dir: Path, out_path: Path, alpha: float, fps: float | None,
           trails: bool, labels: bool, outline: int, trail_len: int,
           video: Path | None) -> None:
    get_frame = open_frame_source(run_dir, video)
    data = MaskStore(run_dir / "masks.npz")
    height, width = data.height, data.width
    if fps is None:
        fps = playback_fps(run_dir) or 30.0

    trail_pts: dict[int, list[tuple[int, int]]] = defaultdict(list)
    # NOT in /tmp: ffmpeg here is a snap, and snaps get a private /tmp namespace,
    # so a file written to the real /tmp is invisible to it. Stage next to the output.
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.parent / f".{out_path.stem}.raw.mp4"
    writer = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"),
                             fps, (width, height))
    if not writer.isOpened():
        raise SystemExit("cv2.VideoWriter failed to open")

    for fidx in data.frames:
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
    print(f"[tracked] {len(data)} frames -> {out_path} "
          f"({out_path.stat().st_size / 1e6:.1f} MB)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", type=Path, help="a runs/<name> dir containing masks.npz")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--video", type=Path, default=None,
                    help="source video, used when frames/ has been deleted")
    ap.add_argument("--alpha", type=float, default=0.65, help="mask fill opacity")
    ap.add_argument("--fps", type=float, default=None,
                    help="playback frame rate (default: the source video's, for real-time)")
    ap.add_argument("--trails", action="store_true", help="draw centroid motion trails")
    ap.add_argument("--no-trails", dest="trails", action="store_false",
                    help="override trails = true from a config file")
    ap.add_argument("--trail-len", type=int, default=60,
                    help="trail length in frames; 0 for unbounded")
    ap.add_argument("--no-labels", action="store_true")
    ap.add_argument("--outline", type=int, default=2, help="contour thickness, 0 to disable")
    cfgmod.add_config_arg(ap)
    cfgmod.apply_to(ap, cfgmod.load_config(cfgmod.preparse_config()), "render")
    args = ap.parse_args()

    # Name the output after the run (and so after the source video), not a
    # bare "tracked.mp4" -- these get moved and shared away from their dir.
    out = args.out or (args.run_dir / f"{args.run_dir.resolve().name}_tracked.mp4")
    render(args.run_dir, out, args.alpha, args.fps, args.trails,
           not args.no_labels, args.outline, args.trail_len, args.video)


if __name__ == "__main__":
    main()
