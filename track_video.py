#!/usr/bin/env python
"""Track objects in a video with SAM 3 and export per-frame centroids.

Stages:
  1. extract  decode a clip from the source video into <out>/frames/
  2. track    run the SAM 3.1 video predictor with a text prompt
  3. export   per-frame, per-object centroids/areas -> <out>/tracks.csv
  4. render   annotated MP4 -> <out>/<out-name>_overlay.mp4

Written against a Drosophila arena assay (1120x1120 grayscale, 60 fps), but
nothing here is species-specific -- --prompt drives what gets segmented.

Example:
    python track_video.py test_video.mp4 --prompt fly --frames 300 --out runs/smoke
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import inspect
import os
import sys

# Set before torch loads: cuts fragmentation, which is what tips a
# borderline clip into OutOfMemoryError on a 16 GB card.
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
import time
from pathlib import Path

import numpy as np

import config as cfgmod


# ---------------------------------------------------------------- stage 1

def extract_clip(video_path: Path, frames_dir: Path, start: int, count: int,
                 stride: int) -> tuple[list[Path], list[int], float]:
    """Decode [start, start + count*stride) into frames_dir as 0.jpg, 1.jpg, ...

    The SAM 3 image-folder loader sorts on int(stem), so the names must be bare
    integers or it silently falls back to a lexicographic sort (0, 1, 10, 100...).
    """
    import decord
    from PIL import Image

    vr = decord.VideoReader(str(video_path))
    fps = float(vr.get_avg_fps())
    src_indices = list(range(start, min(start + count * stride, len(vr)), stride))
    if not src_indices:
        raise SystemExit(f"no frames selected: video has {len(vr)}, asked for start={start}")

    frames_dir.mkdir(parents=True, exist_ok=True)
    for stale in frames_dir.glob("*.jpg"):
        stale.unlink()

    paths = []
    # Batch the decode; decord seeks fine but batching is much faster.
    for chunk_start in range(0, len(src_indices), 64):
        chunk = src_indices[chunk_start:chunk_start + 64]
        batch = vr.get_batch(chunk).asnumpy()
        for offset, arr in enumerate(batch):
            idx = chunk_start + offset
            path = frames_dir / f"{idx}.jpg"
            Image.fromarray(arr).save(path, quality=95)
            paths.append(path)

    print(f"[extract] {len(paths)} frames -> {frames_dir} "
          f"(src {src_indices[0]}..{src_indices[-1]} stride {stride}, {fps:.2f} fps)")
    return paths, src_indices, fps


# ---------------------------------------------------------------- stage 2

def build_predictor(version: str, max_num_objects: int = 16):
    """Build a SAM 3 predictor and patch over two upstream rough edges."""
    from sam3 import model_builder
    from sam3.model_builder import build_sam3_predictor

    # The builders locate the BPE vocab with pkg_resources.resource_filename("sam3", ...),
    # which blows up with "expected str ... not NoneType" whenever the process is run
    # from a directory containing a `sam3/` folder (i.e. the repo clone): Python then
    # imports `sam3` as a namespace package whose __file__ is None. Deriving the asset
    # path from the module file instead is CWD-independent.
    bpe_path = Path(model_builder.__file__).parent / "assets" / "bpe_simple_vocab_16e6.txt.gz"
    if not bpe_path.is_file():
        raise SystemExit(f"BPE vocab not found at {bpe_path}")

    # use_fa3 defaults to True, but Flash Attention 3 needs a Hopper-class GPU.
    # On Ada (sm_89) and earlier this must be off or the build fails / falls over.
    build_kwargs = dict(version=version, use_fa3=False, bpe_path=str(bpe_path))
    if version == "sam3.1":
        # max_num_objects is a runtime cap and is safe to lower.
        # multiplex_count is NOT exposed: it is baked into the checkpoint
        # (no_obj_embed_spatial, iou_token, ... are all [16, 256]), so any
        # value but 16 fails with a state_dict size mismatch.
        build_kwargs.update(max_num_objects=max_num_objects)
    predictor = build_sam3_predictor(**build_kwargs)

    # Upstream bug: the shared start_session always forwards offload_state_to_cpu,
    # but the SAM 3.1 multiplex tracker's init_state does not accept it -- so
    # start_session raises TypeError for version="sam3.1" no matter what is passed.
    # (The plain "sam3" tracker does accept it.) Drop kwargs it can't take.
    init_params = inspect.signature(predictor.model.init_state).parameters
    unsupported = [k for k in ("offload_state_to_cpu", "offload_video_to_cpu")
                   if k not in init_params]
    if unsupported:
        inner_init_state = predictor.model.init_state

        def init_state_compat(*args, **kwargs):
            for key in unsupported:
                kwargs.pop(key, None)
            return inner_init_state(*args, **kwargs)

        predictor.model.init_state = init_state_compat
        print(f"[track] note: {version!r} ignores {unsupported}", file=sys.stderr)
    return predictor


def track(frames_dir: Path, prompt: str, version: str, offload_video: bool,
          offload_state: bool, prob_thresh: float,
          max_num_objects: int) -> dict[int, dict]:
    """Run SAM 3 video propagation. Returns {frame_index: outputs}."""
    t0 = time.time()
    predictor = build_predictor(version, max_num_objects)
    print(f"[track] model ready in {time.time() - t0:.1f}s")

    session = predictor.handle_request({
        "type": "start_session",
        "resource_path": str(frames_dir),
        "offload_video_to_cpu": offload_video,
        "offload_state_to_cpu": offload_state,
    })
    sid = session["session_id"]

    outputs: dict[int, dict] = {}
    first = predictor.handle_request({
        "type": "add_prompt",
        "session_id": sid,
        "frame_index": 0,
        "text": prompt,
        "output_prob_thresh": prob_thresh,
    })
    outputs[first["frame_index"]] = first["outputs"]
    n_seed = len(first["outputs"]["out_obj_ids"])
    print(f"[track] prompt {prompt!r} matched {n_seed} object(s) on frame 0 "
          f"(thresh {prob_thresh})")
    if n_seed == 0:
        print("[track] WARNING: nothing matched the prompt; propagation will be empty.",
              file=sys.stderr)

    t0 = time.time()
    for out in predictor.handle_stream_request({
        "type": "propagate_in_video",
        "session_id": sid,
    }):
        outputs[out["frame_index"]] = out["outputs"]

    n = len(outputs)
    print(f"[track] propagated {n} frames in {time.time() - t0:.1f}s "
          f"({n / max(time.time() - t0, 1e-9):.1f} fps)")

    with contextlib.suppress(Exception):
        predictor.handle_request({"type": "close_session", "session_id": sid})
    return outputs


# ---------------------------------------------------------------- mask persistence

# masks.npz layout version. 1 packed whole frames; 2 packs each object cropped
# to its bounding box and carries a c{frame} array of (y0, x0, h, w) alongside.
# Readers must keep handling 1: runs tracked before the change are still v1.
MASK_FORMAT = 2


def mask_bboxes(masks: np.ndarray) -> np.ndarray:
    """Tight (y0, x0, h, w) per mask; all-zero for an empty one."""
    crops = np.zeros((len(masks), 4), dtype=np.int32)
    for i, m in enumerate(masks):
        ys, xs = np.nonzero(m)
        if len(ys):
            y0, x0 = int(ys.min()), int(xs.min())
            crops[i] = (y0, x0, int(ys.max()) - y0 + 1, int(xs.max()) - x0 + 1)
    return crops


def pack_cropped(masks: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Bit-pack masks cropped to their bounding boxes -> (packed, crops).

    A full-frame bool mask packs to ~160 KB per object per frame however small
    the subject is, so a long run piles up tens of GB before it ever reaches
    savez -- enough to get a 57k-frame run OOM-killed. Cropping first makes the
    cost proportional to the subject rather than to the arena.

    Objects are padded to the largest box in the frame so the result stays
    rectangular; each object's true extent is its own row in `crops`.
    """
    crops = mask_bboxes(masks)
    mh, mw = (int(crops[:, 2].max()), int(crops[:, 3].max())) if len(crops) else (0, 0)
    tiles = np.zeros((len(masks), mh, mw), dtype=bool)
    for i, (y0, x0, h, w) in enumerate(crops.tolist()):
        if h:
            tiles[i, :h, :w] = masks[i][y0:y0 + h, x0:x0 + w]
    return np.packbits(tiles, axis=-1), crops


def save_masks(outputs: dict[int, dict], path: Path, frame_shape: tuple[int, int]) -> None:
    """Persist per-frame masks so re-rendering never requires re-tracking."""
    height, width = frame_shape
    arrays: dict[str, np.ndarray] = {}
    for fidx, out in outputs.items():
        masks = np.asarray(out["out_binary_masks"]).astype(bool)
        arrays[f"m{fidx}"], arrays[f"c{fidx}"] = pack_cropped(masks)
        arrays[f"i{fidx}"] = np.asarray(out["out_obj_ids"])
        arrays[f"b{fidx}"] = np.asarray(out.get("out_boxes_xywh", np.zeros((len(masks), 4))))
        arrays[f"p{fidx}"] = np.asarray(out.get("out_probs", np.ones(len(masks))))
    np.savez_compressed(path, height=height, width=width, format=MASK_FORMAT,
                        frames=np.array(sorted(outputs)), **arrays)
    size_mb = path.stat().st_size / 1e6
    print(f"[masks] {len(outputs)} frames -> {path} ({size_mb:.1f} MB)")


# ---------------------------------------------------------------- stage 3

def export_tracks(outputs: dict[int, dict], src_indices: list[int], fps: float,
                  frame_shape: tuple[int, int], csv_path: Path) -> int:
    """Per-frame, per-object centroid + area -> tidy CSV."""
    height, width = frame_shape
    checked_shape = False
    rows = 0

    with csv_path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["frame", "src_frame", "time_s", "obj_id",
                         "cx", "cy", "area_px", "score",
                         "box_x", "box_y", "box_w", "box_h"])

        for fidx in sorted(outputs):
            out = outputs[fidx]
            masks = np.asarray(out["out_binary_masks"])
            obj_ids = np.asarray(out["out_obj_ids"])
            boxes = out.get("out_boxes_xywh")
            probs = out.get("out_probs")

            if masks.size and not checked_shape:
                # Masks are documented to come back at orig_height/orig_width.
                # Assert rather than assume -- if that ever changes, centroids
                # would be silently wrong in model-resolution pixels.
                if masks.shape[-2:] != (height, width):
                    raise SystemExit(
                        f"mask resolution {masks.shape[-2:]} != frame {(height, width)}; "
                        "centroids would need rescaling -- refusing to write bad data."
                    )
                checked_shape = True

            src = src_indices[fidx] if fidx < len(src_indices) else -1
            for i, obj_id in enumerate(np.asarray(obj_ids).tolist()):
                mask = np.asarray(masks[i]).astype(bool)
                area = int(mask.sum())
                if area == 0:
                    continue  # object not visible this frame
                ys, xs = np.nonzero(mask)
                box = np.asarray(boxes[i]).tolist() if boxes is not None else [""] * 4
                score = float(np.asarray(probs)[i]) if probs is not None else ""
                writer.writerow([fidx, src, round(src / fps, 6), int(obj_id),
                                 round(float(xs.mean()), 3), round(float(ys.mean()), 3),
                                 area, score, *box])
                rows += 1

    ids = sorted({int(o) for out in outputs.values()
                  for o in np.asarray(out["out_obj_ids"]).tolist()})
    print(f"[export] {rows} rows, {len(ids)} object ids {ids} -> {csv_path}")
    return rows


# ---------------------------------------------------------------- stage 4

def render(frame_paths: list[Path], outputs: dict[int, dict], out_path: Path,
           fps: float) -> None:
    """Annotated MP4 via the repo's own renderer."""
    from sam3.visualization_utils import save_masklet_video

    # render_masklet_frame indexes out_probs unconditionally; the tracker does
    # not always supply it. Fill with 1.0 so rendering can't crash the run after
    # the expensive part is already done.
    for out in outputs.values():
        if "out_probs" not in out:
            out["out_probs"] = np.ones(len(out["out_obj_ids"]), dtype=np.float32)

    # save_masklet_video writes a scratch "temp.mp4" into the *current* directory
    # before re-encoding, so run it from the output dir to avoid littering.
    # Resolve *before* the chdir: Path.resolve() is relative to the current
    # directory, and load_frame() rejects a non-existent path with a misleading
    # "invalid frame type" error rather than a missing-file one.
    abs_frames = [str(p.resolve()) for p in frame_paths]
    out_name = out_path.name
    out_dir = out_path.parent.resolve()

    cwd = Path.cwd()
    try:
        os.chdir(out_dir)
        save_masklet_video(abs_frames, outputs, out_name, fps=fps)
    finally:
        os.chdir(cwd)
    print(f"[render] {out_path}")


# ---------------------------------------------------------------- plumbing test

def threshold_outputs(frame_paths: list[Path], max_objects: int,
                      min_area: int) -> dict[int, dict]:
    """Crude dark-blob segmentation, for exercising stages 3-4 without the model.

    This is NOT a tracker: ids are per-frame component ranks and will swap
    between frames. It exists only to prove the export/render plumbing works
    while the SAM 3 checkpoint is gated.
    """
    from PIL import Image
    from skimage.measure import label, regionprops

    outputs = {}
    for fidx, path in enumerate(frame_paths):
        gray = np.asarray(Image.open(path).convert("L"))
        # Arena is backlit: subjects are dark blobs well below the median.
        binary = gray < (np.median(gray) * 0.55)
        regions = [r for r in regionprops(label(binary)) if r.area >= min_area]
        regions.sort(key=lambda r: r.area, reverse=True)
        regions = regions[:max_objects]

        masks = np.zeros((len(regions), *gray.shape), dtype=bool)
        boxes = np.zeros((len(regions), 4), dtype=np.float32)
        for i, region in enumerate(regions):
            masks[i][tuple(region.coords.T)] = True
            minr, minc, maxr, maxc = region.bbox
            boxes[i] = [minc, minr, maxc - minc, maxr - minr]

        outputs[fidx] = {
            "out_obj_ids": np.arange(len(regions), dtype=np.int64),
            "out_binary_masks": masks,
            "out_boxes_xywh": boxes,
            "out_probs": np.ones(len(regions), dtype=np.float32),
        }
    print(f"[threshold] plumbing-only segmentation over {len(frame_paths)} frames")
    return outputs


# ---------------------------------------------------------------- main

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video", type=Path)
    ap.add_argument("--prompt", default="fly", help="open-vocabulary text prompt")
    ap.add_argument("--out", type=Path, default=Path("runs/smoke"))
    ap.add_argument("--start", type=int, default=0, help="first source frame")
    ap.add_argument("--frames", type=int, default=300, help="frames to process")
    ap.add_argument("--stride", type=int, default=1,
                    help="take every Nth source frame (2 = halve the frame rate)")
    ap.add_argument("--version", default="sam3.1", choices=["sam3", "sam3.1"])
    ap.add_argument("--offload-video", action="store_true",
                    help="keep the frame tensor on CPU (needed for long clips)")
    ap.add_argument("--offload-state", action="store_true",
                    help="keep tracker state on CPU (slower, much less VRAM)")
    ap.add_argument("--prob-thresh", type=float, default=0.5,
                    help="detection confidence threshold for the seeding prompt")
    ap.add_argument("--max-num-objects", type=int, default=16,
                    help="sam3.1 tracker capacity; lower saves VRAM")
    ap.add_argument("--save-masks", action="store_true",
                    help="write masks.npz so the video can be re-rendered without re-tracking")
    ap.add_argument("--no-render", action="store_true")
    ap.add_argument("--no-model", action="store_true",
                    help="skip SAM 3; use threshold blobs to test stages 1/3/4")
    ap.add_argument("--max-objects", type=int, default=16,
                    help="--no-model only: cap on blobs per frame")
    ap.add_argument("--min-area", type=int, default=150,
                    help="--no-model only: minimum blob area in px")
    cfgmod.add_config_arg(ap)
    cfgmod.apply_to(ap, cfgmod.load_config(cfgmod.preparse_config()), "track")
    args = ap.parse_args()

    if not args.video.is_file():
        raise SystemExit(f"no such video: {args.video}")

    args.out.mkdir(parents=True, exist_ok=True)
    frames_dir = args.out / "frames"

    frame_paths, src_indices, fps = extract_clip(
        args.video, frames_dir, args.start, args.frames, args.stride)
    out_fps = fps / args.stride

    from PIL import Image
    with Image.open(frame_paths[0]) as probe:
        width, height = probe.size

    if args.no_model:
        outputs = threshold_outputs(frame_paths, args.max_objects, args.min_area)
    else:
        outputs = track(frames_dir, args.prompt, args.version,
                        args.offload_video, args.offload_state,
                        args.prob_thresh, args.max_num_objects)

    export_tracks(outputs, src_indices, fps, (height, width), args.out / "tracks.csv")

    if args.save_masks:
        save_masks(outputs, args.out / "masks.npz", (height, width))

    if not args.no_render:
        # Named after the run rather than a bare "overlay.mp4": these get
        # moved and shared away from the directory that identifies them.
        # Kept as _overlay, not _tracked: render_colors.py writes
        # <run>_tracked.mp4 into this same directory from masks.npz.
        render(frame_paths, outputs,
               args.out / f"{args.out.resolve().name}_overlay.mp4", out_fps)


if __name__ == "__main__":
    main()
