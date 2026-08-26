#!/usr/bin/env python
"""Track a long video with SAM 3 by chunking, stitching object ids across chunks.

A single SAM 3 session grows GPU memory with sequence length: on a 16 GB card
this video OOMs at roughly 500 frames even with --offload-video/--offload-state.
This driver runs overlapping chunks in separate sessions and reconciles ids by
mask IoU on the overlapping frames, where a subject has barely moved and the
correspondence is unambiguous.

The prompt is re-run at the start of every chunk, so subjects that were occluded
or off-floor earlier get picked up; new detections that match nothing existing
are given fresh ids.

Example:
    python track_long.py test_video.mp4 --prompt insect --stride 3 --out runs/full_s3
"""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

import track_video as tv


def chunk_bounds(total: int, size: int, overlap: int) -> list[tuple[int, int]]:
    """Contiguous chunks of `size` that share `overlap` frames with the previous."""
    step = size - overlap
    bounds, start = [], 0
    while start < total:
        end = min(start + size, total)
        bounds.append((start, end))
        if end == total:
            break
        start += step
    return bounds


def link_chunk(frames_dir: Path, chunk_dir: Path, lo: int, hi: int) -> None:
    """Symlink global frames lo..hi-1 as 0.jpg..N-1.jpg (the loader needs int names)."""
    if chunk_dir.exists():
        shutil.rmtree(chunk_dir)
    chunk_dir.mkdir(parents=True)
    for local, glob_idx in enumerate(range(lo, hi)):
        (chunk_dir / f"{local}.jpg").symlink_to((frames_dir / f"{glob_idx}.jpg").resolve())


def iou_matrix(prev: dict[int, np.ndarray], new: dict[int, np.ndarray]) -> np.ndarray:
    """Mean IoU between two id->mask-stack mappings accumulated over shared frames.

    `prev`/`new` map object id -> boolean mask summed across the overlap window,
    so a single IoU per pair covers the whole overlap rather than one frame.
    """
    prev_ids, new_ids = sorted(prev), sorted(new)
    mat = np.zeros((len(prev_ids), len(new_ids)), dtype=np.float32)
    for i, pid in enumerate(prev_ids):
        pm = prev[pid]
        for j, nid in enumerate(new_ids):
            nm = new[nid]
            union = np.logical_or(pm, nm).sum()
            mat[i, j] = (np.logical_and(pm, nm).sum() / union) if union else 0.0
    return mat


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video", type=Path)
    ap.add_argument("--prompt", default="insect")
    ap.add_argument("--out", type=Path, default=Path("runs/full"))
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--frames", type=int, default=10**9, help="frames to process")
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--version", default="sam3", choices=["sam3", "sam3.1"])
    ap.add_argument("--chunk", type=int, default=450, help="frames per session")
    ap.add_argument("--overlap", type=int, default=15, help="shared frames between chunks")
    ap.add_argument("--iou-thresh", type=float, default=0.3,
                    help="minimum overlap IoU to consider two tracks the same object")
    ap.add_argument("--prob-thresh", type=float, default=0.5)
    ap.add_argument("--reuse-frames", action="store_true",
                    help="skip extraction if <out>/frames is already populated")
    args = ap.parse_args()

    if args.overlap >= args.chunk:
        raise SystemExit("--overlap must be smaller than --chunk")

    args.out.mkdir(parents=True, exist_ok=True)
    frames_dir = args.out / "frames"

    existing = sorted(frames_dir.glob("*.jpg")) if frames_dir.is_dir() else []
    if args.reuse_frames and existing:
        import decord
        fps = float(decord.VideoReader(str(args.video)).get_avg_fps())
        n_frames = len(existing)
        src_indices = list(range(args.start, args.start + n_frames * args.stride, args.stride))
        print(f"[extract] reusing {n_frames} frames in {frames_dir}")
    else:
        frame_paths, src_indices, fps = tv.extract_clip(
            args.video, frames_dir, args.start, args.frames, args.stride)
        n_frames = len(frame_paths)

    from PIL import Image
    with Image.open(frames_dir / "0.jpg") as probe:
        width, height = probe.size

    bounds = chunk_bounds(n_frames, args.chunk, args.overlap)
    print(f"[plan] {n_frames} frames -> {len(bounds)} chunks of <={args.chunk} "
          f"(overlap {args.overlap})")

    t_model = time.time()
    predictor = tv.build_predictor(args.version)
    print(f"[track] model ready in {time.time() - t_model:.1f}s")

    csv_rows: list[list] = []
    packed: dict[str, np.ndarray] = {}
    written: set[int] = set()          # global frame indices already recorded
    next_gid = 0
    prev_union: dict[int, np.ndarray] = {}   # global id -> mask union over last overlap
    chunk_dir = args.out / "_chunk"
    t_all = time.time()

    for ci, (lo, hi) in enumerate(bounds):
        link_chunk(frames_dir, chunk_dir, lo, hi)
        session = predictor.handle_request({
            "type": "start_session",
            "resource_path": str(chunk_dir),
            "offload_video_to_cpu": True,
            "offload_state_to_cpu": True,
        })
        sid = session["session_id"]

        seed = predictor.handle_request({
            "type": "add_prompt", "session_id": sid, "frame_index": 0,
            "text": args.prompt, "output_prob_thresh": args.prob_thresh,
        })
        local_outputs = {seed["frame_index"]: seed["outputs"]}
        t0 = time.time()
        for out in predictor.handle_stream_request({
            "type": "propagate_in_video", "session_id": sid,
        }):
            local_outputs[out["frame_index"]] = out["outputs"]
        predictor.handle_request({"type": "close_session", "session_id": sid})

        # --- stitch this chunk's local ids onto global ids ---
        # Union each object's masks across the overlap window (the first
        # `overlap` frames of this chunk, which the previous chunk also covered).
        new_union: dict[int, np.ndarray] = {}
        if ci > 0:
            for local_f in range(min(args.overlap, hi - lo)):
                out = local_outputs.get(local_f)
                if out is None:
                    continue
                for i, oid in enumerate(np.asarray(out["out_obj_ids"]).tolist()):
                    m = np.asarray(out["out_binary_masks"][i]).astype(bool)
                    new_union[int(oid)] = np.logical_or(new_union[int(oid)], m) if int(oid) in new_union else m

        id_map: dict[int, int] = {}
        if ci > 0 and prev_union and new_union:
            prev_ids, new_ids = sorted(prev_union), sorted(new_union)
            mat = iou_matrix(prev_union, new_union)
            rows, cols = linear_sum_assignment(-mat)
            matched = 0
            for r, c in zip(rows, cols):
                if mat[r, c] >= args.iou_thresh:
                    id_map[new_ids[c]] = prev_ids[r]
                    matched += 1
            print(f"[chunk {ci}] matched {matched}/{len(new_ids)} tracks to previous "
                  f"(best IoU {mat.max():.2f})" if mat.size else f"[chunk {ci}] no overlap")
        for oid in sorted({int(o) for out in local_outputs.values()
                           for o in np.asarray(out["out_obj_ids"]).tolist()}):
            if oid not in id_map:
                id_map[oid] = next_gid
                next_gid += 1

        # --- record frames this chunk owns (skip ones an earlier chunk wrote) ---
        for local_f in sorted(local_outputs):
            gframe = lo + local_f
            if gframe in written:
                continue
            written.add(gframe)
            out = local_outputs[local_f]
            obj_ids = np.asarray(out["out_obj_ids"])
            masks = np.asarray(out["out_binary_masks"]).astype(bool)
            boxes = np.asarray(out.get("out_boxes_xywh", np.zeros((len(masks), 4))))
            probs = np.asarray(out.get("out_probs", np.ones(len(masks))))
            gids = np.array([id_map[int(o)] for o in obj_ids.tolist()], dtype=np.int64)

            packed[f"m{gframe}"] = np.packbits(masks, axis=-1)
            packed[f"i{gframe}"] = gids
            packed[f"b{gframe}"] = boxes
            packed[f"p{gframe}"] = probs

            src = src_indices[gframe] if gframe < len(src_indices) else -1
            for i, gid in enumerate(gids.tolist()):
                area = int(masks[i].sum())
                if area == 0:
                    continue
                ys, xs = np.nonzero(masks[i])
                csv_rows.append([gframe, src, round(src / fps, 6), gid,
                                 round(float(xs.mean()), 3), round(float(ys.mean()), 3),
                                 area, float(probs[i]), *np.asarray(boxes[i]).tolist()])

        # carry this chunk's tail forward as the next comparison window
        prev_union = {}
        for local_f in range(max(0, (hi - lo) - args.overlap), hi - lo):
            out = local_outputs.get(local_f)
            if out is None:
                continue
            for i, oid in enumerate(np.asarray(out["out_obj_ids"]).tolist()):
                gid = id_map[int(oid)]
                m = np.asarray(out["out_binary_masks"][i]).astype(bool)
                prev_union[gid] = np.logical_or(prev_union[gid], m) if gid in prev_union else m

        done = len(written)
        rate = (hi - lo) / max(time.time() - t0, 1e-9)
        eta = (n_frames - done) / max(rate, 1e-9)
        print(f"[chunk {ci + 1}/{len(bounds)}] frames {lo}..{hi - 1} "
              f"({rate:.1f} fps) | {done}/{n_frames} done | ETA {eta / 60:.1f} min "
              f"| {next_gid} ids so far", flush=True)

    shutil.rmtree(chunk_dir, ignore_errors=True)

    csv_path = args.out / "tracks.csv"
    with csv_path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["frame", "src_frame", "time_s", "obj_id", "cx", "cy",
                    "area_px", "score", "box_x", "box_y", "box_w", "box_h"])
        w.writerows(csv_rows)
    print(f"[export] {len(csv_rows)} rows, {next_gid} ids -> {csv_path}")

    npz_path = args.out / "masks.npz"
    np.savez_compressed(npz_path, height=height, width=width,
                        frames=np.array(sorted(written)), **packed)
    print(f"[masks] {len(written)} frames -> {npz_path} "
          f"({npz_path.stat().st_size / 1e6:.1f} MB)")
    print(f"[done] {n_frames} frames in {(time.time() - t_all) / 60:.1f} min")


if __name__ == "__main__":
    main()
