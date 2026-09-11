#!/usr/bin/env python
"""Track or render every video under a data directory, one at a time.

Replaces run_batch.sh and render_batch.sh, whose paths were hardcoded to one
machine. Videos are found recursively and the input tree is mirrored into the
output tree, so <data>/30_7_26/CS_T5.mp4 becomes <out>/30_7_26/CS_T5/ and a
flat folder of videos works just as well.

Sequential by design: a single run already saturates a 16 GB card, so two at
once would OOM. Resumable -- a video whose output exists is skipped, so the
script can be killed and restarted without losing finished work.

    uv run batch.py track              # run (or resume) the whole batch
    uv run batch.py track --dry-run    # print the plan, run nothing
    uv run batch.py render             # overlay video for every tracked run
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import config as cfgmod

HERE = Path(__file__).resolve().parent


# ---------------------------------------------------------------- discovery

def discover(data_dir: Path, extensions: list[str],
             exclude: Path | None = None) -> list[Path]:
    """Every video under data_dir, at any depth, sorted for a stable order.

    `exclude`, when given, is skipped entirely -- this is for out_dir when a
    user nests it inside data_dir. Without it, every rendered video becomes a
    new input on the next run, and each render adds more: the batch would
    track and re-render its own output forever.
    """
    if not data_dir.is_dir():
        raise SystemExit(f"[batch] no such data directory: {data_dir}")
    wanted = {e.lower() if e.startswith(".") else f".{e.lower()}" for e in extensions}
    exclude_r = exclude.resolve() if exclude is not None else None
    found = [p for p in data_dir.rglob("*")
             if p.is_file() and p.suffix.lower() in wanted
             and not (exclude_r is not None and p.resolve().is_relative_to(exclude_r))]
    return sorted(found)


def out_dir_for(video: Path, data_dir: Path, out_dir: Path) -> Path:
    """<data>/<rel>/<stem>.mp4 -> <out>/<rel>/<stem>/"""
    rel = video.relative_to(data_dir)
    return out_dir / rel.parent / rel.stem


# ---------------------------------------------------------------- plumbing

class Reporter:
    """Timestamped status lines, to the terminal and to a master log."""

    def __init__(self, log_path: Path):
        self.log_path = log_path
        log_path.parent.mkdir(parents=True, exist_ok=True)

    def say(self, msg: str) -> None:
        line = f"{datetime.now():%F %T} {msg}"
        print(line, flush=True)
        with self.log_path.open("a") as fh:
            fh.write(line + "\n")


def free_gb(path: Path) -> int:
    return shutil.disk_usage(path).free // (1024 ** 3)


def is_cuda_oom(log: Path) -> bool:
    """Did this video die of GPU memory rather than something else?"""
    if not log.is_file():
        return False
    tail = log.read_text(errors="replace").splitlines()[-40:]
    return any("torch.OutOfMemoryError" in line for line in tail)


def run_script(script: str, argv: list[str], log: Path) -> int:
    """Run a pipeline script, appending its output to `log`.

    Invoked as [sys.executable, <abs path>] rather than `uv run <name>` so the
    batch works from any working directory and does not require uv on PATH.
    """
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a") as fh:
        return subprocess.run([sys.executable, str(HERE / script), *argv],
                              stdout=fh, stderr=subprocess.STDOUT).returncode


def final_ids(run: Path) -> int | None:
    """Ids in a finished run, from run.json if present, else from tracks.csv."""
    meta = run / "run.json"
    if meta.is_file():
        try:
            return int(json.loads(meta.read_text())["final_ids"])
        except (ValueError, KeyError):
            pass
    csv_path = run / "tracks.csv"
    if not csv_path.is_file():
        return None
    lines = csv_path.read_text().splitlines()[1:]
    return len({line.split(",")[3] for line in lines}) if lines else 0


def summarise(rep: Reporter, rows: list[tuple[str, str, int, int | None]],
              expect_n: int | None) -> None:
    """The end-of-batch table: which recordings need a human to look at them."""
    rep.say("")
    rep.say(f"{'video':<44} {'status':<8} {'min':>4} {'ids':>4}")
    for name, status, mins, ids in rows:
        flag = ""
        if expect_n is not None and ids is not None and ids != expect_n:
            flag = f"  <- expected {expect_n}"
        rep.say(f"{name:<44} {status:<8} {mins:>4} {'' if ids is None else ids:>4}{flag}")
    bad = sum(1 for _, s, _, _ in rows if s != "ok")
    odd = sum(1 for _, _, _, i in rows
              if expect_n is not None and i is not None and i != expect_n)
    rep.say(f"[batch] {len(rows)} videos, {bad} failed, {odd} with an unexpected id count")


# ---------------------------------------------------------------- track

def cmd_track(args: argparse.Namespace) -> None:
    rep = Reporter(args.out_dir / "_logs" / "batch.log")
    videos = discover(args.data_dir, args.extensions, exclude=args.out_dir)
    rep.say(f"[batch] {len(videos)} videos, prompt={args.prompt} stride={args.stride} "
            f"chunk={args.chunk} overlap={args.overlap}")

    rows: list[tuple[str, str, int, int | None]] = []
    aborted = False
    for video in videos:
        out = out_dir_for(video, args.data_dir, args.out_dir)
        name = video.relative_to(args.data_dir).as_posix()
        log = args.out_dir / "_logs" / (name.replace("/", ".") + ".log")

        if (out / "tracks.csv").is_file():
            rep.say(f"[skip] {name} already has tracks.csv")
            rows.append((name, "skip", 0, final_ids(out)))
            continue

        # out_dir exists by now: Reporter created out_dir/_logs on construction.
        have = free_gb(args.out_dir)
        if have < args.min_free_gb:
            rep.say(f"[abort] only {have}G free, need {args.min_free_gb}G "
                    f"— stopping before {name}")
            aborted = True
            break

        if args.dry_run:
            rep.say(f"[dry-run] would track {name} -> {out}")
            continue

        rep.say(f"[start] {name} -> {out}  (log: {log})")
        started = time.time()
        argv = [str(video), "--out", str(out), "--prompt", args.prompt,
                "--stride", str(args.stride), "--overlap", str(args.overlap),
                "--version", args.version, "--prob-thresh", str(args.prob_thresh),
                "--iou-thresh", str(args.iou_thresh),
                "--relink-dist", str(args.relink_dist),
                "--relink-chunks", str(args.relink_chunks),
                "--reuse-frames"]
        if args.expect_n is not None:
            argv += ["--expect-n", str(args.expect_n)]
        status = run_script("track_long.py", argv + ["--chunk", str(args.chunk)], log)

        # GPU memory scales with objects x frames per session, so a video with
        # more subjects than usual can OOM on chunk 1 at a chunk size that suits
        # the rest of the batch (10 flies vs the usual 5-7 was enough to do it on
        # a 16 GB card). Halving the chunk trades a little speed for the run
        # finishing at all.
        #
        # Chunk size also selects which way identity stitching fails at a seam,
        # so do not tune it on speed alone. Larger chunks mean fewer seams but a
        # longer absence per missed one: a subject lost for a whole chunk returns
        # beyond IoU range and is handed a new id (a split -- one animal, two
        # ids). Smaller chunks double the seams, so unmatched detections show up
        # more often as short-lived stubs, but nothing is ever gone long enough
        # to split. Measured on this batch: 450 gave 1 split and 0 stubs, 225
        # gave 0 splits and 1-4 stubs per video. track_long.py
        # --relink-dist/--relink-chunks recovers the split case; nothing recovers
        # a stub, so prefer the smaller chunk and filter stubs by lifetime.
        if status != 0 and not (out / "tracks.csv").is_file() and is_cuda_oom(log):
            rep.say(f"[retry] {name} hit CUDA OOM at chunk {args.chunk} "
                    f"— retrying at {args.retry_chunk}")
            status = run_script("track_long.py",
                                argv + ["--chunk", str(args.retry_chunk)], log)

        mins = int((time.time() - started) / 60)
        if status == 0 and (out / "tracks.csv").is_file():
            shutil.rmtree(out / "frames", ignore_errors=True)   # ~26 GB for the
            # longest video; kept only on failure, where a rerun resumes the
            # extract via --reuse-frames.
            ids = final_ids(out)
            rep.say(f"[done] {name} in {mins}m — {ids} ids")
            rows.append((name, "ok", mins, ids))
        else:
            # Frames are deliberately left in place so a rerun resumes.
            rep.say(f"[FAIL] {name} exit={status} after {mins}m — see {log}")
            rows.append((name, "FAIL", mins, None))

    if not args.dry_run:
        summarise(rep, rows, args.expect_n)
    rep.say("[batch] finished")
    # Propagate nonzero exit if the batch was aborted early. The bash driver this
    # replaced exited 1 here, so a wrapper watching the exit code can tell a
    # truncated batch from a complete one.
    if aborted:
        raise SystemExit(1)


# ---------------------------------------------------------------- render

def cmd_render(args: argparse.Namespace) -> None:
    """Overlay video for every run that tracked successfully.

    A run counts as successful if it has masks.npz. cmd_track deletes frames/
    on success, so each render decodes its pictures back out of the source
    video (--video). Playback fps is left to render_colors.py, which infers the
    source rate from tracks.csv.
    """
    rep = Reporter(args.out_dir / "_logs" / "render.log")
    videos = discover(args.data_dir, args.extensions, exclude=args.out_dir)
    rep.say(f"[render] {len(videos)} videos, trail-len={args.trail_len}")

    rows: list[tuple[str, str, int, int | None]] = []
    aborted = False
    for video in videos:
        run = out_dir_for(video, args.data_dir, args.out_dir)
        name = video.relative_to(args.data_dir).as_posix()
        log = args.out_dir / "_logs" / (name.replace("/", ".") + ".render.log")
        out = run / f"{run.name}_tracked.mp4"

        if not (run / "masks.npz").is_file():
            rep.say(f"[skip] {name} has no masks.npz — not tracked yet")
            continue
        if out.is_file():
            rep.say(f"[skip] {name} already has {out.name}")
            rows.append((name, "skip", 0, final_ids(run)))
            continue

        have = free_gb(args.out_dir)
        if have < args.min_free_gb:
            rep.say(f"[abort] only {have}G free, need {args.min_free_gb}G "
                    f"— stopping before {name}")
            aborted = True
            break

        if args.dry_run:
            rep.say(f"[dry-run] would render {name} -> {out}")
            continue

        rep.say(f"[start] {name}  (log: {log})")
        started = time.time()
        argv = [str(run), "--video", str(video), "--out", str(out),
                "--trail-len", str(args.trail_len), "--alpha", str(args.alpha),
                "--outline", str(args.outline)]
        # Always pass trails explicitly. batch.py does not forward --config, and
        # render_colors.py independently loads ./segmentation.toml -- if that file
        # sets trails = true, an omitted flag is not "unset", it is the child's
        # own config-derived default, which is True. Only an explicit
        # --trails/--no-trails can override it.
        argv.append("--trails" if args.trails else "--no-trails")
        status = run_script("render_colors.py", argv, log)
        mins = int((time.time() - started) / 60)

        if status == 0 and out.is_file():
            mb = out.stat().st_size / 1e6
            rep.say(f"[done] {name} in {mins}m — {mb:.0f} MB")
            rows.append((name, "ok", mins, final_ids(run)))
        else:
            # Drop the half-written output so a rerun does not skip it.
            out.unlink(missing_ok=True)
            (run / f".{run.name}_tracked.raw.mp4").unlink(missing_ok=True)
            rep.say(f"[FAIL] {name} exit={status} after {mins}m — see {log}")
            rows.append((name, "FAIL", mins, None))

    if not args.dry_run:
        summarise(rep, rows, args.expect_n)
    rep.say("[render] finished")
    # Propagate nonzero exit if the render was aborted early, mirroring
    # cmd_track: the bash driver this replaced exited 1 here.
    if aborted:
        raise SystemExit(1)


# ---------------------------------------------------------------- main

def build_parser() -> tuple[argparse.ArgumentParser,
                            dict[str, argparse.ArgumentParser]]:
    """-> (root parser, {subcommand name: its parser}).

    The subparsers come back because config defaults have to be applied to the
    subparser that declares the flag, not to the root.
    """
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    cfgmod.add_config_arg(ap)
    sub = ap.add_subparsers(dest="command", required=True)

    def common(p: argparse.ArgumentParser) -> None:
        # Declared here too (not via add_config_arg) so --config also works after
        # the subcommand, which is where most people reach for it. default must be
        # SUPPRESS, not None: argparse copies every subparser attribute onto the
        # main namespace, and a plain None default would clobber a value already
        # set by --config on the root parser. preparse_config() scans the whole
        # argv regardless of position, so the file is found either way -- this is
        # only about argparse accepting and advertising the flag here too.
        p.add_argument("--config", type=Path, default=argparse.SUPPRESS,
                       help="TOML config file (also accepted before the subcommand)")
        p.add_argument("--data-dir", type=Path, default=None,
                       help="root searched recursively for videos")
        p.add_argument("--out-dir", type=Path, default=None,
                       help="root the input tree is mirrored into")
        p.add_argument("--extensions", nargs="+", default=[".mp4"],
                       help="video extensions to look for")
        p.add_argument("--expect-n", type=int, default=None,
                       help="animals per recording; flags runs whose id count differs")
        p.add_argument("--dry-run", action="store_true",
                       help="print the plan, run nothing")

    t = sub.add_parser("track", help="track every video")
    common(t)
    t.add_argument("--prompt", default="insect")
    t.add_argument("--stride", type=int, default=1)
    t.add_argument("--chunk", type=int, default=450)
    t.add_argument("--overlap", type=int, default=15)
    t.add_argument("--version", default="sam3", choices=["sam3", "sam3.1"])
    t.add_argument("--prob-thresh", type=float, default=0.5)
    t.add_argument("--iou-thresh", type=float, default=0.3)
    t.add_argument("--relink-dist", type=float, default=200.0)
    t.add_argument("--relink-chunks", type=int, default=3)
    t.add_argument("--retry-chunk", type=int, default=225,
                   help="second attempt for a video that OOMs the GPU at --chunk")
    t.add_argument("--min-free-gb", type=int, default=40,
                   help="frames for the longest video need ~26 GB")
    t.set_defaults(func=cmd_track)

    r = sub.add_parser("render", help="render an overlay video for every run")
    common(r)
    r.add_argument("--trails", action="store_true", help="draw centroid motion trails")
    r.add_argument("--no-trails", dest="trails", action="store_false",
                   help="override trails = true from a config file")
    r.add_argument("--trail-len", type=int, default=20,
                   help="trail length in frames; 0 for unbounded")
    r.add_argument("--alpha", type=float, default=0.65, help="mask fill opacity")
    r.add_argument("--outline", type=int, default=2,
                   help="contour thickness, 0 to disable")
    r.add_argument("--min-free-gb", type=int, default=20,
                   help="the mp4v intermediate is transient but not small")
    r.set_defaults(func=cmd_render)
    return ap, {"track": t, "render": r}


def main() -> None:
    ap, subparsers = build_parser()
    cfg = cfgmod.load_config(cfgmod.preparse_config())
    for name, parser in subparsers.items():
        cfgmod.apply_to(parser, cfg, "paths", "subjects", "track", "render",
                        f"batch.{name}")
    args = ap.parse_args()

    for required in ("data_dir", "out_dir"):
        if getattr(args, required) is None:
            raise SystemExit(f"[batch] --{required.replace('_', '-')} is required "
                             f"(set paths.{required} in {cfgmod.CONFIG_NAME}, "
                             f"or pass the flag)")
    args.func(args)


if __name__ == "__main__":
    main()
