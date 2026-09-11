# Making the segmentation pipeline shippable

**Date:** 2026-09-11
**Status:** approved design, not yet implemented

## Problem

The pipeline works, but only on the machine it was written on. Three things
block anyone else from using it:

1. `run_batch.sh` and `render_batch.sh` hardcode `/home/kyle/projects/segmentation`
   and `/home/kyle/projects/test_data`, and assume videos sit exactly two
   directories deep.
2. Tracking parameters that a new recording would need to change (prompt,
   stride, chunk size) are shell constants inside those scripts.
3. There is no way to say how many animals a recording contains, so nothing
   can tell a good run from one that lost a fly or invented an extra id.

The Python scripts themselves are in reasonable shape — `--out`, `--prompt`
and `--stride` already exist. What they lack is a way to avoid retyping eight
flags per invocation, which is how the constants ended up baked into a shell
script in the first place.

## Audience and scope

The target user is a lab collaborator on a comparable Linux + NVIDIA setup who
clones the repo and runs `uv run`. That rules several things out of scope:
PyPI publication, console-script entry points, CI, and support for hardware the
pipeline has never seen.

Also out of scope, deliberately:

- `make_figures.py` and `build_report.py` stay gitignored as local validation.
- LICENSE choice is deferred to a later decision.
- No change to the tracking algorithm, the model, or the output formats.

## Design

### 1. Config layer (`config.py`)

A new top-level `config.py`, a committed `segmentation.example.toml`, and a
gitignored `segmentation.toml`. Parsing uses stdlib `tomllib`; no new
dependency.

**Config keys are spelled exactly like the flags they back.** `stride` backs
`--stride`, `chunk` backs `--chunk`. There is no second vocabulary and no
per-script translation table, so `--help` remains the only reference a user
needs.

```toml
[paths]
data_dir = "~/recordings"      # searched recursively for videos
out_dir  = "runs"              # relative paths resolve against THIS FILE's dir
extensions = [".mp4", ".avi", ".mov"]

[subjects]
expected = 7                   # animals per video; omit to disable the checks
prompt   = "insect"

[track]
version = "sam3"
stride = 1
chunk = 450
overlap = 15
prob_thresh = 0.5
iou_thresh = 0.3
relink_dist = 200.0
relink_chunks = 3

[render]
trails = true
trail_len = 20
alpha = 0.65
outline = 2

[filter]
min_fraction = 0.5

[batch.track]
retry_chunk = 225
min_free_gb = 40     # the longest video needs ~26 GB of extracted frames

[batch.render]
min_free_gb = 20     # the mp4v intermediate is transient but not small
```

Public API:

```python
def load_config(path: Path | None = None) -> dict
def apply_to(parser: argparse.ArgumentParser, cfg: dict, *sections: str) -> None
```

**Resolution order for the config file:** `--config PATH`, then
`./segmentation.toml`, then none. A missing config file is not an error; it
simply means every default is the built-in one. `--config PATH` naming a file
that does not exist *is* an error.

**Precedence is CLI flag > config file > built-in default.** Implemented via
`parser.set_defaults(**section)` after a small `--config`-only pre-parser
(`parse_known_args`, then the real parser declared with `parents=[pre]`). Using
argparse's own mechanism means `--help` prints the effective default rather
than the hardcoded one.

**Relative paths resolve against the config file's own directory**, not the
process CWD. `~` is expanded. Without this a config file only works when
invoked from one directory.

**Unknown keys in a known section raise `SystemExit` with the offending key
and the section's valid keys.** Silently ignoring typos is the primary way
config files waste an afternoon. Unknown *sections* are likewise an error.

`subjects.prompt` is read by both the track and filter paths, so
`apply_to` accepts several section names and applies them in order.

No packaging change is required for `import config` to work: Python places a
script's own directory on `sys.path[0]`, so the existing flat layout already
resolves sibling imports from any CWD. This is the same mechanism that makes
`track_long.py`'s existing `import track_video` work today.

### 2. Expected subject count

A new `--expect-n` flag, backed by `subjects.expected`, defaulting to `None`
(all checks disabled). It does not touch the model: SAM 3's `max_num_objects`
only binds on the `sam3.1` path, which does not fit in 16 GB at this
resolution, so a flag claiming to cap object count would be a no-op that lies.
`--expect-n` is a post-hoc expectation instead. Three consumers:

**`track_long.py`** — after each chunk, if the live object count differs from
N, print `[warn] chunk 3: 9 objects, expected 7`. Non-fatal; a chunk that
disagrees is information, not a failure.

**`filter_tracks.py`** — a new selection mode that keeps the N longest-lived
ids and drops the rest, complementing the existing lifetime cutoff rather than
replacing it. When both `--expect-n` and `--min-fraction` are given,
`--expect-n` decides which ids to keep and `--min-fraction` becomes a sanity
check: if the Nth-longest track falls below the cutoff, warn. That combination
is the signal that a real animal was lost, not that an artefact was found.

**`batch.py`** — the end-of-batch summary flags every video whose final id
count differs from N. On a multi-video batch this is the practical deliverable:
which recordings need a human to look at them.

### 3. Run provenance (`run.json`)

`track_long.py` writes one JSON file into each run directory:

```json
{
  "video": "/data/30_7_26/CS_T5.mp4",
  "created": "2026-09-11T14:02:11",
  "prompt": "insect", "version": "sam3",
  "stride": 1, "chunk": 450, "overlap": 15,
  "prob_thresh": 0.5, "iou_thresh": 0.3,
  "expected": 7,
  "frames": 12500, "chunks": 28,
  "objects_per_chunk": [7, 7, 9, 7],
  "final_ids": 8
}
```

The config layer is what makes this necessary. Today every parameter is typed
on the command line, so a run is recoverable from shell history. Once defaults
come from a file that may since have been edited, `uv run track_long.py
video.mp4` no longer records what it did. `run.json` closes that gap and makes
a run interpretable by a collaborator whose config differs.

Scope is fixed at "settings actually used, plus object counts". It is
explicitly not a general metadata system: no schema versioning, no query tool.

### 4. Batch driver (`batch.py`)

`run_batch.sh` and `render_batch.sh` are replaced by one `batch.py` with
`track` and `render` subcommands. The two shell scripts already duplicate their
logger, free-space guard, resumability check and dry-run handling; recursive
discovery with path mirroring would have become a third duplicated piece.

The port is mechanical. These behaviours are preserved exactly:

- Sequential execution — one run already saturates a 16 GB card.
- Resumability — a video whose `tracks.csv` (or `<stem>_tracked.mp4`) exists is
  skipped, so the batch can be killed and restarted.
- CUDA OOM retry at `batch.track.retry_chunk`, detected by scanning the tail of the
  video's log for `torch.OutOfMemoryError`.
- Free-space guard before each video at the per-subcommand threshold
  (`batch.track.min_free_gb` = 40, `batch.render.min_free_gb` = 20 — the two
  shell scripts used different values and that distinction is preserved),
  aborting the batch rather than failing
  mid-write.
- `frames/` deleted on success, left in place on failure so a rerun resumes the
  extract via `--reuse-frames`.
- Per-video logs plus a master log, with timestamped status lines.
- `DRY_RUN=1` becomes `--dry-run`.

The comments explaining *why* — the OOM retry, and the chunk-size tuning note
about splits versus stubs — carry over verbatim. They record measurements that
cannot be recovered by reading the code.

Changed behaviour:

- Data dir, output dir, and every tracking parameter come from config or CLI.
- **Discovery is recursive.** Videos are found by `rglob` over
  `paths.extensions` at any depth under `data_dir`. A flat folder of videos
  works; the existing `<day>/<stem>.mp4` layout keeps working unchanged.
- **Output mirrors input structure:** a video at `<data_dir>/<rel>/<stem>.mp4`
  produces `<out_dir>/<rel>/<stem>/`.
- No `cd` into a hardcoded project root. Child scripts are invoked as
  `[sys.executable, str(Path(__file__).parent / "track_long.py"), ...]`, which
  is CWD-independent and does not assume `uv` is on PATH.
- A final summary table: video, status, elapsed minutes, final id count, and
  whether it matched `--expect-n`.

### 5. Bug fixed in passing

`filter_tracks.py --backup` writes originals to `backup / run.name`, using only
the leaf directory name. In a mirrored `<day>/<stem>/` tree, two recordings
sharing a stem on different days overwrite each other's backup, silently, after
the originals have been modified in place. It must mirror the relative path the
same way `batch.py` does.

### 6. Documentation

README restructured to: Setup → Configure → Track one video → Run a batch →
Outputs → Adapting to your own recordings → Things that cost time to discover
→ Validation.

- A new "Configure" section covering the config file, resolution order, and
  precedence.
- A new "Adapting to your own recordings" section: prompt wording, expected
  count, stride, and the chunk-size/VRAM relationship.
- `filter_tracks.py` documented as a pipeline step (track → filter → upsample
  → render) and added to the script table.
- Examples use a neutral path rather than `test_video.mp4`.
- "Things that cost time to discover" is preserved intact. It is the repo's
  most valuable section and none of it is affected by this work.

`.gitignore` gains `segmentation.toml`.

## Testing

The repo currently has no tests. The GPU path cannot be exercised routinely, so
coverage is on pure logic only — stated plainly rather than implying more:

| Test | Covers |
|---|---|
| Precedence: CLI beats file beats default | The single most breakable property of the config layer |
| Relative path resolves against config dir | Config usable from any CWD |
| `~` expansion in `data_dir` | Common in a hand-written config |
| Unknown key / unknown section rejected | Typo detection |
| Missing config file is fine; missing `--config` target is an error | Resolution order |
| Discovery + mirroring over a temp tree | Flat, nested, and mixed-extension layouts |
| `--expect-n` keeps the N longest-lived ids | Synthetic `tracks.csv`, no GPU |
| `--expect-n` with `--min-fraction` warns below cutoff | The lost-animal signal |
| `filter_tracks --backup` mirrors relative paths | The collision bug above |
| `--help` exits 0 on every script | Catches config wiring breaking argparse |

`pytest` is added to the `dev` extra. GPU-dependent code — model building,
tracking, mask export — remains untested, and the README says so.

## Risks

**Porting 250 lines of tuned bash.** The OOM retry and free-space guard encode
real operational experience. Mitigation: mechanical translation preserving
comments, and a `--dry-run` comparison against the shell scripts' plan on the
existing `runs/batch` tree before the shell scripts are deleted.

**A config file can hide what a run did.** Mitigated by `run.json`.

**Silent behaviour change from new defaults.** The example config ships the
values currently hardcoded in the shell scripts, so an existing batch behaves
identically.
