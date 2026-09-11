# Fly arena segmentation with SAM 3

Segment and track individual flies in a backlit *Drosophila* arena recording using
Meta's [Segment Anything 3](https://github.com/facebookresearch/sam3), and export
per-frame centroid trajectories for behavioural analysis.

Built against a 1120×1120 grayscale, 60 fps arena assay with ~7 flies, but nothing
is species-specific — `--prompt` drives what gets segmented.

**! code was written with help from claude**

## Setup from scratch

Written for Linux with an NVIDIA GPU. Other platforms are untested: there is no
CUDA path on Apple silicon, and nothing here has been run under WSL.

Budget about **11 GB of disk** — ~7.7 GB for the environment and ~3.3 GB for the
model checkpoints — and expect the Hugging Face access request in step 4 to take
anywhere from minutes to a day to be approved.

### 1. System prerequisites

```bash
nvidia-smi          # should print your GPU and driver; if not, install the driver first
ffmpeg -version     # needed by render_colors.py, which shells out to it
git --version
```

You need a **16 GB GPU or larger** at this resolution, CUDA **12.6+**, and
`ffmpeg` on `PATH`. `ffmpeg` is a system package (`apt install ffmpeg`), not a
Python dependency, so `uv sync` will not supply it — without it, tracking
succeeds and only rendering fails.

Do not install a system Python for this. Triton JIT-compiles a CUDA helper at
runtime and needs `Python.h`; the uv-managed interpreter in step 2 ships the
headers, while a bare `python3.12` needs `python3.12-dev`.

### 2. Install uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then restart your shell, or `source $HOME/.local/bin/env`.

### 3. Clone and install

```bash
git clone https://github.com/kylethieringer/segmentation.git
cd segmentation
uv sync                  # the pipeline
uv sync --extra dev      # ...plus pytest and jupyter, for the tests and sam3's notebooks
```

This resolves the Python interpreter, torch, and SAM 3 from `uv.lock`. SAM 3 is
installed from a pinned commit rather than vendored, so no manual clone is
needed. To read or modify its source, clone it and swap the `[tool.uv.sources]`
entry in `pyproject.toml` for `sam3 = { path = "sam3", editable = true }`.

Check the install without touching the GPU:

```bash
uv run --extra dev pytest      # 50 tests
uv run batch.py track --help
```

### 4. Get access to the SAM 3 checkpoints

The weights are gated, so this cannot be automated:

1. Create an account at [huggingface.co](https://huggingface.co/join).
2. Go to [facebook/sam3](https://huggingface.co/facebook/sam3) and accept the
   terms to request access. **Wait for the approval email** — the next step
   fails until it arrives.
3. Create a token at
   [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).
   A **read** token is enough.
4. Log in with it:

```bash
uv run hf auth login
```

Nothing is downloaded yet. The checkpoint (~3.3 GB) is fetched on your **first
tracking run**, not during `uv sync`, and cached under `~/.cache/huggingface`
— set `HF_HOME` to move that. So an auth or access problem surfaces partway
into the first run rather than at install time.

### 5. First real run

Point it at any recording you have. This tracks 60 frames, which is enough to
prove the checkpoint download, the GPU path, and video decoding all work:

```bash
uv run track_video.py YOUR_VIDEO.mp4 --version sam3 --prompt insect \
    --frames 60 --out runs/smoke
```

Pass `--version sam3` explicitly: the built-in default is `sam3.1`, which will
not fit in 16 GB at this resolution (see *Things that cost time to discover*).
A `runs/smoke/tracks.csv` with a handful of distinct `obj_id` values means the
install is good.

If the prompt finds nothing, try another wording before assuming the install is
broken — `insect` works where `fly` returns zero. The same section explains why.

### 6. Configure it for your own data

Set `data_dir`, `out_dir` and `expect_n` as described under **Configure** below,
then use `batch.py` for anything more than one recording.

## Configure

Every setting is a command-line flag, and a config file only changes what
those flags default to:

```
CLI flag  >  segmentation.toml  >  built-in default
```

```bash
cp segmentation.example.toml segmentation.toml   # then edit it
```

`segmentation.toml` is found in the current directory, or named explicitly with
`--config PATH`. It is optional — without one, every script behaves exactly as
its `--help` describes. Relative paths inside it resolve against the file's own
directory, so the same config works from anywhere. An unknown section or key is
an error rather than a silent no-op, which is what catches typos.

The three settings most people need to change are in `[paths]` (`data_dir`,
`out_dir`) and `[subjects]` (`expect_n`).

## Pipeline

One recording:

```bash
# 1. track (chunked; handles arbitrarily long video)
uv run track_long.py recording.mp4 --out runs/rec1

# 2. drop short-lived spurious tracks
uv run filter_tracks.py runs/rec1 --dry-run    # check first
uv run filter_tracks.py runs/rec1 --backup runs/superseded

# 3. fill in the frames skipped by --stride (skip if stride is 1)
uv run upsample_tracks.py runs/rec1/tracks.csv --kind pchip

# 4. render a video, one colour per fly
uv run render_colors.py runs/rec1 --trails
```

A folder of recordings, tracked one at a time and resumable:

```bash
uv run batch.py track --dry-run    # print the plan
uv run batch.py track              # run it
uv run batch.py render             # overlay video for every finished run
```

`batch.py` searches `data_dir` recursively and mirrors what it finds into
`out_dir`, so `<data>/30_7_26/CS_T5.mp4` produces `<out>/30_7_26/CS_T5/`. A flat
folder of videos works the same way. A video whose `tracks.csv` already exists
is skipped, so the batch can be killed and restarted. It ends with a table
flagging every recording whose track count differs from `expect_n`.

Discovery matches every file under `data_dir` with a configured extension, at
any depth. `out_dir` is skipped automatically even when it is nested inside
`data_dir`. Rendered videos left loose in `data_dir` are not: they have the
same extension as a recording, so they get picked up as new inputs on the
next run. Keep outputs out of the data directory, or narrow `extensions` to
exclude them.

| Script | Does |
|---|---|
| `batch.py` | Tracks or renders a whole folder, one video at a time. **Use this for more than one recording.** |
| `track_video.py` | Single-session tracking. Fine up to ~500 frames; the building block for `track_long.py`. |
| `track_long.py` | Chunked tracking with IoU identity stitching across chunk seams. **Use this for real runs.** |
| `filter_tracks.py` | Drops short-lived spurious ids from `tracks.csv` and `masks.npz`. |
| `upsample_tracks.py` | Interpolates strided tracks back to every source frame. |
| `render_colors.py` | Renders `masks.npz` to a video with one stable colour per object. |

## Adapting to your own recordings

**Prompt.** Try two or three wordings on one recording before starting a batch;
see the note below on how much the wording matters. `--prompt` takes any
open-vocabulary noun, so nothing here is *Drosophila*-specific.

**Expected count.** `expect_n` (or `--expect-n`) is a post-hoc expectation, not
an instruction to the model: SAM 3's object cap is a `sam3.1` knob, and `sam3.1`
does not fit in 16 GB at this resolution. It warns per chunk when the live
object count differs, flags finished runs whose id count differs, and tells
`filter_tracks.py` to keep the N longest-lived ids.

**Stride.** `1` tracks every frame. Higher values trade temporal resolution for
speed and are recoverable with `upsample_tracks.py` to well under a pixel on
this assay — validate that on your own footage before relying on it.

**Chunk size.** This is a VRAM knob first: lower it if you hit CUDA OOM. It also
selects *how* identity stitching fails at a seam, so do not tune it on speed
alone. Larger chunks mean fewer seams but a longer absence per missed one, so a
subject lost for a whole chunk comes back beyond IoU range and is handed a new
id — one animal, two ids. Smaller chunks double the seams but nothing is ever
gone long enough to split, at the cost of more short-lived stubs. Splits are
recoverable with `--relink-dist`/`--relink-chunks`; stubs are not, but
`filter_tracks.py` removes them by lifetime.

### Outputs

#### `tracks.csv`

One row per object per frame. Rows are omitted where an object's mask is empty, so a
lost object leaves a gap rather than a zero-area row.

| Column | Units | Meaning |
|---|---|---|
| `frame` | index | Position in the **tracked** sequence, `0 … N-1`. This is *not* a video frame number — with `--stride 3`, frame 1 is the fourth frame of the video. |
| `src_frame` | index | Frame number in the **source video** (`start + frame × stride`). Use this to seek back into the original recording. |
| `time_s` | seconds | Elapsed time from the start of the recording, `src_frame / source_fps`. |
| `obj_id` | id | Track identity, stable for the whole recording including across chunk seams. Assigned by the tracker, not a biological identity. |
| `cx`, `cy` | **pixels** | Mask centroid in image coordinates: origin top-left, `y` increasing **downward**. |
| `area_px` | pixels | Mask area, i.e. the number of pixels in the mask. |
| `score` | 0–1 | SAM 3 confidence for that object on that frame. |
| `box_x`, `box_y` | **normalised 0–1** | **Top-left corner** of the bounding box, as a fraction of frame width and height. |
| `box_w`, `box_h` | **normalised 0–1** | Box width and height, as fractions of frame width and height. |

> **The units are mixed.** `cx`/`cy`/`area_px` are in pixels; the four `box_*` columns
> are normalised to 0–1. Multiply the box columns by frame width/height to get pixels —
> at 1120×1120, `box_x * 1120`. This comes from SAM 3's own output format and is
> preserved rather than silently converted.

Nothing here is spatially calibrated: distances and areas are in image units. Multiply
by (arena diameter in mm) / (arena diameter in px) to get millimetres.

#### `tracks_upsampled.csv`

Written by `upsample_tracks.py`. Same columns with two differences: there is **no
`frame` column** (only `src_frame`, now contiguous), and a `measured` column is added.

| Column | Units | Meaning |
|---|---|---|
| `measured` | 0 or 1 | `1` for a real tracked observation, `0` for an interpolated one. With `--stride 3`, one row in three is measured. |

Filter on `measured == 1` for anything that must rest on observations only.

#### `masks.npz`

Bit-packed per-frame masks plus ids, boxes and scores, so a run can be re-rendered
without re-tracking. Keys are `m{frame}`, `i{frame}`, `b{frame}`, `p{frame}`, plus
`height`, `width` and `frames`. Unpack with
`np.unpackbits(z[f"m{i}"], axis=-1)[..., :width]`.

#### `run.json`

Every setting the run actually used, plus its object counts. Once defaults can
come from a config file, the command line no longer records what a run did —
and the config may have been edited since — so this is what makes a run
reproducible by someone whose config differs.

| Key | Meaning |
|---|---|
| `video`, `created` | Absolute source path and run timestamp |
| `prompt`, `version`, `start`, `stride`, `chunk`, `overlap` | Settings used |
| `prob_thresh`, `iou_thresh`, `relink_dist`, `relink_chunks` | Thresholds used |
| `expect_n` | The expected animal count, or `null` |
| `frames`, `chunks` | Frames tracked and chunks they were split into |
| `objects_per_chunk` | Live object count per chunk — where a subject was lost |
| `final_ids` | Distinct `obj_id` values in `tracks.csv` |

## Things that cost time to discover

**SAM 3 under-declares its own dependencies.** Its core inference path imports
`einops`, `pandas`, `matplotlib`, `scikit-learn` and `pycocotools` at module level while
declaring them only in its `notebooks` extra, or not at all. It also imports
`pkg_resources`, which setuptools deleted in v81 — so a modern setuptools breaks every
sam3 import. All of these are pinned explicitly in `pyproject.toml`; the comments there
say why, so nobody "tidies them up" later.

**Use `--version sam3`, not `sam3.1`.** SAM 3.1 will not fit in 16 GB at this
resolution, and shortening the clip does not help: the OOM is a fixed ~14.6 GiB
per-step peak in the detector mask head (byte-identical at 40 and 120 frames). The
multiplex tracker runs 16 hypotheses at full resolution and `multiplex_count` is baked
into the checkpoint (`[16, 256]` tensors), so it cannot be lowered.

**Prompt wording matters more than it should.** `insect` and `fly` both find all seven
flies, but `fly` scores 0.56–0.60 against the default 0.50 threshold and returns *zero*
in the video model. `insect` scores 0.94. `ant` finds nothing — a useful negative control.

**GPU memory grows with sequence length** even with `--offload-video --offload-state`.
A single session dies near 500 frames on a 16 GB card; hence chunking.

**A local `sam3/` clone shadows the installed package.** If you clone it into the project
root, Python imports the clone directory as a namespace package whose `__file__` is
`None`, and SAM 3's `pkg_resources.resource_filename("sam3", ...)` fails with a confusing
`expected str ... not NoneType`. Installing from git (the default here) avoids it; the
scripts also derive the asset path from `model_builder.__file__`, which is CWD-independent
either way.

**`offload_state_to_cpu` breaks `sam3.1`.** The shared `start_session` forwards it
unconditionally but the multiplex `init_state` does not accept it. `build_predictor()`
drops kwargs the underlying `init_state` cannot take.

**Python must have headers.** Triton JIT-compiles a CUDA helper at runtime and needs
`Python.h`. A uv-managed CPython ships them; a system `python3.12` needs
`python3.12-dev`.

**If `ffmpeg` is a snap**, it runs with a private `/tmp` and cannot see files written
to the real one — it fails with a misleading `No such file or directory`.
`render_colors.py` stages its intermediate next to the output.

## Validation

- Interpolation (stride 3 → every frame) checked against a separate stride-1 run:
  median error 0.57 px, p95 1.71 px, max 3.43 px, against a ~60 px fly body.
- Identity across chunk seams: 7/7 tracks matched at all 8 boundaries, overlap IoU
  0.77–0.95 against a 0.30 threshold.

**Not validated:** identity through occlusion. In the test recording no two flies came
within 30 px of each other, so the case that breaks trackers never occurred. For
recordings with real contact, compare against idtracker.ai or SLEAP before trusting
per-individual time series.
