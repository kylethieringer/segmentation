# Fly arena segmentation with SAM 3

Segment and track individual flies in a backlit *Drosophila* arena recording using
Meta's [Segment Anything 3](https://github.com/facebookresearch/sam3), and export
per-frame centroid trajectories for behavioural analysis.

Built against a 1120×1120 grayscale, 60 fps arena assay with ~7 flies, but nothing
is species-specific — `--prompt` drives what gets segmented.

**! code was written with help from claude**

## Setup

Requires an NVIDIA GPU (CUDA 12.6+) and [uv](https://docs.astral.sh/uv/). Everything
else — the Python interpreter, torch, and SAM 3 itself — is resolved from `uv.lock`:

```bash
uv sync                  # the pipeline
uv sync --extra dev      # ...plus jupyter, for sam3's own example notebooks
```

SAM 3 is installed from a pinned commit rather than vendored, so no manual clone is
needed. To read or modify its source, clone it and swap the `[tool.uv.sources]` entry
in `pyproject.toml` for `sam3 = { path = "sam3", editable = true }`.

The checkpoints are gated. Request access to
[facebook/sam3](https://huggingface.co/facebook/sam3), then:

```bash
uv run hf auth login
```

## Pipeline

```bash
# 1. track (chunked; handles arbitrarily long video)
uv run track_long.py test_video.mp4 --prompt insect --stride 3 \
    --chunk 450 --overlap 15 --out runs/full_s3

# 2. fill in the frames skipped by --stride
uv run upsample_tracks.py runs/full_s3/tracks.csv --kind pchip

# 3. render a video, one colour per fly
uv run render_colors.py runs/full_s3 --fps 20 --trails
```

| Script | Does |
|---|---|
| `track_video.py` | Single-session tracking. Fine up to ~500 frames; the building block for `track_long.py`. |
| `track_long.py` | Chunked tracking with IoU identity stitching across chunk seams. **Use this for real runs.** |
| `upsample_tracks.py` | Interpolates strided tracks back to every source frame. |
| `render_colors.py` | Renders `masks.npz` to a video with one stable colour per object. |

### Outputs

- `tracks.csv` — `frame, src_frame, time_s, obj_id, cx, cy, area_px, score, box_*`
- `tracks_upsampled.csv` — same, every source frame, plus a `measured` column marking
  real observations (1) versus interpolated ones (0)
- `masks.npz` — bit-packed per-frame masks; re-render without re-tracking

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
