# Shippable Config Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let anyone point this pipeline at their own video folder, animal count, and output location without editing code.

**Architecture:** A single `config.py` reads an optional `segmentation.toml` and feeds values into each script's existing argparse parser via `set_defaults`, so CLI flags always win and `--help` stays truthful. The two hardcoded bash drivers are replaced by one `batch.py` with `track` and `render` subcommands that discovers videos recursively and mirrors the input tree into the output tree.

**Tech Stack:** Python 3.12, stdlib `tomllib` + `argparse` (no new runtime dependency), pytest for tests.

**Spec:** `docs/superpowers/specs/2026-09-11-shippable-config-design.md`

## Global Constraints

- Python `>=3.12,<3.13`. `tomllib` is stdlib — do **not** add `toml` or `tomli`.
- No new runtime dependencies. `pytest` goes in the `dev` extra only.
- Config keys are spelled exactly like the argparse `dest` they back (`stride` → `--stride`, `prob_thresh` → `--prob-thresh`). No aliases, no translation tables.
- Precedence is always **CLI flag > config file > built-in default**.
- Relative paths in the config resolve against the **config file's own directory**, never the CWD. `~` is expanded.
- Unknown sections and unknown keys are a hard `SystemExit`, never a silent ignore.
- A missing config file is fine. A `--config PATH` that does not exist is an error.
- Existing CLI behaviour must not change when no config file is present. Every current default stays as it is.
- `numpy<2` is pinned repo-wide; do not use numpy-2-only APIs.
- Comments explaining *why* (OOM retry, chunk-size tuning, mixed units) are carried over **verbatim** when code moves. They record measurements that cannot be recovered by reading the code.

## Deviation from the spec (deliberate, carried through every task)

The spec's example config wrote `[subjects] expected = 7`, but the flag it backs is `--expect-n` (dest `expect_n`). Keeping the spec's spelling would have required an alias table for exactly one key, breaking the rule that makes the whole layer simple. **The key is `expect_n`.** Everything below uses that.

Also clarified: `[paths]` is a **batch-level** section. `data_dir`/`out_dir` are consumed only by `batch.py` (dests `data_dir`, `out_dir`). They are deliberately *not* applied to `track_long.py --out`, which names one specific run directory rather than a root.

## File Structure

| File | Status | Responsibility |
|---|---|---|
| `config.py` | create | Locate, parse, validate, and path-resolve the TOML; apply sections to a parser |
| `conftest.py` | create | Empty; its presence puts the repo root on `sys.path` for pytest |
| `segmentation.example.toml` | create | Committed reference config carrying today's hardcoded values |
| `batch.py` | create | Recursive discovery, output mirroring, `track` and `render` subcommands |
| `tests/test_config.py` | create | Precedence, path resolution, validation |
| `tests/test_batch.py` | create | Discovery and mirroring |
| `tests/test_filter_tracks.py` | create | `--expect-n` selection, backup path mirroring |
| `tests/test_cli_smoke.py` | create | `--help` exits 0 on every script |
| `track_video.py` | modify | Accept `--config`, consume `[track]` |
| `track_long.py` | modify | Accept `--config`, consume `[track]`/`[subjects]`, add `--expect-n`, write `run.json` |
| `upsample_tracks.py` | modify | Accept `--config` (no section yet; keeps the flag uniform) |
| `render_colors.py` | modify | Accept `--config`, consume `[render]` |
| `filter_tracks.py` | commit + modify | `--expect-n` mode; fix the backup path collision |
| `run_batch.sh`, `render_batch.sh` | delete (Task 6) | Replaced by `batch.py` |
| `README.md` | modify | Setup → Configure → One video → Batch → Outputs → Adapting → Gotchas |
| `.gitignore` | modify | Add `segmentation.toml` |
| `pyproject.toml` | modify | `pytest` in `dev`, `[tool.pytest.ini_options]` |

---

### Task 1: The config layer

**Files:**
- Create: `config.py`
- Create: `conftest.py`
- Create: `tests/test_config.py`
- Modify: `pyproject.toml`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `config.SCHEMA: dict[str, set[str]]`
  - `config.load_config(path: Path | None = None) -> dict[str, dict]`
  - `config.add_config_arg(parser: argparse.ArgumentParser) -> None`
  - `config.preparse_config(argv: list[str] | None = None) -> Path | None`
  - `config.apply_to(parser: argparse.ArgumentParser, cfg: dict, *sections: str) -> set[str]` (returns the keys it skipped because the parser has no such dest)

- [ ] **Step 1: Add pytest and its config**

In `pyproject.toml`, add `"pytest>=8"` to the `dev` extra list, and append:

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
```

Create an empty-but-commented `conftest.py` at the repo root:

```python
# Present so pytest puts the repo root on sys.path: the pipeline is a flat
# set of top-level modules, so `import config` in tests/ only resolves when
# the rootdir is importable.
```

- [ ] **Step 2: Write the failing tests**

Create `tests/test_config.py`:

```python
from __future__ import annotations

import argparse
from pathlib import Path

import pytest

import config


def write(tmp_path: Path, text: str) -> Path:
    p = tmp_path / "segmentation.toml"
    p.write_text(text)
    return p


def test_load_returns_empty_when_no_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert config.load_config() == {}


def test_explicit_missing_config_is_an_error(tmp_path):
    with pytest.raises(SystemExit):
        config.load_config(tmp_path / "nope.toml")


def test_picks_up_cwd_config(tmp_path, monkeypatch):
    write(tmp_path, "[track]\nstride = 4\n")
    monkeypatch.chdir(tmp_path)
    assert config.load_config()["track"]["stride"] == 4


def test_unknown_section_rejected(tmp_path):
    p = write(tmp_path, "[trak]\nstride = 4\n")
    with pytest.raises(SystemExit, match="trak"):
        config.load_config(p)


def test_unknown_key_rejected(tmp_path):
    p = write(tmp_path, "[track]\nstrid = 4\n")
    with pytest.raises(SystemExit, match="strid"):
        config.load_config(p)


def test_relative_path_resolves_against_config_dir(tmp_path):
    sub = tmp_path / "cfg"
    sub.mkdir()
    p = sub / "segmentation.toml"
    p.write_text('[paths]\nout_dir = "runs"\n')
    assert config.load_config(p)["paths"]["out_dir"] == sub / "runs"


def test_tilde_expanded(tmp_path):
    p = write(tmp_path, '[paths]\ndata_dir = "~/recordings"\n')
    got = config.load_config(p)["paths"]["data_dir"]
    assert got == Path.home() / "recordings"
    assert "~" not in str(got)


def test_absolute_path_left_alone(tmp_path):
    p = write(tmp_path, '[paths]\ndata_dir = "/srv/videos"\n')
    assert config.load_config(p)["paths"]["data_dir"] == Path("/srv/videos")


def test_nested_batch_sections_flattened(tmp_path):
    p = write(tmp_path, "[batch.track]\nmin_free_gb = 5\n[batch.render]\nmin_free_gb = 2\n")
    cfg = config.load_config(p)
    assert cfg["batch.track"]["min_free_gb"] == 5
    assert cfg["batch.render"]["min_free_gb"] == 2


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stride", type=int, default=3)
    return ap


def test_config_overrides_builtin_default(tmp_path):
    p = write(tmp_path, "[track]\nstride = 4\n")
    ap = build_parser()
    config.apply_to(ap, config.load_config(p), "track")
    assert ap.parse_args([]).stride == 4


def test_cli_beats_config(tmp_path):
    p = write(tmp_path, "[track]\nstride = 4\n")
    ap = build_parser()
    config.apply_to(ap, config.load_config(p), "track")
    assert ap.parse_args(["--stride", "9"]).stride == 9


def test_builtin_default_survives_empty_config(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    ap = build_parser()
    config.apply_to(ap, config.load_config(), "track")
    assert ap.parse_args([]).stride == 3


def test_keys_the_parser_does_not_declare_are_skipped(tmp_path):
    p = write(tmp_path, "[track]\nstride = 4\nchunk = 100\n")
    ap = build_parser()
    skipped = config.apply_to(ap, config.load_config(p), "track")
    assert skipped == {"chunk"}
    assert ap.parse_args([]).stride == 4


def test_preparse_finds_config_flag():
    assert config.preparse_config(["--config", "/tmp/x.toml", "--stride", "2"]) == Path("/tmp/x.toml")
    assert config.preparse_config(["--stride", "2"]) is None
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `uv run --extra dev pytest tests/test_config.py -v`
Expected: collection error, `ModuleNotFoundError: No module named 'config'`.

- [ ] **Step 4: Write `config.py`**

```python
#!/usr/bin/env python
"""Optional TOML configuration for the tracking pipeline.

The pipeline stays CLI-first: every setting is still a flag, and a config
file only changes what those flags default to. Precedence is

    CLI flag  >  config file  >  built-in default

which is argparse's own `set_defaults` behaviour, so `--help` keeps printing
the value a run would actually use.

Config keys are spelled exactly like the argparse dest they back, so there is
no second vocabulary to learn: `stride` backs --stride, `prob_thresh` backs
--prob-thresh. The only cost of that rule is that the section a key lives in
has to be chosen once, here, in SCHEMA.
"""

from __future__ import annotations

import argparse
import tomllib
from pathlib import Path
from typing import Any

CONFIG_NAME = "segmentation.toml"

# Every key a config file may contain. Validated strictly: a typo that were
# silently ignored is the classic way a config file wastes an afternoon.
#
# [paths] is batch-level. data_dir/out_dir name the roots batch.py walks and
# mirrors into -- they are deliberately not applied to track_long.py --out,
# which names one specific run directory rather than a root.
SCHEMA: dict[str, set[str]] = {
    "paths": {"data_dir", "out_dir", "extensions"},
    "subjects": {"expect_n", "prompt"},
    "track": {"version", "stride", "chunk", "overlap", "prob_thresh",
              "iou_thresh", "relink_dist", "relink_chunks"},
    "render": {"trails", "trail_len", "alpha", "outline"},
    "filter": {"min_fraction"},
    "batch.track": {"retry_chunk", "min_free_gb"},
    "batch.render": {"min_free_gb"},
}

# Values under these keys are turned into Paths and resolved against the
# config file's directory, so a config is usable from any working directory.
PATH_KEYS = {"data_dir", "out_dir"}

# Sections written as [batch.track] parse as a nested table; flatten them back
# to the dotted names SCHEMA uses.
NESTED = {"batch"}


def add_config_arg(parser: argparse.ArgumentParser) -> None:
    """Declare --config on a parser so it shows up in --help."""
    parser.add_argument("--config", type=Path, default=None,
                        help=f"TOML config file (default: ./{CONFIG_NAME} if present)")


def preparse_config(argv: list[str] | None = None) -> Path | None:
    """Pull --config out of argv before the real parser is built.

    The real parser's defaults depend on the config file, so the file has to
    be located first. parse_known_args on a throwaway parser does that without
    duplicating any other flag.
    """
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=Path, default=None)
    known, _ = pre.parse_known_args(argv)
    return known.config


def _flatten(raw: dict[str, Any]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for name, body in raw.items():
        if not isinstance(body, dict):
            raise SystemExit(f"[config] top-level key {name!r} must be a section, "
                             f"e.g. [track]")
        if name in NESTED:
            for sub, subbody in body.items():
                if not isinstance(subbody, dict):
                    raise SystemExit(f"[config] [{name}] takes subsections only, "
                                     f"e.g. [{name}.track]; got key {sub!r}")
                out[f"{name}.{sub}"] = subbody
        else:
            out[name] = body
    return out


def _validate(cfg: dict[str, dict[str, Any]], path: Path) -> None:
    for section, body in cfg.items():
        if section not in SCHEMA:
            known = ", ".join(sorted(SCHEMA))
            raise SystemExit(f"[config] {path}: unknown section [{section}]. "
                             f"Known sections: {known}")
        unknown = set(body) - SCHEMA[section]
        if unknown:
            bad = ", ".join(sorted(unknown))
            known = ", ".join(sorted(SCHEMA[section]))
            raise SystemExit(f"[config] {path}: unknown key(s) in [{section}]: {bad}. "
                             f"Valid keys: {known}")


def _resolve_paths(cfg: dict[str, dict[str, Any]], base: Path) -> None:
    for body in cfg.values():
        for key in list(body):
            if key in PATH_KEYS:
                p = Path(str(body[key])).expanduser()
                body[key] = p if p.is_absolute() else (base / p).resolve()


def find_config(explicit: Path | None = None) -> Path | None:
    """--config PATH, else ./segmentation.toml, else nothing."""
    if explicit is not None:
        if not explicit.is_file():
            raise SystemExit(f"[config] no such config file: {explicit}")
        return explicit
    local = Path.cwd() / CONFIG_NAME
    return local if local.is_file() else None


def load_config(path: Path | None = None) -> dict[str, dict[str, Any]]:
    """-> {section: {key: value}}. Empty dict when there is no config file."""
    found = find_config(path)
    if found is None:
        return {}
    with found.open("rb") as fh:
        raw = tomllib.load(fh)
    cfg = _flatten(raw)
    _validate(cfg, found)
    _resolve_paths(cfg, found.resolve().parent)
    return cfg


def apply_to(parser: argparse.ArgumentParser, cfg: dict[str, dict[str, Any]],
             *sections: str) -> set[str]:
    """Make the named config sections the parser's defaults. CLI still wins.

    Keys naming a flag this particular parser does not declare are skipped and
    returned -- [track] is shared by track_video.py and track_long.py, which
    do not have identical flag sets. Typos are caught by _validate, not here.
    """
    declared = {a.dest for a in parser._actions}   # noqa: SLF001 - stable, and
    # there is no public way to ask a parser which dests it knows.
    skipped: set[str] = set()
    defaults: dict[str, Any] = {}
    for section in sections:
        for key, value in cfg.get(section, {}).items():
            if key in declared:
                defaults[key] = value
            else:
                skipped.add(key)
    parser.set_defaults(**defaults)
    return skipped
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run --extra dev pytest tests/test_config.py -v`
Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add config.py conftest.py tests/test_config.py pyproject.toml
git commit -m "Add optional TOML config layer"
```

---

### Task 2: Wire the config into the four single-video scripts

**Files:**
- Modify: `track_video.py` (in `main()`, at the `ap = argparse.ArgumentParser(` line)
- Modify: `track_long.py` (in `main()`, same place)
- Modify: `upsample_tracks.py` (in `main()`, same place)
- Modify: `render_colors.py` (in `main()`, same place)
- Create: `tests/test_cli_smoke.py`

**Interfaces:**
- Consumes: `config.add_config_arg`, `config.preparse_config`, `config.load_config`, `config.apply_to` from Task 1.
- Produces: every script accepts `--config PATH`. No change to any other flag name or default.

The same four-line idiom goes into each `main()`, immediately after the parser's arguments are declared and immediately before `args = ap.parse_args()`.

- [ ] **Step 1: Write the failing smoke test**

Create `tests/test_cli_smoke.py`:

```python
"""--help must exit 0 on every script.

This is the test that actually catches config wiring breaking argparse: a
misspelled dest in set_defaults, a --config flag declared twice, or an import
that only fails at parse time.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ["track_video.py", "track_long.py", "upsample_tracks.py",
           "render_colors.py", "filter_tracks.py"]


@pytest.mark.parametrize("script", SCRIPTS)
def test_help_exits_zero(script, tmp_path):
    # Run from tmp_path so a segmentation.toml in the repo root cannot affect
    # the result -- this must pass on a clean checkout and on a configured one.
    proc = subprocess.run([sys.executable, str(ROOT / script), "--help"],
                          cwd=tmp_path, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    assert "--config" in proc.stdout


@pytest.mark.parametrize("script", SCRIPTS)
def test_config_default_reaches_help(script, tmp_path):
    (tmp_path / "segmentation.toml").write_text("[track]\nstride = 7\n[render]\ntrail_len = 7\n")
    proc = subprocess.run([sys.executable, str(ROOT / script), "--help"],
                          cwd=tmp_path, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run --extra dev pytest tests/test_cli_smoke.py -v`
Expected: FAIL — `assert "--config" in proc.stdout` for every script (and `filter_tracks.py` may not be tracked yet; it exists in the working tree, so the run still executes it).

- [ ] **Step 3: Wire `track_video.py`**

Add to the imports at the top of the file:

```python
import config as cfgmod
```

In `main()`, replace this line:

```python
    args = ap.parse_args()
```

with:

```python
    cfgmod.add_config_arg(ap)
    cfgmod.apply_to(ap, cfgmod.load_config(cfgmod.preparse_config()), "track")
    args = ap.parse_args()
```

- [ ] **Step 4: Wire `track_long.py`**

Add `import config as cfgmod` to the imports. In `main()`, replace:

```python
    args = ap.parse_args()

    if args.overlap >= args.chunk:
```

with:

```python
    cfgmod.add_config_arg(ap)
    cfgmod.apply_to(ap, cfgmod.load_config(cfgmod.preparse_config()),
                    "track", "subjects")
    args = ap.parse_args()

    if args.overlap >= args.chunk:
```

`[subjects]` carries `prompt`, which `track_long.py` already declares. `expect_n` is skipped harmlessly until Task 3 adds the flag.

- [ ] **Step 5: Wire `upsample_tracks.py`**

Add `import config as cfgmod` to the imports. In `main()`, replace:

```python
    args = ap.parse_args()

    df = pd.read_csv(args.tracks_csv)
```

with:

```python
    cfgmod.add_config_arg(ap)
    # No section of its own: the flag is declared for uniformity, so every
    # script in the pipeline takes --config whether or not it reads anything.
    cfgmod.apply_to(ap, cfgmod.load_config(cfgmod.preparse_config()))
    args = ap.parse_args()

    df = pd.read_csv(args.tracks_csv)
```

- [ ] **Step 6: Wire `render_colors.py`**

Add `import config as cfgmod` to the imports. In `main()`, replace:

```python
    args = ap.parse_args()

    # Name the output after the run
```

with:

```python
    cfgmod.add_config_arg(ap)
    cfgmod.apply_to(ap, cfgmod.load_config(cfgmod.preparse_config()), "render")
    args = ap.parse_args()

    # Name the output after the run
```

- [ ] **Step 7: Wire `filter_tracks.py`**

Add `import config as cfgmod` to the imports. In `main()`, replace:

```python
    args = ap.parse_args()
```

with:

```python
    cfgmod.add_config_arg(ap)
    cfgmod.apply_to(ap, cfgmod.load_config(cfgmod.preparse_config()), "filter")
    args = ap.parse_args()
```

- [ ] **Step 8: Run the smoke tests to verify they pass**

Run: `uv run --extra dev pytest tests/test_cli_smoke.py -v`
Expected: all PASS.

- [ ] **Step 9: Verify no behaviour changed without a config file**

Run: `uv run track_long.py --help | grep -E "stride|chunk|overlap"`
Expected: `--stride` still shows `(default: 3)`, `--chunk` still `450`, `--overlap` still `15` — identical to before this task.

- [ ] **Step 10: Commit**

```bash
git add track_video.py track_long.py upsample_tracks.py render_colors.py filter_tracks.py tests/test_cli_smoke.py
git commit -m "Accept --config in every pipeline script"
```

---

### Task 3: Expected subject count and run provenance in `track_long.py`

**Files:**
- Modify: `track_long.py` (`main()`: argument block, the chunk loop, the tail after `masks.npz` is written)

**Interfaces:**
- Consumes: `[subjects]` section wiring from Task 2.
- Produces: `--expect-n INT | None`; a `run.json` in every run directory with the keys listed below. `batch.py` (Task 6) reads `final_ids` and `expect_n` from it.

- [ ] **Step 1: Add the flag**

In `main()`, after the `--reuse-frames` argument, add:

```python
    ap.add_argument("--expect-n", type=int, default=None,
                    help="animals you expect per recording; warns per chunk when "
                         "the count differs. Post-hoc only -- SAM 3's object cap "
                         "is a sam3.1 knob and sam3.1 does not fit in 16 GB here.")
```

- [ ] **Step 2: Add `import json` and a per-chunk counter**

Add `import json` to the imports. Before the `for ci, (lo, hi) in enumerate(bounds):` loop, next to the other accumulators, add:

```python
    objects_per_chunk: list[int] = []
```

- [ ] **Step 3: Record and warn inside the chunk loop**

In the chunk loop, immediately after the block that mints new global ids —

```python
        for oid in sorted({int(o) for out in local_outputs.values()
                           for o in np.asarray(out["out_obj_ids"]).tolist()}):
            if oid not in id_map:
                id_map[oid] = next_gid
                next_gid += 1
```

— insert:

```python
        # A chunk that disagrees with the expected count is information, not a
        # failure: it is usually one animal lost to a wall reflection for a few
        # seconds, and the run is still worth finishing.
        objects_per_chunk.append(len(id_map))
        if args.expect_n is not None and len(id_map) != args.expect_n:
            print(f"[warn] chunk {ci}: {len(id_map)} objects, expected {args.expect_n}",
                  flush=True)
```

- [ ] **Step 4: Write `run.json` after `masks.npz`**

After the `print(f"[masks] ...")` line and before the final `[done]` print, insert:

```python
    # Provenance. Once defaults can come from a config file, the command line
    # no longer records what a run did -- and the config file may have been
    # edited since. This is what makes a run reproducible by someone whose
    # config differs.
    final_ids = len({row[3] for row in csv_rows})
    meta = {
        "video": str(args.video.resolve()),
        "created": datetime.now().isoformat(timespec="seconds"),
        "prompt": args.prompt,
        "version": args.version,
        "start": args.start,
        "stride": args.stride,
        "chunk": args.chunk,
        "overlap": args.overlap,
        "prob_thresh": args.prob_thresh,
        "iou_thresh": args.iou_thresh,
        "relink_dist": args.relink_dist,
        "relink_chunks": args.relink_chunks,
        "expect_n": args.expect_n,
        "frames": n_frames,
        "chunks": len(bounds),
        "objects_per_chunk": objects_per_chunk,
        "final_ids": final_ids,
    }
    meta_path = args.out / "run.json"
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    print(f"[meta] {meta_path}")
    if args.expect_n is not None and final_ids != args.expect_n:
        print(f"[warn] {final_ids} ids in the finished run, expected {args.expect_n}")
```

Add `from datetime import datetime` to the imports.

- [ ] **Step 5: Verify the flag and JSON shape without a GPU**

`track_long.py` cannot run without CUDA, so verify the two things that are testable:

Run: `uv run track_long.py --help | grep -A3 expect-n`
Expected: the flag and its help text appear.

Run: `uv run python -c "import json,ast,sys; src=open('track_long.py').read(); ast.parse(src); print('parses')"`
Expected: `parses`.

- [ ] **Step 6: Commit**

```bash
git add track_long.py
git commit -m "Add --expect-n warnings and write run.json provenance"
```

---

### Task 4: `filter_tracks.py` — expect-n mode and the backup path fix

**Files:**
- Modify: `filter_tracks.py` (the `main()` argument block, the cutoff logic, and the `--backup` destination)
- Create: `tests/test_filter_tracks.py`

**Interfaces:**
- Consumes: `--expect-n` semantics established in Task 3.
- Produces: `filter_tracks.select_drop(seen, n_frames, expect_n, min_fraction, min_frames) -> tuple[set[str], str]` returning the ids to drop and a one-line reason; `filter_tracks.backup_dest(root, run, runs) -> Path`.

`filter_tracks.py` was first committed in Task 2 (it is untracked before that),
so this task only modifies it.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_filter_tracks.py`:

```python
from __future__ import annotations

from pathlib import Path

import filter_tracks as ft


def seen_from(lifetimes: dict[str, int]) -> dict[str, set[str]]:
    """{obj_id: frames present} -> the {id: set-of-frame-strings} shape
    read_lifetimes produces."""
    return {oid: {str(f) for f in range(n)} for oid, n in lifetimes.items()}


def test_cutoff_mode_unchanged_without_expect_n():
    seen = seen_from({"0": 1000, "1": 1000, "2": 12})
    drop, _ = ft.select_drop(seen, n_frames=1000, expect_n=None,
                             min_fraction=0.5, min_frames=None)
    assert drop == {"2"}


def test_expect_n_keeps_the_n_longest_lived():
    seen = seen_from({"0": 1000, "1": 990, "2": 12, "3": 8})
    drop, _ = ft.select_drop(seen, n_frames=1000, expect_n=2,
                             min_fraction=0.5, min_frames=None)
    assert drop == {"2", "3"}


def test_expect_n_beats_the_cutoff_when_both_given():
    # Id 2 is above the 0.5 cutoff but is still the third-longest of three.
    seen = seen_from({"0": 1000, "1": 990, "2": 800})
    drop, _ = ft.select_drop(seen, n_frames=1000, expect_n=2,
                             min_fraction=0.5, min_frames=None)
    assert drop == {"2"}


def test_expect_n_warns_when_a_kept_track_is_below_cutoff():
    # Only two ids exist but three were expected, and the shortest kept id is
    # far below the cutoff -- the signal that a real animal was lost.
    seen = seen_from({"0": 1000, "1": 990, "2": 30})
    drop, reason = ft.select_drop(seen, n_frames=1000, expect_n=3,
                                  min_fraction=0.5, min_frames=None)
    assert drop == set()
    assert "below" in reason


def test_expect_n_larger_than_id_count_drops_nothing():
    seen = seen_from({"0": 1000, "1": 990})
    drop, _ = ft.select_drop(seen, n_frames=1000, expect_n=7,
                             min_fraction=0.5, min_frames=None)
    assert drop == set()


def test_ties_broken_deterministically():
    seen = seen_from({"0": 100, "1": 100, "2": 100})
    drop, _ = ft.select_drop(seen, n_frames=100, expect_n=2,
                             min_fraction=0.5, min_frames=None)
    assert drop == {"2"}


def test_backup_mirrors_relative_path_for_multiple_runs(tmp_path):
    a = tmp_path / "batch" / "30_7_26" / "CS_T5"
    b = tmp_path / "batch" / "31_7_26" / "CS_T5"
    runs = [a, b]
    for r in runs:
        r.mkdir(parents=True)
    root = tmp_path / "superseded"
    assert ft.backup_dest(root, a, runs) == root / "30_7_26" / "CS_T5"
    assert ft.backup_dest(root, b, runs) == root / "31_7_26" / "CS_T5"


def test_backup_uses_leaf_name_for_a_single_run(tmp_path):
    run = tmp_path / "batch" / "30_7_26" / "CS_T5"
    run.mkdir(parents=True)
    root = tmp_path / "superseded"
    assert ft.backup_dest(root, run, [run]) == root / "CS_T5"
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run --extra dev pytest tests/test_filter_tracks.py -v`
Expected: FAIL — `AttributeError: module 'filter_tracks' has no attribute 'select_drop'`.

- [ ] **Step 3: Add `select_drop` and `backup_dest` to `filter_tracks.py`**

Add `import os` to the imports, then add both functions after `read_lifetimes`:

```python
def select_drop(seen: dict[str, set[str]], n_frames: int, expect_n: int | None,
                min_fraction: float, min_frames: int | None) -> tuple[set[str], str]:
    """Choose which obj_ids to discard. -> (ids to drop, one-line reason).

    Two modes. Without --expect-n, an id is an artefact if it is present in
    fewer than `cutoff` frames -- the original rule, and the right one when the
    subject count is unknown. With --expect-n, the N longest-lived ids are the
    subjects by definition and everything else goes; the cutoff then only
    reports, because an id that survives the top-N cut while sitting below the
    cutoff means a real animal was lost, not that an artefact was found.
    """
    cutoff = min_frames or int(n_frames * min_fraction)
    if expect_n is None:
        drop = {i for i, frames in seen.items() if len(frames) < cutoff}
        return drop, f"under {cutoff} frames"

    # Longest-lived first; ties broken by id so a rerun drops the same ones.
    ranked = sorted(seen, key=lambda i: (-len(seen[i]), int(i)))
    keep, drop = set(ranked[:expect_n]), set(ranked[expect_n:])
    if len(ranked) < expect_n:
        return drop, (f"expected {expect_n}, found only {len(ranked)} ids "
                      f"-- nothing to drop")
    shortest_kept = min(len(seen[i]) for i in keep)
    if shortest_kept < cutoff:
        return drop, (f"top {expect_n} by lifetime, but the shortest kept id has "
                      f"{shortest_kept} frames, below the {cutoff}-frame cutoff "
                      f"-- a subject was probably lost")
    return drop, f"top {expect_n} by lifetime"


def backup_dest(root: Path, run: Path, runs: list[Path]) -> Path:
    """Where `run`'s originals are copied under `root`.

    Mirrors each run's path below the runs' common ancestor. Using the leaf
    name alone (as this did originally) makes 30_7_26/CS_T5 and 31_7_26/CS_T5
    write to the same place -- the second silently destroys the first's backup,
    after the originals have already been rewritten in place.
    """
    if len(runs) == 1:
        return root / run.name
    base = Path(os.path.commonpath([str(r.resolve()) for r in runs]))
    rel = run.resolve().relative_to(base)
    return root / (rel if rel != Path(".") else Path(run.name))
```

- [ ] **Step 4: Use them in `main()`**

Add the flag after `--min-frames`:

```python
    ap.add_argument("--expect-n", type=int, default=None,
                    help="keep the N longest-lived ids and drop the rest; "
                         "--min-fraction then only warns")
```

Change `--min-fraction`'s default so "given" is distinguishable from "defaulted":

```python
    ap.add_argument("--min-fraction", type=float, default=None,
                    help="drop ids present in fewer than this share of frames "
                         "(default: 0.5)")
```

Then in the per-run loop, replace:

```python
        cutoff = args.min_frames or int(n_frames * args.min_fraction)
        drop = {i for i, frames in seen.items() if len(frames) < cutoff}
        if not drop:
            print(f"[ok]   {run.name}: {len(seen)} ids, none under {cutoff} frames")
            continue
```

with:

```python
        drop, reason = select_drop(seen, n_frames, args.expect_n,
                                   args.min_fraction if args.min_fraction is not None
                                   else 0.5,
                                   args.min_frames)
        if not drop:
            print(f"[ok]   {run.name}: {len(seen)} ids, none dropped ({reason})")
            continue
```

Replace the drop report so it carries the reason:

```python
        print(f"[drop] {run.name}: ids {detail} -> {len(lines) - 1} to "
              f"{len(kept) - 1} rows, {len(seen)} to {len(seen) - len(drop)} ids "
              f"({reason})")
```

And replace the backup destination line:

```python
            dest = args.backup / run.name
```

with:

```python
            dest = backup_dest(args.backup, run, args.run_dir)
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run --extra dev pytest tests/test_filter_tracks.py tests/test_cli_smoke.py -v`
Expected: all PASS.

- [ ] **Step 6: Verify the default path is unchanged end to end**

Run: `uv run filter_tracks.py runs/full_s3 --dry-run`
Expected: the same ids reported as before this task (no `--expect-n`, so cutoff mode), with the reason `(under N frames)` appended.

- [ ] **Step 7: Commit**

```bash
git add filter_tracks.py tests/test_filter_tracks.py
git commit -m "Add filter_tracks.py with an --expect-n mode

Also fixes the --backup destination, which used the run's leaf name and
so let two runs sharing a stem overwrite each other's originals."
```

---

### Task 5: `batch.py` — discovery, mirroring, and the `track` subcommand

**Files:**
- Create: `batch.py`
- Create: `tests/test_batch.py`

**Interfaces:**
- Consumes: `config.load_config`/`apply_to` (Task 1); `track_long.py --expect-n` (Task 3).
- Produces:
  - `batch.discover(data_dir: Path, extensions: list[str]) -> list[Path]` — sorted, recursive
  - `batch.out_dir_for(video: Path, data_dir: Path, out_dir: Path) -> Path`
  - `batch.is_cuda_oom(log: Path) -> bool`
  - `batch.free_gb(path: Path) -> int`
  - `batch.Reporter` with `.say(msg)` appending to a master log
  - Task 6 adds the `render` subcommand to this same file.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_batch.py`:

```python
from __future__ import annotations

from pathlib import Path

import batch


def make(root: Path, *rels: str) -> None:
    for rel in rels:
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"")


def test_discovers_recursively_and_sorts(tmp_path):
    make(tmp_path, "b/2.mp4", "a/1.mp4", "a/deep/deeper/3.mp4")
    got = batch.discover(tmp_path, [".mp4"])
    assert [p.relative_to(tmp_path).as_posix() for p in got] == [
        "a/1.mp4", "a/deep/deeper/3.mp4", "b/2.mp4"]


def test_discovers_a_flat_folder(tmp_path):
    make(tmp_path, "1.mp4", "2.mp4")
    assert len(batch.discover(tmp_path, [".mp4"])) == 2


def test_honours_the_extension_list(tmp_path):
    make(tmp_path, "a.mp4", "b.avi", "c.txt")
    got = batch.discover(tmp_path, [".mp4", ".avi"])
    assert {p.suffix for p in got} == {".mp4", ".avi"}


def test_extension_match_is_case_insensitive(tmp_path):
    make(tmp_path, "a.MP4")
    assert len(batch.discover(tmp_path, [".mp4"])) == 1


def test_missing_data_dir_is_an_error(tmp_path):
    import pytest
    with pytest.raises(SystemExit):
        batch.discover(tmp_path / "nope", [".mp4"])


def test_output_mirrors_input_structure(tmp_path):
    data, out = tmp_path / "data", tmp_path / "runs"
    video = data / "30_7_26" / "CS_T5.mp4"
    assert batch.out_dir_for(video, data, out) == out / "30_7_26" / "CS_T5"


def test_output_for_a_flat_input(tmp_path):
    data, out = tmp_path / "data", tmp_path / "runs"
    assert batch.out_dir_for(data / "CS_T5.mp4", data, out) == out / "CS_T5"


def test_output_for_a_deep_input(tmp_path):
    data, out = tmp_path / "data", tmp_path / "runs"
    video = data / "2026" / "07" / "30" / "CS_T5.mp4"
    assert batch.out_dir_for(video, data, out) == out / "2026" / "07" / "30" / "CS_T5"


def test_oom_detected_in_the_log_tail(tmp_path):
    log = tmp_path / "x.log"
    log.write_text("\n".join(["noise"] * 100 + ["torch.OutOfMemoryError: CUDA out of memory"]))
    assert batch.is_cuda_oom(log) is True


def test_oom_not_detected_far_up_the_log(tmp_path):
    log = tmp_path / "x.log"
    log.write_text("\n".join(["torch.OutOfMemoryError"] + ["noise"] * 100))
    assert batch.is_cuda_oom(log) is False


def test_oom_on_a_missing_log_is_false(tmp_path):
    assert batch.is_cuda_oom(tmp_path / "absent.log") is False
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run --extra dev pytest tests/test_batch.py -v`
Expected: collection error, `ModuleNotFoundError: No module named 'batch'`.

- [ ] **Step 3: Write `batch.py`**

```python
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

def discover(data_dir: Path, extensions: list[str]) -> list[Path]:
    """Every video under data_dir, at any depth, sorted for a stable order."""
    if not data_dir.is_dir():
        raise SystemExit(f"[batch] no such data directory: {data_dir}")
    wanted = {e.lower() if e.startswith(".") else f".{e.lower()}" for e in extensions}
    found = [p for p in data_dir.rglob("*")
             if p.is_file() and p.suffix.lower() in wanted]
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
    videos = discover(args.data_dir, args.extensions)
    rep.say(f"[batch] {len(videos)} videos, prompt={args.prompt} stride={args.stride} "
            f"chunk={args.chunk} overlap={args.overlap}")

    rows: list[tuple[str, str, int, int | None]] = []
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
    return ap, {"track": t}


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
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run --extra dev pytest tests/test_batch.py -v`
Expected: all PASS.

- [ ] **Step 5: Compare the plan against the shell script it replaces**

Run, from the repo root:

```bash
uv run batch.py track --data-dir /home/kyle/projects/test_data \
    --out-dir runs/batch --dry-run
DRY_RUN=1 ./run_batch.sh
```

Expected: the same set of videos, each skipped or planned identically. Any video the shell script skips (`tracks.csv` exists) must also be skipped by `batch.py`. Investigate any difference before continuing — this is the check that the port did not lose behaviour.

- [ ] **Step 6: Commit**

```bash
git add batch.py tests/test_batch.py
git commit -m "Add batch.py with recursive discovery and a track subcommand"
```

---

### Task 6: `batch.py` render subcommand, and retire the shell scripts

**Files:**
- Modify: `batch.py` (add `cmd_render`, register the subparser)
- Delete: `run_batch.sh`, `render_batch.sh`

**Interfaces:**
- Consumes: `discover`, `out_dir_for`, `Reporter`, `free_gb`, `run_script`, `final_ids`, `summarise` from Task 5.
- Produces: `batch.cmd_render(args) -> None`.

- [ ] **Step 1: Add `cmd_render` to `batch.py`**

Insert after `cmd_track`:

```python
# ---------------------------------------------------------------- render

def cmd_render(args: argparse.Namespace) -> None:
    """Overlay video for every run that tracked successfully.

    A run counts as successful if it has masks.npz. cmd_track deletes frames/
    on success, so each render decodes its pictures back out of the source
    video (--video). Playback fps is left to render_colors.py, which infers the
    source rate from tracks.csv.
    """
    rep = Reporter(args.out_dir / "_logs" / "render.log")
    videos = discover(args.data_dir, args.extensions)
    rep.say(f"[render] {len(videos)} videos, trail-len={args.trail_len}")

    rows: list[tuple[str, str, int, int | None]] = []
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
            break

        if args.dry_run:
            rep.say(f"[dry-run] would render {name} -> {out}")
            continue

        rep.say(f"[start] {name}  (log: {log})")
        started = time.time()
        argv = [str(run), "--video", str(video), "--out", str(out),
                "--trail-len", str(args.trail_len), "--alpha", str(args.alpha),
                "--outline", str(args.outline)]
        if args.trails:
            argv.append("--trails")
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
```

- [ ] **Step 2: Register the subparser**

In `build_parser()`, before the `return` statement, add:

```python
    r = sub.add_parser("render", help="render an overlay video for every run")
    common(r)
    r.add_argument("--trails", action="store_true", help="draw centroid motion trails")
    r.add_argument("--trail-len", type=int, default=20,
                   help="trail length in frames; 0 for unbounded")
    r.add_argument("--alpha", type=float, default=0.65, help="mask fill opacity")
    r.add_argument("--outline", type=int, default=2,
                   help="contour thickness, 0 to disable")
    r.add_argument("--min-free-gb", type=int, default=20,
                   help="the mp4v intermediate is transient but not small")
    r.set_defaults(func=cmd_render)
```

Then change the return to hand back both subparsers:

```python
    return ap, {"track": t, "render": r}
```

- [ ] **Step 3: Verify both subcommands parse**

Run: `uv run batch.py track --help && uv run batch.py render --help`
Expected: both exit 0 and list their flags.

Run: `uv run --extra dev pytest tests/test_batch.py -v`
Expected: still all PASS.

- [ ] **Step 4: Compare the render plan against the shell script**

```bash
uv run batch.py render --data-dir /home/kyle/projects/test_data \
    --out-dir runs/batch --dry-run
DRY_RUN=1 ./render_batch.sh
```

Expected: the same runs planned and the same ones skipped. Resolve any difference before deleting the shell scripts.

- [ ] **Step 5: Delete the shell scripts**

```bash
git rm run_batch.sh render_batch.sh
```

- [ ] **Step 6: Commit**

```bash
git add batch.py
git commit -m "Add the render subcommand and retire the bash batch drivers"
```

---

### Task 7: Example config, README, and `.gitignore`

**Files:**
- Create: `segmentation.example.toml`
- Modify: `README.md`
- Modify: `.gitignore`

**Interfaces:**
- Consumes: every flag and section name established in Tasks 1–6.
- Produces: user-facing documentation. Nothing depends on it in code.

- [ ] **Step 1: Write `segmentation.example.toml`**

Values are the ones the shell scripts hardcoded, so an existing batch behaves identically.

```toml
# Copy to segmentation.toml and edit. Every value here is a default: any flag
# you pass on the command line still wins, and deleting a key just restores
# the script's built-in default.
#
# Relative paths resolve against THIS FILE's directory, not your shell's.

[paths]
# Searched recursively, at any depth. The structure under data_dir is mirrored
# into out_dir: <data_dir>/30_7_26/CS_T5.mp4 -> <out_dir>/30_7_26/CS_T5/
data_dir = "~/recordings"
out_dir = "runs/batch"
extensions = [".mp4"]

[subjects]
# How many animals are in each recording. Used to flag runs that came out with
# a different number of tracks, and by filter_tracks.py to keep the N
# longest-lived ids. Delete the key to switch every such check off.
expect_n = 7

# Prompt wording matters more than it should: "insect" scores 0.94 on this
# assay while "fly" scores 0.56-0.60 and returns nothing from the video model.
# Try a few on one recording before committing to a batch.
prompt = "insect"

[track]
# sam3, not sam3.1 -- 3.1 needs ~14.6 GiB in the detector mask head alone and
# will not fit in 16 GB at this resolution.
version = "sam3"
stride = 1
# Frames per session. Raise it for fewer seams, lower it if you hit CUDA OOM.
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
retry_chunk = 225    # second attempt for a video that OOMs the GPU at chunk
min_free_gb = 40     # extracted frames for the longest video need ~26 GB

[batch.render]
min_free_gb = 20     # the mp4v intermediate is transient but not small
```

- [ ] **Step 2: Add the config file to `.gitignore`**

Under the `# Environment` block, add:

```
# Local configuration; segmentation.example.toml is the committed reference.
segmentation.toml
```

- [ ] **Step 3: Rewrite the README's Setup and Pipeline sections**

Replace the existing `## Pipeline` section with a `## Configure` section followed by the pipeline, keeping the existing script table and extending it. Insert after the Setup section:

````markdown
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
````

- [ ] **Step 4: Document `run.json` in the Outputs section**

After the `masks.npz` subsection, add:

````markdown
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
````

- [ ] **Step 5: Update the stale example paths**

In the README, replace remaining `test_video.mp4` examples with `recording.mp4`, and `runs/full_s3` with `runs/rec1`, so no example depends on a file only this machine has. Leave the "Things that cost time to discover" and "Validation" sections untouched.

- [ ] **Step 6: Verify the example config is valid and complete**

Run:

```bash
uv run python -c "
import config, pathlib, shutil, tempfile
d = pathlib.Path(tempfile.mkdtemp())
shutil.copy('segmentation.example.toml', d / 'segmentation.toml')
cfg = config.load_config(d / 'segmentation.toml')
print(sorted(cfg))
missing = {s: sorted(config.SCHEMA[s] - set(cfg.get(s, {}))) for s in config.SCHEMA}
print('keys in SCHEMA but not in the example:', {k: v for k, v in missing.items() if v})
"
```

Expected: all seven sections listed, and no missing keys. A key in `SCHEMA` that the example omits is a documentation gap — add it.

- [ ] **Step 7: Run the whole test suite**

Run: `uv run --extra dev pytest -v`
Expected: all PASS.

- [ ] **Step 8: Commit**

```bash
git add segmentation.example.toml README.md .gitignore
git commit -m "Document configuration and the batch driver"
```

---

## Verification after the last task

Run all of these and confirm the output before calling the work done:

1. `uv run --extra dev pytest -v` — every test passes.
2. `uv run batch.py track --help` — exits 0.
3. With no `segmentation.toml` present: `uv run track_long.py --help` still shows `--stride` defaulting to 3, `--chunk` to 450, `--overlap` to 15.
4. With `segmentation.example.toml` copied to `segmentation.toml`: `uv run track_long.py --help` shows `--stride` defaulting to 1 and `--prompt` to `insect`.
5. `uv run batch.py track --dry-run` on a small folder lists the right videos and output directories.
6. `git status` — `segmentation.toml` is ignored, `segmentation.example.toml` is tracked.

**Not verified by any of this, and the README says so:** anything requiring a
GPU — model building, tracking, mask export, rendering. The end-to-end check is
still a real run on a real recording.
