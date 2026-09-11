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
        if section == "paths" and "extensions" in body:
            # A bare string passes isinstance(x, list) here as False, which is
            # exactly what we want to reject -- but it also passes as an
            # iterable of characters, so a string that slipped through would
            # get zipped apart into single-character "extensions" downstream
            # and silently discover 0 videos. Bad config must be a hard error,
            # never a silent no-op.
            exts = body["extensions"]
            if isinstance(exts, str) or not isinstance(exts, (list, tuple)) or \
                    not all(isinstance(e, str) for e in exts):
                raise SystemExit(
                    f"[config] {path}: [paths] extensions must be a list of "
                    f"strings, e.g. extensions = [\".mp4\", \".avi\"]; got {exts!r}")


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
