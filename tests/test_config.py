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
