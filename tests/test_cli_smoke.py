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
           "render_colors.py", "filter_tracks.py", "batch.py"]


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
