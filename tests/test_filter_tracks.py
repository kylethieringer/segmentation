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
