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
