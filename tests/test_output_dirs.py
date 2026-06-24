from pathlib import Path

from heal_compress.utils.io_utils import ensure_unique_dir


def test_ensure_unique_dir_uses_requested_empty_path(tmp_path):
    requested = tmp_path / "run"

    resolved = ensure_unique_dir(requested)

    assert resolved == requested
    assert resolved.is_dir()


def test_ensure_unique_dir_creates_suffix_for_nonempty_existing_path(tmp_path):
    requested = tmp_path / "run"
    requested.mkdir()
    (requested / "eval_log.txt").write_text("old run", encoding="utf-8")

    resolved = ensure_unique_dir(requested)

    assert resolved == tmp_path / "run_001"
    assert resolved.is_dir()
    assert (requested / "eval_log.txt").read_text(encoding="utf-8") == "old run"


def test_ensure_unique_dir_skips_existing_suffixes(tmp_path):
    requested = tmp_path / "run"
    requested.mkdir()
    (requested / "old.txt").write_text("old", encoding="utf-8")
    (tmp_path / "run_001").mkdir()

    resolved = ensure_unique_dir(requested)

    assert resolved == tmp_path / "run_002"
    assert resolved.is_dir()
