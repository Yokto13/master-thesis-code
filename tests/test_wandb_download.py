import math
from pathlib import Path

import pandas as pd
from wandb_download import build_output_path, download_run, rows_to_df, scan_run_to_rows, should_skip


class MockRun:
    def __init__(self, name, run_id, rows, state="finished"):
        self.name = name
        self.id = run_id
        self.state = state
        self._rows = rows

    def scan_history(self):
        return iter(self._rows)


# ── build_output_path ──


def test_build_output_path_basic():
    p = build_output_path("wandb_exports", "my_group", "run_name", "abc123")
    assert p == Path("wandb_exports") / "my_group" / "run_name__abc123.csv"


def test_build_output_path_separates_name_and_id():
    p = build_output_path("out", "g", "the_run", "id99")
    assert p.name == "the_run__id99.csv"


def test_build_output_path_accepts_path_output_dir():
    p = build_output_path(Path("/tmp/exports"), "grp", "r", "1")
    assert p == Path("/tmp/exports/grp/r__1.csv")


# ── should_skip ──


def test_should_skip_returns_false_when_file_missing(tmp_path):
    p = tmp_path / "nonexistent.csv"
    assert should_skip(p, overwrite=False) is False


def test_should_skip_returns_true_when_file_exists_no_overwrite(tmp_path):
    p = tmp_path / "existing.csv"
    p.write_text("data")
    assert should_skip(p, overwrite=False) is True


def test_should_skip_returns_false_when_file_exists_with_overwrite(tmp_path):
    p = tmp_path / "existing.csv"
    p.write_text("data")
    assert should_skip(p, overwrite=True) is False


def test_should_skip_returns_false_when_file_missing_with_overwrite(tmp_path):
    p = tmp_path / "nonexistent.csv"
    assert should_skip(p, overwrite=True) is False


# ── rows_to_df ──


def test_rows_to_df_basic():
    rows = [
        {"_step": 0, "loss": 1.0},
        {"_step": 1, "loss": 0.5},
    ]
    df = rows_to_df(rows)
    assert list(df.columns) == ["_step", "loss"]
    assert len(df) == 2
    assert df["loss"].tolist() == [1.0, 0.5]


def test_rows_to_df_missing_keys_filled_with_nan():
    rows = [
        {"_step": 0, "a": 1.0},
        {"_step": 1, "b": 2.0},
    ]
    df = rows_to_df(rows)
    assert "a" in df.columns
    assert "b" in df.columns
    assert math.isnan(df.loc[0, "b"])
    assert math.isnan(df.loc[1, "a"])


def test_rows_to_df_empty_returns_empty_dataframe():
    df = rows_to_df([])
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 0


def test_rows_to_df_preserves_step_column():
    rows = [{"_step": 100, "x": 3.0}, {"_step": 200, "x": 4.0}]
    df = rows_to_df(rows)
    assert df["_step"].tolist() == [100, 200]


# ── scan_run_to_rows ──


def test_scan_run_to_rows_returns_list():
    run = MockRun("r", "id1", [{"_step": 0, "v": 1.0}, {"_step": 1, "v": 2.0}])
    rows = scan_run_to_rows(run)
    assert isinstance(rows, list)
    assert rows == [{"_step": 0, "v": 1.0}, {"_step": 1, "v": 2.0}]


def test_scan_run_to_rows_empty():
    run = MockRun("r", "id1", [])
    assert scan_run_to_rows(run) == []


# ── download_run ──


def test_download_run_writes_csv(tmp_path):
    rows = [{"_step": i, "val": float(i)} for i in range(10)]
    run = MockRun("my_run", "abc", rows)
    path, status = download_run(run, str(tmp_path), "my_group", overwrite=False)
    assert status == "ok"
    assert path.exists()
    df = pd.read_csv(path)
    assert len(df) == 10
    assert "_step" in df.columns


def test_download_run_correct_path(tmp_path):
    run = MockRun("the_run", "id42", [{"_step": 0}])
    path, _ = download_run(run, str(tmp_path), "grp", overwrite=False)
    assert path == tmp_path / "grp" / "the_run__id42.csv"


def test_download_run_creates_parent_dirs(tmp_path):
    run = MockRun("r", "id1", [{"_step": 0}])
    path, _ = download_run(run, str(tmp_path / "nested" / "dir"), "g", overwrite=False)
    assert path.exists()


def test_download_run_skips_existing(tmp_path):
    rows = [{"_step": 0, "val": 99.0}]
    run = MockRun("r", "id1", rows)
    path, _ = download_run(run, str(tmp_path), "g", overwrite=False)
    assert path.exists()
    original_mtime = path.stat().st_mtime

    run2 = MockRun("r", "id1", [{"_step": 0, "val": 1.0}])
    _, status = download_run(run2, str(tmp_path), "g", overwrite=False)
    assert status == "skipped"
    assert path.stat().st_mtime == original_mtime  # file not touched


def test_download_run_overwrites_when_flag_set(tmp_path):
    run = MockRun("r", "id1", [{"_step": 0, "val": 1.0}])
    path, _ = download_run(run, str(tmp_path), "g", overwrite=False)

    run2 = MockRun("r", "id1", [{"_step": 0, "val": 99.0}, {"_step": 1, "val": 100.0}])
    _, status = download_run(run2, str(tmp_path), "g", overwrite=True)
    assert status == "ok"
    df = pd.read_csv(path)
    assert len(df) == 2


def test_download_run_empty_history_creates_empty_file(tmp_path):
    # An empty run has no columns, so pandas writes a zero-byte file.
    # The file is created (so the run is recorded as processed), but is not CSV-parseable.
    run = MockRun("r", "id1", [])
    path, status = download_run(run, str(tmp_path), "g", overwrite=False)
    assert status == "ok"
    assert path.exists()
    assert path.read_bytes() in (b"", b"\n")  # pandas writes a bare newline for an empty DataFrame
