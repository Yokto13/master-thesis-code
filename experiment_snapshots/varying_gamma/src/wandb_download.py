from pathlib import Path

import pandas as pd


def build_output_path(output_dir, group: str, run_name: str, run_id: str) -> Path:
    """Return the CSV path for a run: <output_dir>/<group>/<run_name>__<run_id>.csv"""
    return Path(output_dir) / group / f"{run_name}__{run_id}.csv"


def should_skip(path: Path, overwrite: bool) -> bool:
    """Return True if the file already exists and overwrite is not requested."""
    return path.exists() and not overwrite


def scan_run_to_rows(run) -> list[dict]:
    """Return all history rows for a run as a list of dicts."""
    return list(run.scan_history())


def rows_to_df(rows: list[dict]) -> pd.DataFrame:
    """Convert a list of history row dicts to a DataFrame.

    Keys that are absent from some rows become NaN in those positions.
    Returns an empty DataFrame (no columns) when rows is empty.
    """
    return pd.DataFrame(rows)


def download_run(run, output_dir, group: str, overwrite: bool) -> tuple[Path, str]:
    """Download a single run's full history and save it as a CSV.

    Returns (path, status) where status is one of "ok", "skipped", or "error: <msg>".
    """
    path = build_output_path(output_dir, group, run.name, run.id)
    if should_skip(path, overwrite):
        return path, "skipped"
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = scan_run_to_rows(run)
    df = rows_to_df(rows)
    df.to_csv(path, index=False)
    return path, "ok"
