"""Tests for the MLflow path-portability script."""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from make_mlflow_portable import rewrite, to_relative  # noqa: E402

ABSOLUTE = "/Users/someone/work/loan-approval-mlops/mlartifacts/abc123/artifacts"


@pytest.fixture
def database(tmp_path: Path) -> Path:
    path = tmp_path / "mlflow.db"
    connection = sqlite3.connect(path)
    connection.execute("create table experiments (experiment_id integer, artifact_location text)")
    connection.execute("create table runs (run_uuid text, artifact_uri text)")
    connection.execute(
        "insert into experiments values (1, '/Users/someone/work/loan-approval-mlops/mlartifacts')"
    )
    connection.execute("insert into runs values ('abc123', ?)", (ABSOLUTE,))
    connection.execute("insert into runs values ('def456', 'mlartifacts/def456/artifacts')")
    connection.commit()
    connection.close()
    return path


class TestToRelative:
    def test_strips_everything_up_to_the_project(self):
        assert to_relative(ABSOLUTE, "loan-approval-mlops") == "mlartifacts/abc123/artifacts"

    def test_leaves_an_already_relative_path_alone(self):
        assert to_relative("mlartifacts/abc/artifacts", "loan-approval-mlops") == "mlartifacts/abc/artifacts"

    def test_leaves_an_unrelated_path_alone(self):
        assert to_relative("/somewhere/else/mlruns/0", "loan-approval-mlops") == "/somewhere/else/mlruns/0"

    def test_handles_a_file_uri(self):
        uri = "file:///Users/someone/loan-approval-mlops/mlartifacts/run/artifacts"
        assert to_relative(uri, "loan-approval-mlops") == "mlartifacts/run/artifacts"


class TestRewrite:
    def test_makes_stored_paths_relative(self, database: Path):
        rewrite(database, "loan-approval-mlops")
        connection = sqlite3.connect(database)
        locations = [row[0] for row in connection.execute("select artifact_location from experiments")]
        uris = [row[0] for row in connection.execute("select artifact_uri from runs")]
        connection.close()
        assert locations == ["mlartifacts"]
        assert uris == ["mlartifacts/abc123/artifacts", "mlartifacts/def456/artifacts"]

    def test_reports_how_many_rows_changed(self, database: Path):
        changed = rewrite(database, "loan-approval-mlops")
        assert changed["experiments.artifact_location"] == 1
        assert changed["runs.artifact_uri"] == 1      # the already-relative row is skipped

    def test_dry_run_leaves_the_database_untouched(self, database: Path):
        changed = rewrite(database, "loan-approval-mlops", dry_run=True)
        assert changed["runs.artifact_uri"] == 1
        connection = sqlite3.connect(database)
        stored = connection.execute("select artifact_uri from runs limit 1").fetchone()[0]
        connection.close()
        assert stored == ABSOLUTE

    def test_is_idempotent(self, database: Path):
        rewrite(database, "loan-approval-mlops")
        assert rewrite(database, "loan-approval-mlops") == {
            "experiments.artifact_location": 0,
            "runs.artifact_uri": 0,
        }

    def test_missing_database_raises(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            rewrite(tmp_path / "absent.db", "loan-approval-mlops")
