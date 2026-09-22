"""Tests for configuration loading and the IO helpers."""

from __future__ import annotations

import pandas as pd
import pytest

from src.config import PROJECT_ROOT, load_config, resolve_path
from src.utils.io import ensure_parent, read_csv, read_json, write_csv, write_json
from src.utils.logger import get_logger


class TestConfig:
    def test_every_section_is_present(self, config):
        for section in ("base", "data", "validation", "transformation", "training",
                        "evaluation", "serving", "monitoring"):
            assert config.section(section)

    def test_missing_section_raises(self, config):
        with pytest.raises(KeyError):
            config.section("does_not_exist")

    def test_dotted_lookup(self, config):
        assert config.get_nested("base.random_state") == 42
        assert config.get_nested("training.candidates.random_forest.n_estimators") == 300

    def test_dotted_lookup_default(self, config):
        assert config.get_nested("base.nope", "fallback") == "fallback"
        assert config.get_nested("base.random_state.deeper", "fallback") == "fallback"

    def test_paths_resolve_against_the_project_root(self, config):
        assert config.get_path("data.raw_path") == PROJECT_ROOT / "data/raw/loan_applications.csv"

    def test_unknown_path_key_raises(self, config):
        with pytest.raises(KeyError):
            config.get_path("data.nonexistent_path")

    def test_absolute_paths_are_left_alone(self, tmp_path):
        assert resolve_path(tmp_path / "x.csv") == tmp_path / "x.csv"

    def test_override_file_can_be_loaded(self, tmp_path):
        params = tmp_path / "params.yaml"
        params.write_text("base:\n  random_state: 7\n", encoding="utf-8")
        assert load_config(params).get_nested("base.random_state") == 7

    def test_thresholds_are_sane(self, config):
        assert 0 < config.get_nested("serving.approval_threshold") < 1
        assert config.get_nested("evaluation.min_roc_auc") >= 0.5


class TestIoHelpers:
    def test_json_round_trip(self, tmp_path):
        path = write_json(tmp_path / "nested" / "report.json", {"metric": 0.9})
        assert path.exists()
        assert read_json(path) == {"metric": 0.9}

    def test_csv_round_trip(self, tmp_path):
        frame = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
        path = write_csv(tmp_path / "deep" / "data.csv", frame)
        pd.testing.assert_frame_equal(read_csv(path), frame)

    def test_ensure_parent_creates_directories(self, tmp_path):
        target = ensure_parent(tmp_path / "a" / "b" / "c.txt")
        assert target.parent.is_dir()


class TestLogger:
    def test_logger_is_named_and_configured(self):
        logger = get_logger("test.logger")
        assert logger.name == "test.logger"
        assert logger.getEffectiveLevel() > 0
