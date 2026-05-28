"""Security regression tests for the Astra ML System.

Covers:
    * _sanitize_filename — CWE-22 (Path Traversal) on uploaded filenames.
    * validate_model_params — CWE-400 / CWE-20 (DoS via unbounded params,
      injection of unknown solvers/kernels).
    * _effective_cv — graceful handling of degenerate class distributions.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.model_selector import _effective_cv, validate_model_params
from web_app import _sanitize_filename


# ---------------------------------------------------------------------------
# CWE-22: filename sanitiser
# ---------------------------------------------------------------------------

class TestSanitizeFilename:
    def test_strips_parent_directory_traversal(self):
        assert _sanitize_filename("../../etc/passwd").startswith("passwd")
        assert ".." not in _sanitize_filename("../../etc/passwd")

    def test_strips_absolute_paths(self):
        out = _sanitize_filename("/etc/shadow")
        assert "/" not in out and "\\" not in out

    def test_strips_windows_path(self):
        out = _sanitize_filename(r"C:\Windows\System32\evil.csv")
        assert ":" not in out and "\\" not in out

    def test_strips_null_byte(self):
        out = _sanitize_filename("evil\x00.csv")
        assert "\x00" not in out

    def test_strips_shell_metacharacters(self):
        out = _sanitize_filename("a;rm -rf /.csv")
        assert ";" not in out and " " not in out

    def test_returns_csv_extension(self):
        assert _sanitize_filename("anything").endswith(".csv")

    def test_truncates_long_names(self):
        out = _sanitize_filename("a" * 500)
        # 64-char stem cap + ".csv"
        assert len(out) <= 68

    def test_empty_input_does_not_crash(self):
        assert _sanitize_filename("") == "dataset.csv"
        assert _sanitize_filename("...") == "dataset.csv"


# ---------------------------------------------------------------------------
# CWE-400 / CWE-20: model parameter validation
# ---------------------------------------------------------------------------

class TestValidateModelParams:
    def test_clamps_huge_n_estimators(self):
        out = validate_model_params(
            "RandomForest_Specialist", {"n_estimators": 10_000_000}
        )
        assert out["n_estimators"] <= 500

    def test_clamps_huge_max_depth(self):
        out = validate_model_params(
            "RandomForest_Specialist", {"max_depth": 999_999}
        )
        assert out["max_depth"] <= 50

    def test_drops_unknown_numeric_param(self):
        out = validate_model_params(
            "RandomForest_Specialist", {"n_estimators": 50, "bogus": 1}
        )
        assert "bogus" not in out
        assert out["n_estimators"] == 50

    def test_rejects_unknown_solver(self):
        with pytest.raises(ValueError):
            validate_model_params(
                "LogisticRegression_Specialist", {"solver": "drop-table"}
            )

    def test_rejects_unknown_kernel(self):
        with pytest.raises(ValueError):
            validate_model_params("SVM_Specialist", {"kernel": "evil"})

    def test_rejects_unknown_activation(self):
        with pytest.raises(ValueError):
            validate_model_params(
                "NeuralNetwork_Specialist", {"activation": "sigmoid_evil"}
            )

    def test_clamps_hidden_layer_widths(self):
        out = validate_model_params(
            "NeuralNetwork_Specialist", {"hidden_layer_sizes": (100_000, 100_000)}
        )
        assert all(s <= 512 for s in out["hidden_layer_sizes"])

    def test_rejects_too_deep_network(self):
        with pytest.raises(ValueError):
            validate_model_params(
                "NeuralNetwork_Specialist",
                {"hidden_layer_sizes": [10] * 100},
            )

    def test_gamma_string_must_be_known(self):
        with pytest.raises(ValueError):
            validate_model_params("SVM_Specialist", {"gamma": "weird"})

    def test_gamma_numeric_clamped(self):
        out = validate_model_params("SVM_Specialist", {"gamma": 1e9})
        assert out["gamma"] <= 100.0


# ---------------------------------------------------------------------------
# CV folds guard
# ---------------------------------------------------------------------------

class TestEffectiveCV:
    def test_classification_reduces_to_min_class_count(self):
        y = np.array([0, 0, 0, 1, 1])  # min class = 2
        assert _effective_cv(10, y, "classification") == 2

    def test_classification_keeps_requested_when_feasible(self):
        y = np.array([0] * 20 + [1] * 20)
        assert _effective_cv(5, y, "classification") == 5

    def test_regression_keeps_requested(self):
        y = np.arange(100, dtype=float)
        assert _effective_cv(5, y, "regression") == 5

    def test_minimum_is_two(self):
        y = np.array([0, 1])
        assert _effective_cv(1, y, "regression") == 2
