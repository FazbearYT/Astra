import json
import logging
import os
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import Lasso, LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR

logger = logging.getLogger(__name__)


MAX_HIDDEN_LAYERS = 8
MAX_HIDDEN_LAYER_WIDTH = 512


def configure_threading() -> None:
    n = str(os.cpu_count() or 4)
    for var in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ.setdefault(var, n)
    os.environ.setdefault("MKL_DYNAMIC", "FALSE")
    logger.debug("Threading configured: %s threads", n)


_PARAM_LIMITS: Dict[str, Dict[str, Tuple]] = {
    "RandomForest_Specialist": {
        "n_estimators": (int, 1, 500),
        "max_depth": (int, 1, 50),
        "min_samples_leaf": (int, 1, 100),
        "min_samples_split": (int, 2, 200),
        "random_state": (int, 0, 9999),
    },
    "SVM_Specialist": {
        "C": (float, 1e-4, 100.0),
        "gamma": None,
        "degree": (int, 1, 5),
        "random_state": (int, 0, 9999),
    },
    "GradientBoosting_Specialist": {
        "n_estimators": (int, 1, 500),
        "max_depth": (int, 1, 20),
        "learning_rate": (float, 1e-4, 1.0),
        "random_state": (int, 0, 9999),
    },
    "NeuralNetwork_Specialist": {
        "max_iter": (int, 10, 2000),
        "random_state": (int, 0, 9999),
    },
    "LogisticRegression_Specialist": {
        "max_iter": (int, 10, 2000),
        "C": (float, 1e-4, 100.0),
        "random_state": (int, 0, 9999),
    },
    "Ridge_Specialist": {
        "alpha": (float, 1e-6, 1000.0),
    },
    "Lasso_Specialist": {
        "alpha": (float, 1e-6, 1000.0),
        "max_iter": (int, 10, 5000),
    },
}

_ALLOWED_KERNELS = {"rbf", "linear", "poly", "sigmoid"}
_ALLOWED_ACTIVATIONS = {"relu", "tanh", "logistic", "identity"}
_ALLOWED_SOLVERS_LR = {"lbfgs", "liblinear", "saga", "sag", "newton-cg"}
_ALLOWED_SOLVERS_MLP = {"adam", "sgd", "lbfgs"}
_ALLOWED_PENALTIES_LR = {"l1", "l2", "elasticnet", None}


def validate_model_params(model_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
    limits = _PARAM_LIMITS.get(model_name)
    if limits is None:
        logger.warning("No param limits defined for %s - passing params as-is.", model_name)
        return params.copy()

    sanitised: Dict[str, Any] = {}

    if "random_state" in params:
        sanitised["random_state"] = int(params["random_state"])

    for key, value in params.items():
        if key == "random_state":
            continue

        if key == "kernel":
            v = str(value).lower()
            if v not in _ALLOWED_KERNELS:
                raise ValueError(f"Invalid kernel '{value}'. Allowed: {_ALLOWED_KERNELS}")
            sanitised[key] = v
            continue
        if key == "activation":
            v = str(value).lower()
            if v not in _ALLOWED_ACTIVATIONS:
                raise ValueError(f"Invalid activation '{value}'.")
            sanitised[key] = v
            continue
        if key == "solver":
            v = str(value).lower()
            allowed = (
                _ALLOWED_SOLVERS_MLP if "Neural" in model_name else _ALLOWED_SOLVERS_LR
            )
            if v not in allowed:
                raise ValueError(f"Invalid solver '{value}'. Allowed: {allowed}")
            sanitised[key] = v
            continue
        if key == "penalty":
            v = str(value).lower() if value is not None else None
            if v not in _ALLOWED_PENALTIES_LR:
                raise ValueError(f"Invalid penalty '{value}'.")
            sanitised[key] = v
            continue
        if key == "gamma":
            if isinstance(value, str):
                if value not in ("scale", "auto"):
                    raise ValueError(f"Invalid gamma string '{value}'.")
                sanitised[key] = value
            else:
                sanitised[key] = float(np.clip(float(value), 1e-9, 100.0))
            continue
        if key == "class_weight":
            if value not in ("balanced", None):
                raise ValueError(f"Invalid class_weight '{value}'.")
            sanitised[key] = value
            continue
        if key == "hidden_layer_sizes":
            raw = list(value) if isinstance(value, (list, tuple)) else [value]
            if len(raw) > MAX_HIDDEN_LAYERS:
                raise ValueError(
                    f"hidden_layer_sizes has {len(raw)} layers; max allowed is {MAX_HIDDEN_LAYERS}."
                )
            sizes = tuple(max(1, min(int(s), MAX_HIDDEN_LAYER_WIDTH)) for s in raw)
            sanitised[key] = sizes
            continue
        if key == "l1_ratio":
            sanitised[key] = float(np.clip(float(value), 0.0, 1.0))
            continue
        if key in ("probability",):
            sanitised[key] = bool(value)
            continue

        spec = limits.get(key)
        if spec is None:
            logger.debug("Dropping unknown param '%s' for %s", key, model_name)
            continue
        typ, lo, hi = spec
        try:
            sanitised[key] = typ(np.clip(typ(value), lo, hi))
        except (TypeError, ValueError) as exc:
            logger.warning("Param '%s' invalid (%s) - skipping.", key, exc)

    return sanitised


class SpecializedModel:
    def __init__(
        self,
        name: str,
        model: BaseEstimator,
        profile_requirements: Dict[str, Any],
        description: str = "",
        custom_scoring_weights: Optional[Dict[str, float]] = None,
    ) -> None:
        self.name = name
        self.model = model
        self.profile_requirements = profile_requirements
        self.description = description or f"Specialized model: {name}"
        self.performance_metrics: Dict[str, float] = {}
        self.is_trained = False
        self.training_time: Optional[float] = None
        self.custom_scoring_weights = custom_scoring_weights or {}

    def fit(self, X: np.ndarray, y: np.ndarray, **fit_params) -> "SpecializedModel":
        start = time.time()
        if isinstance(self.model, Pipeline):
            estimator_name = self.model.steps[-1][0]
            prefixed = {f"{estimator_name}__{k}": v for k, v in fit_params.items()}
            self.model.fit(X, y, **prefixed)
        else:
            self.model.fit(X, y, **fit_params)
        self.is_trained = True
        self.training_time = time.time() - start
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_trained:
            raise ValueError(f"Model '{self.name}' has not been trained yet.")
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.is_trained:
            raise ValueError(f"Model '{self.name}' has not been trained yet.")
        if not hasattr(self.model, "predict_proba"):
            raise AttributeError(f"Model '{self.name}' does not support predict_proba.")
        return self.model.predict_proba(X)

    def evaluate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        metrics: Optional[List[str]] = None,
        task_type: str = "classification",
    ) -> Dict[str, float]:
        y_pred = self.predict(X)
        results: Dict[str, float] = {}

        if task_type == "classification":
            if metrics is None:
                metrics = ["accuracy", "precision", "recall", "f1"]
            if "accuracy" in metrics:
                results["accuracy"] = float(accuracy_score(y, y_pred))
            if "precision" in metrics:
                results["precision"] = float(
                    precision_score(y, y_pred, average="weighted", zero_division=0)
                )
            if "recall" in metrics:
                results["recall"] = float(
                    recall_score(y, y_pred, average="weighted", zero_division=0)
                )
            if "f1" in metrics:
                results["f1"] = float(
                    f1_score(y, y_pred, average="weighted", zero_division=0)
                )
        else:
            if metrics is None:
                metrics = ["r2", "mse", "rmse", "mae"]
            if "r2" in metrics:
                results["r2"] = float(r2_score(y, y_pred))
            mse_val = float(mean_squared_error(y, y_pred))
            if "mse" in metrics:
                results["mse"] = mse_val
            if "rmse" in metrics:
                results["rmse"] = float(np.sqrt(mse_val))
            if "mae" in metrics:
                results["mae"] = float(mean_absolute_error(y, y_pred))

        self.performance_metrics.update(results)
        return results

    def cross_validate(
        self, X: np.ndarray, y: np.ndarray, cv: int = 5, task_type: str = "classification"
    ) -> Dict[str, Any]:
        scoring = "accuracy" if task_type == "classification" else "r2"
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
            scores = cross_val_score(self.model, X, y, cv=cv, scoring=scoring, n_jobs=-1)
        return {
            "cv_mean": float(scores.mean()),
            "cv_std": float(scores.std()),
            "cv_scores": scores.tolist(),
        }

    def matches_profile(self, data_profile: Dict[str, Any]) -> float:
        score = 0.0
        max_score = 0.0

        if "data_complexity" in self.profile_requirements:
            max_score += 1.0
            if data_profile.get("data_complexity") == self.profile_requirements["data_complexity"]:
                score += 1.0

        if "class_balance_min" in self.profile_requirements:
            max_score += 1.0
            bal = data_profile.get("class_balance_ratio")
            if bal is not None and bal >= self.profile_requirements["class_balance_min"]:
                score += 1.0

        if "min_samples" in self.profile_requirements:
            max_score += 1.0
            if data_profile.get("n_samples", 0) >= self.profile_requirements["min_samples"]:
                score += 1.0

        if "n_features_range" in self.profile_requirements:
            max_score += 1.0
            n_feat = data_profile.get("n_features", 0)
            lo, hi = self.profile_requirements["n_features_range"]
            if lo <= n_feat <= hi:
                score += 1.0

        return score / max_score if max_score > 0 else 0.5

    def get_info(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "profile_requirements": self.profile_requirements,
            "performance_metrics": self.performance_metrics,
            "is_trained": self.is_trained,
            "training_time": self.training_time,
            "custom_scoring_weights": self.custom_scoring_weights,
        }

    def save(self, filepath: str) -> None:
        joblib.dump(
            {
                "name": self.name,
                "model": self.model,
                "profile_requirements": self.profile_requirements,
                "description": self.description,
                "performance_metrics": self.performance_metrics,
                "is_trained": self.is_trained,
                "training_time": self.training_time,
                "custom_scoring_weights": self.custom_scoring_weights,
            },
            filepath,
        )
        logger.debug("Model saved: %s", filepath)

    @classmethod
    def load(cls, filepath: str) -> "SpecializedModel":
        data = joblib.load(filepath)
        obj = cls(
            name=data["name"],
            model=data["model"],
            profile_requirements=data["profile_requirements"],
            description=data["description"],
            custom_scoring_weights=data.get("custom_scoring_weights"),
        )
        obj.performance_metrics = data["performance_metrics"]
        obj.is_trained = data["is_trained"]
        obj.training_time = data["training_time"]
        return obj


class AdaptiveModelSelector:
    def __init__(self) -> None:
        self.models: List[SpecializedModel] = []
        self.best_model: Optional[SpecializedModel] = None
        self.data_profile: Optional[Dict] = None
        self.selection_history: List[Dict] = []

    def register_model(self, model: SpecializedModel) -> None:
        self.models.append(model)

    def _get_sklearn_model_instance(
        self, model_name: str, params: Dict[str, Any], task_type: str = "classification"
    ) -> BaseEstimator:
        p = params.copy()
        if task_type == "classification":
            if model_name == "RandomForest_Specialist":
                return RandomForestClassifier(**p)
            if model_name == "SVM_Specialist":
                return Pipeline([("scaler", StandardScaler()), ("svc", SVC(**p))])
            if model_name == "GradientBoosting_Specialist":
                return GradientBoostingClassifier(**p)
            if model_name == "NeuralNetwork_Specialist":
                return Pipeline([("scaler", StandardScaler()), ("mlp", MLPClassifier(**p))])
            if model_name == "LogisticRegression_Specialist":
                return Pipeline([("scaler", StandardScaler()), ("lr", LogisticRegression(**p))])
        else:
            if model_name == "RandomForest_Specialist":
                return RandomForestRegressor(**p)
            if model_name == "SVM_Specialist":
                return Pipeline([("scaler", StandardScaler()), ("svr", SVR(**p))])
            if model_name == "GradientBoosting_Specialist":
                return GradientBoostingRegressor(**p)
            if model_name == "NeuralNetwork_Specialist":
                return Pipeline([("scaler", StandardScaler()), ("mlp", MLPRegressor(**p))])
            if model_name == "Ridge_Specialist":
                return Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(**p))])
            if model_name == "Lasso_Specialist":
                return Pipeline([("scaler", StandardScaler()), ("lasso", Lasso(**p))])

        raise ValueError(f"Unknown model '{model_name}' for task '{task_type}'")

    def register_models_from_pipeline_config(
        self, pipeline_config, task_type: str = "classification"
    ) -> None:
        self.models = []
        for model_key, model_cfg in pipeline_config.models.items():
            if not model_cfg.enabled:
                continue
            try:
                sklearn_model = self._get_sklearn_model_instance(
                    model_cfg.name, model_cfg.params, task_type
                )
                self.register_model(
                    SpecializedModel(
                        name=model_cfg.name,
                        model=sklearn_model,
                        profile_requirements=model_cfg.profile_requirements,
                        description=model_cfg.description,
                        custom_scoring_weights=model_cfg.custom_scoring_weights,
                    )
                )
            except (ValueError, TypeError) as exc:
                logger.warning("Skipping model '%s': %s", model_key, exc)

        if not self.models:
            raise ValueError("No models were enabled or successfully registered.")

    def profile_and_select(
        self,
        X: np.ndarray,
        y: np.ndarray,
        data_profile: Optional[Dict] = None,
        cv_folds: int = 5,
        test_size: float = 0.2,
        random_state: int = 42,
        use_cv_in_scoring: bool = False,
        global_accuracy_weight: float = 0.7,
        global_f1_weight: float = 0.3,
        global_cv_weight: float = 0.0,
    ) -> Tuple["SpecializedModel", np.ndarray, np.ndarray]:
        configure_threading()

        if data_profile is None:
            data_profile = {
                "n_samples": X.shape[0],
                "n_features": X.shape[1],
                "data_complexity": "medium",
                "class_balance_ratio": 1.0,
                "task_type": "classification",
            }

        self.data_profile = data_profile
        task_type: str = data_profile.get("task_type", "classification")

        stratify_y = y if (task_type == "classification" and len(np.unique(y)) > 1) else None
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify_y,
        )

        logger.info(
            "Starting model selection | task=%s | train=%d test=%d | cv_folds=%d",
            task_type, X_train.shape[0], X_test.shape[0], cv_folds,
        )

        if not self.models:
            raise ValueError("No models are registered for selection.")

        model_scores = [
            (model, model.matches_profile(data_profile)) for model in self.models
        ]
        model_scores.sort(key=lambda x: x[1], reverse=True)

        results_table: List[Dict] = []
        total = len(model_scores)

        for idx, (model, profile_score) in enumerate(model_scores, 1):
            logger.info("[%d/%d] %s - profile_score=%.3f", idx, total, model.name, profile_score)
            try:
                t0 = time.time()
                model.fit(X_train, y_train)
                train_time = time.time() - t0
                logger.info("  Trained in %.2fs", train_time)

                metrics = model.evaluate(X_test, y_test, task_type=task_type)
                logger.info("  Metrics: %s", {k: f"{v:.4f}" for k, v in metrics.items()})

                cv_results: Dict[str, Any] = {"cv_mean": 0.0, "cv_std": 0.0, "cv_scores": []}
                if use_cv_in_scoring or global_cv_weight > 0:
                    cv_results = model.cross_validate(
                        X_train, y_train, cv=cv_folds, task_type=task_type
                    )
                    logger.info("  CV mean=%.4f +/- %.4f", cv_results["cv_mean"], cv_results["cv_std"])

                acc_w = model.custom_scoring_weights.get("accuracy_weight", global_accuracy_weight)
                f1_w = model.custom_scoring_weights.get("f1_weight", global_f1_weight)
                cv_w = model.custom_scoring_weights.get("cv_weight", global_cv_weight)

                if task_type == "classification":
                    raw = (
                        metrics.get("accuracy", 0.0) * acc_w
                        + metrics.get("f1", 0.0) * f1_w
                        + (cv_results["cv_mean"] * cv_w if use_cv_in_scoring else 0.0)
                    )
                    total_w = acc_w + f1_w + (cv_w if use_cv_in_scoring else 0.0)
                    final_score = raw / total_w if total_w > 0 else raw
                else:
                    r2_w = model.custom_scoring_weights.get("r2_weight", global_accuracy_weight)
                    rmse_w = model.custom_scoring_weights.get("rmse_weight", global_f1_weight)
                    rmse_score = 1.0 / (1.0 + metrics.get("rmse", 1.0))
                    raw = (
                        metrics.get("r2", 0.0) * r2_w
                        + rmse_score * rmse_w
                        + (cv_results["cv_mean"] * cv_w if use_cv_in_scoring else 0.0)
                    )
                    total_w = r2_w + rmse_w + (cv_w if use_cv_in_scoring else 0.0)
                    final_score = raw / total_w if total_w > 0 else raw

                logger.info("  Final score=%.4f", final_score)

                results_table.append({
                    "model": model.name,
                    "profile_score": profile_score,
                    "task_type": task_type,
                    **metrics,
                    "cv_mean": cv_results["cv_mean"],
                    "final_score": final_score,
                    "training_time": train_time,
                })
                self.selection_history.append({
                    "model": model.name,
                    "profile_score": profile_score,
                    "metrics": metrics,
                    "cv_results": cv_results,
                    "final_score": final_score,
                    "task_type": task_type,
                    "timestamp": datetime.now().isoformat(),
                })

            except Exception as exc:
                logger.error("Error processing model '%s': %s", model.name, exc, exc_info=True)
                continue

        if not results_table:
            raise ValueError("No models were successfully trained or evaluated.")

        sorted_results = sorted(results_table, key=lambda x: x["final_score"], reverse=True)

        header = f"{'Model':<32} {'Score':<10}"
        logger.info("Model comparison:\n%s\n%s", header, "-" * len(header))
        for r in sorted_results:
            logger.info("  %-32s %.4f", r["model"], r["final_score"])

        best_result = sorted_results[0]
        best_model = next((m for m in self.models if m.name == best_result["model"]), None)
        if best_model is None:
            raise RuntimeError(f"Best model '{best_result['model']}' not found in registered models.")

        self.best_model = best_model
        logger.info("Selected: %s (score=%.4f)", best_model.name, best_result["final_score"])
        return best_model, X_test, y_test

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.best_model is None:
            raise ValueError("Call profile_and_select() first.")
        return self.best_model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.best_model is None:
            raise ValueError("Call profile_and_select() first.")
        return self.best_model.predict_proba(X)

    def get_model_info(self) -> Dict[str, Any]:
        return self.best_model.get_info() if self.best_model else {}

    def save_all_models(self, directory: str = "saved_models") -> int:
        os.makedirs(directory, exist_ok=True)
        saved = 0
        for model in self.models:
            if model.is_trained:
                fp = os.path.join(directory, f"{model.name}.pkl")
                model.save(fp)
                saved += 1
        logger.info("Saved %d trained models to '%s'.", saved, directory)

        history_path = os.path.join(directory, "selection_history.json")
        try:
            with open(history_path, "w", encoding="utf-8") as f:
                json.dump(self.selection_history, f, indent=2, ensure_ascii=False)
        except OSError as exc:
            logger.warning("Could not save history: %s", exc)
        return saved

    def load_models(self, directory: str = "saved_models") -> None:
        self.models = []
        directory_path = Path(directory)
        if not directory_path.exists():
            logger.warning("Model directory '%s' does not exist.", directory)
            return

        trusted_root = directory_path.resolve()
        loaded = 0
        for child in directory_path.iterdir():
            if not child.is_file() or child.suffix != ".pkl":
                continue
            resolved = child.resolve()
            try:
                resolved.relative_to(trusted_root)
            except ValueError:
                logger.warning("Skipping suspicious path: %s", child)
                continue
            try:
                model = SpecializedModel.load(str(resolved))
                self.models.append(model)
                loaded += 1
            except Exception as exc:
                logger.warning("Failed to load '%s': %s", child.name, exc)

        logger.info("Loaded %d models from '%s'.", loaded, directory)

        history_path = directory_path / "selection_history.json"
        if history_path.exists():
            try:
                with open(history_path, "r", encoding="utf-8") as f:
                    history = json.load(f)
                self.selection_history = history
                if history:
                    best_entry = max(
                        history, key=lambda x: x.get("final_score", -1) or -1
                    )
                    best_name = best_entry["model"]
                    for m in self.models:
                        if m.name == best_name:
                            self.best_model = m
                            logger.info("Best model restored: %s", m.name)
                            break
            except (OSError, json.JSONDecodeError, KeyError) as exc:
                logger.warning("Could not load selection history: %s", exc)
