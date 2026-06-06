from dataclasses import dataclass, field
from math import sqrt
from typing import Any, Dict, Optional
import json


@dataclass
class ModelConfig:
    name: str
    enabled: bool = True
    params: Dict[str, Any] = field(default_factory=dict)
    profile_requirements: Dict[str, Any] = field(default_factory=dict)
    description: str = ""
    custom_scoring_weights: Dict[str, float] = field(default_factory=dict)


@dataclass
class PipelineConfig:
    models: Dict[str, ModelConfig] = field(default_factory=dict)
    training: Dict[str, Any] = field(default_factory=dict)
    scoring: Dict[str, float] = field(default_factory=dict)

    def save(self, filepath: str) -> None:
        serializable_models = {k: v.__dict__ for k, v in self.models.items()}
        data = {
            "models": serializable_models,
            "training": self.training,
            "scoring": self.scoring,
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, filepath: str) -> "PipelineConfig":
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        models_data = data.get("models", {})
        models = {k: ModelConfig(**v) for k, v in models_data.items()}
        return cls(
            models=models,
            training=data.get("training", {}),
            scoring=data.get("scoring", {}),
        )


def adapt_params_to_profile(
    model_name: str,
    params: Dict[str, Any],
    profile: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    if not profile:
        return dict(params)

    adapted = dict(params)
    n_samples = int(profile.get("n_samples", 0) or 0)
    n_features = int(profile.get("n_features", 0) or 0)
    complexity = profile.get("data_complexity", "medium")
    notes: list[str] = []

    if model_name in ("RandomForest_Specialist", "GradientBoosting_Specialist"):
        base_n = int(adapted.get("n_estimators", 100))
        if n_samples and n_samples < 200:
            adapted["n_estimators"] = max(20, base_n // 2)
            notes.append(f"n_estimators {base_n}→{adapted['n_estimators']} (мало данных)")
        elif n_samples and n_samples > 50_000:
            adapted["n_estimators"] = min(500, base_n + 50)
            notes.append(f"n_estimators {base_n}→{adapted['n_estimators']} (много данных)")

        base_d = adapted.get("max_depth")
        if isinstance(base_d, int) and n_features > 50:
            new_d = max(3, base_d // 2)
            if new_d != base_d:
                adapted["max_depth"] = new_d
                notes.append(f"max_depth {base_d}→{new_d} (высокая размерность)")

    if model_name == "NeuralNetwork_Specialist":
        sizes = adapted.get("hidden_layer_sizes", (100, 50))
        if isinstance(sizes, list):
            sizes = tuple(sizes)
        if n_samples and n_samples < 500:
            adapted["hidden_layer_sizes"] = tuple(max(10, s // 2) for s in sizes)
            notes.append("hidden_layer_sizes уменьшены (мало данных)")
        base_iter = int(adapted.get("max_iter", 500))
        if n_samples and n_samples > 10_000:
            adapted["max_iter"] = min(2000, max(base_iter, 800))
            if adapted["max_iter"] != base_iter:
                notes.append(f"max_iter {base_iter}→{adapted['max_iter']} (много данных)")

    if model_name == "KNN_Specialist" and n_samples:
        ideal_k = max(3, min(25, int(sqrt(n_samples / 2)) | 1))
        if ideal_k % 2 == 0:
            ideal_k += 1
        adapted["n_neighbors"] = ideal_k
        notes.append(f"n_neighbors={ideal_k} (≈√(n/2))")

    if model_name == "DecisionTree_Specialist":
        base_d = adapted.get("max_depth")
        if complexity == "high" and isinstance(base_d, int):
            new_d = min(30, base_d + 5)
            if new_d != base_d:
                adapted["max_depth"] = new_d
                notes.append(f"max_depth {base_d}→{new_d} (сложные данные)")

    if model_name == "SVM_Specialist":
        if n_samples and n_samples > 20_000 and adapted.get("kernel") == "rbf":
            adapted["kernel"] = "linear"
            notes.append("kernel rbf→linear (большой объём данных)")

    if model_name == "LogisticRegression_Specialist":
        if n_features and n_samples and n_features > n_samples:
            adapted["penalty"] = "l1"
            adapted["solver"] = "saga"
            notes.append("penalty=l1, solver=saga (фичей больше чем строк)")

    adapted["_adaptation_notes"] = notes
    return adapted


def _classification_models_full() -> Dict[str, ModelConfig]:
    return {
        "RandomForest_Specialist": ModelConfig(
            name="RandomForest_Specialist",
            enabled=True,
            params={"n_estimators": 100, "max_depth": 10, "random_state": 42},
            profile_requirements={
                "min_samples": 50,
                "n_features_range": [2, 100],
                "data_complexity": "medium",
            },
            description="Ансамбль деревьев, подходит для большинства табличных задач.",
        ),
        "SVM_Specialist": ModelConfig(
            name="SVM_Specialist",
            enabled=True,
            params={"C": 1.0, "kernel": "rbf", "probability": True, "random_state": 42},
            profile_requirements={
                "min_samples": 30,
                "n_features_range": [2, 50],
                "data_complexity": "medium",
            },
            description="Ядровая модель, хороша на средних, хорошо разделимых данных.",
        ),
        "GradientBoosting_Specialist": ModelConfig(
            name="GradientBoosting_Specialist",
            enabled=True,
            params={"n_estimators": 150, "learning_rate": 0.05, "max_depth": 3, "random_state": 42},
            profile_requirements={
                "min_samples": 100,
                "n_features_range": [5, 100],
                "data_complexity": "high",
            },
            description="Бустинг для сложных нелинейных зависимостей.",
        ),
        "NeuralNetwork_Specialist": ModelConfig(
            name="NeuralNetwork_Specialist",
            enabled=True,
            params={
                "hidden_layer_sizes": (50, 25),
                "max_iter": 500,
                "activation": "relu",
                "solver": "adam",
                "random_state": 42,
            },
            profile_requirements={
                "min_samples": 200,
                "n_features_range": [10, 150],
                "data_complexity": "high",
            },
            description="Многослойный перцептрон для сложных нелинейных данных.",
        ),
        "LogisticRegression_Specialist": ModelConfig(
            name="LogisticRegression_Specialist",
            enabled=True,
            params={"solver": "lbfgs", "penalty": "l2", "max_iter": 500, "random_state": 42},
            profile_requirements={
                "min_samples": 50,
                "n_features_range": [2, 100],
                "data_complexity": "medium",
            },
            description="Линейная модель — интерпретируемый baseline.",
        ),
        "KNN_Specialist": ModelConfig(
            name="KNN_Specialist",
            enabled=True,
            params={"n_neighbors": 5, "weights": "distance", "p": 2},
            profile_requirements={
                "min_samples": 20,
                "n_features_range": [2, 30],
                "data_complexity": "low",
            },
            description="K ближайших соседей — без обучения, опирается на геометрию.",
        ),
        "GaussianNB_Specialist": ModelConfig(
            name="GaussianNB_Specialist",
            enabled=True,
            params={},
            profile_requirements={
                "min_samples": 20,
                "n_features_range": [2, 200],
                "data_complexity": "low",
            },
            description="Наивный Байес — мгновенное обучение, требует слабых корреляций фичей.",
        ),
        "DecisionTree_Specialist": ModelConfig(
            name="DecisionTree_Specialist",
            enabled=True,
            params={"max_depth": 10, "min_samples_leaf": 2, "criterion": "gini", "random_state": 42},
            profile_requirements={
                "min_samples": 30,
                "n_features_range": [2, 100],
                "data_complexity": "medium",
            },
            description="Одиночное дерево — максимально интерпретируемая модель.",
        ),
    }


def get_default_config() -> PipelineConfig:
    full = _classification_models_full()
    selected = {
        k: full[k]
        for k in (
            "RandomForest_Specialist",
            "SVM_Specialist",
            "LogisticRegression_Specialist",
            "KNN_Specialist",
            "GaussianNB_Specialist",
            "DecisionTree_Specialist",
        )
    }
    return PipelineConfig(
        models=selected,
        training={"cv_folds": 5, "test_size": 0.2, "random_state": 42, "use_cv_in_scoring": False},
        scoring={"accuracy_weight": 0.7, "f1_weight": 0.3, "cv_weight": 0.0},
    )


def get_fast_config() -> PipelineConfig:
    full = _classification_models_full()

    rf = full["RandomForest_Specialist"]
    rf.params = {"n_estimators": 50, "max_depth": 5, "random_state": 42}
    rf.description = "Быстрый RandomForest для прототипирования."

    knn = full["KNN_Specialist"]
    knn.params = {"n_neighbors": 5, "weights": "uniform", "p": 2}

    gnb = full["GaussianNB_Specialist"]

    dt = full["DecisionTree_Specialist"]
    dt.params = {"max_depth": 6, "min_samples_leaf": 4, "criterion": "gini", "random_state": 42}

    return PipelineConfig(
        models={
            "RandomForest_Specialist": rf,
            "KNN_Specialist": knn,
            "GaussianNB_Specialist": gnb,
            "DecisionTree_Specialist": dt,
        },
        training={"cv_folds": 3, "test_size": 0.2, "random_state": 42, "use_cv_in_scoring": False},
        scoring={"accuracy_weight": 0.6, "f1_weight": 0.4, "cv_weight": 0.0},
    )


def get_accurate_config() -> PipelineConfig:
    full = _classification_models_full()

    rf = full["RandomForest_Specialist"]
    rf.params = {
        "n_estimators": 200,
        "max_depth": 10,
        "min_samples_leaf": 2,
        "class_weight": "balanced",
        "random_state": 42,
    }
    rf.description = "RandomForest, оптимизированный по точности (balanced classes)."

    svm = full["SVM_Specialist"]
    svm.params = {"C": 5.0, "kernel": "rbf", "gamma": "scale", "probability": True, "random_state": 42}
    svm.description = "RBF SVM с высокой регуляризацией."

    lr = full["LogisticRegression_Specialist"]
    lr.params = {
        "solver": "saga",
        "penalty": "elasticnet",
        "l1_ratio": 0.5,
        "max_iter": 500,
        "random_state": 42,
    }
    lr.description = "Logistic Regression с elasticnet-регуляризацией."

    dt = full["DecisionTree_Specialist"]
    dt.params = {"max_depth": 15, "min_samples_leaf": 2, "criterion": "entropy", "random_state": 42}

    knn = full["KNN_Specialist"]
    knn.params = {"n_neighbors": 7, "weights": "distance", "p": 2, "leaf_size": 30}

    return PipelineConfig(
        models={
            "RandomForest_Specialist": rf,
            "SVM_Specialist": svm,
            "GradientBoosting_Specialist": full["GradientBoosting_Specialist"],
            "NeuralNetwork_Specialist": full["NeuralNetwork_Specialist"],
            "LogisticRegression_Specialist": lr,
            "KNN_Specialist": knn,
            "GaussianNB_Specialist": full["GaussianNB_Specialist"],
            "DecisionTree_Specialist": dt,
        },
        training={"cv_folds": 10, "test_size": 0.2, "random_state": 42, "use_cv_in_scoring": True},
        scoring={"accuracy_weight": 0.5, "f1_weight": 0.4, "cv_weight": 0.1},
    )
