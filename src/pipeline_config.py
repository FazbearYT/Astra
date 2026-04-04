from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
import json
import os


@dataclass
class ModelConfig:
    name: str
    enabled: bool = True
    params: Dict[str, Any] = field(default_factory=dict)
    profile_requirements: Dict[str, Any] = field(default_factory=dict)
    description: str = ""


@dataclass
class PipelineConfig:
    models: Dict[str, ModelConfig] = field(default_factory=dict)
    training: Dict[str, Any] = field(default_factory=dict)
    scoring: Dict[str, float] = field(default_factory=dict)

    def save(self, filepath: str):
        serializable_models = {k: v.__dict__ for k, v in self.models.items()}
        data = {
            "models": serializable_models,
            "training": self.training,
            "scoring": self.scoring
        }
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, filepath: str) -> 'PipelineConfig':
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        models_data = data.get("models", {})
        models = {k: ModelConfig(**v) for k, v in models_data.items()}
        return cls(
            models=models,
            training=data.get("training", {}),
            scoring=data.get("scoring", {})
        )


def get_default_config() -> PipelineConfig:
    return PipelineConfig(
        models={
            "RandomForest_Specialist": ModelConfig(
                name="RandomForest_Specialist",
                enabled=True,
                params={"n_estimators": 100, "max_depth": 10, "random_state": 42},
                profile_requirements={
                    "min_samples": 50,
                    "n_features_range": [2, 50],
                    "data_complexity": "medium"
                },
                description="General-purpose RandomForest for diverse datasets."
            ),
            "SVM_Specialist": ModelConfig(
                name="SVM_Specialist",
                enabled=True,
                params={"C": 1.0, "kernel": "rbf", "probability": True, "random_state": 42},
                profile_requirements={
                    "min_samples": 30,
                    "n_features_range": [2, 20],
                    "data_complexity": "low"
                },
                description="SVM for smaller, well-separated datasets."
            ),
            "LogisticRegression_Specialist": ModelConfig(
                name="LogisticRegression_Specialist",
                enabled=True,
                params={"solver": "liblinear", "penalty": "l1", "random_state": 42},
                profile_requirements={
                    "min_samples": 50,
                    "n_features_range": [2, 100],
                    "data_complexity": "medium"
                },
                description="Linear model for interpretable results."
            )
        },
        training={
            "cv_folds": 5,
            "test_size": 0.2,
            "random_state": 42,
            "use_cv_in_scoring": False
        },
        scoring={
            "accuracy_weight": 0.7,
            "f1_weight": 0.3,
            "cv_weight": 0.0
        }
    )


def get_fast_config() -> PipelineConfig:
    return PipelineConfig(
        models={
            "RandomForest_Specialist": ModelConfig(
                name="RandomForest_Specialist",
                enabled=True,
                params={"n_estimators": 50, "max_depth": 5, "random_state": 42},
                profile_requirements={
                    "min_samples": 20,
                    "n_features_range": [2, 30],
                    "data_complexity": "low"
                },
                description="Fast RandomForest for quick prototyping."
            ),
            "SVM_Specialist": ModelConfig(
                name="SVM_Specialist",
                enabled=True,
                params={"C": 0.5, "kernel": "linear", "probability": True, "random_state": 42},
                profile_requirements={
                    "min_samples": 10,
                    "n_features_range": [2, 10],
                    "data_complexity": "low"
                },
                description="Fast Linear SVM."
            )
        },
        training={
            "cv_folds": 3,
            "test_size": 0.2,
            "random_state": 42,
            "use_cv_in_scoring": False
        },
        scoring={
            "accuracy_weight": 0.6,
            "f1_weight": 0.4,
            "cv_weight": 0.0
        }
    )


def get_accurate_config() -> PipelineConfig:
    return PipelineConfig(
        models={
            "RandomForest_Specialist": ModelConfig(
                name="RandomForest_Specialist",
                enabled=True,
                params={
                    "n_estimators": 200,
                    "max_depth": 10,
                    "min_samples_leaf": 2,
                    "class_weight": "balanced",
                    "random_state": 42
                },
                profile_requirements={
                    "min_samples": 100,
                    "n_features_range": [5, 50],
                    "data_complexity": "medium"
                },
                description="RandomForest optimized for accuracy with balanced classes."
            ),
            "SVM_Specialist": ModelConfig(
                name="SVM_Specialist",
                enabled=True,
                params={"C": 5.0, "kernel": "rbf", "gamma": "scale", "probability": True, "random_state": 42},
                profile_requirements={
                    "min_samples": 50,
                    "n_features_range": [2, 30],
                    "data_complexity": "medium"
                },
                description="Highly accurate RBF SVM."
            ),
            "GradientBoosting_Specialist": ModelConfig(
                name="GradientBoosting_Specialist",
                enabled=True,
                params={"n_estimators": 150, "learning_rate": 0.05, "max_depth": 3, "random_state": 42},
                profile_requirements={
                    "min_samples": 100,
                    "n_features_range": [5, 100],
                    "data_complexity": "high"
                },
                description="Gradient Boosting for complex datasets."
            ),
            "NeuralNetwork_Specialist": ModelConfig(
                name="NeuralNetwork_Specialist",
                enabled=True,
                params={"hidden_layer_sizes": (50, 25), "max_iter": 500, "activation": "relu", "solver": "adam",
                        "random_state": 42},
                profile_requirements={
                    "min_samples": 200,
                    "n_features_range": [10, 150],
                    "data_complexity": "high"
                },
                description="Multi-layer Perceptron for highly non-linear data."
            ),
            "LogisticRegression_Specialist": ModelConfig(
                name="LogisticRegression_Specialist",
                enabled=True,
                params={"solver": "saga", "penalty": "elasticnet", "l1_ratio": 0.5, "max_iter": 200,
                        "random_state": 42},
                profile_requirements={
                    "min_samples": 50,
                    "n_features_range": [2, 100],
                    "data_complexity": "medium"
                },
                description="Robust Logistic Regression with elasticnet regularization."
            )
        },
        training={
            "cv_folds": 10,
            "test_size": 0.2,
            "random_state": 42,
            "use_cv_in_scoring": True
        },
        scoring={
            "accuracy_weight": 0.5,
            "f1_weight": 0.4,
            "cv_weight": 0.1
        }
    )


def interactive_config() -> PipelineConfig:
    print("\n=== Интерактивная настройка конфигурации ===")

    base_config = get_accurate_config()

    base_config.scoring['accuracy_weight'] = 0.7
    base_config.scoring['f1_weight'] = 0.3
    base_config.scoring['cv_weight'] = 0.0
    base_config.training['use_cv_in_scoring'] = False

    print("\nВыберите модели для включения:")
    model_names = list(base_config.models.keys())
    for idx, name in enumerate(model_names, 1):
        current = base_config.models[name].enabled
        print(f"{idx}. {name} (сейчас {'вкл' if current else 'выкл'})")

    choice = input(
        "\nВведите номера моделей для переключения через запятую (или 'all' для всех, 'done' для завершения): ")
    if choice.lower() == 'all':
        for name in model_names:
            base_config.models[name].enabled = True
    elif choice.lower() != 'done':
        try:
            indices = [int(x.strip()) for x in choice.split(',')]
            for idx in indices:
                if 1 <= idx <= len(model_names):
                    base_config.models[model_names[idx - 1]].enabled = not base_config.models[
                        model_names[idx - 1]].enabled
        except:
            print("Неверный ввод, оставляем текущие настройки")

    print("\nНастройка параметров обучения:")
    try:
        cv = int(input(f"CV-Folds (сейчас {base_config.training['cv_folds']}): ") or base_config.training['cv_folds'])
        base_config.training['cv_folds'] = cv
    except:
        pass

    use_cv = input(
        f"Use CV in scoring? (yes/no, сейчас {'yes' if base_config.training.get('use_cv_in_scoring', False) else 'no'}): ").strip().lower()
    if use_cv in ['yes', 'y']:
        base_config.training['use_cv_in_scoring'] = True
    elif use_cv in ['no', 'n']:
        base_config.training['use_cv_in_scoring'] = False

    print("\nВеса метрик (сумма должна быть 1.0):")
    try:
        acc_w = float(
            input(f"Weight Accuracy (сейчас {base_config.scoring['accuracy_weight']}): ") or base_config.scoring[
                'accuracy_weight'])
        f1_w = float(
            input(f"Weight F1 (сейчас {base_config.scoring['f1_weight']}): ") or base_config.scoring['f1_weight'])
        cv_w = float(
            input(f"Weight CV (сейчас {base_config.scoring['cv_weight']}): ") or base_config.scoring['cv_weight'])
        total = acc_w + f1_w + cv_w
        if abs(total - 1.0) > 0.01:
            print(f"Сумма весов {total} не равна 1. Нормируем.")
            acc_w, f1_w, cv_w = acc_w / total, f1_w / total, cv_w / total
        base_config.scoring['accuracy_weight'] = acc_w
        base_config.scoring['f1_weight'] = f1_w
        base_config.scoring['cv_weight'] = cv_w
    except:
        pass

    print("\nИнтерактивная настройка завершена.")
    return base_config


def get_config_by_name(name: str) -> PipelineConfig:
    if name == "default":
        return get_default_config()
    elif name == "fast":
        return get_fast_config()
    elif name == "accurate":
        return get_accurate_config()
    elif name == "interactive":
        return interactive_config()
    else:
        raise ValueError(f"Unknown pipeline config name: {name}")