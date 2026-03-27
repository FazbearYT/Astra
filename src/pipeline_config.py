"""
Конфигурация ML пайплайна
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
import json


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

    preprocessing: Dict[str, Any] = field(default_factory=lambda: {
        'scale_features': True,
        'handle_outliers': False,
        'balance_classes': False
    })

    training: Dict[str, Any] = field(default_factory=lambda: {
        'test_size': 0.2,
        'cv_folds': 5,
        'random_state': 42,
        'use_cv_in_scoring': False
    })

    scoring: Dict[str, float] = field(default_factory=lambda: {
        'accuracy_weight': 0.7,
        'f1_weight': 0.3,
        'cv_weight': 0.0
    })

    def save(self, filepath: str):
        from dataclasses import asdict

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(asdict(self), f, indent=2, default=str)

    @classmethod
    def load(cls, filepath: str) -> 'PipelineConfig':
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        models = {}
        for name, config in data.get('models', {}).items():
            models[name] = ModelConfig(**config)
        data['models'] = models

        return cls(**data)


def get_default_config() -> PipelineConfig:
    config = PipelineConfig()

    config.models['random_forest'] = ModelConfig(
        name="RandomForest_Specialist",
        enabled=True,
        params={
            'n_estimators': 100,
            'max_depth': None,
            'random_state': 42,
            'n_jobs': -1
        },
        profile_requirements={
            'data_complexity': 'simple',
            'class_balance_min': 0.7,
            'min_samples': 50
        },
        description="Random Forest - эффективен на небольших сбалансированных данных"
    )

    config.models['svm'] = ModelConfig(
        name="SVM_Specialist",
        enabled=True,
        params={
            'kernel': 'rbf',
            'probability': True,
            'random_state': 42,
            'C': 1.0,
            'gamma': 'scale'
        },
        profile_requirements={
            'data_complexity': 'medium',
            'n_features_range': (2, 50)
        },
        description="SVM с RBF ядром - эффективен на данных с четкими границами"
    )

    config.models['gradient_boosting'] = ModelConfig(
        name="GradientBoosting_Specialist",
        enabled=True,
        params={
            'n_estimators': 100,
            'max_depth': 3,
            'learning_rate': 0.1,
            'random_state': 42
        },
        profile_requirements={
            'data_complexity': 'complex',
            'min_samples': 100
        },
        description="Gradient Boosting - эффективен на сложных нелинейных данных"
    )

    config.models['neural_network'] = ModelConfig(
        name="NeuralNetwork_Specialist",
        enabled=True,
        params={
            'hidden_layer_sizes': (10, 10),
            'max_iter': 1000,
            'random_state': 42,
            'learning_rate_init': 0.001
        },
        profile_requirements={
            'data_complexity': 'complex',
            'min_samples': 200
        },
        description="Нейронная сеть - универсальная модель"
    )

    config.models['logistic_regression'] = ModelConfig(
        name="LogisticRegression_Specialist",
        enabled=True,
        params={
            'max_iter': 1000,
            'random_state': 42,
            'C': 1.0,
            'solver': 'lbfgs'
        },
        profile_requirements={
            'data_complexity': 'simple',
            'n_features_range': (2, 1000)
        },
        description="Логистическая регрессия - быстрая базовая модель"
    )

    return config


def get_fast_config() -> PipelineConfig:
    config = get_default_config()

    config.models['gradient_boosting'].enabled = False
    config.models['neural_network'].enabled = False

    config.training['cv_folds'] = 3

    return config


def get_accurate_config() -> PipelineConfig:
    config = get_default_config()

    config.training['use_cv_in_scoring'] = True
    config.training['cv_folds'] = 10

    config.models['random_forest'].params['n_estimators'] = 200
    config.models['gradient_boosting'].params['n_estimators'] = 200

    config.models['svm'].params['C'] = 1.5

    return config


def interactive_config() -> PipelineConfig:
    print("\nНастройка Pipeline")
    print("-" * 40)

    config = get_default_config()

    print("\nВыберите режим:")
    print("  1. Быстрый (2 модели, 3 CV folds)")
    print("  2. Стандартный (5 моделей, 5 CV folds)")
    print("  3. Точный (5 моделей, 10 CV folds)")
    print("  4. Свой (настроить вручную)")

    choice = input("\nВаш выбор (1-4): ").strip()

    if choice == "1":
        return get_fast_config()
    elif choice == "2":
        return get_default_config()
    elif choice == "3":
        return get_accurate_config()
    elif choice == "4":
        print("\nНастройка моделей:")

        for model_key, model_config in config.models.items():
            enabled = input(f"  Включить {model_config.name}? [y/N]: ").strip().lower()
            model_config.enabled = (enabled == 'y' or enabled == 'yes')

        print("\nВеса для сравнения моделей:")
        try:
            acc_weight = float(input("  Accuracy вес (0-1) [0.7]: ").strip() or "0.7")
            f1_weight = float(input("  F1-Score вес (0-1) [0.3]: ").strip() or "0.3")

            config.scoring['accuracy_weight'] = acc_weight
            config.scoring['f1_weight'] = f1_weight
            config.scoring['cv_weight'] = 1.0 - acc_weight - f1_weight
        except Exception:
            print("  Использованы значения по умолчанию")

        try:
            cv_folds = int(input("\n  Количество CV folds [5]: ").strip() or "5")
            config.training['cv_folds'] = cv_folds
        except Exception:
            pass

        return config
    else:
        return get_default_config()