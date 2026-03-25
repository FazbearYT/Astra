"""
Конфигурация ML пайплайна
==========================

Позволяет пользователю настраивать:
- Какие модели использовать
- Гиперпараметры моделей
- Параметры предобработки
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    """Конфигурация одной модели"""
    name: str
    enabled: bool = True
    params: Dict[str, Any] = field(default_factory=dict)
    profile_requirements: Dict[str, Any] = field(default_factory=dict)
    description: str = ""


@dataclass
class PipelineConfig:
    """Конфигурация всего пайплайна"""
    # Какие модели использовать
    models: Dict[str, ModelConfig] = field(default_factory=dict)

    # Параметры предобработки
    preprocessing: Dict[str, Any] = field(default_factory=lambda: {
        'scale_features': True,
        'handle_outliers': False,
        'balance_classes': False
    })

    # Параметры обучения
    training: Dict[str, Any] = field(default_factory=lambda: {
        'test_size': 0.2,
        'cv_folds': 5,
        'random_state': 42,
        'use_cv_in_scoring': False
    })

    # Весовые коэффициенты для сравнения
    scoring: Dict[str, float] = field(default_factory=lambda: {
        'accuracy_weight': 0.7,
        'f1_weight': 0.3,
        'cv_weight': 0.0
    })

    def save(self, filepath: str):
        """Сохранение конфигурации в JSON"""
        import json
        from dataclasses import asdict

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(asdict(self), f, indent=2, default=str)
        print(f"✓ Конфигурация сохранена: {filepath}")

    @classmethod
    def load(cls, filepath: str) -> 'PipelineConfig':
        """Загрузка конфигурации из JSON"""
        import json

        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Восстановление ModelConfig объектов
        models = {}
        for name, config in data.get('models', {}).items():
            models[name] = ModelConfig(**config)
        data['models'] = models

        return cls(**data)


def get_default_config() -> PipelineConfig:
    """Конфигурация по умолчанию"""
    config = PipelineConfig()

    # Random Forest
    config.models['random_forest'] = ModelConfig(
        name="RandomForest_Specialist",
        enabled=True,
        params={
            'n_estimators': 100,
            'max_depth': 10,
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

    # SVM
    config.models['svm'] = ModelConfig(
        name="SVM_Specialist",
        enabled=True,
        params={
            'kernel': 'rbf',
            'probability': True,
            'random_state': 42
        },
        profile_requirements={
            'data_complexity': 'medium',
            'n_features_range': (2, 50)
        },
        description="SVM с RBF ядром - эффективен на данных с четкими границами"
    )

    # Gradient Boosting
    config.models['gradient_boosting'] = ModelConfig(
        name="GradientBoosting_Specialist",
        enabled=True,
        params={
            'n_estimators': 100,
            'max_depth': 5,
            'learning_rate': 0.1,
            'random_state': 42
        },
        profile_requirements={
            'data_complexity': 'complex',
            'min_samples': 100
        },
        description="Gradient Boosting - эффективен на сложных нелинейных данных"
    )

    # Neural Network
    config.models['neural_network'] = ModelConfig(
        name="NeuralNetwork_Specialist",
        enabled=True,
        params={
            'hidden_layer_sizes': (100, 50),
            'max_iter': 1000,
            'random_state': 42
        },
        profile_requirements={
            'data_complexity': 'complex',
            'min_samples': 200
        },
        description="Нейронная сеть - универсальная модель"
    )

    # Logistic Regression
    config.models['logistic_regression'] = ModelConfig(
        name="LogisticRegression_Specialist",
        enabled=True,
        params={
            'max_iter': 1000,
            'random_state': 42
        },
        profile_requirements={
            'data_complexity': 'simple',
            'n_features_range': (2, 1000)
        },
        description="Логистическая регрессия - быстрая базовая модель"
    )

    return config


def get_fast_config() -> PipelineConfig:
    """Быстрая конфигурация (только 2 модели для теста)"""
    config = get_default_config()

    # Отключаем некоторые модели для скорости
    config.models['gradient_boosting'].enabled = False
    config.models['neural_network'].enabled = False

    config.training['cv_folds'] = 3

    return config


def get_accurate_config() -> PipelineConfig:
    """Точная конфигурация (все модели + CV в скоре)"""
    config = get_default_config()

    # Включаем CV в итоговый скор
    config.training['use_cv_in_scoring'] = True
    config.training['cv_folds'] = 10

    # Увеличиваем количество деревьев
    config.models['random_forest'].params['n_estimators'] = 200
    config.models['gradient_boosting'].params['n_estimators'] = 200

    return config


def interactive_config() -> PipelineConfig:
    """Интерактивная настройка конфигурации"""
    print("\n" + "=" * 70)
    print("⚙️  НАСТРОЙКА PIPELINE")
    print("=" * 70)

    config = get_default_config()

    # Выбор режима
    print("\n📋 Выберите режим:")
    print("  1. Быстрый (2 модели, 3 CV folds)")
    print("  2. Стандартный (5 моделей, 5 CV folds)")
    print("  3. Точный (5 моделей, 10 CV folds, CV в скоре)")
    print("  4. Свой (настроить вручную)")

    choice = input("\n👉 Ваш выбор (1-4): ").strip()

    if choice == "1":
        return get_fast_config()
    elif choice == "2":
        return get_default_config()
    elif choice == "3":
        return get_accurate_config()
    elif choice == "4":
        # Ручная настройка
        print("\n🔧 Настройка моделей:")

        for model_key, model_config in config.models.items():
            enabled = input(f"  Включить {model_config.name}? [y/N]: ").strip().lower()
            model_config.enabled = (enabled == 'y' or enabled == 'yes')

        # Настройка весов
        print("\n⚖️  Веса для сравнения моделей:")
        try:
            acc_weight = float(input("  Accuracy вес (0-1) [0.7]: ").strip() or "0.7")
            f1_weight = float(input("  F1-Score вес (0-1) [0.3]: ").strip() or "0.3")

            config.scoring['accuracy_weight'] = acc_weight
            config.scoring['f1_weight'] = f1_weight
            config.scoring['cv_weight'] = 1.0 - acc_weight - f1_weight
        except:
            print("  ⚠️  Использованы значения по умолчанию")

        # Настройка CV
        try:
            cv_folds = int(input("\n  Количество CV folds [5]: ").strip() or "5")
            config.training['cv_folds'] = cv_folds
        except:
            pass

        return config
    else:
        return get_default_config()


if __name__ == "__main__":
    # Тест конфигурации
    config = get_default_config()

    print("Конфигурация по умолчанию:")
    print(f"  Моделей: {len(config.models)}")
    print(f"  CV folds: {config.training['cv_folds']}")
    print(f"  Accuracy вес: {config.scoring['accuracy_weight']}")
    print(f"  F1 вес: {config.scoring['f1_weight']}")

    # Сохранение
    config.save("pipeline_config.json")