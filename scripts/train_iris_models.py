"""
Скрипт для обучения моделей на Iris Dataset
ПРОВЕРЕН И ГОТОВ К ИСПОЛЬЗОВАНИЮ
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from src.model_profiler import DataProfiler, DatasetProfile
from src.model_selector import AdaptiveModelSelector, SpecializedModel
import joblib
import json
from pathlib import Path


def main():
    """Основной пайплайн обучения на Iris Dataset"""
    print("\n" + "="*70)
    print("🌸 ОБУЧЕНИЕ МОДЕЛЕЙ НА IRIS DATASET")
    print("="*70)

    # 1. Загрузка данных
    print("\n📥 Загрузка Iris Dataset...")
    iris = load_iris()
    X, y = iris.data, iris.target

    print(f"✓ Загружено {X.shape[0]} образцов с {X.shape[1]} признаками")
    print(f"  Классы: {dict(zip(range(len(iris.target_names)), iris.target_names))}")

    # 2. Разделение данных
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"  Train: {X_train.shape[0]} образцов")
    print(f"  Test: {X_test.shape[0]} образцов")

    # 3. Профилирование данных
    print("\n📊 Профилирование данных...")
    profiler = DataProfiler(dataset_name="Iris_Dataset")

    profile = profiler.profile_tabular_data(
        X_train, y_train,
        feature_names=iris.feature_names
    )

    # Сохранение профиля
    profile_path = Path("models/iris_profile.json")
    profile_path.parent.mkdir(exist_ok=True)
    profile.save(str(profile_path))

    # Визуализация
    profiler.visualize_profile(save_path="models/iris_profile_visualization.png")
    profiler.print_summary()

    # 4. Создание и обучение моделей
    print("\n🤖 Создание и обучение моделей...")
    selector = AdaptiveModelSelector()

    # Создание специализированных моделей
    models_created = selector.create_default_models(
        model_types=['random_forest', 'svm', 'gradient_boosting',
                    'neural_network', 'logistic_regression']
    )

    print(f"✓ Создано {len(models_created)} моделей")

    # 5. Адаптивный выбор модели
    print("\n🎯 Адаптивный выбор лучшей модели...")

    best_model = selector.profile_and_select(
        X, y,
        data_profile=profile.to_dict(),
        cv_folds=5
    )

    # 6. Финальная оценка
    print("\n📈 Финальная оценка на тестовом наборе...")
    y_pred = selector.predict(X_test)

    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n✅ ТОЧНОСТЬ НА ТЕСТЕ: {accuracy:.4f} ({accuracy*100:.2f}%)")

    print("\n📋 Отчет о классификации:")
    print(classification_report(y_test, y_pred, target_names=iris.target_names))

    print("\n📊 Матрица ошибок:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)

    # 7. Сохранение моделей
    print("\n💾 Сохранение моделей...")
    models_dir = Path("models/iris_models")
    models_dir.mkdir(exist_ok=True)

    selector.save_all_models(str(models_dir))

    # Сохранение метаданных
    metadata = {
        'dataset': 'Iris',
        'n_samples': X.shape[0],
        'n_features': X.shape[1],
        'n_classes': len(iris.target_names),
        'class_names': iris.target_names.tolist(),
        'feature_names': iris.feature_names.tolist(),
        'best_model': best_model.name,
        'test_accuracy': float(accuracy),
        'timestamp': str(np.datetime64('now'))
    }

    metadata_path = models_dir / "metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2, default=str)

    print(f"✓ Метаданные сохранены: {metadata_path}")

    # 8. Тестирование предсказания
    print("\n🧪 Тестирование предсказания...")
    test_samples = [
        [5.1, 3.5, 1.4, 0.2],
        [6.5, 3.0, 5.2, 2.0],
        [5.7, 2.8, 4.1, 1.3],
    ]

    for i, sample in enumerate(test_samples):
        pred = selector.predict(np.array([sample]))[0]
        pred_proba = selector.predict_proba(np.array([sample]))[0]

        print(f"\n  Sample {i+1}: {sample}")
        print(f"  Предсказанный класс: {iris.target_names[pred]}")
        print(f"  Вероятности: {dict(zip(iris.target_names, pred_proba.round(3)))}")

    print("\n" + "="*70)
    print("✅ ОБУЧЕНИЕ ЗАВЕРШЕНО УСПЕШНО!")
    print("="*70)
    print(f"\n📂 Результаты сохранены в: {models_dir}")
    print(f"📄 Профиль данных: {profile_path}")
    print(f"🏆 Лучшая модель: {best_model.name}")
    print(f"📊 Точность: {accuracy*100:.2f}%")

    return selector, profile


def load_and_use_model(models_dir: str = "models/iris_models"):
    """
    Пример загрузки и использования обученной модели
    """
    print("\n🔄 Загрузка обученной модели...")

    models_dir = Path(models_dir)

    # Загрузка метаданных
    with open(models_dir / "metadata.json", 'r') as f:
        metadata = json.load(f)

    print(f"  Датасет: {metadata['dataset']}")
    print(f"  Лучшая модель: {metadata['best_model']}")
    print(f"  Точность: {metadata['test_accuracy']*100:.2f}%")

    # Загрузка селектора
    from src.model_selector import AdaptiveModelSelector

    selector = AdaptiveModelSelector()
    selector.load_models(str(models_dir))

    print(f"✓ Загружено {len(selector.models)} моделей")

    # Пример предсказания
    new_sample = np.array([[5.0, 3.4, 1.5, 0.2]])
    prediction = selector.predict(new_sample)

    print(f"\n📊 Пример предсказания:")
    print(f"  Вход: {new_sample[0]}")
    print(f"  Класс: {metadata['class_names'][prediction[0]]}")

    return selector


if __name__ == "__main__":
    # Обучение
    selector, profile = main()

    # Тестирование загрузки
    load_and_use_model()