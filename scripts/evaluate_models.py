"""
Скрипт для оценки и сравнения обученных моделей
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import json
from pathlib import Path
from sklearn.datasets import load_iris
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)
import matplotlib
matplotlib.use('Agg')  # Неинтерактивный режим
import matplotlib.pyplot as plt
import seaborn as sns
from src.model_selector import AdaptiveModelSelector
from sklearn.model_selection import train_test_split


def evaluate_saved_models(models_dir: str = "models/iris_models"):
    """
    Оценка сохранённых моделей
    """
    print("\n" + "="*70)
    print("📊 ОЦЕНКА СОХРАНЁННЫХ МОДЕЛЕЙ")
    print("="*70)

    models_dir = Path(models_dir)

    if not models_dir.exists():
        print(f"❌ Директория не найдена: {models_dir}")
        return

    # Загрузка метаданных
    metadata_path = models_dir / "metadata.json"
    if not metadata_path.exists():
        print(f"❌ Метаданные не найдены: {metadata_path}")
        return

    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    print(f"\n📋 Информация о датасете:")
    print(f"  • Название: {metadata['dataset']}")
    print(f"  • Образцов: {metadata['n_samples']}")
    print(f"  • Признаков: {metadata['n_features']}")
    print(f"  • Классов: {metadata['n_classes']}")
    print(f"  • Лучшая модель: {metadata['best_model']}")

    # Загрузка данных
    print(f"\n📥 Загрузка данных...")
    iris = load_iris()
    X, y = iris.data, iris.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Загрузка селектора
    print(f"\n🔄 Загрузка моделей...")
    selector = AdaptiveModelSelector()
    selector.load_models(str(models_dir))

    print(f"✓ Загружено {len(selector.models)} моделей")

    # Оценка каждой модели
    print("\n" + "="*70)
    print("📈 РЕЗУЛЬТАТЫ ОЦЕНКИ")
    print("="*70)

    results = []

    for model in selector.models:
        print(f"\n🤖 Модель: {model.name}")
        print("-" * 50)

        try:
            # 🔧 ИСПРАВЛЕНИЕ 1: Если модель не обучена, пробуем обучить
            if not model.is_trained:
                print(f"   ⚠️  Модель не обучена, обучаем...")
                model.fit(X_train, y_train)
                print(f"   ✅ Модель обучена")

            # Предсказание
            y_pred = model.predict(X_test)

            # Метрики
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

            print(f"  Accuracy:  {accuracy:.4f}")
            print(f"  Precision: {precision:.4f}")
            print(f"  Recall:    {recall:.4f}")
            print(f"  F1-Score:  {f1:.4f}")

            results.append({
                'model': model.name,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'training_time': model.training_time
            })

        except Exception as e:
            print(f"  ⚠️  Ошибка: {e}")
            import traceback
            traceback.print_exc()

    # Сводная таблица
    print("\n" + "="*70)
    print("📊 СВОДНАЯ ТАБЛИЦА")
    print("="*70)

    print(f"\n{'Модель':<35} {'Accuracy':<10} {'F1-Score':<10} {'Время (с)':<10}")
    print("-" * 70)

    for result in sorted(results, key=lambda x: x['accuracy'], reverse=True):
        time_str = f"{result['training_time']:.3f}" if result['training_time'] else "N/A"
        print(f"{result['model']:<35} {result['accuracy']:<10.4f} {result['f1']:<10.4f} {time_str:<10}")

    # Визуализация
    print("\n📈 Создание визуализаций...")

    # График сравнения моделей
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy comparison
    models_names = [r['model'].replace('_Specialist', '') for r in results]
    accuracies = [r['accuracy'] for r in results]

    axes[0].barh(models_names, accuracies, color='steelblue')
    axes[0].set_xlabel('Accuracy')
    axes[0].set_title('Сравнение точности моделей')
    axes[0].set_xlim([0, 1])

    for i, v in enumerate(accuracies):
        axes[0].text(v + 0.01, i, f'{v:.3f}', va='center')

    # F1-Score comparison
    f1_scores = [r['f1'] for r in results]

    axes[1].barh(models_names, f1_scores, color='coral')
    axes[1].set_xlabel('F1-Score')
    axes[1].set_title('Сравнение F1-Score моделей')
    axes[1].set_xlim([0, 1])

    for i, v in enumerate(f1_scores):
        axes[1].text(v + 0.01, i, f'{v:.3f}', va='center')

    plt.tight_layout()

    viz_path = models_dir / "models_comparison.png"
    plt.savefig(str(viz_path), dpi=300, bbox_inches='tight')
    print(f"✓ Визуализация сохранена: {viz_path}")
    plt.close('all')

    # Матрица ошибок для лучшей модели
    print("\n📊 Матрица ошибок для лучшей модели...")

    best_result = max(results, key=lambda x: x['accuracy'])
    best_model = next(m for m in selector.models if m.name == best_result['model'])

    y_pred_best = best_model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred_best)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=iris.target_names,
                yticklabels=iris.target_names, ax=ax)
    ax.set_xlabel('Предсказанный класс')
    ax.set_ylabel('Истинный класс')
    ax.set_title(f'Матрица ошибок - {best_model.name}')

    cm_path = models_dir / "confusion_matrix.png"
    plt.savefig(str(cm_path), dpi=300, bbox_inches='tight')
    print(f"✓ Матрица ошибок сохранена: {cm_path}")
    plt.close('all')

    # Сохранение результатов
    results_path = models_dir / "evaluation_results.json"
    with open(results_path, 'w') as f:
        json.dump({
            'metadata': metadata,
            'results': results,
            'best_model': best_result['model'],
            'best_accuracy': best_result['accuracy']
        }, f, indent=2)

    print(f"✓ Результаты сохранены: {results_path}")

    print("\n" + "="*70)
    print("✅ ОЦЕНКА ЗАВЕРШЕНА")
    print("="*70)

    return results


def compare_with_baseline(models_dir: str = "models/iris_models"):
    """
    Сравнение с базовой моделью
    """
    print("\n" + "="*70)
    print("📊 СРАВНЕНИЕ С БАЗОВОЙ МОДЕЛЬЮ")
    print("="*70)

    from sklearn.dummy import DummyClassifier

    iris = load_iris()
    X, y = iris.data, iris.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Базовая модель (стратегия most_frequent)
    baseline = DummyClassifier(strategy='most_frequent', random_state=42)
    baseline.fit(X_train, y_train)
    baseline_pred = baseline.predict(X_test)
    baseline_accuracy = accuracy_score(y_test, baseline_pred)

    print(f"\n📌 Базовая модель (Dummy - most_frequent):")
    print(f"   Accuracy: {baseline_accuracy:.4f}")

    # Загрузка лучших моделей
    results = evaluate_saved_models(models_dir)

    if results:
        best_accuracy = max(r['accuracy'] for r in results)
        improvement = (best_accuracy - baseline_accuracy) / baseline_accuracy * 100

        print(f"\n🏆 Лучшая модель:")
        print(f"   Accuracy: {best_accuracy:.4f}")
        print(f"   Улучшение: +{improvement:.2f}%")

    return baseline_accuracy


def main():
    """Основная функция"""
    import argparse

    parser = argparse.ArgumentParser(description='Оценка обученных моделей')
    parser.add_argument('--models-dir', type=str, default='models/iris_models',
                       help='Директория с моделями')
    parser.add_argument('--compare-baseline', action='store_true',
                       help='Сравнить с базовой моделью')
    parser.add_argument('--output', type=str, default='models/evaluation_report.json',
                       help='Путь для отчёта')

    args = parser.parse_args()

    # Оценка
    results = evaluate_saved_models(args.models_dir)

    # Сравнение с базовой
    if args.compare_baseline:
        compare_with_baseline(args.models_dir)

    print(f"\n✅ Все отчёты сохранены в: {args.models_dir}")


if __name__ == "__main__":
    main()