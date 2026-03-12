"""
Скрипт для оценки и сравнения обученных моделей
ИСПРАВЛЕННАЯ ВЕРСИЯ
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import json
from pathlib import Path
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from src.model_selector import AdaptiveModelSelector
from sklearn.model_selection import train_test_split
import traceback


def evaluate_saved_models(models_dir: str = "models/iris_models"):
    """
    Оценка сохранённых моделей
    """
    print("\n" + "="*70)
    print("📊 ОЦЕНКА СОХРАНЁННЫХ МОДЕЛЕЙ")
    print("="*70)

    models_dir = Path(models_dir)

    # Проверка директории
    if not models_dir.exists():
        print(f"❌ ОШИБКА: Директория не найдена: {models_dir.absolute()}")
        print(f"💡 Подсказка: Сначала запустите python scripts/train_iris_models.py")
        return None

    print(f"📂 Директория: {models_dir.absolute()}")

    # Проверка метаданных
    metadata_path = models_dir / "metadata.json"
    if not metadata_path.exists():
        print(f"❌ ОШИБКА: Метаданные не найдены: {metadata_path}")
        return None

    # Загрузка метаданных
    try:
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        print(f"✅ Метаданные загружены")
    except Exception as e:
        print(f"❌ ОШИБКА загрузки метаданных: {e}")
        return None

    print(f"\n📋 Информация о датасете:")
    print(f"  • Название: {metadata.get('dataset', 'N/A')}")
    print(f"  • Образцов: {metadata.get('n_samples', 'N/A')}")
    print(f"  • Признаков: {metadata.get('n_features', 'N/A')}")
    print(f"  • Классов: {metadata.get('n_classes', 'N/A')}")
    print(f"  • Лучшая модель: {metadata.get('best_model', 'N/A')}")

    # Загрузка данных
    print(f"\n📥 Загрузка данных Iris...")
    try:
        iris = load_iris()
        X, y = iris.data, iris.target
        print(f"✅ Загружено {len(X)} образцов")
    except Exception as e:
        print(f"❌ ОШИБКА загрузки данных: {e}")
        return None

    # Разделение данных
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"  Train: {len(X_train)}, Test: {len(X_test)}")

    # Загрузка моделей
    print(f"\n🔄 Загрузка моделей...")
    selector = AdaptiveModelSelector()

    try:
        selector.load_models(str(models_dir))
        print(f"✅ Загружено {len(selector.models)} моделей")
    except Exception as e:
        print(f"❌ ОШИБКА загрузки моделей: {e}")
        traceback.print_exc()
        return None

    if len(selector.models) == 0:
        print("❌ ОШИБКА: Не загружено ни одной модели!")
        return None

    # Оценка моделей
    print("\n" + "="*70)
    print("📈 РЕЗУЛЬТАТЫ ОЦЕНКИ")
    print("="*70)

    results = []

    for idx, model in enumerate(selector.models, 1):
        print(f"\n[{idx}/{len(selector.models)}] Модель: {model.name}")
        print("-" * 60)

        try:
            # Обучение если нужно
            if not model.is_trained:
                print(f"   ⚠️  Модель не обучена, обучаем...")
                try:
                    model.fit(X_train, y_train)
                    print(f"   ✅ Модель обучена")
                except Exception as e:
                    print(f"   ❌ Ошибка обучения: {e}")
                    continue

            # Предсказание
            y_pred = model.predict(X_test)

            # Метрики
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

            print(f"   ✓ Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
            print(f"   ✓ Precision: {precision:.4f}")
            print(f"   ✓ Recall:    {recall:.4f}")
            print(f"   ✓ F1-Score:  {f1:.4f}")

            results.append({
                'model': model.name,
                'accuracy': float(accuracy),
                'precision': float(precision),
                'recall': float(recall),
                'f1': float(f1),
                'training_time': float(model.training_time) if model.training_time else None
            })

        except Exception as e:
            print(f"   ❌ ОШИБКА оценки: {e}")
            traceback.print_exc()
            continue

    # Проверка результатов
    if len(results) == 0:
        print("\n❌ ОШИБКА: Ни одна модель не была оценена!")
        return None

    print(f"\n✅ Оценено моделей: {len(results)} из {len(selector.models)}")

    # Сводная таблица
    print("\n" + "="*70)
    print("📊 СВОДНАЯ ТАБЛИЦА")
    print("="*70)

    print(f"\n{'Модель':<35} {'Accuracy':<12} {'F1-Score':<12} {'Время (с)':<10}")
    print("-" * 70)

    for result in sorted(results, key=lambda x: x['accuracy'], reverse=True):
        time_str = f"{result['training_time']:.3f}" if result['training_time'] else "N/A"
        print(f"{result['model']:<35} {result['accuracy']:<12.4f} {result['f1']:<12.4f} {time_str:<10}")

    # Визуализация
    print("\n" + "="*70)
    print("📈 СОЗДАНИЕ ВИЗУАЛИЗАЦИЙ")
    print("="*70)

    try:
        # График сравнения
        print("\n📊 Создание графика сравнения моделей...")
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        models_names = [r['model'].replace('_Specialist', '') for r in results]
        accuracies = [r['accuracy'] for r in results]
        f1_scores = [r['f1'] for r in results]

        # Accuracy
        axes[0].barh(models_names, accuracies, color='steelblue', alpha=0.7)
        axes[0].set_xlabel('Accuracy', fontsize=10)
        axes[0].set_title('Сравнение точности моделей', fontsize=12, fontweight='bold')
        axes[0].set_xlim([0, 1])
        axes[0].grid(axis='x', alpha=0.3)

        for i, v in enumerate(accuracies):
            axes[0].text(v + 0.01, i, f'{v:.3f}', va='center', fontsize=9)

        # F1-Score
        axes[1].barh(models_names, f1_scores, color='coral', alpha=0.7)
        axes[1].set_xlabel('F1-Score', fontsize=10)
        axes[1].set_title('Сравнение F1-Score моделей', fontsize=12, fontweight='bold')
        axes[1].set_xlim([0, 1])
        axes[1].grid(axis='x', alpha=0.3)

        for i, v in enumerate(f1_scores):
            axes[1].text(v + 0.01, i, f'{v:.3f}', va='center', fontsize=9)

        plt.tight_layout()

        # Сохранение
        viz_path = models_dir / "models_comparison.png"
        plt.savefig(str(viz_path), dpi=300, bbox_inches='tight')
        plt.close('all')
        print(f"✅ График сохранен: {viz_path.absolute()}")

    except Exception as e:
        print(f"⚠️  Не удалось создать график: {e}")
        traceback.print_exc()
        viz_path = None

    # Матрица ошибок
    try:
        print("\n📊 Создание матрицы ошибок...")

        # Находим лучшую модель
        best_result = max(results, key=lambda x: x['accuracy'])
        best_model = next(m for m in selector.models if m.name == best_result['model'])

        y_pred_best = best_model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred_best)

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=iris.target_names,
                    yticklabels=iris.target_names, ax=ax)
        ax.set_xlabel('Предсказанный класс', fontsize=10)
        ax.set_ylabel('Истинный класс', fontsize=10)
        ax.set_title(f'Матрица ошибок\n{best_model.name} (Accuracy: {best_result["accuracy"]:.4f})',
                    fontsize=12, fontweight='bold')

        plt.tight_layout()

        cm_path = models_dir / "confusion_matrix.png"
        plt.savefig(str(cm_path), dpi=300, bbox_inches='tight')
        plt.close('all')
        print(f"✅ Матрица ошибок сохранена: {cm_path.absolute()}")

    except Exception as e:
        print(f"⚠️  Не удалось создать матрицу ошибок: {e}")
        traceback.print_exc()
        cm_path = None

    # Сохранение результатов в JSON
    try:
        print("\n💾 Сохранение результатов в JSON...")

        results_path = models_dir / "evaluation_results.json"
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump({
                'metadata': metadata,
                'results': results,
                'best_model': best_result['model'],
                'best_accuracy': best_result['accuracy'],
                'n_models_evaluated': len(results),
                'timestamp': str(np.datetime64('now'))
            }, f, indent=2, ensure_ascii=False)

        print(f"✅ Результаты сохранены: {results_path.absolute()}")

    except Exception as e:
        print(f"⚠️  Не удалось сохранить JSON: {e}")
        traceback.print_exc()
        results_path = None

    # Итоговый вывод
    print("\n" + "="*70)
    print("✅ ОЦЕНКА ЗАВЕРШЕНА УСПЕШНО!")
    print("="*70)
    print(f"\n📂 Созданные файлы:")
    if viz_path and viz_path.exists():
        print(f"  ✓ {viz_path.absolute()}")
    if 'cm_path' in locals() and cm_path and cm_path.exists():
        print(f"  ✓ {cm_path.absolute()}")
    if results_path and results_path.exists():
        print(f"  ✓ {results_path.absolute()}")

    print(f"\n🏆 Лучшая модель: {best_result['model']}")
    print(f"📊 Точность: {best_result['accuracy']*100:.2f}%")

    return results


def main():
    """Основная функция"""
    import argparse

    parser = argparse.ArgumentParser(description='Оценка обученных моделей')
    parser.add_argument('--models-dir', type=str, default='models/iris_models',
                       help='Директория с моделями')

    args = parser.parse_args()

    # Оценка
    results = evaluate_saved_models(args.models_dir)

    if results is None:
        print("\n❌ ОШИБКА: Оценка не завершена!")
        sys.exit(1)

    print(f"\n✅ Все отчёты сохранены в: {args.models_dir}")


if __name__ == "__main__":
    main()