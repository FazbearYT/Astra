"""
Загрузка большого датасета для тестирования
============================================

Запуск:
    python scripts/download_large_dataset.py

Создаст:
    data/tabular/covertype.csv (581,012 строк, 7 классов)
"""

from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_covtype


def download_covertype():
    """Загрузка Forest Covertype Dataset"""
    print("=" * 70)
    print("🌲 ЗАГРУЗКА БОЛЬШОГО ДАТАСЕТА (Forest Covertype)")
    print("=" * 70)

    print("\n📊 Характеристики:")
    print("   • Строк: 581,012")
    print("   • Признаков: 54")
    print("   • Классов: 7")
    print("   • Размер: ~250 MB")
    print("\n⏱️  Время загрузки: 1-5 минут")

    choice = input("\nПродолжить? [y/N]: ").strip().lower()
    if choice != 'y' and choice != 'yes':
        print("❌ Отменено")
        return None

    try:
        print("\n📥 Загрузка...")
        covertype = fetch_covtype()

        # Создаём DataFrame
        df = pd.DataFrame(covertype.data, columns=covertype.feature_names)
        df['target'] = covertype.target
        df['target_name'] = df['target'].map({
            1: 'Spruce_Fir',
            2: 'Lodgepole_Pine',
            3: 'Ponderosa_Pine',
            4: 'Cottonwood_Willow',
            5: 'Aspen',
            6: 'Douglas_Fir',
            7: 'Krummholz'
        })

        # Сохранение
        tabular_dir = Path("data/tabular")
        tabular_dir.mkdir(parents=True, exist_ok=True)

        filepath = tabular_dir / "covertype.csv"
        print(f"\n💾 Сохранение в {filepath}...")
        df.to_csv(filepath, index=False, encoding='utf-8')

        # README
        readme_path = tabular_dir / "covertype_info.txt"
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write("Forest Covertype Dataset\n")
            f.write("=" * 50 + "\n\n")
            f.write("Предсказание типа леса по картографическим данным.\n\n")
            f.write(f"Rows: {len(df)}\n")
            f.write(f"Columns: {len(df.columns)}\n")
            f.write(f"Classes: 7\n\n")
            f.write("7 классов деревьев:\n")
            f.write("  1. Spruce/Fir\n")
            f.write("  2. Lodgepole Pine\n")
            f.write("  3. Ponderosa Pine\n")
            f.write("  4. Cottonwood/Willow\n")
            f.write("  5. Aspen\n")
            f.write("  6. Douglas Fir\n")
            f.write("  7. Krummholz\n")

        print(f"\n✅ Датасет загружен: {filepath}")
        print(f"📄 Описание: {readme_path}")

        return filepath

    except Exception as e:
        print(f"\n❌ Ошибка: {e}")
        return None


def create_synthetic_large():
    """Создание синтетического большого датасета"""
    print("\n" + "=" * 70)
    print("🎨 СОЗДАНИЕ СИНТЕТИЧЕСКОГО БОЛЬШОГО ДАТАСЕТА")
    print("=" * 70)

    print("\n📊 Параметры:")
    print("   • Строк: 100,000")
    print("   • Признаков: 20")
    print("   • Классов: 5")

    choice = input("\nПродолжить? [y/N]: ").strip().lower()
    if choice != 'y' and choice != 'yes':
        return None

    try:
        from sklearn.datasets import make_classification

        print("\n🔧 Генерация...")
        X, y = make_classification(
            n_samples=100000,
            n_features=20,
            n_informative=15,
            n_redundant=5,
            n_classes=5,
            n_clusters_per_class=2,
            random_state=42
        )

        df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(20)])
        df['target'] = y
        df['target_name'] = df['target'].map({
            i: f'class_{chr(65 + i)}' for i in range(5)
        })

        # Сохранение
        tabular_dir = Path("data/tabular")
        tabular_dir.mkdir(parents=True, exist_ok=True)

        filepath = tabular_dir / "synthetic_large.csv"
        df.to_csv(filepath, index=False, encoding='utf-8')

        print(f"\n✅ Создан: {filepath}")
        print(f"   📏 {len(df)} строк, {len(df.columns)} колонок")

        return filepath

    except Exception as e:
        print(f"\n❌ Ошибка: {e}")
        return None


def main():
    print("\n📥 ЗАГРУЗКА БОЛЬШИХ ДАТАСЕТОВ")
    print("=" * 70)

    print("\nВыберите датасет:")
    print("  1. Forest Covertype (581K строк, реальный)")
    print("  2. Synthetic Large (100K строк, синтетический)")
    print("  3. Оба")

    choice = input("\n👉 Ваш выбор (1-3): ").strip()

    if choice == "1":
        download_covertype()
    elif choice == "2":
        create_synthetic_large()
    elif choice == "3":
        download_covertype()
        create_synthetic_large()

    print("\n" + "=" * 70)
    print("✅ ГОТОВО!")
    print("=" * 70)
    print("\n💡 Запуск с большим датасетом:")
    print("   python app.py")
    print("   → Выбрать covertype.csv или synthetic_large.csv")
    print("\n⏱️  Ожидаемое время обучения:")
    print("   • Covertype: 5-15 минут")
    print("   • Synthetic: 2-5 минут")


if __name__ == "__main__":
    main()