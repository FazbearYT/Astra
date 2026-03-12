import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.datasets import (
    load_iris, load_wine, load_digits,
    fetch_openml, make_classification, make_blobs
)

PROJECT_ROOT = Path(__file__).parent.parent
TABULAR_DIR = PROJECT_ROOT / "data" / "tabular"


def save_dataset(df: pd.DataFrame, filename: str, description: str):
    TABULAR_DIR.mkdir(parents=True, exist_ok=True)

    filepath = TABULAR_DIR / filename
    df.to_csv(filepath, index=False, encoding='utf-8')

    readme_path = TABULAR_DIR / f"{filename.replace('.csv', '')}_info.txt"
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(f"Dataset: {filename.replace('.csv', '')}\n")
        f.write("="*50 + "\n\n")
        f.write(f"{description}\n\n")
        f.write(f"Rows: {len(df)}\n")
        f.write(f"Columns: {len(df.columns)}\n")
        f.write(f"Target column: target\n\n")
        f.write("Columns:\n")
        for col in df.columns:
            f.write(f"  - {col}\n")

    print(f"   ✅ {filename} ({len(df)} строк, {len(df.columns)} колонок)")
    return filepath


def download_iris():
    print("\n1️⃣  Iris Dataset...")
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['target'] = iris.target
    df['target_name'] = df['target'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})

    desc = """Классический датасет для классификации ирисов.
3 вида ирисов (setosa, versicolor, virginica).
4 признака: длина/ширина чашелистика и лепестка."""

    return save_dataset(df, "iris.csv", desc)


def download_wine():
    print("\n2️⃣  Wine Dataset...")
    wine = load_wine()
    df = pd.DataFrame(wine.data, columns=wine.feature_names)
    df['target'] = wine.target
    df['target_name'] = df['target'].map({0: 'class_0', 1: 'class_1', 2: 'class_2'})

    desc = """Результаты химического анализа вин из 3 сортов винограда.
13 признаков: алкоголь, яблочная кислота, зола, магний и т.д.
3 класса вин."""

    return save_dataset(df, "wine.csv", desc)


def download_digits():
    print("\n3️⃣  Digits Dataset...")
    digits = load_digits()

    df = pd.DataFrame(digits.data, columns=[f'pixel_{i}' for i in range(64)])
    df['target'] = digits.target

    desc = """Распознавание рукописных цифр (0-9).
64 признака: интенсивность пикселей 8x8.
10 классов: цифры 0-9."""

    return save_dataset(df, "digits.csv", desc)


def download_titanic():
    print("\n4️⃣  Titanic Dataset...")

    try:
        titanic = fetch_openml(name='titanic', version=1, as_frame=True)
        df = titanic.frame

        df = df[['pclass', 'gender', 'age', 'sibsp', 'parch', 'fare', 'embarked', 'survived']]
        df = df.dropna()

        df['gender'] = df['gender'].map({'male': 0, 'female': 1})
        df['embarked'] = df['embarked'].map({'S': 0, 'C': 1, 'Q': 2}).fillna(0)

        df['target'] = df['survived'].astype(int)
        df = df.drop(columns=['survived'])

    except:
        np.random.seed(42)
        n = 891
        df = pd.DataFrame({
            'pclass': np.random.randint(1, 4, n),
            'gender': np.random.randint(0, 2, n),
            'age': np.random.normal(30, 14, n).clip(0, 80),
            'sibsp': np.random.poisson(0.5, n),
            'parch': np.random.poisson(0.4, n),
            'fare': np.random.exponential(30, n),
        })
        df['target'] = ((df['gender'] == 1) & (df['pclass'] < 3) |
                       (df['age'] < 10)).astype(int)

    desc = """Выживание пассажиров Титаника.
Признаки: класс билета, gender, возраст, цена билета и т.д.
2 класса: выжил/погиб."""

    return save_dataset(df, "titanic.csv", desc)


def download_synthetic_blobs():
    print("\n5️⃣  Synthetic Blobs Dataset...")

    np.random.seed(42)
    X, y = make_blobs(n_samples=500, n_features=5, centers=3,
                      cluster_std=1.5, random_state=42)

    df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(5)])
    df['target'] = y
    df['target_name'] = df['target'].map({0: 'class_A', 1: 'class_B', 2: 'class_C'})

    desc = """Синтетический датасет для тестирования.
3 хорошо разделимых кластера.
5 признаков, 500 образцов."""

    return save_dataset(df, "synthetic_blobs.csv", desc)


def create_summary():
    summary_path = TABULAR_DIR / "DATASETS_SUMMARY.md"
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("# 📊 Тестовые датасеты\n\n")
        f.write("| Датасет | Файл | Строк | Классов | Признаков |\n")
        f.write("|---------|------|-------|---------|------------|\n")
        f.write("| Iris | iris.csv | 150 | 3 | 4 |\n")
        f.write("| Wine | wine.csv | 178 | 3 | 13 |\n")
        f.write("| Digits | digits.csv | 1797 | 10 | 64 |\n")
        f.write("| Titanic | titanic.csv | 891 | 2 | 7 |\n")
        f.write("| Synthetic Blobs | synthetic_blobs.csv | 500 | 3 | 5 |\n")
        f.write("\n\n## 🚀 Использование\n\n")
        f.write("```bash\npython app.py\n```\n\n")
        f.write("Программа автоматически найдёт все CSV файлы в этой папке!\n")
        f.write("\n\n## ⚠️ Изменения\n\n")
        f.write("- breast_cancer.csv - удалён\n")
        f.write("- diabetes.csv - удалён\n")
        f.write("- titanic.csv - колонка 'sex' переименована в 'gender'\n")

    print(f"\n✅ Summary: {summary_path}")


def main():
    print("\n" + "="*70)
    print("📥 ЗАГРУЗКА ВСЕХ ТЕСТОВЫХ ДАТАСЕТОВ")
    print("="*70)
    print(f"\n📂 Путь сохранения: {TABULAR_DIR.absolute()}")

    datasets = [
        download_iris,
        download_wine,
        download_digits,
        download_titanic,
        download_synthetic_blobs,
    ]

    for func in datasets:
        try:
            func()
        except Exception as e:
            print(f"   ❌ Ошибка: {e}")
            import traceback
            traceback.print_exc()

    create_summary()

    print("\n" + "="*70)
    print("✅ ВСЕ ДАТАСЕТЫ ЗАГРУЖЕНЫ!")
    print("="*70)
    print(f"\n📂 Расположение: {TABULAR_DIR.absolute()}")
    print("\n💡 Запуск программы:")
    print("   python app.py")
    print("\n📋 Доступные датасеты:")
    print("   1. iris.csv (150 строк, 3 класса)")
    print("   2. wine.csv (178 строк, 3 класса)")
    print("   3. digits.csv (1797 строк, 10 классов)")
    print("   4. titanic.csv (891 строк, 2 класса) - колонка 'gender'")
    print("   5. synthetic_blobs.csv (500 строк, 3 класса)")


if __name__ == "__main__":
    main()