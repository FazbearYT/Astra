"""
Создание тестовых датасетов
"""

import os
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from PIL import Image, ImageDraw
import random


def create_tabular_datasets():
    print("="*70)
    print("📊 СОЗДАНИЕ ТАБЛИЧНЫХ ДАТАСЕТОВ")
    print("="*70)

    tabular_dir = Path("data/tabular")
    tabular_dir.mkdir(parents=True, exist_ok=True)

    # Iris
    print("\n1️⃣  Iris Dataset (150 образцов)")
    iris = load_iris()
    df_iris = pd.DataFrame(iris.data, columns=iris.feature_names)
    df_iris['target'] = iris.target
    df_iris['target_name'] = df_iris['target'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})

    iris_path = tabular_dir / "iris_test.csv"
    df_iris.to_csv(iris_path, index=False, encoding='utf-8')
    print(f"   ✅ {iris_path}")

    # Flowers
    print("\n2️⃣  Flowers Dataset (300 образцов)")
    np.random.seed(42)

    data = {
        'petal_length': np.concatenate([
            np.random.normal(2, 0.5, 75), np.random.normal(5, 0.8, 75),
            np.random.normal(3.5, 0.6, 75), np.random.normal(6, 1.0, 75),
        ]),
        'petal_width': np.concatenate([
            np.random.normal(1.5, 0.3, 75), np.random.normal(2.5, 0.4, 75),
            np.random.normal(1.8, 0.3, 75), np.random.normal(3.0, 0.5, 75),
        ]),
        'sepal_length': np.concatenate([
            np.random.normal(5, 0.6, 75), np.random.normal(6, 0.7, 75),
            np.random.normal(4.5, 0.5, 75), np.random.normal(7, 0.8, 75),
        ]),
        'sepal_width': np.concatenate([
            np.random.normal(3.5, 0.4, 75), np.random.normal(3.0, 0.5, 75),
            np.random.normal(3.2, 0.4, 75), np.random.normal(3.8, 0.6, 75),
        ]),
        'color_intensity': np.concatenate([
            np.random.normal(0.8, 0.1, 75), np.random.normal(0.6, 0.15, 75),
            np.random.normal(0.9, 0.08, 75), np.random.normal(0.5, 0.12, 75),
        ]),
        'flower_type': np.array([0]*75 + [1]*75 + [2]*75 + [3]*75),
    }

    df_flowers = pd.DataFrame(data)
    df_flowers['flower_name'] = df_flowers['flower_type'].map({0: 'rose', 1: 'tulip', 2: 'daisy', 3: 'lily'})

    flowers_path = tabular_dir / "flowers_test.csv"
    df_flowers.to_csv(flowers_path, index=False, encoding='utf-8')
    print(f"   ✅ {flowers_path}")

    print("\n✅ Табличные датасеты готовы!")
    return tabular_dir


def create_image_datasets():
    print("\n" + "="*70)
    print("🖼️  СОЗДАНИЕ ТЕСТОВЫХ ИЗОБРАЖЕНИЙ")
    print("="*70)

    images_dir = Path("data/images/test_flowers")
    classes = {0: 'rose', 1: 'tulip'}

    for class_id, class_name in classes.items():
        class_dir = images_dir / str(class_id) / class_name
        class_dir.mkdir(parents=True, exist_ok=True)

        for i in range(10):
            img = Image.new('RGB', (200, 200), color='white')
            draw = ImageDraw.Draw(img)

            if class_id == 0:
                center_color = (255, random.randint(50, 100), random.randint(50, 100))
                petal_color = (255, random.randint(100, 150), random.randint(100, 150))
            else:
                center_color = (random.randint(200, 255), random.randint(150, 200), 0)
                petal_color = (255, random.randint(200, 220), 0)

            for j in range(5):
                angle = (j / 5) * 2 * 3.14159
                offset_x = int(30 * np.cos(angle))
                offset_y = int(30 * np.sin(angle))
                draw.ellipse([
                    100 + offset_x - 25, 100 + offset_y - 25,
                    100 + offset_x + 25, 100 + offset_y + 25
                ], fill=petal_color)

            draw.ellipse([80, 80, 120, 120], fill=center_color)

            img_path = class_dir / f"{class_name}_{i+1:03d}.png"
            img.save(img_path)

        print(f"   ✅ Класс {class_id}: {class_name} (10 изображений)")

    print(f"\n✅ Изображения: {images_dir}")
    return images_dir


def main():
    print("\n🌸 СОЗДАНИЕ ТЕСТОВЫХ ДАТАСЕТОВ")
    create_tabular_datasets()
    create_image_datasets()

    print("\n" + "="*70)
    print("✅ ВСЕ ГОТОВО!")
    print("="*70)
    print("\n📊 ТАБЛИЧНЫЕ:")
    print("   • data/tabular/iris_test.csv")
    print("   • data/tabular/flowers_test.csv")
    print("\n🖼️  ИЗОБРАЖЕНИЯ:")
    print("   • data/images/test_flowers/")
    print("\n💡 Запуск:")
    print("   python app.py")


if __name__ == "__main__":
    main()