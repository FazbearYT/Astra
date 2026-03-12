from pathlib import Path
from PIL import Image, ImageDraw
import numpy as np
import random


def create_flower_image(flower_type: int, img_id: int) -> Image.Image:
    """
    Создание простого изображения цветка

    Args:
        flower_type: 0 - роза, 1 - тюльпан
        img_id: номер изображения

    Returns:
        PIL.Image объект
    """
    img = Image.new('RGB', (200, 200), color='white')
    draw = ImageDraw.Draw(img)

    if flower_type == 0:
        center_color = (
            255,
            random.randint(50, 120),
            random.randint(80, 150)
        )
        petal_color = (
            255,
            random.randint(100, 180),
            random.randint(120, 200)
        )
        stem_color = (random.randint(20, 60), random.randint(100, 150), random.randint(20, 60))
    else:
        center_color = (
            random.randint(255, 255),
            random.randint(140, 180),
            0
        )
        petal_color = (
            255,
            random.randint(180, 220),
            random.randint(0, 50)
        )
        stem_color = (random.randint(20, 60), random.randint(100, 150), random.randint(20, 60))

    stem_x = 100
    draw.rectangle([stem_x - 5, 120, stem_x + 5, 190], fill=stem_color)

    draw.polygon([
        (stem_x, 140),
        (stem_x + 30, 130),
        (stem_x, 150)
    ], fill=stem_color)

    draw.polygon([
        (stem_x, 160),
        (stem_x - 25, 150),
        (stem_x, 170)
    ], fill=stem_color)

    center_x, center_y = 100, 100
    num_petals = 5 if flower_type == 0 else 6

    for j in range(num_petals):
        angle = (j / num_petals) * 2 * np.pi
        radius = 35 if flower_type == 0 else 30

        offset_x = int(radius * np.cos(angle))
        offset_y = int(radius * np.sin(angle))

        petal_size = 25 if flower_type == 0 else 22

        draw.ellipse([
            center_x + offset_x - petal_size,
            center_y + offset_y - petal_size,
            center_x + offset_x + petal_size,
            center_y + offset_y + petal_size
        ], fill=petal_color)

    center_size = 20 if flower_type == 0 else 18
    draw.ellipse([
        center_x - center_size,
        center_y - center_size,
        center_x + center_size,
        center_y + center_size
    ], fill=center_color)

    if flower_type == 1:
        for j in range(3):
            angle = (j / 3) * 2 * np.pi + 0.3
            offset_x = int(25 * np.cos(angle))
            offset_y = int(25 * np.sin(angle))
            draw.ellipse([
                center_x + offset_x - 15,
                center_y + offset_y - 15,
                center_x + offset_x + 15,
                center_y + offset_y + 15
            ], fill=petal_color)

    return img


def create_sample_flower_datasets():
    print("=" * 70)
    print(" СОЗДАНИЕ ТЕСТОВОГО ДАТАСЕТА ИЗОБРАЖЕНИЙ")
    print("=" * 70)

    dataset_dir = Path("data/images/sample_flowers")

    classes = {
        0: 'rose',
        1: 'tulip'
    }

    num_samples_per_class = 10

    print(f"\n📁 Создание {len(classes)} классов по {num_samples_per_class} изображений...")

    for class_id, class_name in classes.items():
        class_dir = dataset_dir / str(class_id) / class_name
        class_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n   Класс {class_id}: {class_name}")

        for i in range(num_samples_per_class):
            img = create_flower_image(class_id, i)

            img_path = class_dir / f"{class_name}_{i + 1:03d}.png"
            img.save(img_path)

            if i < 3:
                print(f"      ✓ {img_path.name}")

        if num_samples_per_class > 3:
            print(f"      ... и еще {num_samples_per_class - 3} изображений")

    readme_path = dataset_dir / "README.txt"
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write("Тестовый датасет изображений цветов\n")
        f.write("=" * 50 + "\n\n")
        f.write("Структура:\n")
        f.write("  0/rose/ - изображения роз (10 шт)\n")
        f.write("  1/tulip/ - изображения тюльпанов (10 шт)\n\n")
        f.write("Формат:\n")
        f.write("  - PNG 200x200 пикселей\n")
        f.write("  - RGB цвет\n")
        f.write("  - Простая синтетическая генерация\n")

    print(f"\n✅ Датасет создан: {dataset_dir}")
    print(f"📄 Информация: {readme_path}")
    print(f"\n💡 Для использования в app.py:")
    print(f"   1. Поместите изображения в data/images/sample_flowers/")
    print(f"   2. Запустите: python app.py")
    print(f"   3. Выберите опцию 'Автоматическое обнаружение'")

    return dataset_dir


def create_advanced_sample_images():
    print("\n" + "=" * 70)
    print("🎨 СОЗДАНИЕ РАСШИРЕННОГО ДАТАСЕТА")
    print("=" * 70)

    dataset_dir = Path("data/images/advanced_flowers")
    classes = {
        0: 'rose',
        1: 'tulip',
        2: 'daisy'
    }

    num_samples = 15

    for class_id, class_name in classes.items():
        class_dir = dataset_dir / str(class_id) / class_name
        class_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n   Класс {class_id}: {class_name} ({num_samples} изображений)")

        for i in range(num_samples):
            img = create_flower_image(class_id % 2, i)

            if class_id == 2:
                img = Image.new('RGB', (200, 200), color='skyblue')
                draw = ImageDraw.Draw(img)

                for j in range(8):
                    angle = (j / 8) * 2 * np.pi
                    offset_x = int(40 * np.cos(angle))
                    offset_y = int(40 * np.sin(angle))
                    draw.ellipse([
                        100 + offset_x - 20, 100 + offset_y - 20,
                        100 + offset_x + 20, 100 + offset_y + 20
                    ], fill='white')

                draw.ellipse([85, 85, 115, 115], fill='yellow')

            img_path = class_dir / f"{class_name}_{i + 1:03d}.png"
            img.save(img_path)

    print(f"\n✅ Расширенный датасет: {dataset_dir}")


def main():
    print("\n🌸 ГЕНЕРАТОР ТЕСТОВЫХ ИЗОБРАЖЕНИЙ ЦВЕТОВ")
    print("=" * 70)

    create_sample_flower_datasets()

    print("\n" + "=" * 70)
    choice = input("Создать расширенный датасет (3 класса)? [y/N]: ").strip().lower()

    if choice == 'y' or choice == 'yes':
        create_advanced_sample_images()

    print("\n" + "=" * 70)
    print("✅ ГОТОВО!")
    print("=" * 70)
    print("\n📂 Созданные датасеты:")
    print("   • data/images/sample_flowers/ (2 класса)")
    if choice == 'y':
        print("   • data/images/advanced_flowers/ (3 класса)")

    print("\n💡 Следующие шаги:")
    print("   1. Проверьте данные в папке data/images/")
    print("   2. Запустите: python app.py")
    print("   3. Выберите автоматическое обнаружение")


if __name__ == "__main__":
    main()