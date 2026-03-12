"""
Скрипт для обучения YOLO на Oxford 102 Flower Dataset
"""
import os
import sys
from pathlib import Path
import yaml


def train_yolo_flowers(epochs: int = 100, img_size: int = 640, batch_size: int = 16):
    """
    Обучение YOLOv5 на Oxford Flowers

    Требования:
    - Датасет должен быть подготовлен через download_oxford_flowers.py
    - Установлен ultralytics или yolov5
    """
    print("\n" + "="*70)
    print("🌺 ОБУЧЕНИЕ YOLO НА OXFORD FLOWERS")
    print("="*70)

    # Проверка наличия датасета
    data_yaml = Path("data/oxford_flowers/data.yaml")

    if not data_yaml.exists():
        print("\n⚠️  Датасет не найден!")
        print("Запустите сначала: python scripts/download_oxford_flowers.py")
        return

    print(f"\n✅ Датасет найден: {data_yaml}")

    # Создание директории для моделей
    models_dir = Path("models/yolo_flowers")
    models_dir.mkdir(exist_ok=True)

    # Команда для обучения
    print(f"\n🚀 Параметры обучения:")
    print(f"  • Epochs: {epochs}")
    print(f"  • Image size: {img_size}")
    print(f"  • Batch size: {batch_size}")
    print(f"  • Dataset: {data_yaml}")

    # Обучение с использованием Ultralytics YOLOv8
    try:
        from ultralytics import YOLO

        print("\n📦 Использование YOLOv8 (Ultralytics)...")

        # Загрузка предобученной модели
        model = YOLO('yolov8n.pt')

        # Обучение
        results = model.train(
            data=str(data_yaml),
            epochs=epochs,
            imgsz=img_size,
            batch=batch_size,
            name='oxford_flowers',
            project=str(models_dir),
            exist_ok=True
        )

        print("\n✅ Обучение завершено!")
        print(f"📂 Результаты сохранены в: {models_dir}")

        return model

    except ImportError:
        print("\n⚠️  Ultralytics не установлен!")
        print("Установите: pip install ultralytics")

        # Альтернатива: YOLOv5
        print("\n🔄 Попробуйте YOLOv5:")
        print("  git clone https://github.com/ultralytics/yolov5")
        print("  cd yolov5")
        print("  python train.py --img 640 --batch 16 --epochs 100 \\")
        print(f"    --data {data_yaml} --weights yolov5s.pt")

        return None


def evaluate_yolo_model(model_path: str = "models/yolo_flowers"):
    """
    Оценка обученной YOLO модели
    """
    print("\n📊 Оценка модели...")

    try:
        from ultralytics import YOLO

        # Поиск лучшего веса
        weights_path = Path(model_path) / "oxford_flowers" / "weights" / "best.pt"

        if not weights_path.exists():
            print(f"⚠️  Модель не найдена: {weights_path}")
            return

        model = YOLO(str(weights_path))

        # Валидация
        metrics = model.val(data="data/oxford_flowers/data.yaml")

        print(f"\n📈 Метрики:")
        print(f"  • mAP50: {metrics.box.map50:.4f}")
        print(f"  • mAP50-95: {metrics.box.map:.4f}")

    except Exception as e:
        print(f"⚠️  Ошибка оценки: {e}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Обучение YOLO на Oxford Flowers')
    parser.add_argument('--epochs', type=int, default=100, help='Количество эпох')
    parser.add_argument('--img-size', type=int, default=640, help='Размер изображения')
    parser.add_argument('--batch', type=int, default=16, help='Размер батча')

    args = parser.parse_args()

    # Обучение
    model = train_yolo_flowers(
        epochs=args.epochs,
        img_size=args.img_size,
        batch_size=args.batch
    )

    # Оценка (если модель обучена)
    if model is not None:
        evaluate_yolo_model()