"""
Скрипт для загрузки и подготовки Oxford 102 Flower Dataset
"""
import os
import tarfile
import urllib.request
import shutil
from pathlib import Path
import numpy as np
from PIL import Image
import yaml


def download_oxford_flowers(download_dir: str = "data/oxford_flowers_raw"):
    """
    Загрузка Oxford 102 Flower Dataset

    Ссылки:
    - Изображения: http://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz
    - Метки: http://www.robots.ox.ac.uk/~vgg/data/flowers/102/imagelabels.mat
    - Сплит: http://www.robots.ox.ac.uk/~vgg/data/flowers/102/setid.mat
    """
    download_dir = Path(download_dir)
    download_dir.mkdir(parents=True, exist_ok=True)

    print("📥 Загрузка Oxford 102 Flower Dataset...")

    # URL для загрузки
    images_url = "http://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz"
    labels_url = "http://www.robots.ox.ac.uk/~vgg/data/flowers/102/imagelabels.mat"
    setid_url = "http://www.robots.ox.ac.uk/~vgg/data/flowers/102/setid.mat"

    # Загрузка изображений
    images_path = download_dir / "102flowers.tgz"
    if not images_path.exists():
        print("Загрузка изображений...")
        urllib.request.urlretrieve(images_url, images_path)
        print("✓ Изображения загружены")
    else:
        print("✓ Изображения уже загружены")

    # Распаковка
    extract_dir = download_dir / "images"
    if not extract_dir.exists():
        print("Распаковка изображений...")
        with tarfile.open(images_path, 'r:gz') as tar:
            tar.extractall(extract_dir)
        print("✓ Изображения распакованы")

    # Загрузка меток
    labels_path = download_dir / "imagelabels.mat"
    if not labels_path.exists():
        print("Загрузка меток...")
        urllib.request.urlretrieve(labels_url, labels_path)
        print("✓ Метки загружены")

    # Загрузка split
    setid_path = download_dir / "setid.mat"
    if not setid_path.exists():
        print("Загрузка split...")
        urllib.request.urlretrieve(setid_url, setid_path)
        print("✓ Split загружен")

    print(f"\n✅ Датасет загружен в: {download_dir}")
    return download_dir


def prepare_yolo_format(raw_dir: str = "data/oxford_flowers_raw",
                        output_dir: str = "data/oxford_flowers"):
    """
    Подготовка датасета в YOLO формате

    Создает структуру:
    data/oxford_flowers/
        ├── train/
        │   ├── images/
        │   └── labels/
        ├── val/
        │   ├── images/
        │   └── labels/
        └── data.yaml
    """
    from scipy.io import loadmat

    raw_dir = Path(raw_dir)
    output_dir = Path(output_dir)

    print("\n🔄 Подготовка датасета в YOLO формате...")

    # Загрузка меток и split
    labels = loadmat(raw_dir / "imagelabels.mat")['labels'][0]
    setid = loadmat(raw_dir / "setid.mat")

    trnid = setid['trnid'][0] - 1  # Индексы для train (0-based)
    valid = setid['valid'][0] - 1   # Индексы для validation
    tstid = setid['tstid'][0] - 1   # Индексы для test

    # Создание директорий
    splits = {
        'train': trnid,
        'val': valid,
        'test': tstid
    }

    for split_name in splits.keys():
        (output_dir / split_name / 'images').mkdir(parents=True, exist_ok=True)
        (output_dir / split_name / 'labels').mkdir(parents=True, exist_ok=True)

    # Копирование изображений и создание меток
    images_dir = raw_dir / "images" / "jpg"

    print("Копирование изображений и создание меток...")

    for split_name, indices in splits.items():
        print(f"  Обработка {split_name} ({len(indices)} изображений)...")

        for idx in indices:
            # Имя файла (image_00001.jpg)
            img_num = idx + 1
            img_name = f"image_{img_num:05d}.jpg"
            src_path = images_dir / img_name

            if not src_path.exists():
                print(f"  ⚠️  Файл не найден: {src_path}")
                continue

            # Копирование изображения
            dst_img_path = output_dir / split_name / 'images' / img_name
            shutil.copy2(src_path, dst_img_path)

            # Создание YOLO метки (класс = label - 1, т.к. в датасете классы 1-102)
            class_id = labels[idx] - 1

            # Для детекции нужен bounding box
            # Для начала создаем dummy bbox на все изображение
            # В реальности нужно использовать реальные аннотации или разметить вручную
            label_content = f"{class_id} 0.5 0.5 1.0 1.0\n"

            label_name = img_name.replace('.jpg', '.txt')
            dst_label_path = output_dir / split_name / 'labels' / label_name

            with open(dst_label_path, 'w') as f:
                f.write(label_content)

    # Создание data.yaml для YOLO
    data_yaml = {
        'path': str(output_dir.absolute()),
        'train': 'train/images',
        'val': 'val/images',
        'test': 'test/images',
        'nc': 102,
        'names': [f"class_{i}" for i in range(102)]
    }

    yaml_path = output_dir / "data.yaml"
    with open(yaml_path, 'w') as f:
        yaml.dump(data_yaml, f, default_flow_style=False)

    print(f"\n✅ Датасет подготовлен в: {output_dir}")
    print(f"📄 Конфиг YOLO: {yaml_path}")

    return output_dir


def main():
    """Основной пайплайн загрузки и подготовки"""
    print("="*70)
    print("ЗАГРУЗКА OXFORD 102 FLOWER DATASET")
    print("="*70)

    # Загрузка
    raw_dir = download_oxford_flowers()

    # Подготовка
    output_dir = prepare_yolo_format(raw_dir)

    print("\n" + "="*70)
    print("✅ ГОТОВО!")
    print("="*70)
    print(f"\nДатасет готов для обучения YOLO:")
    print(f"  • Директория: {output_dir}")
    print(f"  • Конфиг: {output_dir / 'data.yaml'}")
    print(f"\nДля обучения выполните:")
    print(f"  python scripts/train_yolo_flowers.py")


if __name__ == "__main__":
    main()