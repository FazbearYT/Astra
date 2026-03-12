import os
import tarfile
import urllib.request
from pathlib import Path
from scipy.io import loadmat
import yaml


def download_oxford_flowers():
    """Загрузка Oxford 102 Flowers"""
    print("=" * 70)
    print("🌺 ЗАГРУЗКА OXFORD 102 FLOWER DATASET")
    print("=" * 70)

    download_dir = Path("data/images/oxford_flowers_raw")
    download_dir.mkdir(parents=True, exist_ok=True)

    images_url = "http://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz"
    labels_url = "http://www.robots.ox.ac.uk/~vgg/data/flowers/102/imagelabels.mat"
    setid_url = "http://www.robots.ox.ac.uk/~vgg/data/flowers/102/setid.mat"

    images_path = download_dir / "102flowers.tgz"
    if not images_path.exists():
        print("\n📥 Загрузка изображений (~350 MB)...")
        print("   Это займет 2-10 минут в зависимости от скорости")

        def download_with_progress(url, filepath):
            import urllib.request
            from tqdm import tqdm

            class ProgressHook:
                def __init__(self, total):
                    self.pbar = tqdm(total=total, unit='B', unit_scale=True)

                def __call__(self, block_num, block_size, total_size):
                    if total_size > 0:
                        self.pbar.total = total_size
                    self.pbar.update(block_size)

            urllib.request.urlretrieve(url, filepath, ProgressHook(0))

        try:
            urllib.request.urlretrieve(images_url, images_path)
            print("   ✅ Загружено")
        except Exception as e:
            print(f"   ❌ Ошибка: {e}")
            return None
    else:
        print(f"\n✅ Изображения уже загружены")

    extract_dir = download_dir / "images"
    if not extract_dir.exists():
        print("\n📦 Распаковка...")
        with tarfile.open(images_path, 'r:gz') as tar:
            tar.extractall(extract_dir)
        print("   ✅ Распаковано")
    else:
        print(f"✅ Уже распаковано")

    labels_path = download_dir / "imagelabels.mat"
    if not labels_path.exists():
        print("\n📥 Загрузка меток...")
        urllib.request.urlretrieve(labels_url, labels_path)
        print("   ✅ Загружено")
    else:
        print(f"✅ Метки уже загружены")

    setid_path = download_dir / "setid.mat"
    if not setid_path.exists():
        print("\n📥 Загрузка split...")
        urllib.request.urlretrieve(setid_url, setid_path)
        print("   ✅ Загружено")
    else:
        print(f"✅ Split уже загружен")

    print("\n" + "=" * 70)
    print("✅ OXFORD 102 FLOWERS ЗАГРУЖЕН")
    print("=" * 70)
    print(f"\n📂 Расположение: {download_dir}")
    print(f"📊 102 класса, ~8000 изображений")
    print(f"\n💡 Для подготовки к YOLO:")
    print(f"   python scripts/prepare_oxford_for_yolo.py")

    return download_dir


def main():
    download_oxford_flowers()


if __name__ == "__main__":
    main()