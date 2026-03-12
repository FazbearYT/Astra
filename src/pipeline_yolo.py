"""
Подсистема адаптивного выбора YOLO моделей для детекции объектов
"""
import torch
import cv2
import numpy as np
import os
from pathlib import Path
from typing import Dict, List, Tuple, Any
import json
import yaml
from datetime import datetime


class YOLOModelWrapper:
    """Обертка для YOLO модели"""

    def __init__(self, name: str, model_path: str, config: Dict[str, Any]):
        self.name = name
        self.model_path = model_path
        self.config = config
        self.model = None
        self.performance_metrics = {}

    def load(self):
        """Загрузка модели"""
        if self.model_path.endswith('.pt'):
            # YOLOv5/v8 PyTorch модель
            self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=self.model_path)
        else:
            # ONNX или другие форматы
            raise NotImplementedError("Поддерживаются только .pt модели")

        print(f"✓ Загружена модель: {self.name}")
        return self

    def predict(self, image: np.ndarray, conf_threshold: float = 0.5) -> List[Dict]:
        """
        Детекция объектов на изображении

        Args:
            image: входное изображение (BGR)
            conf_threshold: порог уверенности

        Returns:
            список детекций
        """
        if self.model is None:
            self.load()

        # Предсказание
        results = self.model(image, conf=conf_threshold)

        # Парсинг результатов
        detections = []
        df = results.pandas().xyxy[0]  # pandas DataFrame

        for _, row in df.iterrows():
            detections.append({
                'class': row['name'],
                'confidence': float(row['confidence']),
                'bbox': {
                    'x1': int(row['xmin']),
                    'y1': int(row['ymin']),
                    'x2': int(row['xmax']),
                    'y2': int(row['ymax'])
                }
            })

        return detections

    def evaluate(self, val_dataset_path: str) -> Dict[str, float]:
        """
        Оценка качества модели на валидационном датасете

        Returns:
            метрики качества (mAP, precision, recall)
        """
        # Здесь должна быть логика оценки на датасете
        # Для примера возвращаем заглушку
        metrics = {
            'mAP_0.5': 0.85,
            'mAP_0.5:0.95': 0.65,
            'precision': 0.88,
            'recall': 0.82
        }
        self.performance_metrics = metrics
        return metrics


class ImageDataProfiler:
    """Профилировщик изображений для выбора модели"""

    @staticmethod
    def profile_image(image: np.ndarray) -> Dict[str, Any]:
        """Профилирование одного изображения"""
        return {
            'width': image.shape[1],
            'height': image.shape[0],
            'channels': image.shape[2] if len(image.shape) == 3 else 1,
            'aspect_ratio': image.shape[1] / image.shape[0],
            'avg_brightness': np.mean(image),
            'contrast': np.std(image)
        }

    @staticmethod
    def profile_dataset(dataset_path: str) -> Dict[str, Any]:
        """
        Профилирование датасета изображений

        Args:
            dataset_path: путь к датасету

        Returns:
            профиль датасета
        """
        profile = {
            'n_images': 0,
            'avg_resolution': {'width': 0, 'height': 0},
            'object_density': 0,
            'dominant_classes': [],
            'image_quality': 'good'
        }

        # Анализ датасета
        image_paths = list(Path(dataset_path).glob('*.jpg')) + \
                      list(Path(dataset_path).glob('*.png'))

        profile['n_images'] = len(image_paths)

        if len(image_paths) == 0:
            return profile

        total_width = 0
        total_height = 0

        for img_path in image_paths[:100]:
            img = cv2.imread(str(img_path))
            if img is not None:
                total_height += img.shape[0]
                total_width += img.shape[1]

        profile['avg_resolution'] = {
            'width': total_width / min(len(image_paths), 100),
            'height': total_height / min(len(image_paths), 100)
        }

        return profile


class AdaptiveYOLOSelector:
    """Адаптивный селектор YOLO моделей"""

    def __init__(self):
        self.models: List[YOLOModelWrapper] = []
        self.best_model = None
        self.data_profile = None

    def register_model(self, model: YOLOModelWrapper):
        """Регистрация YOLO модели"""
        self.models.append(model)
        print(f"✓ Зарегистрирована YOLO модель: {model.name}")

    def create_specialized_models(self, models_config: Dict[str, str]):
        """
        Создание набора специализированных YOLO моделей

        Args:
            models_config: dict {model_name: model_path}
        """
        for name, path in models_config.items():
            config = {
                'description': f'Specialized model for {name}',
                'input_size': 640,
                'classes': []
            }

            model = YOLOModelWrapper(
                name=name,
                model_path=path,
                config=config
            )
            self.register_model(model)

    def profile_data(self, dataset_path: str) -> Dict:
        """Профилирование датасета"""
        profiler = ImageDataProfiler()
        self.data_profile = profiler.profile_dataset(dataset_path)

        print("\n" + "=" * 60)
        print("📊 ПРОФИЛЬ ДАТАСЕТА ИЗОБРАЖЕНИЙ:")
        print("=" * 60)
        print(f"• Количество изображений: {self.data_profile['n_images']}")
        print(f"• Среднее разрешение: {self.data_profile['avg_resolution']}")
        print("=" * 60)

        return self.data_profile

    def select_best_model(self, val_dataset_path: str) -> YOLOModelWrapper:
        """
        Выбор лучшей модели на валидационном датасете
        """
        print("\n🔄 ТЕСТИРОВАНИЕ YOLO МОДЕЛЕЙ...")
        print("=" * 60)

        best_map = 0

        for model in self.models:
            print(f"\n🤖 Тестирование модели: {model.name}")

            # Оценка модели
            metrics = model.evaluate(val_dataset_path)

            print(f"   mAP@0.5: {metrics['mAP_0.5']:.4f}")
            print(f"   mAP@0.5:0.95: {metrics['mAP_0.5:0.95']:.4f}")
            print(f"   Precision: {metrics['precision']:.4f}")
            print(f"   Recall: {metrics['recall']:.4f}")

            if metrics['mAP_0.5'] > best_map:
                best_map = metrics['mAP_0.5']
                self.best_model = model

        print("\n" + "=" * 60)
        print(f"🏆 ЛУЧШАЯ МОДЕЛЬ: {self.best_model.name}")
        print(f"   mAP@0.5: {best_map:.4f}")
        print("=" * 60)

        return self.best_model

    def detect(self, image: np.ndarray) -> List[Dict]:
        """Детекция объектов на изображении"""
        if self.best_model is None:
            raise ValueError("Сначала выберите модель через select_best_model()")

        return self.best_model.predict(image)

    def visualize_detection(self, image: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """Визуализация детекций"""
        output = image.copy()

        for det in detections:
            bbox = det['bbox']
            label = f"{det['class']}: {det['confidence']:.2f}"

            cv2.rectangle(output,
                          (bbox['x1'], bbox['y1']),
                          (bbox['x2'], bbox['y2']),
                          (0, 255, 0), 2)

            cv2.putText(output, label, (bbox['x1'], bbox['y1'] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        return output


def main_yolo():
    """Пример использования YOLO пайплайна"""
    print("\n" + "=" * 60)
    print("🚀 АДАПТИВНЫЙ ВЫБОР YOLO МОДЕЛЕЙ")
    print("=" * 60)

    # Конфигурация моделей
    models_config = {
        'YOLOv5_Iris_Specialist': 'models/yolo_iris_v1.pt',
        'YOLOv5_Flower_General': 'models/yolo_flowers_v2.pt',
        'YOLOv8_Plant_Detector': 'models/yolov8_plants.pt'
    }

    # Создание селектора
    selector = AdaptiveYOLOSelector()

    # Создание моделей
    selector.create_specialized_models(models_config)

    # Профилирование датасета
    # selector.profile_data('data/iris_images/')

    # Выбор лучшей модели
    # best_model = selector.select_best_model('data/val/')

    # Детекция на изображении
    # image = cv2.imread('test_image.jpg')
    # detections = selector.detect(image)
    # result = selector.visualize_detection(image, detections)
    # cv2.imwrite('result.jpg', result)

    print("\n✅ YOLO пайплайн готов к работе!")

    return selector


if __name__ == "__main__":
    selector = main_yolo()