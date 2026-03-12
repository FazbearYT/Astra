"""
Тесты для pipeline модулей
"""
import pytest
import numpy as np
import sys
import os
from pathlib import Path
import tempfile

# Добавляем src в path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score


class TestTabularPipeline:
    """Тесты для pipeline_tabular.py"""

    @pytest.fixture
    def iris_data(self):
        """Загрузка Iris dataset"""
        iris = load_iris()
        return iris.data, iris.target, iris

    def test_full_pipeline(self, iris_data):
        """Тест полного пайплайна"""
        from pipeline_tabular import AdaptiveModelSelector, DataProfiler

        X, y, iris = iris_data

        selector = AdaptiveModelSelector()
        selector.create_specialized_models()

        profiler = DataProfiler()
        profile = profiler.profile(X, y)

        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        best_model = selector.select_best_model(
            X_train, y_train, X_test, y_test
        )

        predictions = selector.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)

        assert accuracy > 0.8
        assert best_model is not None

    def test_model_saving(self, iris_data, tmp_path):
        """Тест сохранения моделей"""
        from pipeline_tabular import AdaptiveModelSelector
        from sklearn.model_selection import train_test_split

        X, y, _ = iris_data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        selector = AdaptiveModelSelector()
        selector.create_specialized_models()
        selector.select_best_model(X_train, y_train, X_test, y_test)

        models_dir = tmp_path / "models"
        selector.save_models(str(models_dir))

        assert models_dir.exists()
        assert len(list(models_dir.glob("*.pkl"))) > 0

    def test_model_loading(self, iris_data, tmp_path):
        """Тест загрузки моделей"""
        from pipeline_tabular import AdaptiveModelSelector
        from sklearn.model_selection import train_test_split

        X, y, _ = iris_data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        selector1 = AdaptiveModelSelector()
        selector1.create_specialized_models()
        selector1.select_best_model(X_train, y_train, X_test, y_test)

        models_dir = tmp_path / "models"
        selector1.save_models(str(models_dir))

        selector2 = AdaptiveModelSelector()
        selector2.load_models(str(models_dir))

        assert len(selector2.models) > 0


class TestYOLOPipeline:
    """Тесты для pipeline_yolo.py"""

    def test_yolo_wrapper_init(self):
        """Тест инициализации YOLO обёртки"""
        from pipeline_yolo import YOLOModelWrapper

        model = YOLOModelWrapper(
            name="Test_YOLO",
            model_path="dummy.pt",
            config={'input_size': 640}
        )

        assert model.name == "Test_YOLO"
        assert model.model is None

    def test_image_profiler(self):
        """Тест профилировщика изображений"""
        from pipeline_yolo import ImageDataProfiler
        import numpy as np

        profiler = ImageDataProfiler()

        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        profile = profiler.profile_image(image)

        assert profile['width'] == 640
        assert profile['height'] == 480
        assert profile['channels'] == 3

    def test_adaptive_yolo_selector_init(self):
        """Тест инициализации YOLO селектора"""
        from pipeline_yolo import AdaptiveYOLOSelector

        selector = AdaptiveYOLOSelector()

        assert len(selector.models) == 0
        assert selector.best_model is None


class TestIntegration:
    """Интеграционные тесты"""

    def test_end_to_end_tabular(self):
        """Сквозной тест для табличных данных"""
        from src.model_profiler import DataProfiler
        from src.model_selector import AdaptiveModelSelector
        from sklearn.datasets import load_iris
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score

        iris = load_iris()
        X, y = iris.data, iris.target

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        profiler = DataProfiler("Iris_Integration_Test")
        profile = profiler.profile_tabular_data(X_train, y_train)

        selector = AdaptiveModelSelector()
        selector.create_default_models(model_types=['random_forest', 'svm'])
        best_model = selector.profile_and_select(
            X, y,
            data_profile=profile.to_dict(),
            cv_folds=3
        )

        predictions = selector.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)

        assert accuracy > 0.85
        assert best_model.is_trained
        assert len(selector.selection_history) > 0

    def test_error_handling(self):
        """Тест обработки ошибок"""
        from src.model_selector import AdaptiveModelSelector
        import numpy as np

        selector = AdaptiveModelSelector()

        with pytest.raises(ValueError):
            selector.predict(np.array([[1, 2, 3, 4]]))

        with pytest.raises(Exception):
            selector.profile_and_select(
                np.array([]).reshape(0, 4),
                np.array([])
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])