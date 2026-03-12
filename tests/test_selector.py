"""
Тесты для модуля model_selector.py
"""
import pytest
import numpy as np
import sys
import os
from pathlib import Path
import tempfile
import shutil

# Добавляем src в path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from model_selector import AdaptiveModelSelector, SpecializedModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris


class TestSpecializedModel:
    """Тесты для SpecializedModel"""

    @pytest.fixture
    def sample_model(self):
        """Создание тестовой модели"""
        model = SpecializedModel(
            name="Test_Model",
            model=RandomForestClassifier(n_estimators=10, random_state=42),
            profile_requirements={'data_complexity': 'simple'},
            description="Test model for unit tests"
        )
        return model

    @pytest.fixture
    def sample_data(self):
        """Создание тестовых данных"""
        from sklearn.datasets import make_classification
        X, y = make_classification(n_samples=100, n_features=5,
                                   n_classes=2, random_state=42)
        return X, y

    def test_init(self, sample_model):
        """Тест инициализации"""
        assert sample_model.name == "Test_Model"
        assert sample_model.is_trained == False
        assert sample_model.performance_metrics == {}
        assert sample_model.training_time is None

    def test_fit(self, sample_model, sample_data):
        """Тест обучения модели"""
        X, y = sample_data

        sample_model.fit(X, y)

        assert sample_model.is_trained == True
        assert sample_model.training_time is not None
        assert sample_model.training_time > 0

    def test_predict(self, sample_model, sample_data):
        """Тест предсказания"""
        X, y = sample_data
        sample_model.fit(X, y)

        predictions = sample_model.predict(X)

        assert len(predictions) == len(X)
        assert predictions.shape[0] == X.shape[0]

    def test_predict_untrained(self, sample_model, sample_data):
        """Тест предсказания без обучения"""
        X, _ = sample_data

        with pytest.raises(ValueError):
            sample_model.predict(X)

    def test_predict_proba(self, sample_model, sample_data):
        """Тест вероятностей"""
        X, y = sample_data
        sample_model.fit(X, y)

        probas = sample_model.predict_proba(X)

        assert probas.shape[0] == X.shape[0]
        assert probas.shape[1] == 2

    def test_evaluate(self, sample_model, sample_data):
        """Тест оценки модели"""
        X, y = sample_data
        sample_model.fit(X, y)

        metrics = sample_model.evaluate(X, y)

        assert 'accuracy' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1' in metrics
        assert 0 <= metrics['accuracy'] <= 1

    def test_cross_validate(self, sample_model, sample_data):
        """Тест кросс-валидации"""
        X, y = sample_data

        cv_results = sample_model.cross_validate(X, y, cv=3)

        assert 'cv_mean' in cv_results
        assert 'cv_std' in cv_results
        assert 'cv_scores' in cv_results
        assert len(cv_results['cv_scores']) == 3
        assert 0 <= cv_results['cv_mean'] <= 1

    def test_matches_profile(self, sample_model):
        """Тест соответствия профилю"""
        data_profile = {
            'data_complexity': 'simple',
            'class_balance_ratio': 0.8,
            'n_samples': 100,
            'n_features': 5
        }

        score = sample_model.matches_profile(data_profile)

        assert 0 <= score <= 1

    def test_get_info(self, sample_model, sample_data):
        """Тест получения информации"""
        X, y = sample_data
        sample_model.fit(X, y)

        info = sample_model.get_info()

        assert info['name'] == "Test_Model"
        assert info['is_trained'] == True
        assert 'profile_requirements' in info

    def test_save_load(self, sample_model, sample_data, tmp_path):
        """Тест сохранения и загрузки"""
        X, y = sample_data
        sample_model.fit(X, y)

        filepath = tmp_path / "test_model.pkl"
        sample_model.save(str(filepath))

        assert filepath.exists()

        loaded_model = SpecializedModel.load(str(filepath))

        assert loaded_model.name == sample_model.name
        assert loaded_model.is_trained == sample_model.is_trained

    def test_predict_proba_untrained(self, sample_model, sample_data):
        """Тест predict_proba без обучения"""
        X, _ = sample_data

        with pytest.raises(ValueError):
            sample_model.predict_proba(X)


class TestAdaptiveModelSelector:
    """Тесты для AdaptiveModelSelector"""

    @pytest.fixture
    def selector(self):
        """Создание селектора"""
        return AdaptiveModelSelector()

    @pytest.fixture
    def iris_data(self):
        """Загрузка Iris dataset"""
        iris = load_iris()
        return iris.data, iris.target

    def test_init(self, selector):
        """Тест инициализации"""
        assert len(selector.models) == 0
        assert selector.best_model is None
        assert selector.data_profile is None

    def test_register_model(self, selector):
        """Тест регистрации модели"""
        model = SpecializedModel(
            name="Test_Model",
            model=RandomForestClassifier(),
            profile_requirements={}
        )

        selector.register_model(model)

        assert len(selector.models) == 1
        assert selector.models[0].name == "Test_Model"

    def test_create_default_models(self, selector):
        """Тест создания моделей по умолчанию"""
        models = selector.create_default_models()

        assert len(models) > 0
        assert len(selector.models) > 0

    def test_create_specific_models(self, selector):
        """Тест создания конкретных типов моделей"""
        models = selector.create_default_models(
            model_types=['random_forest', 'svm']
        )

        assert len(models) == 2
        model_names = [m.name for m in models]
        assert any('RandomForest' in name for name in model_names)
        assert any('SVM' in name for name in model_names)

    def test_profile_and_select(self, selector, iris_data):
        """Тест профилирования и выбора модели"""
        X, y = iris_data

        data_profile = {
            'n_samples': X.shape[0],
            'n_features': X.shape[1],
            'data_complexity': 'simple',
            'class_balance_ratio': 1.0
        }

        selector.create_default_models(model_types=['random_forest', 'svm'])
        best_model = selector.profile_and_select(X, y, data_profile, cv_folds=3)

        assert selector.best_model is not None
        assert best_model is not None
        assert best_model.is_trained == True

    def test_predict(self, selector, iris_data):
        """Тест предсказания"""
        X, y = iris_data

        selector.create_default_models(model_types=['random_forest'])
        selector.profile_and_select(X, y, cv_folds=3)

        predictions = selector.predict(X)

        assert len(predictions) == len(X)
        assert predictions.shape[0] == X.shape[0]

    def test_predict_without_selection(self, selector, iris_data):
        """Тест предсказания без выбора модели"""
        X, _ = iris_data

        with pytest.raises(ValueError):
            selector.predict(X)

    def test_predict_proba(self, selector, iris_data):
        """Тест вероятностей"""
        X, y = iris_data

        selector.create_default_models(model_types=['random_forest'])
        selector.profile_and_select(X, y, cv_folds=3)

        probas = selector.predict_proba(X)

        assert probas.shape[0] == X.shape[0]
        assert probas.shape[1] == 3

    def test_get_model_info(self, selector, iris_data):
        """Тест получения информации о модели"""
        X, y = iris_data

        selector.create_default_models(model_types=['random_forest'])
        selector.profile_and_select(X, y, cv_folds=3)

        info = selector.get_model_info()

        assert info != {}
        assert 'name' in info
        assert 'is_trained' in info

    def test_save_all_models(self, selector, iris_data, tmp_path):
        """Тест сохранения всех моделей"""
        X, y = iris_data

        selector.create_default_models(model_types=['random_forest', 'svm'])
        selector.profile_and_select(X, y, cv_folds=3)

        models_dir = tmp_path / "saved_models"
        selector.save_all_models(str(models_dir))

        assert models_dir.exists()
        assert len(list(models_dir.glob("*.pkl"))) >= 2
        assert (models_dir / "selection_history.json").exists()

    def test_load_models(self, selector, iris_data, tmp_path):
        """Тест загрузки моделей"""
        X, y = iris_data

        selector.create_default_models(model_types=['random_forest'])
        selector.profile_and_select(X, y, cv_folds=3)

        models_dir = tmp_path / "saved_models"
        selector.save_all_models(str(models_dir))

        new_selector = AdaptiveModelSelector()
        new_selector.load_models(str(models_dir))

        assert len(new_selector.models) > 0

    def test_selection_history(self, selector, iris_data):
        """Тест истории выбора"""
        X, y = iris_data

        selector.create_default_models(model_types=['random_forest', 'svm'])
        selector.profile_and_select(X, y, cv_folds=3)

        assert len(selector.selection_history) > 0
        assert 'model' in selector.selection_history[0]
        assert 'metrics' in selector.selection_history[0]

    def test_invalid_model_types(self, selector):
        """Тест некорректных типов моделей"""
        models = selector.create_default_models(model_types=['invalid_type'])

        assert len(models) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])