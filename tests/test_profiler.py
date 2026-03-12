"""
Тесты для модуля model_profiler.py
"""
import pytest
import numpy as np
import sys
import os
from pathlib import Path

# Добавляем src в path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from model_profiler import DataProfiler, DatasetProfile, FeatureProfile, profile_from_csv


class TestDataProfiler:
    """Тесты для DataProfiler"""

    @pytest.fixture
    def sample_data(self):
        """Создание тестовых данных"""
        np.random.seed(42)
        X = np.random.randn(100, 5)
        y = np.random.randint(0, 3, 100)
        return X, y

    @pytest.fixture
    def profiler(self):
        """Создание профилировщика"""
        return DataProfiler(dataset_name="Test_Dataset")

    def test_init(self, profiler):
        """Тест инициализации"""
        assert profiler.dataset_name == "Test_Dataset"
        assert profiler.profile is None

    def test_profile_tabular_data(self, profiler, sample_data):
        """Тест профилирования табличных данных"""
        X, y = sample_data

        profile = profiler.profile_tabular_data(X, y)

        assert isinstance(profile, DatasetProfile)
        assert profile.n_samples == 100
        assert profile.n_features == 5
        assert profile.n_classes == 3
        assert len(profile.feature_profiles) == 5
        assert profile.data_complexity in ['simple', 'medium', 'complex']

    def test_profile_without_target(self, profiler, sample_data):
        """Тест профилирования без целевой переменной"""
        X, _ = sample_data

        profile = profiler.profile_tabular_data(X, y=None)

        assert profile.n_classes is None
        assert profile.class_distribution is None
        assert profile.class_balance_ratio is None

    def test_feature_profiles(self, profiler, sample_data):
        """Тест профилей признаков"""
        X, y = sample_data

        profile = profiler.profile_tabular_data(X, y)

        for fp in profile.feature_profiles:
            assert isinstance(fp, FeatureProfile)
            assert isinstance(fp.name, str)
            assert isinstance(fp.mean, float)
            assert isinstance(fp.std, float)
            assert fp.outliers_count >= 0
            assert fp.missing_values >= 0

    def test_recommended_models(self, profiler, sample_data):
        """Тест рекомендаций моделей"""
        X, y = sample_data

        profile = profiler.profile_tabular_data(X, y)

        assert isinstance(profile.recommended_models, list)
        assert len(profile.recommended_models) > 0
        assert all(isinstance(m, str) for m in profile.recommended_models)

    def test_preprocessing_needs(self, profiler, sample_data):
        """Тест потребностей в предобработке"""
        X, y = sample_data

        profile = profiler.profile_tabular_data(X, y)

        assert isinstance(profile.preprocessing_needs, list)
        assert len(profile.preprocessing_needs) > 0

    def test_profile_save_load(self, profiler, sample_data, tmp_path):
        """Тест сохранения и загрузки профиля"""
        X, y = sample_data

        profile = profiler.profile_tabular_data(X, y)

        # Сохранение
        filepath = tmp_path / "test_profile.json"
        profile.save(str(filepath))

        assert filepath.exists()

        # Загрузка
        loaded_profile = DatasetProfile.load(str(filepath))

        assert loaded_profile.n_samples == profile.n_samples
        assert loaded_profile.n_features == profile.n_features
        assert loaded_profile.dataset_name == profile.dataset_name

    def test_print_summary(self, profiler, sample_data, capsys):
        """Тест вывода сводки"""
        X, y = sample_data

        profiler.profile_tabular_data(X, y)
        profiler.print_summary()

        captured = capsys.readouterr()
        assert "ПРОФИЛЬ ДАТАСЕТА" in captured.out
        assert "Образцов:" in captured.out
        assert "Признаков:" in captured.out

    def test_visualize_profile(self, profiler, sample_data, tmp_path):
        """Тест визуализации профиля"""
        X, y = sample_data

        profiler.profile_tabular_data(X, y)

        filepath = tmp_path / "test_profile.png"
        profiler.visualize_profile(save_path=str(filepath))

        assert filepath.exists()

    def test_profile_from_csv(self, tmp_path):
        """Тест профилирования из CSV"""
        import pandas as pd

        # Создание тестового CSV
        csv_path = tmp_path / "test_data.csv"
        df = pd.DataFrame({
            'feature1': np.random.randn(50),
            'feature2': np.random.randn(50),
            'target': np.random.randint(0, 2, 50)
        })
        df.to_csv(csv_path, index=False)

        # Профилирование
        profile = profile_from_csv(
            str(csv_path),
            target_column='target',
            dataset_name='CSV_Test'
        )

        assert profile.n_samples == 50
        assert profile.n_features == 2
        assert profile.n_classes == 2

    def test_invalid_input(self, profiler):
        """Тест некорректных входных данных"""
        # Пустой массив
        X = np.array([]).reshape(0, 5)
        y = np.array([])

        with pytest.raises(Exception):
            profiler.profile_tabular_data(X, y)

    def test_profile_without_visualization(self, profiler, sample_data):
        """Тест что visualize_profile требует профиль"""
        with pytest.raises(ValueError):
            profiler.visualize_profile()

    def test_print_summary_without_profile(self, profiler):
        """Тест что print_summary требует профиль"""
        with pytest.raises(ValueError):
            profiler.print_summary()


class TestDatasetProfile:
    """Тесты для DatasetProfile"""

    def test_to_dict(self):
        """Тест конвертации в словарь"""
        from dataclasses import asdict

        profile = DatasetProfile(
            n_samples=100,
            n_features=5,
            n_classes=3,
            memory_size_mb=0.01,
            feature_profiles=[],
            class_distribution={'0': 33, '1': 33, '2': 34},
            class_balance_ratio=0.97,
            data_complexity='simple',
            feature_correlation_matrix=None,
            recommended_models=['RandomForest'],
            preprocessing_needs=['none_required'],
            created_at='2024-01-01',
            dataset_name='Test'
        )

        profile_dict = profile.to_dict()

        assert isinstance(profile_dict, dict)
        assert profile_dict['n_samples'] == 100
        assert profile_dict['dataset_name'] == 'Test'


class TestFeatureProfile:
    """Тесты для FeatureProfile"""

    def test_creation(self):
        """Тест создания профиля признака"""
        fp = FeatureProfile(
            name='test_feature',
            mean=0.5,
            std=0.2,
            min=0.0,
            max=1.0,
            median=0.5,
            skewness=0.0,
            kurtosis=0.0,
            missing_values=0,
            outliers_count=0
        )

        assert fp.name == 'test_feature'
        assert fp.mean == 0.5
        assert fp.std == 0.2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])