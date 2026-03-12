"""
Модуль профилирования данных для адаптивной системы выбора моделей
Анализирует характеристики датасета и определяет его профиль
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from typing import Dict, List, Any, Tuple, Optional
import json
import os
from dataclasses import dataclass, asdict
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')


@dataclass
class FeatureProfile:
    """Профиль одного признака"""
    name: str
    mean: float
    std: float
    min: float
    max: float
    median: float
    skewness: float
    kurtosis: float
    missing_values: int
    outliers_count: int
    correlation_with_target: Optional[float] = None


@dataclass
class DatasetProfile:
    """Полный профиль датасета"""
    # Основная информация
    n_samples: int
    n_features: int
    n_classes: Optional[int]
    memory_size_mb: float

    # Статистики
    feature_profiles: List[FeatureProfile]

    # Распределение классов
    class_distribution: Optional[Dict[str, int]]
    class_balance_ratio: Optional[float]

    # Сложность данных
    data_complexity: str  # 'simple', 'medium', 'complex'
    feature_correlation_matrix: Optional[List[List[float]]]

    # Рекомендации
    recommended_models: List[str]
    preprocessing_needs: List[str]

    # Метаданные
    created_at: str
    dataset_name: str

    def to_dict(self) -> Dict:
        """Конвертация в словарь"""
        return asdict(self)

    def save(self, filepath: str):
        """Сохранение профиля в JSON"""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, default=str)
        print(f"✓ Профиль сохранен: {filepath}")

    @classmethod
    def load(cls, filepath: str) -> 'DatasetProfile':
        """Загрузка профиля из JSON"""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return cls(**data)


class DataProfiler:
    """
    Профилировщик данных для ML моделей

    Анализирует:
    - Статистики признаков
    - Распределение классов
    - Корреляции
    - Выбросы
    - Сложность данных
    """

    def __init__(self, dataset_name: str = "unknown"):
        self.dataset_name = dataset_name
        self.profile = None

    def profile_tabular_data(self, X: np.ndarray, y: Optional[np.ndarray] = None,
                             feature_names: Optional[List[str]] = None) -> DatasetProfile:
        """
        Профилирование табличных данных

        Args:
            X: матрица признаков (n_samples, n_features)
            y: вектор целевых значений (опционально)
            feature_names: имена признаков

        Returns:
            DatasetProfile с полным описанием данных
        """
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(X.shape[1])]

        # Основная информация
        n_samples, n_features = X.shape
        memory_size_mb = X.nbytes / (1024 ** 2)

        # Профилирование признаков
        feature_profiles = []
        for i in range(n_features):
            feature = X[:, i]

            # Вычисление статистик
            profile = FeatureProfile(
                name=feature_names[i],
                mean=float(np.mean(feature)),
                std=float(np.std(feature)),
                min=float(np.min(feature)),
                max=float(np.max(feature)),
                median=float(np.median(feature)),
                skewness=float(stats.skew(feature)),
                kurtosis=float(stats.kurtosis(feature)),
                missing_values=int(np.sum(np.isnan(feature))),
                outliers_count=self._count_outliers(feature),
                correlation_with_target=None
            )

            # Корреляция с целевой переменной
            if y is not None:
                try:
                    corr, _ = stats.pearsonr(feature, y)
                    profile.correlation_with_target = float(corr)
                except:
                    profile.correlation_with_target = None

            feature_profiles.append(profile)

        # Распределение классов
        class_distribution = None
        class_balance_ratio = None
        n_classes = None

        if y is not None:
            unique, counts = np.unique(y, return_counts=True)
            class_distribution = dict(zip([str(u) for u in unique], [int(c) for c in counts]))
            n_classes = len(unique)
            class_balance_ratio = float(min(counts) / max(counts)) if max(counts) > 0 else 0

        # Корреляционная матрица
        corr_matrix = None
        if n_features <= 20:  # Только для небольшого числа признаков
            corr_matrix = np.corrcoef(X.T)
            # Замена NaN на 0
            corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)
            corr_matrix = corr_matrix.tolist()

        # Определение сложности данных
        complexity = self._assess_complexity(X, y, n_samples, n_features)

        # Рекомендации моделей
        recommended_models = self._recommend_models(X, y, complexity, class_balance_ratio)

        # Потребности в предобработке
        preprocessing_needs = self._detect_preprocessing_needs(X, feature_profiles)

        # Создание профиля
        self.profile = DatasetProfile(
            n_samples=n_samples,
            n_features=n_features,
            n_classes=n_classes,
            memory_size_mb=memory_size_mb,
            feature_profiles=feature_profiles,
            class_distribution=class_distribution,
            class_balance_ratio=class_balance_ratio,
            data_complexity=complexity,
            feature_correlation_matrix=corr_matrix,
            recommended_models=recommended_models,
            preprocessing_needs=preprocessing_needs,
            created_at=datetime.now().isoformat(),
            dataset_name=self.dataset_name
        )

        return self.profile

    def _count_outliers(self, feature: np.ndarray) -> int:
        """Подсчет выбросов методом IQR"""
        Q1 = np.percentile(feature, 25)
        Q3 = np.percentile(feature, 75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return int(np.sum((feature < lower_bound) | (feature > upper_bound)))

    def _assess_complexity(self, X: np.ndarray, y: Optional[np.ndarray],
                           n_samples: int, n_features: int) -> str:
        """Оценка сложности датасета"""
        ratio = n_samples / n_features if n_features > 0 else 0

        if ratio < 50:
            return "simple"
        elif ratio < 200:
            return "medium"
        else:
            return "complex"

    def _recommend_models(self, X: np.ndarray, y: Optional[np.ndarray],
                          complexity: str, balance_ratio: Optional[float]) -> List[str]:
        """Рекомендации моделей на основе профиля"""
        recommendations = []

        if complexity == "simple":
            recommendations.extend(["RandomForest", "SVM", "LogisticRegression"])
        elif complexity == "medium":
            recommendations.extend(["GradientBoosting", "RandomForest", "NeuralNetwork"])
        else:
            recommendations.extend(["NeuralNetwork", "GradientBoosting", "XGBoost"])

        if balance_ratio is not None and balance_ratio < 0.5:
            recommendations.append("BalancedRandomForest")
            recommendations.append("SMOTE_Enhanced")

        return list(set(recommendations))  # Удаление дубликатов

    def _detect_preprocessing_needs(self, X: np.ndarray,
                                    feature_profiles: List[FeatureProfile]) -> List[str]:
        """Определение необходимых шагов предобработки"""
        needs = []

        # Проверка на масштабирование
        stds = [fp.std for fp in feature_profiles]
        if max(stds) / min(stds) > 10:
            needs.append("feature_scaling")

        # Проверка на нормализацию
        skewness_values = [abs(fp.skewness) for fp in feature_profiles]
        if any(s > 1 for s in skewness_values):
            needs.append("skewness_correction")

        # Проверка на выбросы
        total_outliers = sum(fp.outliers_count for fp in feature_profiles)
        if total_outliers > len(X) * 0.05:
            needs.append("outlier_treatment")

        # Проверка на пропуски
        if any(fp.missing_values > 0 for fp in feature_profiles):
            needs.append("missing_value_imputation")

        return needs if needs else ["none_required"]

    def visualize_profile(self, save_path: Optional[str] = None):
        """Визуализация профиля данных"""
        if self.profile is None:
            raise ValueError("Сначала создайте профиль через profile_tabular_data()")

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'Профиль датасета: {self.dataset_name}', fontsize=16, fontweight='bold')

        # 1. Распределение признаков
        axes[0, 0].hist([fp.mean for fp in self.profile.feature_profiles],
                        bins=20, color='steelblue', alpha=0.7)
        axes[0, 0].set_title('Распределение средних значений признаков')
        axes[0, 0].set_xlabel('Среднее')
        axes[0, 0].set_ylabel('Частота')

        # 2. Корреляционная матрица (если есть)
        if self.profile.feature_correlation_matrix:
            corr_matrix = np.array(self.profile.feature_correlation_matrix)
            im = axes[0, 1].imshow(corr_matrix, cmap='coolwarm', aspect='auto',
                                   vmin=-1, vmax=1)
            axes[0, 1].set_title('Корреляционная матрица')
            plt.colorbar(im, ax=axes[0, 1])
        else:
            axes[0, 1].text(0.5, 0.5, 'Слишком много признаков\nдля визуализации',
                            ha='center', va='center', transform=axes[0, 1].transAxes)
            axes[0, 1].set_title('Корреляционная матрица')

        # 3. Распределение классов
        if self.profile.class_distribution:
            classes = list(self.profile.class_distribution.keys())
            counts = list(self.profile.class_distribution.values())
            axes[1, 0].bar(classes, counts, color='coral', alpha=0.7)
            axes[1, 0].set_title('Распределение классов')
            axes[1, 0].set_xlabel('Класс')
            axes[1, 0].set_ylabel('Количество')
            plt.setp(axes[1, 0].xaxis.get_majorticklabels(), rotation=45)
        else:
            axes[1, 0].text(0.5, 0.5, 'Нет целевой переменной',
                            ha='center', va='center', transform=axes[1, 0].transAxes)
            axes[1, 0].set_title('Распределение классов')

        # 4. Выбросы по признакам
        outlier_counts = [fp.outliers_count for fp in self.profile.feature_profiles]
        feature_names = [fp.name for fp in self.profile.feature_profiles]
        axes[1, 1].barh(feature_names[:10], outlier_counts[:10], color='green', alpha=0.7)
        axes[1, 1].set_title('Количество выбросов (первые 10 признаков)')
        axes[1, 1].set_xlabel('Выбросов')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Визуализация сохранена: {save_path}")

        plt.show()

    def print_summary(self):
        """Вывод краткой сводки профиля"""
        if self.profile is None:
            raise ValueError("Сначала создайте профиль")

        print("\n" + "=" * 70)
        print(f"📊 ПРОФИЛЬ ДАТАСЕТА: {self.dataset_name.upper()}")
        print("=" * 70)

        print(f"\n📐 РАЗМЕРЫ:")
        print(f"   • Образцов: {self.profile.n_samples:,}")
        print(f"   • Признаков: {self.profile.n_features}")
        if self.profile.n_classes:
            print(f"   • Классов: {self.profile.n_classes}")
        print(f"   • Размер в памяти: {self.profile.memory_size_mb:.2f} MB")

        print(f"\n⚖️  БАЛАНС КЛАССОВ:")
        if self.profile.class_distribution:
            for cls, count in self.profile.class_distribution.items():
                pct = count / self.profile.n_samples * 100
                print(f"   • Класс {cls}: {count} ({pct:.1f}%)")
            print(f"   • Ratio баланса: {self.profile.class_balance_ratio:.3f}")
        else:
            print("   • Регрессия (нет классов)")

        print(f"\n🎯 СЛОЖНОСТЬ ДАННЫХ: {self.profile.data_complexity.upper()}")

        print(f"\n🤖 РЕКОМЕНДУЕМЫЕ МОДЕЛИ:")
        for model in self.profile.recommended_models:
            print(f"   • {model}")

        print(f"\n🔧 ТРЕБУЕМАЯ ПРЕДОБРАБОТКА:")
        for need in self.profile.preprocessing_needs:
            print(f"   • {need.replace('_', ' ').title()}")

        print("\n" + "=" * 70)


def profile_from_csv(filepath: str, target_column: Optional[str] = None,
                     dataset_name: Optional[str] = None) -> DatasetProfile:
    """
    Удобная функция для профилирования CSV файла

    Args:
        filepath: путь к CSV файлу
        target_column: имя целевой колонки (опционально)
        dataset_name: имя датасета

    Returns:
        DatasetProfile
    """
    if dataset_name is None:
        dataset_name = os.path.basename(filepath)

    # Загрузка данных
    df = pd.read_csv(filepath)

    # Разделение на X и y
    if target_column and target_column in df.columns:
        X = df.drop(columns=[target_column]).values
        y = df[target_column].values
    else:
        X = df.values
        y = None

    # Профилирование
    profiler = DataProfiler(dataset_name=dataset_name)
    profile = profiler.profile_tabular_data(X, y, feature_names=list(df.columns))

    return profile


# Пример использования
if __name__ == "__main__":
    from sklearn.datasets import load_iris

    # Загрузка Iris dataset
    iris = load_iris()
    X, y = iris.data, iris.target

    # Создание профилировщика
    profiler = DataProfiler(dataset_name="Iris_Dataset")

    # Профилирование
    profile = profiler.profile_tabular_data(
        X, y,
        feature_names=iris.feature_names
    )

    # Вывод сводки
    profiler.print_summary()

    # Визуализация
    profiler.visualize_profile(save_path="iris_profile.png")

    # Сохранение профиля
    profile.save("iris_profile.json")