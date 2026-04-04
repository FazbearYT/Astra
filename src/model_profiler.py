import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from typing import Dict, List, Any, Optional
import json
from dataclasses import dataclass, asdict
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

@dataclass
class FeatureProfile:
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
    n_samples: int
    n_features: int
    n_classes: Optional[int]
    memory_size_mb: float
    feature_profiles: List[FeatureProfile]
    class_distribution: Optional[Dict[str, int]]
    class_balance_ratio: Optional[float]
    data_complexity: str
    feature_correlation_matrix: Optional[List[List[float]]]
    recommended_models: List[str]
    preprocessing_needs: List[str]
    created_at: str
    dataset_name: str

    def to_dict(self) -> Dict:
        return asdict(self)

    def save(self, filepath: str):
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, default=str)

    @classmethod
    def load(cls, filepath: str) -> 'DatasetProfile':
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return cls(**data)

class DataProfiler:
    def __init__(self, dataset_name: str = "unknown"):
        self.dataset_name = dataset_name
        self.profile = None

    def profile_tabular_data(self, X: np.ndarray, y: Optional[np.ndarray] = None,
                            feature_names: Optional[List[str]] = None) -> DatasetProfile:
        if X.dtype == object or X.dtype.kind == 'U':
            X = X.astype(np.float64)
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(X.shape[1])]

        n_samples, n_features = X.shape
        memory_size_mb = X.nbytes / (1024 ** 2)

        X_work, y_work = X, y
        if n_samples > 10000 and y is not None:
            idx = np.random.choice(n_samples, 10000, replace=False)
            X_work, y_work = X[idx], y[idx]

        feature_profiles = []
        for i in range(n_features):
            feature = X_work[:, i]
            std_val = float(np.std(feature)) if np.std(feature) != 0 else 1e-10
            mean_val = float(np.mean(feature))
            try:
                skewness_val = float(stats.skew(feature))
            except Exception:
                skewness_val = 0.0
            try:
                kurtosis_val = float(stats.kurtosis(feature))
            except Exception:
                kurtosis_val = 0.0

            profile = FeatureProfile(
                name=feature_names[i], mean=mean_val, std=std_val,
                min=float(np.min(feature)), max=float(np.max(feature)),
                median=float(np.median(feature)), skewness=skewness_val,
                kurtosis=kurtosis_val, missing_values=int(np.sum(np.isnan(feature))),
                outliers_count=self._count_outliers(feature), correlation_with_target=None
            )

            if y_work is not None:
                try:
                    corr, _ = stats.pearsonr(feature, y_work)
                    profile.correlation_with_target = float(corr)
                except Exception:
                    profile.correlation_with_target = None

            feature_profiles.append(profile)

        class_distribution = None
        class_balance_ratio = None
        n_classes = None
        if y is not None:
            unique, counts = np.unique(y, return_counts=True)
            class_distribution = dict(zip([str(u) for u in unique], [int(c) for c in counts]))
            n_classes = len(unique)
            class_balance_ratio = float(min(counts) / max(counts)) if max(counts) > 0 else 0

        corr_matrix = None
        if n_features <= 20:
            try:
                corr_matrix = np.corrcoef(X_work.T)
                corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)
                corr_matrix = corr_matrix.tolist()
            except Exception:
                corr_matrix = None

        complexity = self._assess_complexity(n_samples, n_features)
        recommended_models = self._recommend_models(complexity, class_balance_ratio)
        preprocessing_needs = self._detect_preprocessing_needs(X, feature_profiles)

        self.profile = DatasetProfile(
            n_samples=n_samples, n_features=n_features, n_classes=n_classes,
            memory_size_mb=memory_size_mb, feature_profiles=feature_profiles,
            class_distribution=class_distribution, class_balance_ratio=class_balance_ratio,
            data_complexity=complexity, feature_correlation_matrix=corr_matrix,
            recommended_models=recommended_models, preprocessing_needs=preprocessing_needs,
            created_at=datetime.now().isoformat(), dataset_name=self.dataset_name
        )
        return self.profile

    def _count_outliers(self, feature: np.ndarray) -> int:
        Q1 = np.percentile(feature, 25)
        Q3 = np.percentile(feature, 75)
        IQR = Q3 - Q1
        if IQR == 0:
            return 0
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return int(np.sum((feature < lower_bound) | (feature > upper_bound)))

    def _assess_complexity(self, n_samples: int, n_features: int) -> str:
        ratio = n_samples / n_features if n_features > 0 else 0
        if ratio < 50:
            return "simple"
        elif ratio < 200:
            return "medium"
        else:
            return "complex"

    def _recommend_models(self, complexity: str, balance_ratio: Optional[float]) -> List[str]:
        recommendations = []
        if complexity == "simple":
            recommendations.extend(["RandomForest", "SVM", "LogisticRegression"])
        elif complexity == "medium":
            recommendations.extend(["GradientBoosting", "RandomForest", "NeuralNetwork"])
        else:
            recommendations.extend(["NeuralNetwork", "GradientBoosting"])
        if balance_ratio is not None and balance_ratio < 0.5:
            recommendations.append("BalancedRandomForest")
        return list(set(recommendations))

    def _detect_preprocessing_needs(self, X: np.ndarray,
                                   feature_profiles: List[FeatureProfile]) -> List[str]:
        needs = []
        stds = [fp.std for fp in feature_profiles if fp.std > 1e-9]
        if len(stds) > 0:
            if max(stds) / min(stds) > 10:
                needs.append("feature_scaling")
        else:
            needs.append("feature_scaling")
        skewness_values = [abs(fp.skewness) for fp in feature_profiles]
        if any(s > 1 for s in skewness_values):
            needs.append("skewness_correction")
        total_outliers = sum(fp.outliers_count for fp in feature_profiles)
        if total_outliers > len(X) * 0.05:
            needs.append("outlier_treatment")
        if any(fp.missing_values > 0 for fp in feature_profiles):
            needs.append("missing_value_imputation")
        return needs if needs else ["none_required"]

    def visualize_profile(self, save_path: Optional[str] = None):
        if self.profile is None:
            raise ValueError("Сначала создайте профиль")
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'Профиль датасета: {self.dataset_name}', fontsize=16, fontweight='bold')
        axes[0, 0].hist([fp.mean for fp in self.profile.feature_profiles],
                       bins=20, color='steelblue', alpha=0.7)
        axes[0, 0].set_title('Распределение средних значений признаков')
        axes[0, 0].set_xlabel('Среднее')
        axes[0, 0].set_ylabel('Частота')
        if self.profile.feature_correlation_matrix:
            corr_matrix = np.array(self.profile.feature_correlation_matrix)
            im = axes[0, 1].imshow(corr_matrix, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
            axes[0, 1].set_title('Корреляционная матрица')
            plt.colorbar(im, ax=axes[0, 1])
        else:
            axes[0, 1].text(0.5, 0.5, 'Слишком много признаков', ha='center', va='center')
            axes[0, 1].set_title('Корреляционная матрица')
        if self.profile.class_distribution:
            classes = list(self.profile.class_distribution.keys())
            counts = list(self.profile.class_distribution.values())
            axes[1, 0].bar(classes, counts, color='coral', alpha=0.7)
            axes[1, 0].set_title('Распределение классов')
            axes[1, 0].set_xlabel('Класс')
            axes[1, 0].set_ylabel('Количество')
        else:
            axes[1, 0].text(0.5, 0.5, 'Нет целевой переменной', ha='center', va='center')
            axes[1, 0].set_title('Распределение классов')
        outlier_counts = [fp.outliers_count for fp in self.profile.feature_profiles]
        feature_names = [fp.name for fp in self.profile.feature_profiles]
        axes[1, 1].barh(feature_names[:10], outlier_counts[:10], color='green', alpha=0.7)
        axes[1, 1].set_title('Количество выбросов (первые 10 признаков)')
        axes[1, 1].set_xlabel('Выбросов')
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close('all')

    def print_summary(self):
        if self.profile is None:
            raise ValueError("Сначала создайте профиль")
        print("\nПрофиль датасета: ")
        print(f"  Образцов: {self.profile.n_samples:,} ")
        print(f"  Признаков: {self.profile.n_features} ")
        if self.profile.n_classes:
            print(f"  Классов: {self.profile.n_classes} ")
        print(f"  Размер: {self.profile.memory_size_mb:.2f} MB ")
        if self.profile.class_distribution:
            print("\nРаспределение классов: ")
            for cls, count in self.profile.class_distribution.items():
                pct = count / self.profile.n_samples * 100
                print(f"  Класс {cls}: {count} ({pct:.1f}%) ")
        print(f"\nСложность: {self.profile.data_complexity.upper()} ")
        print("\nРекомендуемые модели: ")
        for model in self.profile.recommended_models:
            print(f"  - {model} ")
        print("\nПредобработка: ")
        for need in self.profile.preprocessing_needs:
            print(f"  - {need.replace('_', ' ').title()} ")