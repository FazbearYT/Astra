"""
Подсистема адаптивного выбора моделей для табличных данных
"""
import numpy as np
import pandas as pd
import joblib
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
import json
import os
from typing import Dict, List, Tuple, Any
import warnings

warnings.filterwarnings('ignore')


class DataProfiler:
    """Профилировщик данных для определения характеристик датасета"""

    @staticmethod
    def profile(X: np.ndarray, y: np.ndarray = None) -> Dict[str, Any]:
        """
        Создает профиль данных

        Returns:
            dict с характеристиками данных
        """
        profile = {
            'n_samples': X.shape[0],
            'n_features': X.shape[1],
            'feature_stats': [],
            'class_distribution': None,
            'data_complexity': None
        }

        # Статистики по признакам
        for i in range(X.shape[1]):
            feature = X[:, i]
            profile['feature_stats'].append({
                'mean': float(np.mean(feature)),
                'std': float(np.std(feature)),
                'min': float(np.min(feature)),
                'max': float(np.max(feature)),
                'skewness': float(np.mean(((feature - np.mean(feature)) / np.std(feature)) ** 3))
            })

        # Распределение классов
        if y is not None:
            unique, counts = np.unique(y, return_counts=True)
            profile['class_distribution'] = dict(zip(unique.astype(str), counts.astype(int)))

            # Баланс классов
            profile['class_balance'] = float(min(counts) / max(counts))

        # Сложность данных (на основе соотношения samples/features)
        profile['data_complexity'] = 'simple' if X.shape[0] / X.shape[1] < 50 else 'complex'

        return profile


class SpecializedModel:
    """Специализированная модель с профилем эффективности"""

    def __init__(self, name: str, model, profile_requirements: Dict[str, Any]):
        self.name = name
        self.model = model
        self.profile_requirements = profile_requirements
        self.performance_metrics = {}

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def evaluate(self, X, y) -> float:
        """Оценка качества модели"""
        accuracy = accuracy_score(y, self.predict(X))
        self.performance_metrics['accuracy'] = accuracy
        return accuracy

    def get_specialization(self) -> str:
        """Возвращает описание специализации модели"""
        return self.profile_requirements.get('description', 'General purpose model')


class AdaptiveModelSelector:
    """Адаптивный селектор моделей на основе профилирования данных"""

    def __init__(self):
        self.models: List[SpecializedModel] = []
        self.data_profile = None
        self.best_model = None

    def register_model(self, model: SpecializedModel):
        """Регистрация специализированной модели"""
        self.models.append(model)
        print(f"✓ Зарегистрирована модель: {model.name}")

    def create_specialized_models(self):
        """Создание набора специализированных моделей"""

        # Модель 1: Random Forest - хороша для небольших датасетов
        rf_model = SpecializedModel(
            name="RandomForest_Specialist",
            model=RandomForestClassifier(n_estimators=100, random_state=42),
            profile_requirements={
                'description': 'Лучше работает на небольших сбалансированных датасетах',
                'min_samples': 50,
                'max_samples': 1000,
                'class_balance_min': 0.7
            }
        )
        self.register_model(rf_model)

        # Модель 2: SVM - хороша для данных с четкими границами
        svm_model = SpecializedModel(
            name="SVM_Specialist",
            model=Pipeline([
                ('scaler', StandardScaler()),
                ('svc', SVC(kernel='rbf', probability=True, random_state=42))
            ]),
            profile_requirements={
                'description': 'Эффективна на данных с четкими линейными/нелинейными границами',
                'feature_separability': 'high'
            }
        )
        self.register_model(svm_model)

        # Модель 3: Gradient Boosting - для сложных паттернов
        gb_model = SpecializedModel(
            name="GradientBoosting_Specialist",
            model=GradientBoostingClassifier(n_estimators=100, random_state=42),
            profile_requirements={
                'description': 'Лучше на сложных нелинейных данных',
                'data_complexity': 'complex'
            }
        )
        self.register_model(gb_model)

        # Модель 4: Neural Network - универсальная
        nn_model = SpecializedModel(
            name="NeuralNetwork_Specialist",
            model=Pipeline([
                ('scaler', StandardScaler()),
                ('mlp', MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42))
            ]),
            profile_requirements={
                'description': 'Универсальная модель для различных типов данных',
                'versatile': True
            }
        )
        self.register_model(nn_model)

    def profile_data(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """Профилирование входных данных"""
        profiler = DataProfiler()
        self.data_profile = profiler.profile(X, y)

        print("\n" + "=" * 60)
        print("📊 ПРОФИЛЬ ДАННЫХ:")
        print("=" * 60)
        print(f"• Образцов: {self.data_profile['n_samples']}")
        print(f"• Признаков: {self.data_profile['n_features']}")
        print(f"• Распределение классов: {self.data_profile['class_distribution']}")
        print(f"• Сложность: {self.data_profile['data_complexity']}")
        print("=" * 60)

        return self.data_profile

    def select_best_model(self, X_train, y_train, X_test, y_test) -> SpecializedModel:
        """
        Выбор лучшей модели на основе профиля данных и тестирования
        """
        if self.data_profile is None:
            raise ValueError("Сначала выполните profile_data()")

        print("\n🔄 ОБУЧЕНИЕ И ТЕСТИРОВАНИЕ МОДЕЛЕЙ...")
        print("=" * 60)

        best_accuracy = 0

        for model in self.models:
            # Обучение модели
            model.fit(X_train, y_train)

            # Оценка
            accuracy = model.evaluate(X_test, y_test)

            # Кросс-валидация
            cv_scores = cross_val_score(model.model, X_train, y_train, cv=3)

            print(f"\n🤖 Модель: {model.name}")
            print(f"   Специализация: {model.get_specialization()}")
            print(f"   Accuracy: {accuracy:.4f}")
            print(f"   CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                self.best_model = model

        print("\n" + "=" * 60)
        print(f"🏆 ЛУЧШАЯ МОДЕЛЬ: {self.best_model.name}")
        print(f"   Точность: {best_accuracy:.4f}")
        print("=" * 60)

        return self.best_model

    def predict(self, X) -> np.ndarray:
        """Предсказание с помощью выбранной модели"""
        if self.best_model is None:
            raise ValueError("Сначала выберите модель через select_best_model()")

        return self.best_model.predict(X)

    def save_models(self, path: str = "models/"):
        """Сохранение всех моделей"""
        os.makedirs(path, exist_ok=True)

        for model in self.models:
            model_path = os.path.join(path, f"{model.name}.pkl")
            joblib.dump({
                'model': model.model,
                'profile': model.profile_requirements,
                'metrics': model.performance_metrics
            }, model_path)
            print(f"✓ Сохранена модель: {model_path}")

        # Сохранение профиля данных
        if self.data_profile:
            with open(os.path.join(path, "data_profile.json"), 'w') as f:
                json.dump(self.data_profile, f, indent=2)

    def load_models(self, path: str = "models/"):
        """Загрузка сохраненных моделей"""
        self.models = []

        for filename in os.listdir(path):
            if filename.endswith('.pkl'):
                model_data = joblib.load(os.path.join(path, filename))
                model = SpecializedModel(
                    name=filename.replace('.pkl', ''),
                    model=model_data['model'],
                    profile_requirements=model_data['profile']
                )
                model.performance_metrics = model_data['metrics']
                self.models.append(model)
                print(f"✓ Загружена модель: {model.name}")


def main():
    """Основной пайплайн"""
    print("\n" + "=" * 60)
    print("🚀 АДАПТИВНАЯ ПОДСИСТЕМА ВЫБОРА МОДЕЛЕЙ")
    print("=" * 60)

    # Загрузка данных Iris
    print("\n📥 Загрузка данных...")
    iris = load_iris()
    X, y = iris.data, iris.target

    # Разделение на train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # Создание селектора
    selector = AdaptiveModelSelector()

    # Создание специализированных моделей
    selector.create_specialized_models()

    # Профилирование данных
    selector.profile_data(X_train, y_train)

    # Выбор лучшей модели
    best_model = selector.select_best_model(X_train, y_train, X_test, y_test)

    # Финальное предсказание
    print("\n📈 ФИНАЛЬНОЕ ТЕСТИРОВАНИЕ:")
    predictions = selector.predict(X_test)
    print(f"\nТочность на тесте: {accuracy_score(y_test, predictions):.4f}")
    print("\nОтчет о классификации:")
    print(classification_report(y_test, predictions, target_names=iris.target_names))

    # Сохранение моделей
    selector.save_models()

    print("\n✅ Пайплайн завершен успешно!")

    return selector


if __name__ == "__main__":
    selector = main()