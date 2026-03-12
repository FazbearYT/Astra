"""
Модуль адаптивного выбора и управления ML моделями
Автоматически выбирает лучшую модель на основе профиля данных
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Callable
import joblib
import os
import json
from datetime import datetime
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import warnings

warnings.filterwarnings('ignore')


class SpecializedModel:
    """
    Обертка для специализированной ML модели

    Хранит:
    - Саму модель
    - Профиль данных, для которых модель эффективна
    - Метрики производительности
    """

    def __init__(self, name: str, model: BaseEstimator,
                 profile_requirements: Dict[str, Any],
                 description: str = ""):
        self.name = name
        self.model = model
        self.profile_requirements = profile_requirements
        self.description = description or f"Specialized model: {name}"
        self.performance_metrics = {}
        self.is_trained = False
        self.training_time = None

    def fit(self, X: np.ndarray, y: np.ndarray, **fit_params) -> 'SpecializedModel':
        """Обучение модели"""
        import time
        start_time = time.time()

        self.model.fit(X, y, **fit_params)
        self.is_trained = True
        self.training_time = time.time() - start_time

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Предсказание"""
        if not self.is_trained:
            raise ValueError(f"Модель {self.name} не обучена!")
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Вероятности классов"""
        if not self.is_trained:
            raise ValueError(f"Модель {self.name} не обучена!")

        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)
        else:
            raise AttributeError(f"Модель {self.name} не поддерживает predict_proba")

    def evaluate(self, X: np.ndarray, y: np.ndarray,
                 metrics: Optional[List[str]] = None) -> Dict[str, float]:
        """
        Оценка качества модели

        Args:
            X: данные
            y: истинные метки
            metrics: список метрик ('accuracy', 'precision', 'recall', 'f1')

        Returns:
            словарь с метриками
        """
        if metrics is None:
            metrics = ['accuracy', 'precision', 'recall', 'f1']

        y_pred = self.predict(X)

        results = {}

        if 'accuracy' in metrics:
            results['accuracy'] = accuracy_score(y, y_pred)

        if 'precision' in metrics:
            results['precision'] = precision_score(y, y_pred, average='weighted', zero_division=0)

        if 'recall' in metrics:
            results['recall'] = recall_score(y, y_pred, average='weighted', zero_division=0)

        if 'f1' in metrics:
            results['f1'] = f1_score(y, y_pred, average='weighted', zero_division=0)

        self.performance_metrics.update(results)

        return results

    def cross_validate(self, X: np.ndarray, y: np.ndarray, cv: int = 5) -> Dict[str, float]:
        """Кросс-валидация модели"""
        scores = cross_val_score(self.model, X, y, cv=cv, scoring='accuracy')

        results = {
            'cv_mean': float(scores.mean()),
            'cv_std': float(scores.std()),
            'cv_scores': scores.tolist()
        }

        return results

    def matches_profile(self, data_profile: Dict[str, Any]) -> float:
        """
        Проверка соответствия модели профилю данных

        Returns:
            score соответствия (0-1)
        """
        score = 0.0
        max_score = 0.0

        # Проверка сложности данных
        if 'data_complexity' in self.profile_requirements:
            max_score += 1.0
            if data_profile.get('data_complexity') == self.profile_requirements['data_complexity']:
                score += 1.0

        # Проверка баланса классов
        if 'class_balance_min' in self.profile_requirements:
            max_score += 1.0
            if data_profile.get('class_balance_ratio', 0) >= self.profile_requirements['class_balance_min']:
                score += 1.0

        # Проверка размера данных
        if 'min_samples' in self.profile_requirements:
            max_score += 1.0
            if data_profile.get('n_samples', 0) >= self.profile_requirements['min_samples']:
                score += 1.0

        # Проверка количества признаков
        if 'n_features_range' in self.profile_requirements:
            max_score += 1.0
            n_feat = data_profile.get('n_features', 0)
            min_f, max_f = self.profile_requirements['n_features_range']
            if min_f <= n_feat <= max_f:
                score += 1.0

        return score / max_score if max_score > 0 else 0.5

    def get_info(self) -> Dict[str, Any]:
        """Получение информации о модели"""
        return {
            'name': self.name,
            'description': self.description,
            'profile_requirements': self.profile_requirements,
            'performance_metrics': self.performance_metrics,
            'is_trained': self.is_trained,
            'training_time': self.training_time
        }

    def save(self, filepath: str):
        """Сохранение модели"""
        joblib.dump({
            'name': self.name,
            'model': self.model,
            'profile_requirements': self.profile_requirements,
            'description': self.description,
            'performance_metrics': self.performance_metrics,
            'is_trained': self.is_trained,
            'training_time': self.training_time
        }, filepath)
        print(f"✓ Модель сохранена: {filepath}")

    @classmethod
    def load(cls, filepath: str) -> 'SpecializedModel':
        """Загрузка модели"""
        data = joblib.load(filepath)
        model = cls(
            name=data['name'],
            model=data['model'],
            profile_requirements=data['profile_requirements'],
            description=data['description']
        )
        model.performance_metrics = data['performance_metrics']
        model.is_trained = data['is_trained']
        model.training_time = data['training_time']
        print(f"✓ Модель загружена: {filepath}")
        return model


class AdaptiveModelSelector:
    """
    Адаптивный селектор моделей

    Автоматически выбирает лучшую модель на основе:
    - Профиля данных
    - Кросс-валидации
    - Производительности
    """

    def __init__(self):
        self.models: List[SpecializedModel] = []
        self.best_model: Optional[SpecializedModel] = None
        self.data_profile = None
        self.selection_history = []

    def register_model(self, model: SpecializedModel):
        """Регистрация модели в селекторе"""
        self.models.append(model)
        print(f"✓ Зарегистрирована модель: {model.name}")

    def create_default_models(self, model_types: Optional[List[str]] = None) -> List[SpecializedModel]:
        """
        Создание набора стандартных специализированных моделей

        Args:
            model_types: список типов моделей для создания

        Returns:
            список созданных моделей
        """
        if model_types is None:
            model_types = ['random_forest', 'svm', 'gradient_boosting',
                           'neural_network', 'logistic_regression', 'knn']

        created_models = []

        if 'random_forest' in model_types:
            # Random Forest - для небольших сбалансированных датасетов
            rf_model = SpecializedModel(
                name="RandomForest_Specialist",
                model=RandomForestClassifier(
                    n_estimators=100,
                    max_depth=10,
                    random_state=42,
                    n_jobs=-1
                ),
                profile_requirements={
                    'data_complexity': 'simple',
                    'class_balance_min': 0.7,
                    'min_samples': 50,
                    'n_features_range': (4, 100)
                },
                description="Random Forest - эффективен на небольших сбалансированных датасетах"
            )
            self.register_model(rf_model)
            created_models.append(rf_model)

        if 'svm' in model_types:
            # SVM - для данных с четкими границами
            svm_model = SpecializedModel(
                name="SVM_Specialist",
                model=Pipeline([
                    ('scaler', StandardScaler()),
                    ('svc', SVC(kernel='rbf', probability=True, random_state=42))
                ]),
                profile_requirements={
                    'data_complexity': 'medium',
                    'n_features_range': (2, 50)
                },
                description="SVM с RBF ядром - эффективен на данных с четкими границами"
            )
            self.register_model(svm_model)
            created_models.append(svm_model)

        if 'gradient_boosting' in model_types:
            # Gradient Boosting - для сложных паттернов
            gb_model = SpecializedModel(
                name="GradientBoosting_Specialist",
                model=GradientBoostingClassifier(
                    n_estimators=100,
                    max_depth=5,
                    learning_rate=0.1,
                    random_state=42
                ),
                profile_requirements={
                    'data_complexity': 'complex',
                    'min_samples': 100
                },
                description="Gradient Boosting - эффективен на сложных нелинейных данных"
            )
            self.register_model(gb_model)
            created_models.append(gb_model)

        if 'neural_network' in model_types:
            # Neural Network - универсальная модель
            nn_model = SpecializedModel(
                name="NeuralNetwork_Specialist",
                model=Pipeline([
                    ('scaler', StandardScaler()),
                    ('mlp', MLPClassifier(
                        hidden_layer_sizes=(100, 50),
                        max_iter=1000,
                        random_state=42
                    ))
                ]),
                profile_requirements={
                    'data_complexity': 'complex',
                    'min_samples': 200
                },
                description="Нейронная сеть - универсальная модель для различных типов данных"
            )
            self.register_model(nn_model)
            created_models.append(nn_model)

        if 'logistic_regression' in model_types:
            # Logistic Regression - базовая модель
            lr_model = SpecializedModel(
                name="LogisticRegression_Specialist",
                model=Pipeline([
                    ('scaler', StandardScaler()),
                    ('lr', LogisticRegression(
                        max_iter=1000,
                        random_state=42,
                        # multi_class удален в sklearn 1.5+, теперь автоматически
                    ))
                ]),
                profile_requirements={
                    'data_complexity': 'simple',
                    'n_features_range': (2, 1000)
                },
                description="Логистическая регрессия - быстрая базовая модель"
            )
            self.register_model(lr_model)
            created_models.append(lr_model)

        if 'knn' in model_types:
            # K-Nearest Neighbors
            knn_model = SpecializedModel(
                name="KNN_Specialist",
                model=Pipeline([
                    ('scaler', StandardScaler()),
                    ('knn', KNeighborsClassifier(n_neighbors=5))
                ]),
                profile_requirements={
                    'data_complexity': 'simple',
                    'min_samples': 100,
                    'n_features_range': (2, 50)
                },
                description="K-Nearest Neighbors - простая интерпретируемая модель"
            )
            self.register_model(knn_model)
            created_models.append(knn_model)

        return created_models

    def profile_and_select(self, X: np.ndarray, y: np.ndarray,
                           data_profile: Optional[Dict] = None,
                           cv_folds: int = 5) -> SpecializedModel:
        """
        Профилирование данных и выбор лучшей модели

        Args:
            X: данные
            y: метки
            data_profile: готовый профиль (опционально)
            cv_folds: число фолдов для кросс-валидации

        Returns:
            лучшая модель
        """
        print("\n" + "=" * 70)
        print("🔄 АДАПТИВНЫЙ ВЫБОР МОДЕЛИ")
        print("=" * 70)

        # Если профиль не предоставлен, создаем базовый
        if data_profile is None:
            data_profile = {
                'n_samples': X.shape[0],
                'n_features': X.shape[1],
                'data_complexity': 'medium',
                'class_balance_ratio': 1.0
            }

        self.data_profile = data_profile

        # Разделение на train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        print(f"\n📊 Данные:")
        print(f"   • Train: {X_train.shape[0]} образцов")
        print(f"   • Test: {X_test.shape[0]} образцов")

        # Оценка соответствия моделей профилю
        print("\n📋 Оценка соответствия моделей профилю данных:")
        model_scores = []

        for model in self.models:
            profile_score = model.matches_profile(data_profile)
            model_scores.append((model, profile_score))
            print(f"   • {model.name}: {profile_score:.3f}")

        # Сортировка по соответствию профилю
        model_scores.sort(key=lambda x: x[1], reverse=True)

        # Тестирование топ-3 моделей
        print("\n🤖 Тестирование топ-3 моделей:")
        print("=" * 70)

        best_accuracy = 0.0
        best_model = None

        for model, profile_score in model_scores[:3]:
            print(f"\n🔬 Тестирование: {model.name}")
            print(f"   Соответствие профилю: {profile_score:.3f}")

            try:
                # Обучение
                model.fit(X_train, y_train)

                # Оценка
                metrics = model.evaluate(X_test, y_test)

                # Кросс-валидация
                cv_results = model.cross_validate(X_train, y_train, cv=cv_folds)

                print(f"   Accuracy: {metrics['accuracy']:.4f}")
                print(f"   F1-Score: {metrics['f1']:.4f}")
                print(f"   CV Score: {cv_results['cv_mean']:.4f} (+/- {cv_results['cv_std']:.4f})")
                print(f"   Время обучения: {model.training_time:.3f}s")

                # Итоговый скор (комбинация метрик)
                final_score = (
                        metrics['accuracy'] * 0.4 +
                        metrics['f1'] * 0.3 +
                        cv_results['cv_mean'] * 0.3
                )

                print(f"   Итоговый скор: {final_score:.4f}")

                if final_score > best_accuracy:
                    best_accuracy = final_score
                    best_model = model

                # Сохранение истории
                self.selection_history.append({
                    'model': model.name,
                    'profile_score': profile_score,
                    'metrics': metrics,
                    'cv_results': cv_results,
                    'final_score': final_score,
                    'timestamp': datetime.now().isoformat()
                })

            except Exception as e:
                print(f"   ⚠️ Ошибка: {str(e)}")
                continue

        if best_model is None:
            raise ValueError("Ни одна модель не прошла тестирование!")

        self.best_model = best_model

        print("\n" + "=" * 70)
        print(f"🏆 ВЫБРАНА МОДЕЛЬ: {best_model.name}")
        print(f"   Итоговый скор: {best_accuracy:.4f}")
        print("=" * 70)

        return best_model

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Предсказание с помощью выбранной модели"""
        if self.best_model is None:
            raise ValueError("Сначала выполните profile_and_select()!")
        return self.best_model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Вероятности классов"""
        if self.best_model is None:
            raise ValueError("Сначала выполните profile_and_select()!")
        return self.best_model.predict_proba(X)

    def get_model_info(self) -> Dict[str, Any]:
        """Получение информации о выбранной модели"""
        if self.best_model is None:
            return {}
        return self.best_model.get_info()

    def save_all_models(self, directory: str = "saved_models"):
        """Сохранение всех моделей"""
        os.makedirs(directory, exist_ok=True)

        for model in self.models:
            filepath = os.path.join(directory, f"{model.name}.pkl")
            model.save(filepath)

        # Сохранение истории выбора
        history_path = os.path.join(directory, "selection_history.json")
        with open(history_path, 'w') as f:
            json.dump(self.selection_history, f, indent=2)

        print(f"\n✓ Все модели сохранены в: {directory}")

    def load_models(self, directory: str = "saved_models"):
        """Загрузка моделей"""
        self.models = []

        for filename in os.listdir(directory):
            if filename.endswith('.pkl'):
                filepath = os.path.join(directory, filename)
                model = SpecializedModel.load(filepath)
                self.models.append(model)

        print(f"\n✓ Загружено моделей: {len(self.models)}")


# Пример использования
if __name__ == "__main__":
    from sklearn.datasets import load_iris

    print("\n" + "=" * 70)
    print("🚀 АДАПТИВНЫЙ СЕЛЕКТОР МОДЕЛЕЙ")
    print("=" * 70)

    # Загрузка данных
    iris = load_iris()
    X, y = iris.data, iris.target

    # Создание селектора
    selector = AdaptiveModelSelector()

    # Создание моделей
    selector.create_default_models()

    # Профилирование и выбор
    best_model = selector.profile_and_select(
        X, y,
        data_profile={
            'n_samples': X.shape[0],
            'n_features': X.shape[1],
            'data_complexity': 'simple',
            'class_balance_ratio': 1.0
        },
        cv_folds=5
    )

    # Финальное тестирование
    print("\n📈 ФИНАЛЬНОЕ ТЕСТИРОВАНИЕ:")
    X_test = X[-20:]
    y_test = y[-20:]

    predictions = selector.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    print(f"\nТочность на тесте: {accuracy:.4f}")
    print(f"Правильных: {np.sum(predictions == y_test)} из {len(y_test)}")

    # Сохранение
    selector.save_all_models()

    print("\n✅ Готово!")