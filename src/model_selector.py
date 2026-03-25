"""
Модуль адаптивного выбора и управления ML моделями
"""

import numpy as np
from typing import Dict, List, Any, Optional
import joblib
import os
import json
from datetime import datetime
from sklearn.base import BaseEstimator
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')


class SpecializedModel:
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
        import time
        start_time = time.time()
        self.model.fit(X, y, **fit_params)
        self.is_trained = True
        self.training_time = time.time() - start_time
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_trained:
            raise ValueError(f"Модель {self.name} не обучена!")
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.is_trained:
            raise ValueError(f"Модель {self.name} не обучена!")
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)
        else:
            raise AttributeError(f"Модель {self.name} не поддерживает predict_proba")

    def evaluate(self, X: np.ndarray, y: np.ndarray,
                metrics: Optional[List[str]] = None) -> Dict[str, float]:
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
        scores = cross_val_score(self.model, X, y, cv=cv, scoring='accuracy')
        return {
            'cv_mean': float(scores.mean()),
            'cv_std': float(scores.std()),
            'cv_scores': scores.tolist()
        }

    def matches_profile(self, data_profile: Dict[str, Any]) -> float:
        score = 0.0
        max_score = 0.0

        if 'data_complexity' in self.profile_requirements:
            max_score += 1.0
            if data_profile.get('data_complexity') == self.profile_requirements['data_complexity']:
                score += 1.0

        if 'class_balance_min' in self.profile_requirements:
            max_score += 1.0
            if data_profile.get('class_balance_ratio', 0) >= self.profile_requirements['class_balance_min']:
                score += 1.0

        if 'min_samples' in self.profile_requirements:
            max_score += 1.0
            if data_profile.get('n_samples', 0) >= self.profile_requirements['min_samples']:
                score += 1.0

        if 'n_features_range' in self.profile_requirements:
            max_score += 1.0
            n_feat = data_profile.get('n_features', 0)
            min_f, max_f = self.profile_requirements['n_features_range']
            if min_f <= n_feat <= max_f:
                score += 1.0

        return score / max_score if max_score > 0 else 0.5

    def get_info(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'description': self.description,
            'profile_requirements': self.profile_requirements,
            'performance_metrics': self.performance_metrics,
            'is_trained': self.is_trained,
            'training_time': self.training_time
        }

    def save(self, filepath: str):
        joblib.dump({
            'name': self.name,
            'model': self.model,
            'profile_requirements': self.profile_requirements,
            'description': self.description,
            'performance_metrics': self.performance_metrics,
            'is_trained': self.is_trained,
            'training_time': self.training_time
        }, filepath)

    @classmethod
    def load(cls, filepath: str) -> 'SpecializedModel':
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
        return model


class AdaptiveModelSelector:
    def __init__(self):
        self.models: List[SpecializedModel] = []
        self.best_model: Optional[SpecializedModel] = None
        self.data_profile = None
        self.selection_history = []

    def register_model(self, model: SpecializedModel):
        self.models.append(model)

    def create_default_models(self, model_types: Optional[List[str]] = None) -> List[SpecializedModel]:
        if model_types is None:
            model_types = ['random_forest', 'svm', 'gradient_boosting',
                          'neural_network', 'logistic_regression']

        created_models = []

        if 'random_forest' in model_types:
            rf_model = SpecializedModel(
                name="RandomForest_Specialist",
                model=RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1),
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
            gb_model = SpecializedModel(
                name="GradientBoosting_Specialist",
                model=GradientBoostingClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42),
                profile_requirements={
                    'data_complexity': 'complex',
                    'min_samples': 100
                },
                description="Gradient Boosting - эффективен на сложных нелинейных данных"
            )
            self.register_model(gb_model)
            created_models.append(gb_model)

        if 'neural_network' in model_types:
            nn_model = SpecializedModel(
                name="NeuralNetwork_Specialist",
                model=Pipeline([
                    ('scaler', StandardScaler()),
                    ('mlp', MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42))
                ]),
                profile_requirements={
                    'data_complexity': 'complex',
                    'min_samples': 200
                },
                description="Нейронная сеть - универсальная модель"
            )
            self.register_model(nn_model)
            created_models.append(nn_model)

        if 'logistic_regression' in model_types:
            lr_model = SpecializedModel(
                name="LogisticRegression_Specialist",
                model=Pipeline([
                    ('scaler', StandardScaler()),
                    ('lr', LogisticRegression(max_iter=1000, random_state=42))
                ]),
                profile_requirements={
                    'data_complexity': 'simple',
                    'n_features_range': (2, 1000)
                },
                description="Логистическая регрессия - быстрая базовая модель"
            )
            self.register_model(lr_model)
            created_models.append(lr_model)

        return created_models

    def profile_and_select(self, X: np.ndarray, y: np.ndarray,
                          data_profile: Optional[Dict] = None,
                          cv_folds: int = 5,
                          use_cv_in_scoring: bool = False) -> SpecializedModel:
        print("\n" + "="*70)
        print("АДАПТИВНЫЙ ВЫБОР МОДЕЛИ")
        print("="*70)

        if data_profile is None:
            data_profile = {
                'n_samples': X.shape[0],
                'n_features': X.shape[1],
                'data_complexity': 'medium',
                'class_balance_ratio': 1.0
            }

        self.data_profile = data_profile

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        print(f"\nДанные:")
        print(f"   - Train: {X_train.shape[0]} образцов")
        print(f"   - Test: {X_test.shape[0]} образцов")

        print("\nОценка соответствия моделей профилю данных:")
        model_scores = []

        for model in self.models:
            profile_score = model.matches_profile(data_profile)
            model_scores.append((model, profile_score))
            print(f"   - {model.name}: {profile_score:.3f}")

        model_scores.sort(key=lambda x: x[1], reverse=True)

        print("\nТестирование ВСЕХ моделей:")
        print("="*70)

        results_table = []

        for model, profile_score in model_scores:
            print(f"\nТестирование: {model.name}")
            print(f"   Соответствие профилю: {profile_score:.3f}")

            try:
                model.fit(X_train, y_train)
                metrics = model.evaluate(X_test, y_test)
                cv_results = model.cross_validate(X_train, y_train, cv=cv_folds)

                print(f"   Accuracy:  {metrics['accuracy']:.4f}")
                print(f"   F1-Score:  {metrics['f1']:.4f}")
                print(f"   Precision: {metrics['precision']:.4f}")
                print(f"   Recall:    {metrics['recall']:.4f}")
                print(f"   CV Score:  {cv_results['cv_mean']:.4f} (+/- {cv_results['cv_std']:.4f})")
                print(f"   Время:     {model.training_time:.3f}s")

                if use_cv_in_scoring:
                    final_score = (
                        metrics['accuracy'] * 0.5 +
                        metrics['f1'] * 0.3 +
                        cv_results['cv_mean'] * 0.2
                    )
                else:
                    final_score = (
                        metrics['accuracy'] * 0.7 +
                        metrics['f1'] * 0.3
                    )

                print(f"   Итоговый скор: {final_score:.4f}")

                results_table.append({
                    'model': model.name,
                    'profile_score': profile_score,
                    'accuracy': metrics['accuracy'],
                    'f1': metrics['f1'],
                    'precision': metrics['precision'],
                    'recall': metrics['recall'],
                    'cv_mean': cv_results['cv_mean'],
                    'final_score': final_score,
                    'training_time': model.training_time
                })

                self.selection_history.append({
                    'model': model.name,
                    'profile_score': profile_score,
                    'metrics': metrics,
                    'cv_results': cv_results,
                    'final_score': final_score,
                    'timestamp': datetime.now().isoformat()
                })

            except Exception as e:
                print(f"   Ошибка: {str(e)}")
                continue

        print("\n" + "="*70)
        print("СРАВНЕНИЕ МОДЕЛЕЙ (отсортировано по итоговому скору)")
        print("="*70)
        print(f"{'Модель':<30} {'Accuracy':<10} {'F1-Score':<10} {'Скор':<10} {'Время (с)':<10}")
        print("-"*70)

        for result in sorted(results_table, key=lambda x: x['final_score'], reverse=True):
            print(f"{result['model']:<30} {result['accuracy']:<10.4f} {result['f1']:<10.4f} {result['final_score']:<10.4f} {result['training_time']:<10.3f}")

        best_result = max(results_table, key=lambda x: x['final_score'])
        best_model = next(m for m in self.models if m.name == best_result['model'])

        self.best_model = best_model

        print("\n" + "="*70)
        print(f"ВЫБРАНА МОДЕЛЬ: {best_model.name}")
        print(f"   Accuracy: {best_result['accuracy']:.4f}")
        print(f"   F1-Score: {best_result['f1']:.4f}")
        print(f"   Итоговый скор: {best_result['final_score']:.4f}")
        print("="*70)

        return best_model

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.best_model is None:
            raise ValueError("Сначала выполните profile_and_select()!")
        return self.best_model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.best_model is None:
            raise ValueError("Сначала выполните profile_and_select()!")
        return self.best_model.predict_proba(X)

    def get_model_info(self) -> Dict[str, Any]:
        if self.best_model is None:
            return {}
        return self.best_model.get_info()

    def save_all_models(self, directory: str = "saved_models"):
        os.makedirs(directory, exist_ok=True)

        saved_count = 0
        for model in self.models:
            if model.is_trained:
                filepath = os.path.join(directory, f"{model.name}.pkl")
                model.save(filepath)
                saved_count += 1

        history_path = os.path.join(directory, "selection_history.json")
        with open(history_path, 'w') as f:
            json.dump(self.selection_history, f, indent=2)

    def load_models(self, directory: str = "saved_models"):
        self.models = []

        for filename in os.listdir(directory):
            if filename.endswith('.pkl'):
                filepath = os.path.join(directory, filename)
                model = SpecializedModel.load(filepath)
                self.models.append(model)

        history_path = os.path.join(directory, "selection_history.json")
        if os.path.exists(history_path):
            try:
                with open(history_path, 'r') as f:
                    history = json.load(f)

                if history:
                    best_history_entry = max(history, key=lambda x: x.get('final_score', 0))
                    best_model_name = best_history_entry['model']

                    for model in self.models:
                        if model.name == best_model_name:
                            self.best_model = model
                            break
            except Exception as e:
                pass