import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import joblib
import os
import json
import time
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
os.environ['OMP_NUM_THREADS'] = str(os.cpu_count() or 4)
os.environ['OPENBLAS_NUM_THREADS'] = str(os.cpu_count() or 4)
os.environ['MKL_NUM_THREADS'] = str(os.cpu_count() or 4)
os.environ['NUMEXPR_NUM_THREADS'] = str(os.cpu_count() or 4)
os.environ['MKL_DYNAMIC'] = 'FALSE'

class SpecializedModel:
    def __init__(self, name: str, model: BaseEstimator,
                 profile_requirements: Dict[str, Any],
                 description: str = "",
                 custom_scoring_weights: Optional[Dict[str, float]] = None):
        self.name = name
        self.model = model
        self.profile_requirements = profile_requirements
        self.description = description or f"Specialized model: {name}"
        self.performance_metrics = {}
        self.is_trained = False
        self.training_time = None
        self.custom_scoring_weights = custom_scoring_weights or {}

    def fit(self, X: np.ndarray, y: np.ndarray, **fit_params) -> 'SpecializedModel':
        start_time = time.time()
        if isinstance(self.model, Pipeline):
            estimator_name = self.model.steps[-1][0]
            step_fit_params = {f"{estimator_name}__{k}": v for k, v in fit_params.items()}
            self.model.fit(X, y, **step_fit_params)
        else:
            self.model.fit(X, y, **fit_params)
        self.is_trained = True
        self.training_time = time.time() - start_time
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_trained:
            raise ValueError(f"Model {self.name} not trained!")
        try:
            return self.model.predict(X)
        except Exception as e:
            raise RuntimeError(f"Prediction error for {self.name}: {e}")

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.is_trained:
            raise ValueError(f"Model {self.name} not trained!")
        try:
            if hasattr(self.model, 'predict_proba'):
                return self.model.predict_proba(X)
            else:
                raise AttributeError(f"Model {self.name} does not support predict_proba")
        except Exception as e:
            raise RuntimeError(f"Probability prediction error for {self.name}: {e}")

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
        scores = cross_val_score(self.model, X, y, cv=cv, scoring='accuracy', n_jobs=-1)
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
            if data_profile.get('class_balance_ratio') is not None and \
               data_profile['class_balance_ratio'] >= self.profile_requirements['class_balance_min']:
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
            'training_time': self.training_time,
            'custom_scoring_weights': self.custom_scoring_weights
        }

    def save(self, filepath: str):
        joblib.dump({
             'name': self.name,
            'model': self.model,
            'profile_requirements': self.profile_requirements,
            'description': self.description,
            'performance_metrics': self.performance_metrics,
            'is_trained': self.is_trained,
            'training_time': self.training_time,
            'custom_scoring_weights': self.custom_scoring_weights
        }, filepath)

    @classmethod
    def load(cls, filepath: str) -> 'SpecializedModel':
        data = joblib.load(filepath)
        model = cls(
            name=data['name'],
            model=data['model'],
            profile_requirements=data['profile_requirements'],
            description=data['description'],
            custom_scoring_weights=data.get('custom_scoring_weights')
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

    def _get_sklearn_model_instance(self, model_name: str, params: Dict[str, Any]) -> BaseEstimator:
        current_params = params.copy()
        if model_name == "RandomForest_Specialist":
            return RandomForestClassifier(**current_params)
        elif model_name == "SVM_Specialist":
            return Pipeline([
                ('scaler', StandardScaler()),
                ('svc', SVC(**current_params))
            ])
        elif model_name == "GradientBoosting_Specialist":
            return GradientBoostingClassifier(**current_params)
        elif model_name == "NeuralNetwork_Specialist":
            return Pipeline([
                ('scaler', StandardScaler()),
                ('mlp', MLPClassifier(**current_params))
            ])
        elif model_name == "LogisticRegression_Specialist":
            return Pipeline([
                ('scaler', StandardScaler()),
                ('lr', LogisticRegression(**current_params))
            ])
        else:
            raise ValueError(f"Unknown model name: '{model_name}'")

    def register_models_from_pipeline_config(self, pipeline_config):
        self.models = []
        for model_key, model_cfg in pipeline_config.models.items():
            if model_cfg.enabled:
                try:
                    sklearn_model_instance = self._get_sklearn_model_instance(model_cfg.name, model_cfg.params)
                    specialized_model = SpecializedModel(
                        name=model_cfg.name,
                        model=sklearn_model_instance,
                        profile_requirements=model_cfg.profile_requirements,
                        description=model_cfg.description,
                        custom_scoring_weights=model_cfg.custom_scoring_weights
                    )
                    self.register_model(specialized_model)
                except (ValueError, TypeError):
                    continue
        if not self.models:
            raise ValueError("No models were enabled or successfully registered from the pipeline config.")

    def profile_and_select(self, X: np.ndarray, y: np.ndarray,
                          data_profile: Optional[Dict] = None,
                          cv_folds: int = 5,
                          use_cv_in_scoring: bool = False,
                          global_accuracy_weight: float = 0.7,
                          global_f1_weight: float = 0.3,
                          global_cv_weight: float = 0.0) -> Tuple[SpecializedModel, np.ndarray, np.ndarray]:
        print("\nAdaptive model selection")
        print(f"Available threads: {os.cpu_count()}")

        if data_profile is None:
            data_profile = {
                'n_samples': X.shape[0],
                'n_features': X.shape[1],
                'data_complexity': 'medium',
                'class_balance_ratio': 1.0
            }

        self.data_profile = data_profile
        stratify_y = y if len(np.unique(y)) > 1 else None
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=stratify_y
        )

        print(f"Data: Train={X_train.shape[0]}, Test={X_test.shape[0]}")
        print("\nEvaluating model profiles:")
        model_scores = []
        if not self.models:
            raise ValueError("No models are registered for selection.")

        for model in self.models:
            profile_score = model.matches_profile(data_profile)
            model_scores.append((model, profile_score))
            print(f"  {model.name}: {profile_score:.3f}")

        model_scores.sort(key=lambda x: x[1], reverse=True)
        print("\nTesting models:")

        results_table = []
        total_models = len(model_scores)

        for idx, (model, profile_score) in enumerate(model_scores, 1):
            print(f"\n[{idx}/{total_models}] {model.name}")
            try:
                print("  Training...", end="   ", flush=True)
                train_start = time.time()
                model.fit(X_train, y_train)
                train_time = time.time() - train_start
                print(f"({train_time:.2f}s)")

                print("  Evaluating...", end="   ", flush=True)
                metrics = model.evaluate(X_test, y_test)
                print("done")

                print(f"  Cross-validation ({cv_folds} folds)...", end="   ", flush=True)
                cv_start = time.time()
                cv_results = model.cross_validate(X_train, y_train, cv=cv_folds)
                cv_time = time.time() - cv_start
                print(f"({cv_time:.2f}s)")

                print(f"  Accuracy: {metrics['accuracy']:.4f}")
                print(f"  F1-Score: {metrics['f1']:.4f}")
                print(f"  CV Score (Accuracy): {cv_results['cv_mean']:.4f}")

                acc_w = model.custom_scoring_weights.get('accuracy_weight', global_accuracy_weight)
                f1_w = model.custom_scoring_weights.get('f1_weight', global_f1_weight)
                cv_w = model.custom_scoring_weights.get('cv_weight', global_cv_weight)

                if use_cv_in_scoring:
                    final_score = (
                        metrics['accuracy'] * acc_w +
                        metrics['f1'] * f1_w +
                        cv_results['cv_mean'] * cv_w
                    )
                else:
                    final_score = (
                        metrics['accuracy'] * acc_w +
                        metrics['f1'] * f1_w
                    )

                print(f"  Final Score: {final_score:.4f}")

                results_table.append({
                    'model': model.name,
                    'profile_score': profile_score,
                    'accuracy': metrics['accuracy'],
                    'f1': metrics['f1'],
                    'precision': metrics['precision'],
                    'recall': metrics['recall'],
                    'cv_mean': cv_results['cv_mean'],
                    'final_score': final_score,
                    'training_time': train_time
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
                print(f"  Error processing model {model.name}: {str(e)}")
                continue

        print("\nModel Comparison:")
        print(f"{'Model':<30} {'Accuracy':<10} {'F1-Score':<10} {'Score':<10}")
        print("-" * 65)

        if not results_table:
            raise ValueError("No models were successfully trained or evaluated!")

        sorted_results = sorted(results_table, key=lambda x: x['final_score'], reverse=True)
        for result in sorted_results:
            print(f"{result['model']:<30} {result['accuracy']:<10.4f} {result['f1']:<10.4f} {result['final_score']:<10.4f}")

        best_result = sorted_results[0]
        best_model = next((m for m in self.models if m.name == best_result['model']), None)

        if best_model is None:
            raise RuntimeError(f"Best model '{best_result['model']}' found in results but not registered.")

        self.best_model = best_model
        print(f"\nSelected model: {best_model.name}")
        print(f"Accuracy: {best_result['accuracy']:.4f}")
        print(f"F1-Score: {best_result['f1']:.4f}")
        return best_model, X_test, y_test

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.best_model is None:
            raise ValueError("Run profile_and_select() first!")
        return self.best_model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.best_model is None:
            raise ValueError("Run profile_and_select() first!")
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
        print(f"Saved {saved_count} trained models.")
        history_path = os.path.join(directory, "selection_history.json")
        try:
            with open(history_path, 'w', encoding='utf-8') as f:
                json.dump(self.selection_history, f, indent=2, ensure_ascii=False)
            print(f"Saved selection history to {history_path}")
        except Exception:
            pass

    def load_models(self, directory: str = "saved_models"):
        self.models = []
        if not os.path.exists(directory):
            print(f"Directory {directory} does not exist.")
            return
        loaded_count = 0
        for filename in os.listdir(directory):
            if filename.endswith('.pkl'):
                filepath = os.path.join(directory, filename)
                try:
                    model = SpecializedModel.load(filepath)
                    self.models.append(model)
                    loaded_count += 1
                except Exception:
                    pass
        print(f"Loaded {loaded_count} models.")
        history_path = os.path.join(directory, "selection_history.json")
        if os.path.exists(history_path):
            try:
                with open(history_path, 'r', encoding='utf-8') as f:
                    history = json.load(f)
                self.selection_history = history
                if history:
                    best_history_entry = max(history, key=lambda x: x.get('final_score', 0) if x.get('final_score') is not None else -1)
                    best_model_name = best_history_entry['model']
                    for model in self.models:
                        if model.name == best_model_name:
                            self.best_model = model
                            print(f"Set best model to {best_model.name} based on history.")
                            break
            except Exception:
                pass