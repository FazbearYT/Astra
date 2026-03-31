from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
import json
import os

@dataclass
class ModelConfig:
    name: str
    enabled: bool = True
    params: Dict[str, Any] = field(default_factory=dict)
    profile_requirements: Dict[str, Any] = field(default_factory=dict)
    description: str = ""

@dataclass
class PipelineConfig:
    models: Dict[str, ModelConfig] = field(default_factory=dict)
    training: Dict[str, Any] = field(default_factory=dict)
    scoring: Dict[str, float] = field(default_factory=dict)

    def save(self, filepath: str):
        # Convert ModelConfig objects to dictionaries for JSON serialization
        serializable_models = {k: v.__dict__ for k, v in self.models.items()}
        data = {
            "models": serializable_models,
            "training": self.training,
            "scoring": self.scoring
        }
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, filepath: str) -> 'PipelineConfig':
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        models_data = data.get("models", {})
        # Convert dictionaries back to ModelConfig objects
        models = {k: ModelConfig(**v) for k, v in models_data.items()}
        return cls(
            models=models,
            training=data.get("training", {}),
            scoring=data.get("scoring", {})
        )

# --- Default Configurations ---
def get_default_config() -> PipelineConfig:
    return PipelineConfig(
        models={
            "RandomForest_Specialist": ModelConfig(
                name="RandomForest_Specialist",
                enabled=True,
                params={"n_estimators": 100, "max_depth": 10, "random_state": 42},
                profile_requirements={
                    "min_samples": 50,
                    "n_features_range": [2, 50],
                    "data_complexity": "medium"
                },
                description="General-purpose RandomForest for diverse datasets."
            ),
            "SVM_Specialist": ModelConfig(
                name="SVM_Specialist",
                enabled=True,
                params={"C": 1.0, "kernel": "rbf", "probability": True, "random_state": 42},
                profile_requirements={
                    "min_samples": 30,
                    "n_features_range": [2, 20],
                    "data_complexity": "low"
                },
                description="SVM for smaller, well-separated datasets."
            ),
            "LogisticRegression_Specialist": ModelConfig(
                name="LogisticRegression_Specialist",
                enabled=False, # Changed to False for faster default
                params={"solver": "liblinear", "penalty": "l1", "random_state": 42},
                profile_requirements={
                    "min_samples": 50,
                    "n_features_range": [2, 100],
                    "data_complexity": "medium"
                },
                description="Linear model for interpretable results."
            )
        },
        training={
            "cv_folds": 5,
            "test_size": 0.2,
            "random_state": 42,
            "use_cv_in_scoring": False # Only use test set for final score
        },
        scoring={
            "accuracy_weight": 0.7,
            "f1_weight": 0.3,
            "cv_weight": 0.0 # Only use test set for final score
        }
    )

def get_fast_config() -> PipelineConfig:
    return PipelineConfig(
        models={
            "RandomForest_Specialist": ModelConfig(
                name="RandomForest_Specialist",
                enabled=True,
                params={"n_estimators": 50, "max_depth": 5, "random_state": 42},
                profile_requirements={
                    "min_samples": 20,
                    "n_features_range": [2, 30],
                    "data_complexity": "low"
                },
                description="Fast RandomForest for quick prototyping."
            ),
            "SVM_Specialist": ModelConfig(
                name="SVM_Specialist",
                enabled=True,
                params={"C": 0.5, "kernel": "linear", "probability": True, "random_state": 42},
                profile_requirements={
                    "min_samples": 10,
                    "n_features_range": [2, 10],
                    "data_complexity": "low"
                },
                description="Fast Linear SVM."
            )
        },
        training={
            "cv_folds": 3,
            "test_size": 0.2,
            "random_state": 42,
            "use_cv_in_scoring": False
        },
        scoring={
            "accuracy_weight": 0.6,
            "f1_weight": 0.4,
            "cv_weight": 0.0
        }
    )

def get_accurate_config() -> PipelineConfig:
    """Provides a configuration for high-accuracy, comprehensive model search."""
    return PipelineConfig(
        models={
            "RandomForest_Specialist": ModelConfig(
                name="RandomForest_Specialist",
                enabled=True,
                params={
                    "n_estimators": 200,
                    "max_depth": 10,
                    "min_samples_leaf": 2,
                    "class_weight": "balanced",
                    "random_state": 42 # <-- ДОБАВЛЕНО
                },
                profile_requirements={
                    "min_samples": 100,
                    "n_features_range": [5, 50],
                    "data_complexity": "medium"
                },
                description="RandomForest optimized for accuracy with balanced classes."
            ),
            "SVM_Specialist": ModelConfig(
                name="SVM_Specialist",
                enabled=True,
                params={"C": 5.0, "kernel": "rbf", "gamma": "scale", "probability": True, "random_state": 42},
                profile_requirements={
                    "min_samples": 50,
                    "n_features_range": [2, 30],
                    "data_complexity": "medium"
                },
                description="Highly accurate RBF SVM."
            ),
            "GradientBoosting_Specialist": ModelConfig(
                name="GradientBoosting_Specialist",
                enabled=True,
                params={"n_estimators": 150, "learning_rate": 0.05, "max_depth": 3, "random_state": 42},
                profile_requirements={
                    "min_samples": 100,
                    "n_features_range": [5, 100],
                    "data_complexity": "high"
                },
                description="Gradient Boosting for complex datasets."
            ),
            "NeuralNetwork_Specialist": ModelConfig(
                name="NeuralNetwork_Specialist",
                enabled=True,
                params={"hidden_layer_sizes": (50, 25), "max_iter": 500, "activation": "relu", "solver": "adam", "random_state": 42},
                profile_requirements={
                    "min_samples": 200,
                    "n_features_range": [10, 150],
                    "data_complexity": "high"
                },
                description="Multi-layer Perceptron for highly non-linear data."
            ),
            "LogisticRegression_Specialist": ModelConfig(
                name="LogisticRegression_Specialist",
                enabled=True,
                params={"solver": "saga", "penalty": "elasticnet", "l1_ratio": 0.5, "max_iter": 200, "random_state": 42},
                profile_requirements={
                    "min_samples": 50,
                    "n_features_range": [2, 100],
                    "data_complexity": "medium"
                },
                description="Robust Logistic Regression with elasticnet regularization."
            )
        },
        training={
            "cv_folds": 10,
            "test_size": 0.2,
            "random_state": 42,
            "use_cv_in_scoring": True # Use CV score in final model selection
        },
        scoring={
            "accuracy_weight": 0.5,
            "f1_weight": 0.4,
            "cv_weight": 0.1
        }
    )

def interactive_config() -> PipelineConfig:
    # This function would allow user to interactively build a config
    # For now, it returns a default config
    print("\n[Interactive Configuration Not Yet Implemented, Returning Default Config]")
    return get_default_config()

# Helper to dynamically get config by name (e.g., from command line or env)
def get_config_by_name(name: str) -> PipelineConfig:
    if name == "default":
        return get_default_config()
    elif name == "fast":
        return get_fast_config()
    elif name == "accurate":
        return get_accurate_config()
    elif name == "interactive":
        return interactive_config()
    else:
        raise ValueError(f"Unknown pipeline config name: {name}")
