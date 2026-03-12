"""
Adaptive ML System - Подсистема адаптации моделей под рабочие профили

Version: 1.0.0
"""

__version__ = "1.0.0"
__author__ = "Adaptive ML Team"
__email__ = "team@adaptiveml.system"

from .model_profiler import DataProfiler, DatasetProfile, FeatureProfile, profile_from_csv
from .model_selector import AdaptiveModelSelector, SpecializedModel

__all__ = [
    # Version info
    "__version__",
    "__author__",
    "__email__",

    # Model Profiler
    "DataProfiler",
    "DatasetProfile",
    "FeatureProfile",
    "profile_from_csv",

    # Model Selector
    "AdaptiveModelSelector",
    "SpecializedModel",
]


def get_version():
    """Возвращает версию пакета"""
    return __version__


def info():
    """Выводит информацию о пакете"""
    print(f"Adaptive ML System v{__version__}")
    print(f"Author: {__author__}")
    print(f"Email: {__email__}")
    print("\nAvailable modules:")
    print("  - DataProfiler: Профилирование данных")
    print("  - AdaptiveModelSelector: Адаптивный выбор моделей")
    print("  - SpecializedModel: Обёртка для специализированных моделей")