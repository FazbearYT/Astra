
__version__ = "2.1.0"
__author__ = "fazbear_r"

from .model_profiler import DataProfiler, DatasetProfile, FeatureProfile, profile_from_csv
from .model_selector import AdaptiveModelSelector, SpecializedModel

__all__ = [
    "__version__",
    "__author__",
    "DataProfiler",
    "DatasetProfile",
    "FeatureProfile",
    "profile_from_csv",
    "AdaptiveModelSelector",
    "SpecializedModel",
]


def get_version():
    return __version__


def info():
    print(f"Adaptive ML System v{__version__}")
    print(f"Author: {__author__}")