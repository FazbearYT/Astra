__version__ = "2.0.0"
__author__ = "Adaptive ML Team"

from .model_profiler import DataProfiler, DatasetProfile, FeatureProfile
from .model_selector import AdaptiveModelSelector, SpecializedModel
from .pipeline_config import PipelineConfig, get_default_config, get_fast_config, get_accurate_config
from .progress import PROGRESS_ENABLED, enable_progress, disable_progress

__all__ = [
    "DataProfiler",
    "DatasetProfile",
    "FeatureProfile",
    "AdaptiveModelSelector",
    "SpecializedModel",
    "PipelineConfig",
    "get_default_config",
    "get_fast_config",
    "get_accurate_config",
    "PROGRESS_ENABLED",
    "enable_progress",
    "disable_progress",
]