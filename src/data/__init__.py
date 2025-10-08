"""
Data module för CNN-LSTM-LSTM träning med VitalDB.
Centraliserade imports för att eliminera kodduplicering.
"""

# Standard imports
import os
import glob
import logging
from typing import Optional, Tuple, List, Dict, Any

# Data processing imports
import pandas as pd
import numpy as np

# VitalDB imports (optional)
try:
    import vitaldb as vdb
    VITALDB_AVAILABLE = True
except ImportError:
    VITALDB_AVAILABLE = False

# Expose commonly used classes
from data.data_loader import VitalDBDataLoader
from data.loaders import VitalLoader, CSVLoader, ClinicalLoader
from data.mappers import FeatureMapper
from data.validators import DataValidator, ClinicalValidator
from data.utils import (
    FileFinder,
    handle_errors,
    safe_pandas_read,
    validate_dataframe
)

# Import all main components
from data.preprocessing import DataPreprocessor
from data.temporal_sync import TemporalSynchronizer
from data.validation import DataValidator
from data.feature_mapping import FeatureMapper
from data.unit_conversion import UnitConverter
from data.tfrecord_utils import TFRecordUtils

# New pipeline components
from data.processing_pipeline import (
    DataProcessingPipeline,
    PipelineConfig,
    ProcessingStep,
    ProcessingStatus,
    ProcessingResult,
    ValidationStep,
    InterpolationStep,
    SynchronizationStep,
    QualityAssuranceStep
)

from data.interpolation_strategies import (
    InterpolationStrategy,
    InterpolationContext,
    InterpolationResult,
    LinearInterpolationStrategy,
    CubicInterpolationStrategy,
    NearestInterpolationStrategy,
    ForwardFillStrategy,
    InterpolationStrategyFactory
)

__all__ = [
    'VitalDBDataLoader',
    'VitalLoader',
    'CSVLoader',
    'ClinicalLoader',
    'FeatureMapper',
    'DataValidator',
    'ClinicalValidator',
    'FileFinder',
    'handle_errors',
    'safe_pandas_read',
    'validate_dataframe',
    'pd',
    'np',
    'VITALDB_AVAILABLE',
    'DataPreprocessor',
    'TemporalSynchronizer',
    'FeatureMapper',
    'UnitConverter',
    'TFRecordUtils',
    # Pipeline components
    'DataProcessingPipeline',
    'PipelineConfig',
    'ProcessingStep',
    'ProcessingStatus',
    'ProcessingResult',
    'ValidationStep',
    'InterpolationStep',
    'SynchronizationStep',
    'QualityAssuranceStep',
    # Interpolation strategies
    'InterpolationStrategy',
    'InterpolationContext',
    'InterpolationResult',
    'LinearInterpolationStrategy',
    'CubicInterpolationStrategy',
    'NearestInterpolationStrategy',
    'ForwardFillStrategy',
    'InterpolationStrategyFactory'
]
