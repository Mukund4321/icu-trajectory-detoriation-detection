"""
ICU Trajectory Deterioration Detection System
==============================================

A research-grade system for detecting physiological deterioration in ICU patients
through multivariate time-series analysis using classical ML, deep learning (LSTM),
and trajectory-based detection logic.

Main Components:
- data_loader: CSV loading and validation
- preprocessing: Data cleaning, resampling, normalization
- feature_engineering: Trajectory-based feature extraction
- ml_models: Logistic Regression and Random Forest baselines
- dl_models: LSTM model for temporal pattern recognition
- trajectory_logic: Physiological deterioration detection algorithm
- evaluation: Metrics computation and visualization
"""

__version__ = "1.0.0"
__author__ = "Research Team"

from .data_loader import ICUDataLoader, generate_synthetic_icu_data
from .preprocessing import ICUPreprocessor
from .feature_engineering import TrajectoryFeatureEngineer
from .ml_models import MLBaselines
from .dl_models import LSTMTrajectoryModel, LSTMTrainer
from .trajectory_logic import TrajectoryBasedDetector
from .evaluation import TrajectoryEvaluator

__all__ = [
    'ICUDataLoader',
    'generate_synthetic_icu_data',
    'ICUPreprocessor',
    'TrajectoryFeatureEngineer',
    'MLBaselines',
    'LSTMTrajectoryModel',
    'LSTMTrainer',
    'TrajectoryBasedDetector',
    'TrajectoryEvaluator'
]
