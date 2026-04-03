"""
Configuration Module
====================
Centralized configuration for the entire pipeline.
"""

# Random Seeds (for reproducibility)
RANDOM_SEED = 42

# ===== DATA GENERATION =====
SYNTHETIC_DATA = {
    'num_patients': 50,           # Number of synthetic patients
    'records_per_patient': 100,   # Records per patient (at original granularity)
    'positive_class_ratio': 0.4   # 40% deteriorating patients
}

# ===== PREPROCESSING =====
PREPROCESSING = {
    'resample_interval': '5min',           # Resampling frequency
    'smoothing_window': 3,                 # Rolling mean window size
    'missing_value_method': 'forward_fill' # 'forward_fill', 'backward_fill', 'interpolate'
}

# ===== NORMALIZATION =====
# Per-patient z-score normalization
NORMALIZATION = {
    'method': 'zscore',  # 'zscore', 'minmax', 'robust'
    'per_patient': True  # Patient-specific normalization
}

# ===== FEATURE ENGINEERING =====
FEATURE_ENGINEERING = {
    'window_size': 12,    # Number of records per window (1 hour at 5-min intervals)
    'window_step': 1,     # Step size for sliding windows
    'lstm_seq_length': 12 # Sequence length for LSTM
}

# ===== TRAJECTORY DETECTION =====
TRAJECTORY_DETECTION = {
    'baseline_window': 12,            # Records for baseline computation
    'instability_window': 3,          # Rolling window for instability
    'persistence_threshold': 2,       # Consecutive windows to flag deterioration
    'instability_threshold': 0.3,     # Abnormality fraction threshold
    'abnormality_thresholds': {       # Clinical thresholds (in SD units)
        'heart_rate': 1.5,
        'systolic_bp': 2.0,
        'diastolic_bp': 2.0,
        'spo2': 2.5,
        'respiratory_rate': 2.0,
        'temperature': 2.0
    }
}

# ===== MACHINE LEARNING =====
ML_MODELS = {
    'test_size': 0.2,
    'logistic_regression': {
        'max_iter': 1000,
        'class_weight': 'balanced'
    },
    'random_forest': {
        'n_estimators': 100,
        'max_depth': 10,
        'class_weight': 'balanced',
        'n_jobs': -1  # Use all CPUs
    }
}

# ===== DEEP LEARNING (LSTM) =====
LSTM_MODEL = {
    'input_size': 6,         # Number of vital signs
    'hidden_size': 64,       # LSTM hidden dimension
    'num_layers': 2,         # Number of LSTM layers
    'dropout': 0.3,          # Dropout rate
    'output_size': 1         # Binary classification
}

LSTM_TRAINING = {
    'learning_rate': 0.001,
    'batch_size': 32,
    'epochs': 50,
    'patience': 10,         # Early stopping patience
    'optimizer': 'adam',
    'loss': 'bce_with_logits'
}

# ===== EVALUATION =====
EVALUATION = {
    'sample_patients_to_plot': 3,
    'top_n_features': 10,
    'confusion_matrix_figsize': (12, 10),
    'roc_curve_figsize': (10, 8),
    'dpi': 300  # Resolution for saved plots
}

# ===== PATHS =====
PATHS = {
    'data_raw': 'data/raw',
    'data_processed': 'data/processed',
    'models': 'models',
    'results': 'results'
}

# ===== VITAL SIGNS CONFIG =====
VITAL_SIGNS = [
    'heart_rate',
    'systolic_bp',
    'diastolic_bp',
    'spo2',
    'respiratory_rate',
    'temperature'
]

VITAL_DISPLAY_NAMES = {
    'heart_rate': 'Heart Rate (bpm)',
    'systolic_bp': 'Systolic BP (mmHg)',
    'diastolic_bp': 'Diastolic BP (mmHg)',
    'spo2': 'SpO₂ (%)',
    'respiratory_rate': 'Respiratory Rate (breaths/min)',
    'temperature': 'Temperature (°C)'
}

# ===== CLINICAL NORMAL RANGES =====
CLINICAL_RANGES = {
    'heart_rate': (60, 100),
    'systolic_bp': (90, 140),
    'diastolic_bp': (60, 90),
    'spo2': (95, 100),
    'respiratory_rate': (12, 20),
    'temperature': (36.5, 37.5)
}
