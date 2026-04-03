"""
Feature Engineering Module
===========================
Creates trajectory-based features from multivariate time series data.
"""

import pandas as pd
import numpy as np
from typing import Tuple, List
from scipy import stats


class TrajectoryFeatureEngineer:
    """Engineer features for trajectory analysis."""

    VITAL_COLUMNS = [
        'heart_rate', 'systolic_bp', 'diastolic_bp', 'spo2',
        'respiratory_rate', 'temperature'
    ]

    # SOFA-score lab markers: mean and trend over window.
    # Sparse (measured infrequently) but highly predictive for sepsis.
    LAB_COLUMNS = ['wbc', 'lactate', 'glucose', 'creatinine', 'bun']
    
    def __init__(self, window_size: int = 12, window_step: int = 1):
        """
        Initialize feature engineer.
        
        Parameters:
            window_size: Number of records per sliding window (e.g., 12 = 1 hour at 5-min intervals)
            window_step: Step size for sliding window (1 = every record, 2 = every other, etc.)
        """
        self.window_size = window_size
        self.window_step = window_step
        self.feature_names = []
    
    def compute_trend_slope(self, values: np.ndarray) -> float:
        """
        Compute linear regression slope of vital signs over window.
        Positive slope indicates deterioration trend.
        
        Parameters:
            values: Sequence of vital sign values
            
        Returns:
            Slope value
        """
        if len(values) < 2:
            return 0.0
        
        x = np.arange(len(values))
        try:
            slope, _ = np.polyfit(x, values, 1)
            return slope
        except (ValueError, np.linalg.LinAlgError):
            return 0.0
    
    def compute_rate_of_change(self, values: np.ndarray) -> float:
        """
        Compute average rate of change (percent change per time step).
        
        Parameters:
            values: Sequence of vital sign values
            
        Returns:
            Average rate of change
        """
        if len(values) < 2:
            return 0.0

        diffs = np.diff(values)
        denominators = values[:-1]
        nonzero = denominators != 0
        if not np.any(nonzero):
            return 0.0
        roc = np.mean(diffs[nonzero] / denominators[nonzero])
        return roc
    
    def compute_volatility(self, values: np.ndarray) -> float:
        """
        Compute standard deviation within window (volatility metric).
        High volatility suggests instability.
        
        Parameters:
            values: Sequence of vital sign values
            
        Returns:
            Standard deviation
        """
        if len(values) < 2:
            return 0.0
        return np.std(values)
    
    def compute_correlation_hr_bp(self, window_df: pd.DataFrame) -> float:
        """
        Compute correlation between heart rate and systolic BP.
        Loss of correlation indicates cardiovascular instability.
        
        Parameters:
            window_df: DataFrame slice for one window
            
        Returns:
            Pearson correlation coefficient
        """
        if len(window_df) < 2:
            return 0.0
        
        try:
            corr = window_df['heart_rate'].corr(window_df['systolic_bp'])
            return corr if not np.isnan(corr) else 0.0
        except (TypeError, ValueError):
            return 0.0
    
    def compute_deviation_from_baseline(self, patient_df: pd.DataFrame, 
                                       baseline_window: int = 5) -> pd.Series:
        """
        Compute deviation of each vital from patient's initial baseline.
        
        Parameters:
            patient_df: DataFrame for one patient
            baseline_window: Window size for baseline computation
            
        Returns:
            Series of baseline deviation scores
        """
        baseline = patient_df[self.VITAL_COLUMNS].head(baseline_window).mean()
        
        deviations = pd.Series(0.0, index=patient_df.index)
        
        for idx in patient_df.index:
            vital_values = patient_df.loc[idx, self.VITAL_COLUMNS].values
            deviation = np.mean(np.abs((vital_values - baseline.values) / baseline.values))
            deviations[idx] = deviation
        
        return deviations
    
    def create_sliding_windows(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Create sliding windows from trajectory data.
        Each window represents a time period for a patient.

        Parameters:
            df: Preprocessed DataFrame

        Returns:
            Tuple of (X: features, y: labels, patient_ids: for tracking)
        """
        print("\n" + "=" * 60)
        print("FEATURE ENGINEERING PIPELINE")
        print("=" * 60)

        patients = df['patient_id'].unique()
        n_patients = len(patients)
        print(f"Creating sliding windows (size={self.window_size}, step={self.window_step}) "
              f"for {n_patients} patients...")

        X_list = []
        y_list = []
        patient_id_list = []

        for idx, patient_id in enumerate(patients):
            patient_df = (df[df['patient_id'] == patient_id]
                          .sort_values('timestamp')
                          .reset_index(drop=True))

            n = len(patient_df)
            if n < self.window_size:
                continue

            vitals = patient_df[self.VITAL_COLUMNS].values  # (n, 6)
            labels = patient_df['label'].values              # (n,)

            # Build all window start indices at once
            starts = np.arange(0, n - self.window_size + 1, self.window_step)

            for start in starts:
                window_df = patient_df.iloc[start:start + self.window_size]
                features  = self._extract_window_features(window_df)
                X_list.append(features)
                y_list.append(labels[start + self.window_size - 1])
                patient_id_list.append(patient_id)

            # Progress every 10% of patients
            if (idx + 1) % max(1, n_patients // 10) == 0 or (idx + 1) == n_patients:
                print(f"  {idx + 1}/{n_patients} patients  "
                      f"({len(X_list)} windows so far)")

        X = np.array(X_list, dtype=np.float32)
        y = np.array(y_list, dtype=np.float32)
        patient_ids = np.array(patient_id_list)

        print(f"Created {len(X)} windows from {n_patients} patients")
        print(f"  Positive class (deteriorating): {int(np.sum(y))} ({100*np.mean(y):.1f}%)")

        return X, y, patient_ids
    
    def _extract_window_features(self, window_df: pd.DataFrame) -> np.ndarray:
        """
        Extract all features for one window.
        
        Parameters:
            window_df: DataFrame for one time window
            
        Returns:
            Feature vector
        """
        features = []

        # Check once whether raw columns exist
        raw_cols = [f'{v}_raw' for v in self.VITAL_COLUMNS]
        has_raw = all(c in window_df.columns for c in raw_cols)

        # For each vital: slope/volatility/roc from z-scored values (trend indicators),
        # mean/min/max from raw values (absolute clinical anchors for cross-patient generalisation).
        # z-scored mean/min/max cluster near 0 for every patient regardless of true severity;
        # raw values preserve the clinical distinction between HR=70 and HR=130.
        for vital, raw_col in zip(self.VITAL_COLUMNS, raw_cols):
            z_values  = window_df[vital].values
            raw_values = window_df[raw_col].values if has_raw else z_values

            # Trend slope (deterioration indicator) — z-scored is fine
            features.append(self.compute_trend_slope(z_values))

            # Volatility (instability indicator) — z-scored is fine
            features.append(self.compute_volatility(z_values))

            # Rate of change — z-scored is fine
            features.append(self.compute_rate_of_change(z_values))

            # Summary statistics — use raw so absolute vital levels are preserved
            features.append(float(np.mean(raw_values)))
            features.append(float(np.min(raw_values)))
            features.append(float(np.max(raw_values)))

        # HR-BP correlation
        hr_bp_corr = self.compute_correlation_hr_bp(window_df)
        features.append(hr_bp_corr)

        # Baseline deviation (mean absolute percent deviation from window start)
        # Computed on z-scored values: measures relative change within the window.
        baseline = window_df[self.VITAL_COLUMNS].iloc[0].values
        nonzero_mask = baseline != 0
        if np.any(nonzero_mask):
            diffs = window_df[self.VITAL_COLUMNS].values[:, nonzero_mask] - baseline[nonzero_mask]
            deviation = np.mean(np.abs(diffs / baseline[nonzero_mask]))
        else:
            deviation = 0.0
        features.append(deviation)

        # SOFA lab markers: last known value + trend slope within window.
        # Last-known captures current severity; slope captures trajectory direction.
        # Missing labs (NaN after forward-fill) are replaced with 0.
        for lab in self.LAB_COLUMNS:
            if lab in window_df.columns:
                vals = window_df[lab].values.astype(np.float32)
                last_val = float(vals[-1]) if not np.isnan(vals[-1]) else 0.0
                slope = self.compute_trend_slope(vals[~np.isnan(vals)]) if np.any(~np.isnan(vals)) else 0.0
                features.append(last_val)
                features.append(slope)
            else:
                features.append(0.0)
                features.append(0.0)

        # Lab missingness: fraction of window hours where each lab was measured.
        # High measurement frequency signals clinical concern for that marker.
        for lab in self.LAB_COLUMNS:
            mcol = f'{lab}_measured'
            if mcol in window_df.columns:
                features.append(float(window_df[mcol].mean()))
            else:
                features.append(0.0)

        # Patient age (constant per patient, highly correlated with sepsis risk)
        if 'age' in window_df.columns:
            features.append(float(window_df['age'].iloc[-1]))
        else:
            features.append(0.0)

        return np.array(features, dtype=np.float32)
    
    def get_feature_names(self) -> List[str]:
        """
        Get names of engineered features.
        
        Returns:
            List of feature names
        """
        names = []
        
        for vital in self.VITAL_COLUMNS:
            names.extend([
                f"{vital}_slope",       # z-scored trend
                f"{vital}_volatility",  # z-scored instability
                f"{vital}_roc",         # z-scored rate of change
                f"{vital}_mean_raw",    # raw absolute level
                f"{vital}_min_raw",     # raw absolute min
                f"{vital}_max_raw",     # raw absolute max
            ])

        names.extend(['hr_bp_correlation', 'baseline_deviation'])

        for lab in self.LAB_COLUMNS:
            names.append(f'{lab}_last')
            names.append(f'{lab}_slope')

        for lab in self.LAB_COLUMNS:
            names.append(f'{lab}_measurement_freq')

        names.append('age')

        return names
    
    def create_sequence_data(self, df: pd.DataFrame, seq_length: int = 12) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for LSTM models (univariate time series per patient).
        
        Parameters:
            df: Preprocessed DataFrame
            seq_length: Length of sequence
            
        Returns:
            Tuple of (X: sequences, y: labels)
        """
        print(f"\nCreating sequences for LSTM (length={seq_length})...")

        # Include raw vitals (absolute clinical levels) alongside z-scored vitals
        # (relative change from patient's baseline). Using only z-scored values means
        # HR=70 and HR=130 look identical if both are 1 SD above their own baseline —
        # the LSTM can't learn that "high heart rate is bad". Raw values break this.
        raw_cols = [f'{v}_raw' for v in self.VITAL_COLUMNS]
        has_raw = all(c in df.columns for c in raw_cols)
        seq_cols = self.VITAL_COLUMNS + (raw_cols if has_raw else [])

        X_list = []
        y_list = []

        for _, patient_df in df.groupby('patient_id'):
            patient_df = patient_df.sort_values('timestamp').reset_index(drop=True)

            X_all = patient_df[seq_cols].values   # (n, 6 or 12)
            y_vital = patient_df['label'].values

            for i in range(len(X_all) - seq_length + 1):
                X_list.append(X_all[i:i + seq_length])
                y_list.append(y_vital[i + seq_length - 1])

        X = np.array(X_list, dtype=np.float32)
        y = np.array(y_list, dtype=np.float32)

        print(f"OK Created {len(X)} sequences")
        print(f"  Shape: {X.shape}  ({'z+raw' if has_raw else 'z-only'} vitals)")

        return X, y
