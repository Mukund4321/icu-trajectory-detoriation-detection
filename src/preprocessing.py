"""
Preprocessing Module
====================
Handles data cleaning, resampling, smoothing, and normalization for trajectory data.
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple


class ICUPreprocessor:
    """Preprocess ICU vital signs data for modeling."""

    VITAL_COLUMNS = [
        'heart_rate', 'systolic_bp', 'diastolic_bp', 'spo2',
        'respiratory_rate', 'temperature'
    ]

    # Key sepsis lab markers (SOFA score components). Measured infrequently
    # (once per 12-24h) — carried forward between measurements.
    LAB_COLUMNS = ['wbc', 'lactate', 'glucose', 'creatinine', 'bun']
    
    def __init__(self, resample_interval: str = '5min', smoothing_window: int = 3):
        """
        Initialize preprocessor.
        
        Parameters:
            resample_interval: Resampling frequency (e.g., '5min', '15min')
            smoothing_window: Window size for rolling mean
        """
        self.resample_interval = resample_interval
        self.smoothing_window = smoothing_window
        self.normalization_stats = {}
    
    def resample_per_patient(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Resample each patient's data to fixed time intervals.
        Forward-filling missing values within gaps.
        
        Parameters:
            df: Input DataFrame with patient trajectories
            
        Returns:
            Resampled DataFrame
        """
        print("\n" + "=" * 60)
        print("PREPROCESSING PIPELINE")
        print("=" * 60)
        print(f"Resampling to {self.resample_interval} intervals...")
        
        resampled_dfs = []
        
        for patient_id, group in df.groupby('patient_id'):
            # Set timestamp as index and resample
            group = group.set_index('timestamp')

            # Resample vital signs
            vital_resampled = group[self.VITAL_COLUMNS].resample(self.resample_interval).mean()
            vital_resampled = vital_resampled.ffill().bfill()

            # Resample label (take mode/most frequent)
            label_resampled = group['label'].resample(self.resample_interval).agg(
                lambda x: x.mode()[0] if len(x) > 0 else 0
            )

            # Resample lab columns if present — heavy forward-fill since labs are
            # measured infrequently (once per 12-24h). Last known value is carried
            # forward until the next measurement.
            present_labs = [c for c in self.LAB_COLUMNS if c in group.columns]
            # Missingness flags and age pass through (resample by max to keep 1s)
            miss_cols = [f'{l}_measured' for l in self.LAB_COLUMNS if f'{l}_measured' in group.columns]
            resampled = vital_resampled.copy()
            if present_labs:
                lab_data = group[present_labs].apply(pd.to_numeric, errors='coerce')
                lab_resampled = lab_data.resample(self.resample_interval).mean()
                lab_resampled = lab_resampled.ffill().bfill()
                for col in present_labs:
                    resampled[col] = lab_resampled[col]
            if miss_cols:
                miss_resampled = group[miss_cols].resample(self.resample_interval).max()
                miss_resampled = miss_resampled.fillna(0)
                for col in miss_cols:
                    resampled[col] = miss_resampled[col]
            if 'age' in group.columns:
                age_resampled = group[['age']].resample(self.resample_interval).mean()
                age_resampled = age_resampled.ffill().bfill()
                resampled['age'] = age_resampled['age']

            resampled['label'] = label_resampled
            resampled['patient_id'] = patient_id
            resampled = resampled.reset_index()

            resampled_dfs.append(resampled)
        
        result = pd.concat(resampled_dfs, ignore_index=True)
        result = result.sort_values(by=['patient_id', 'timestamp']).reset_index(drop=True)
        
        print(f"? Resampled to {len(result)} records")
        return result
    
    def smooth_vitals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply rolling mean smoothing to vital signs per patient.
        Reduces noise while preserving deterioration patterns.
        
        Parameters:
            df: Input DataFrame
            
        Returns:
            Smoothed DataFrame
        """
        print(f"Applying rolling smoothing (window={self.smoothing_window})...")
        
        df = df.copy()
        
        for patient_id, group_idx in df.groupby('patient_id').groups.items():
            for vital in self.VITAL_COLUMNS:
                df.loc[group_idx, vital] = (
                    df.loc[group_idx, vital]
                    .rolling(window=self.smoothing_window, center=False, min_periods=1)
                    .mean()
                    .values
                )
        
        print(f"? Smoothing applied")
        return df
    
    def normalize_per_patient(self, df: pd.DataFrame, fit: bool = True,
                              baseline_window: int = None) -> pd.DataFrame:
        """
        Normalize vital signs per patient using z-score normalization.

        Parameters:
            df: Input DataFrame
            fit: If True, compute and store normalization stats from this data.
                 If False, use previously stored stats (for test/val patients
                 that have their own records but should not re-fit global stats).
            baseline_window: If set, compute mean/std from only the first N records
                             of each patient instead of the full timeline.
                             This prevents look-ahead bias: a deteriorating patient's
                             late-stage values would otherwise shift their own mean,
                             making early records look artificially normal.

        Returns:
            Normalized DataFrame
        """
        print("Normalizing vitals per patient (z-score)...")

        df = df.copy()

        if fit:
            self.normalization_stats = {}

        for patient_id, group_idx in df.groupby('patient_id').groups.items():
            patient_stats = {}

            for vital in self.VITAL_COLUMNS:
                all_values = df.loc[group_idx, vital]

                if baseline_window is not None:
                    # Use only the first baseline_window records to compute stats
                    stat_values = all_values.iloc[:baseline_window]
                else:
                    stat_values = all_values

                mean = stat_values.mean()
                std = stat_values.std()

                if std == 0 or np.isnan(std):
                    std = 1.0

                patient_stats[vital] = {'mean': mean, 'std': std}
                df.loc[group_idx, vital] = (all_values - mean) / std

            if fit:
                self.normalization_stats[patient_id] = patient_stats

        print(f"? Normalized {len(self.normalization_stats)} patients"
              + (f" (baseline_window={baseline_window})" if baseline_window else ""))
        return df
    
    def denormalize_values(self, values: np.ndarray, patient_id: int, vital_name: str) -> np.ndarray:
        """
        Reverse normalization for a specific vital sign.
        
        Parameters:
            values: Normalized values
            patient_id: Patient identifier
            vital_name: Name of vital sign
            
        Returns:
            Denormalized values
        """
        if patient_id not in self.normalization_stats:
            return values
        
        stats = self.normalization_stats[patient_id].get(vital_name, {})
        mean = stats.get('mean', 0)
        std = stats.get('std', 1)
        
        return values * std + mean
    
    def handle_missing_values(self, df: pd.DataFrame, method: str = 'forward_fill') -> pd.DataFrame:
        """
        Handle remaining missing values.
        
        Parameters:
            df: Input DataFrame
            method: 'forward_fill', 'backward_fill', or 'interpolate'
            
        Returns:
            DataFrame with missing values handled
        """
        df = df.copy()
        
        for vital in self.VITAL_COLUMNS:
            if method == 'forward_fill':
                df[vital] = df[vital].ffill()
            elif method == 'backward_fill':
                df[vital] = df[vital].bfill()
            elif method == 'interpolate':
                df[vital] = df[vital].interpolate()
        
        # Fill any remaining NaNs
        df = df.fillna(df.mean(numeric_only=True))
        
        return df
    
    def process(self, df: pd.DataFrame, baseline_window: int = 12) -> Tuple[pd.DataFrame, Dict]:
        """
        Full preprocessing pipeline.
        1. Resample to fixed intervals
        2. Handle missing values
        3. Smooth vitals
        4. Normalize per patient (using only baseline_window records for stats)

        Parameters:
            df: Raw input DataFrame
            baseline_window: Number of initial records used to compute normalization
                             stats. Defaults to 12 (1 hour at 5-min intervals).

        Returns:
            Tuple of (processed DataFrame, preprocessing stats)
        """
        df = self.resample_per_patient(df)
        df = self.handle_missing_values(df)
        df = self.smooth_vitals(df)

        # Save smoothed-but-unnormalised values before z-scoring.
        # These are used as globally-anchored features in feature engineering —
        # they give the model absolute vital sign levels (e.g. HR=130 is always
        # high regardless of this patient's baseline), which transfers better to
        # unseen patients than per-patient z-scores alone.
        for vital in self.VITAL_COLUMNS:
            df[f'{vital}_raw'] = df[vital]

        df = self.normalize_per_patient(df, fit=True, baseline_window=baseline_window)
        
        stats = {
            'total_records': len(df),
            'num_patients': df['patient_id'].nunique(),
            'normalization_stats': self.normalization_stats
        }
        
        print(f"\n? Preprocessing complete: {stats['total_records']} records, {stats['num_patients']} patients")
        
        return df, stats
