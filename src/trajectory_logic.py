"""
Trajectory Logic Module
=======================
Implements physiological deterioration detection based on temporal trajectories.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, List


class TrajectoryBasedDetector:
    """
    Detect physiological deterioration from temporal trajectories.
    
    Key principle: Deterioration is flagged when there is sustained
    deviation from patient baseline across multiple consecutive windows.
    """
    
    VITAL_COLUMNS = [
        'heart_rate', 'systolic_bp', 'diastolic_bp', 'spo2',
        'respiratory_rate', 'temperature'
    ]
    
    # How many SDs from baseline triggers abnormality flag
    ABNORMALITY_THRESHOLDS = {
        'heart_rate':        1.5,
        'systolic_bp':       2.0,
        'diastolic_bp':      2.0,
        'spo2':              2.0,   # Lowered — SpO2 drop is most time-critical
        'respiratory_rate':  2.0,
        'temperature':       2.0,
    }

    # Direction that counts as deterioration per vital.
    # 'decrease' = only flag when value drops below baseline (e.g. BP, SpO2)
    # 'increase' = only flag when value rises above baseline (e.g. HR, RR)
    # 'both'     = flag in either direction (e.g. temperature: fever OR hypothermia)
    DETERIORATION_DIRECTION = {
        'heart_rate':        'increase',
        'systolic_bp':       'decrease',
        'diastolic_bp':      'decrease',
        'spo2':              'decrease',
        'respiratory_rate':  'increase',
        'temperature':       'both',
    }

    # Clinical importance weights — SpO2 and BP drops are more urgent than HR
    VITAL_WEIGHTS = {
        'spo2':              3.0,
        'systolic_bp':       2.5,
        'diastolic_bp':      2.0,
        'respiratory_rate':  2.0,
        'heart_rate':        1.5,
        'temperature':       1.0,
    }
    
    def __init__(self, baseline_window: int = 12, instability_window: int = 3,
                 persistence_threshold: int = 2, instability_threshold: float = 0.3):
        """
        Initialize trajectory detector.

        Parameters:
            baseline_window: Number of records for computing baseline
            instability_window: Window for rolling instability score
            persistence_threshold: Number of consecutive abnormal windows to flag deterioration
            instability_threshold: Fraction of abnormal vitals to flag instability (0-1)
        """
        self.baseline_window = baseline_window
        self.instability_window = instability_window
        self.persistence_threshold = persistence_threshold
        self.instability_threshold = instability_threshold
    
    def compute_patient_baseline(self, patient_df: pd.DataFrame) -> Dict[str, Tuple[float, float]]:
        """
        Compute patient-specific baseline (mean, std) from initial period.
        
        Parameters:
            patient_df: DataFrame for one patient, sorted by timestamp
            
        Returns:
            Dictionary mapping vital names to (mean, std) tuples
        """
        baseline_period = patient_df.head(self.baseline_window)
        
        baseline_stats = {}
        for vital in self.VITAL_COLUMNS:
            values = baseline_period[vital].values
            baseline_stats[vital] = {
                'mean': np.mean(values),
                'std': np.std(values)
            }
        
        return baseline_stats
    
    def compute_abnormality_score(self, values: np.ndarray, baseline_mean: float,
                                  baseline_std: float, threshold_multiplier: float,
                                  direction: str = 'both') -> float:
        """
        Compute abnormality score for a vital sign sequence.

        Uses directional thresholds so only clinically meaningful deviations
        are flagged:
          - SpO2 dropping below baseline = bad (direction='decrease')
          - HR rising above baseline = bad (direction='increase')
          - Temperature either direction = bad (direction='both')

        Parameters:
            values: Vital sign values in current window
            baseline_mean: Patient baseline mean
            baseline_std: Patient baseline std
            threshold_multiplier: Number of SDs to flag as abnormal
            direction: 'increase', 'decrease', or 'both'

        Returns:
            Fraction of values in window that are abnormal (0-1)
        """
        if baseline_std == 0 or np.isnan(baseline_std):
            baseline_std = 1.0

        z_scores = (values - baseline_mean) / baseline_std  # signed

        if direction == 'increase':
            abnormal = z_scores > threshold_multiplier
        elif direction == 'decrease':
            abnormal = z_scores < -threshold_multiplier
        else:  # both
            abnormal = np.abs(z_scores) > threshold_multiplier

        return float(np.mean(abnormal))
    
    def detect_sustained_deviation(self, patient_df: pd.DataFrame) -> Tuple[np.ndarray, List[int]]:
        """
        Detect sustained deviation from baseline across patient trajectory.
        
        Parameters:
            patient_df: DataFrame for one patient, sorted by timestamp
            
        Returns:
            Tuple of (instability_scores, deterioration_window_indices)
        """
        baseline_stats = self.compute_patient_baseline(patient_df)
        
        # Compute sliding instability score
        instability_scores = []
        
        for i in range(self.baseline_window, len(patient_df)):
            current_window = patient_df.iloc[max(0, i-self.instability_window):i+1]
            
            # Compute weighted abnormality — clinically critical vitals (SpO2, BP)
            # contribute more to the instability score than temperature
            weighted_sum = 0.0
            weight_total = 0.0
            for vital in self.VITAL_COLUMNS:
                values    = current_window[vital].values
                baseline  = baseline_stats[vital]
                threshold = self.ABNORMALITY_THRESHOLDS.get(vital, 2.0)
                direction = self.DETERIORATION_DIRECTION.get(vital, 'both')
                weight    = self.VITAL_WEIGHTS.get(vital, 1.0)

                abnorm = self.compute_abnormality_score(
                    values, baseline['mean'], baseline['std'], threshold, direction
                )
                weighted_sum  += abnorm * weight
                weight_total  += weight

            instability = weighted_sum / weight_total if weight_total > 0 else 0.0
            instability_scores.append(instability)
        
        instability_scores = np.array(instability_scores)
        
        # Find sustained deterioration windows
        # Flag as deteriorating if consecutive windows exceed threshold
        deterioration_windows = []
        consecutive_count = 0

        for i, score in enumerate(instability_scores):
            if score > self.instability_threshold:
                consecutive_count += 1
                if consecutive_count >= self.persistence_threshold:
                    deterioration_windows.append(self.baseline_window + i)
            else:
                consecutive_count = 0
        
        return instability_scores, deterioration_windows
    
    def flag_deterioration(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Flag deterioration status for entire dataset using trajectory logic.
        
        Parameters:
            df: Preprocessed DataFrame with all patients
            
        Returns:
            DataFrame with added 'predicted_deterioration' column
        """
        print("\n" + "=" * 60)
        print("TRAJECTORY-BASED DETERIORATION DETECTION")
        print("=" * 60)
        
        df = df.copy()
        df['predicted_deterioration'] = 0
        df['instability_score'] = 0.0
        
        for patient_id, patient_df in df.groupby('patient_id'):
            patient_indices = patient_df.index
            patient_df_sorted = patient_df.sort_values('timestamp').reset_index(drop=True)
            
            # Detect deterioration
            instability_scores, deterioration_windows = self.detect_sustained_deviation(patient_df_sorted)
            
            # Assign instability scores
            for i, score in enumerate(instability_scores):
                idx = patient_indices[self.baseline_window + i] if self.baseline_window + i < len(patient_indices) else None
                if idx is not None:
                    df.loc[idx, 'instability_score'] = score
            
            # Mark deterioration windows
            for win_idx in deterioration_windows:
                if win_idx < len(patient_indices):
                    idx = patient_indices[win_idx]
                    df.loc[idx, 'predicted_deterioration'] = 1
        
        num_deteriorated = np.sum(df['predicted_deterioration'])
        print(f"\n? Deterioration detection complete")
        print(f"  Patients flagged as deteriorating: {num_deteriorated}")
        print(f"  Deterioration rate: {100*num_deteriorated/len(df):.1f}%")
        
        return df
    
    def compute_trajectory_similarity(self, traj1: np.ndarray, traj2: np.ndarray) -> float:
        """
        Compute similarity between two trajectories using normalized Euclidean distance.
        Used for grouping similar deterioration patterns.

        Parameters:
            traj1, traj2: Time series trajectories

        Returns:
            Similarity score (0-1, higher is more similar)
        """
        # Euclidean distance converted to similarity
        if len(traj1) != len(traj2):
            min_len = min(len(traj1), len(traj2))
            traj1 = traj1[:min_len]
            traj2 = traj2[:min_len]
        
        distance = np.linalg.norm(traj1 - traj2)
        similarity = 1.0 / (1.0 + distance)  # Convert distance to similarity
        
        return similarity
    
    def identify_deterioration_phenotypes(self, df: pd.DataFrame, 
                                         num_clusters: int = 3) -> Dict:
        """
        Attempt to identify distinct deterioration phenotypes (patterns).
        
        Parameters:
            df: DataFrame with deteriorating patients
            num_clusters: Number of phenotypes to identify
            
        Returns:
            Dictionary with phenotype information
        """
        print(f"\nIdentifying deterioration phenotypes...")
        
        deteriorating = df[df['predicted_deterioration'] == 1]
        
        vitals_data = deteriorating[self.VITAL_COLUMNS].values
        
        # Simple clustering based on feature profiles
        from sklearn.cluster import KMeans
        
        if len(vitals_data) < num_clusters:
            return {'message': 'Not enough deteriorating samples for clustering'}
        
        kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
        phenotypes = kmeans.fit_predict(vitals_data)
        
        phenotype_info = {}
        for phenotype_id in range(num_clusters):
            phenotype_samples = deteriorating[phenotypes == phenotype_id]
            phenotype_info[f'Phenotype_{phenotype_id+1}'] = {
                'count': len(phenotype_samples),
                'vital_profiles': phenotype_samples[self.VITAL_COLUMNS].mean().to_dict()
            }
        
        print(f"  Identified {num_clusters} deterioration phenotypes")
        
        return phenotype_info
    
    def compare_predictions(self, df: pd.DataFrame) -> Dict:
        """
        Compare trajectory-based predictions with actual labels.
        
        Parameters:
            df: DataFrame with 'label' (actual) and 'predicted_deterioration' columns
            
        Returns:
            Dictionary with comparison metrics
        """
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score, f1_score,
            roc_auc_score, confusion_matrix
        )
        
        y_true = df['label'].values
        y_pred = df['predicted_deterioration'].values
        
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0),
            'confusion_matrix': confusion_matrix(y_true, y_pred)
        }
        
        print(f"\nTrajectory-Based Detection Metrics:")
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1-Score:  {metrics['f1']:.4f}")
        
        cm = metrics['confusion_matrix']
        print(f"  Confusion Matrix:")
        print(f"    TN={cm[0,0]}, FP={cm[0,1]}")
        print(f"    FN={cm[1,0]}, TP={cm[1,1]}")
        
        return metrics
