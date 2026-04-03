"""
Data Loader Module
==================
Handles CSV loading, validation, and initial data organization for ICU trajectory data.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional


class ICUDataLoader:
    """Load and validate ICU vital signs data."""
    
    REQUIRED_COLUMNS = [
        'patient_id', 'timestamp',
        'heart_rate', 'systolic_bp', 'diastolic_bp', 'spo2',
        'respiratory_rate', 'temperature', 'label'
    ]
    
    VITAL_COLUMNS = [
        'heart_rate', 'systolic_bp', 'diastolic_bp', 'spo2',
        'respiratory_rate', 'temperature'
    ]
    
    def __init__(self, random_seed: int = 42):
        """Initialize data loader with random seed for reproducibility."""
        self.random_seed = random_seed
        np.random.seed(random_seed)
    
    def load_csv(self, filepath: str) -> pd.DataFrame:
        """
        Load CSV file and perform initial validation.
        
        Parameters:
            filepath: Path to CSV file
            
        Returns:
            DataFrame with validated data
        """
        try:
            df = pd.read_csv(filepath)
            print(f"? Loaded {len(df)} records from {Path(filepath).name}")
            return df
        except FileNotFoundError:
            raise FileNotFoundError(f"File not found: {filepath}")
        except Exception as e:
            raise ValueError(f"Error loading CSV: {str(e)}")
    
    def validate_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate that all required columns are present.
        
        Parameters:
            df: Input DataFrame
            
        Returns:
            Validated DataFrame
        """
        missing_cols = set(self.REQUIRED_COLUMNS) - set(df.columns)
        
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        print(f"? All required columns present")
        return df
    
    def format_timestamps(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert timestamp column to datetime.
        
        Parameters:
            df: Input DataFrame
            
        Returns:
            DataFrame with datetime timestamps
        """
        df = df.copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        print(f"? Timestamps formatted as datetime")
        return df
    
    def sort_by_trajectory(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Sort by patient_id and timestamp to establish temporal order.
        
        Parameters:
            df: Input DataFrame
            
        Returns:
            Sorted DataFrame
        """
        df = df.sort_values(by=['patient_id', 'timestamp']).reset_index(drop=True)
        print(f"? Data sorted by patient_id and timestamp")
        return df
    
    def check_data_quality(self, df: pd.DataFrame) -> dict:
        """
        Perform data quality checks.
        
        Parameters:
            df: Input DataFrame
            
        Returns:
            Dictionary with quality metrics
        """
        stats = {
            'total_records': len(df),
            'num_patients': df['patient_id'].nunique(),
            'date_range': (df['timestamp'].min(), df['timestamp'].max()),
            'missing_values': df[self.VITAL_COLUMNS].isnull().sum().to_dict(),
            'class_distribution': df['label'].value_counts().to_dict()
        }
        
        print(f"\n? Data Quality Metrics:")
        print(f"   Total Records: {stats['total_records']}")
        print(f"   Unique Patients: {stats['num_patients']}")
        print(f"   Date Range: {stats['date_range'][0]} to {stats['date_range'][1]}")
        print(f"   Missing Values: {stats['missing_values']}")
        print(f"   Label Distribution: {stats['class_distribution']}")
        
        return stats
    
    def process_raw_data(self, filepath: str) -> Tuple[pd.DataFrame, dict]:
        """
        Full data processing pipeline: load -> validate -> sort.
        
        Parameters:
            filepath: Path to raw CSV
            
        Returns:
            Tuple of (processed DataFrame, quality stats)
        """
        print("=" * 60)
        print("DATA LOADING PIPELINE")
        print("=" * 60)
        
        df = self.load_csv(filepath)
        df = self.validate_columns(df)
        df = self.format_timestamps(df)
        df = self.sort_by_trajectory(df)
        
        stats = self.check_data_quality(df)
        
        return df, stats


def generate_synthetic_icu_data(
    num_patients: int = 50,
    records_per_patient: int = 100,
    output_path: str = None,
    random_seed: int = 42
) -> pd.DataFrame:
    """
    Generate synthetic ICU vital signs data for development and testing.
    
    Parameters:
        num_patients: Number of patients in dataset
        records_per_patient: Number of records per patient
        output_path: Optional path to save CSV
        random_seed: Random seed for reproducibility
        
    Returns:
        Synthetic ICU DataFrame
    """
    np.random.seed(random_seed)
    
    data = []
    
    for patient_id in range(1, num_patients + 1):
        # Assign trajectory type: 0 = stable, 1 = deteriorating
        trajectory_type = np.random.choice([0, 1], p=[0.6, 0.4])
        
        for record_idx in range(records_per_patient):
            timestamp = pd.Timestamp('2025-01-01') + pd.Timedelta(minutes=5 * record_idx)
            
            # Generate vital signs with trajectory
            if trajectory_type == 0:  # Stable
                hr = 70 + np.random.normal(0, 5)
                systolic = 120 + np.random.normal(0, 5)
                diastolic = 80 + np.random.normal(0, 3)
                spo2 = 97 + np.random.normal(0, 1)
                rr = 16 + np.random.normal(0, 2)
                temp = 37 + np.random.normal(0, 0.3)
                label = 0
            else:  # Deteriorating
                # Deterioration trend over time
                deterioration_factor = record_idx / records_per_patient
                hr = 70 + deterioration_factor * 30 + np.random.normal(0, 5)
                systolic = 120 - deterioration_factor * 20 + np.random.normal(0, 5)
                diastolic = 80 - deterioration_factor * 12 + np.random.normal(0, 3)
                spo2 = 97 - deterioration_factor * 5 + np.random.normal(0, 1)
                rr = 16 + deterioration_factor * 10 + np.random.normal(0, 2)
                temp = 37 + deterioration_factor * 1.5 + np.random.normal(0, 0.3)
                label = 1 if deterioration_factor > 0.5 else 0
            
            data.append({
                'patient_id': patient_id,
                'timestamp': timestamp,
                'heart_rate': max(40, min(180, hr)),
                'systolic_bp': max(70, min(200, systolic)),
                'diastolic_bp': max(40, min(130, diastolic)),
                'spo2': max(80, min(100, spo2)),
                'respiratory_rate': max(8, min(50, rr)),
                'temperature': max(35, min(41, temp)),
                'label': label
            })
    
    df = pd.DataFrame(data)
    
    if output_path:
        df.to_csv(output_path, index=False)
        print(f"? Synthetic data saved to {output_path}")
    
    return df
