"""
Data Formatter Module
=====================
Converts between different dataset formats and calculates derived fields.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional


class ExtendedDataFormatter:
    """Format and transform medical data with derived clinical features."""
    
    REQUIRED_BASE_COLUMNS = [
        'patient_id', 'timestamp', 'heart_rate', 'systolic_bp', 'diastolic_bp',
        'spo2', 'respiratory_rate', 'temperature'
    ]
    
    EXTENDED_COLUMNS = [
        'Patient ID', 'Heart Rate', 'Respiratory Rate', 'Timestamp',
        'Body Temperature', 'Oxygen Saturation', 'Systolic Blood Pressure',
        'Diastolic Blood Pressure', 'Age', 'Gender', 'Weight (kg)', 'Height (m)',
        'Derived_HRV', 'Derived_Pulse_Pressure', 'Derived_BMI', 'Derived_MAP',
        'Risk Category'
    ]
    
    def __init__(self):
        """Initialize formatter."""
        self.patient_metadata = {}
    
    def set_patient_metadata(self, patient_id: str, age: int, gender: str, 
                           weight: float, height: float) -> None:
        """
        Store patient demographic info for later use.
        
        Parameters:
            patient_id: Unique patient identifier
            age: Age in years
            gender: M/F/Other
            weight: Weight in kg
            height: Height in meters
        """
        self.patient_metadata[patient_id] = {
            'age': age,
            'gender': gender,
            'weight': weight,
            'height': height
        }
    
    def calculate_derived_fields(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate clinical derived fields.
        
        Parameters:
            df: DataFrame with base vitals
            
        Returns:
            DataFrame with added derived columns
        """
        df = df.copy()
        
        # Pulse Pressure = Systolic - Diastolic
        df['Derived_Pulse_Pressure'] = df['systolic_bp'] - df['diastolic_bp']
        
        # Mean Arterial Pressure (MAP) = Diastolic + (Pulse Pressure / 3)
        df['Derived_MAP'] = (
            df['diastolic_bp'] + (df['Derived_Pulse_Pressure'] / 3)
        )
        
        # Heart Rate Variability (simplified: standard deviation over rolling window)
        window_size = max(3, len(df) // 10)  # Adaptive window
        df['Derived_HRV'] = df['heart_rate'].rolling(
            window=window_size, center=True
        ).std().fillna(0)
        
        # BMI and Body measurements (per patient if available)
        df['Derived_BMI'] = np.nan
        df['Age'] = np.nan
        df['Gender'] = 'Unknown'
        df['Weight (kg)'] = np.nan
        df['Height (m)'] = np.nan
        
        if 'patient_id' in df.columns:
            for patient_id in df['patient_id'].unique():
                mask = df['patient_id'] == patient_id
                if patient_id in self.patient_metadata:
                    meta = self.patient_metadata[patient_id]
                    weight = meta.get('weight', np.nan)
                    height = meta.get('height', np.nan)
                    
                    df.loc[mask, 'Age'] = meta.get('age', np.nan)
                    df.loc[mask, 'Gender'] = meta.get('gender', 'Unknown')
                    df.loc[mask, 'Weight (kg)'] = weight
                    df.loc[mask, 'Height (m)'] = height
                    
                    if height > 0:
                        df.loc[mask, 'Derived_BMI'] = weight / (height ** 2)
        
        # Risk Category based on vitals
        df['Risk Category'] = self._calculate_risk_category(df)
        
        return df
    
    def _calculate_risk_category(self, df: pd.DataFrame) -> pd.Series:
        """
        Assign risk category based on vital signs.
        
        Parameters:
            df: DataFrame with vitals
            
        Returns:
            Series with risk categories
        """
        risk = []
        
        for idx, row in df.iterrows():
            hr = row.get('heart_rate', 0)
            systolic = row.get('systolic_bp', 0)
            diastolic = row.get('diastolic_bp', 0)
            spo2 = row.get('spo2', 100)
            temp = row.get('temperature', 37)
            rr = row.get('respiratory_rate', 16)
            
            score = 0
            
            # Heart rate risk
            if hr < 50 or hr > 120:
                score += 2
            elif hr < 60 or hr > 100:
                score += 1
            
            # Blood pressure risk
            if systolic < 90 or systolic > 160:
                score += 2
            elif systolic < 100 or systolic > 140:
                score += 1
            
            if diastolic < 60 or diastolic > 100:
                score += 1
            
            # SpO2 risk
            if spo2 < 90:
                score += 3
            elif spo2 < 94:
                score += 2
            
            # Temperature risk
            if temp < 36.5 or temp > 38.5:
                score += 2
            elif temp < 37 or temp > 38:
                score += 1
            
            # Respiratory rate risk
            if rr < 12 or rr > 20:
                score += 1
            
            # Assign category
            if score >= 8:
                risk.append('High')
            elif score >= 4:
                risk.append('Medium')
            else:
                risk.append('Low')
        
        return pd.Series(risk, index=df.index)
    
    def convert_to_extended_format(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert base format to extended format with all derived fields.
        
        Parameters:
            df: DataFrame in base format
            
        Returns:
            DataFrame in extended format
        """
        df = df.copy()
        
        # Rename columns to match extended format
        column_mapping = {
            'patient_id': 'Patient ID',
            'heart_rate': 'Heart Rate',
            'respiratory_rate': 'Respiratory Rate',
            'timestamp': 'Timestamp',
            'temperature': 'Body Temperature',
            'spo2': 'Oxygen Saturation',
            'systolic_bp': 'Systolic Blood Pressure',
            'diastolic_bp': 'Diastolic Blood Pressure',
        }
        
        df = df.rename(columns=column_mapping)
        
        # Calculate derived fields
        df = self.calculate_derived_fields(df)
        
        # Select and order columns
        available_cols = [col for col in self.EXTENDED_COLUMNS if col in df.columns]
        df = df[available_cols]
        
        return df
    
    def convert_from_extended_format(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert extended format back to base format.
        
        Parameters:
            df: DataFrame in extended format
            
        Returns:
            DataFrame in base format
        """
        df = df.copy()
        
        column_mapping = {
            'Patient ID': 'patient_id',
            'Heart Rate': 'heart_rate',
            'Respiratory Rate': 'respiratory_rate',
            'Timestamp': 'timestamp',
            'Body Temperature': 'temperature',
            'Oxygen Saturation': 'spo2',
            'Systolic Blood Pressure': 'systolic_bp',
            'Diastolic Blood Pressure': 'diastolic_bp',
        }
        
        # Rename to base format
        for ext_col, base_col in column_mapping.items():
            if ext_col in df.columns:
                df[base_col] = df[ext_col]
        
        # Keep metadata columns if present
        if 'Age' in df.columns:
            df['age'] = df['Age']
        if 'Gender' in df.columns:
            df['gender'] = df['Gender']
        if 'Weight (kg)' in df.columns:
            df['weight'] = df['Weight (kg)']
        if 'Height (m)' in df.columns:
            df['height'] = df['Height (m)']
        
        # Add label if Risk Category exists
        if 'Risk Category' in df.columns:
            df['label'] = (df['Risk Category'] == 'High').astype(int)
        
        base_cols = [col for col in self.REQUIRED_BASE_COLUMNS if col in df.columns]
        extra_cols = ['age', 'gender', 'weight', 'height', 'label']
        extra_cols = [col for col in extra_cols if col in df.columns]
        
        return df[base_cols + extra_cols]
    
    def export_to_csv(self, df: pd.DataFrame, filepath: str, 
                     extended_format: bool = True) -> None:
        """
        Export DataFrame to CSV file.
        
        Parameters:
            df: DataFrame to export
            filepath: Output file path
            extended_format: If True, convert to extended format
        """
        if extended_format:
            df = self.convert_to_extended_format(df)
        
        df.to_csv(filepath, index=False, sep='\t')
        print(f"? Data exported to {filepath}")
    
    def import_from_csv(self, filepath: str, 
                       is_extended_format: bool = True) -> pd.DataFrame:
        """
        Import CSV file.
        
        Parameters:
            filepath: Input file path
            is_extended_format: If True, expects extended format
            
        Returns:
            DataFrame in base format
        """
        df = pd.read_csv(filepath, sep='\t', parse_dates=['timestamp'] if not is_extended_format else ['Timestamp'])
        
        if is_extended_format:
            # Rename Timestamp for consistency before converting
            if 'Timestamp' in df.columns:
                df['Timestamp'] = pd.to_datetime(df['Timestamp'])
                df = self.convert_from_extended_format(df)
        
        print(f"? Data imported from {filepath}")
        return df
