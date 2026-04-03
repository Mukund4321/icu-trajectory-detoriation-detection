"""
CSV Format Converter
====================
Convert extended format CSV to training format.
"""

import pandas as pd
import sys
from pathlib import Path


def convert_extended_to_training_format(input_file: str, output_file: str) -> None:
    """
    Convert extended format CSV to training format.
    
    Extended format columns:
        Patient ID, Heart Rate, Respiratory Rate, Timestamp, Body Temperature,
        Oxygen Saturation, Systolic Blood Pressure, Diastolic Blood Pressure,
        Age, Gender, Weight (kg), Height (m), Derived_HRV, Derived_Pulse_Pressure,
        Derived_BMI, Derived_MAP, Risk Category
    
    Training format columns:
        patient_id, timestamp, heart_rate, systolic_bp, diastolic_bp,
        spo2, respiratory_rate, temperature, label
    
    Parameters:
        input_file: Path to input CSV (extended format)
        output_file: Path to output CSV (training format)
    """
    print(f"Reading: {input_file}")
    df = pd.read_csv(input_file)
    
    print(f"Loaded {len(df)} records")
    print(f"Columns: {list(df.columns)}")
    
    # Map columns
    df_converted = pd.DataFrame()
    
    df_converted['patient_id'] = df['Patient ID'].astype(str)
    df_converted['timestamp'] = pd.to_datetime(df['Timestamp'])
    df_converted['heart_rate'] = pd.to_numeric(df['Heart Rate'], errors='coerce')
    df_converted['systolic_bp'] = pd.to_numeric(df['Systolic Blood Pressure'], errors='coerce')
    df_converted['diastolic_bp'] = pd.to_numeric(df['Diastolic Blood Pressure'], errors='coerce')
    df_converted['spo2'] = pd.to_numeric(df['Oxygen Saturation'], errors='coerce')
    df_converted['respiratory_rate'] = pd.to_numeric(df['Respiratory Rate'], errors='coerce')
    df_converted['temperature'] = pd.to_numeric(df['Body Temperature'], errors='coerce')
    
    # Create label from Risk Category
    # High risk = 1 (deteriorating), Low/Medium = 0 (stable)
    df_converted['label'] = (df['Risk Category'] == 'High').astype(int)
    
    # Remove any rows with NaN values in critical columns
    critical_cols = ['patient_id', 'timestamp', 'heart_rate', 'systolic_bp', 
                     'diastolic_bp', 'spo2', 'respiratory_rate', 'temperature']
    df_converted = df_converted.dropna(subset=critical_cols)
    
    print(f"\nAfter conversion: {len(df_converted)} records")
    print(f"Label distribution:")
    print(df_converted['label'].value_counts())
    
    # Save
    df_converted.to_csv(output_file, index=False)
    print(f"\n✓ Saved to: {output_file}")
    print(f"Columns: {list(df_converted.columns)}")


if __name__ == "__main__":
    # Usage: python convert_csv.py input.csv output.csv
    
    if len(sys.argv) < 2:
        # Default: convert from data/raw to data/processed
        input_file = "data/raw/human_vital_signs_dataset_2024.csv"
        output_file = "data/processed/training_data.csv"
        print(f"No arguments provided. Using defaults:")
        print(f"  Input: {input_file}")
        print(f"  Output: {output_file}")
    else:
        input_file = sys.argv[1]
        output_file = sys.argv[2] if len(sys.argv) > 2 else "data/processed/training_data.csv"
    
    if not Path(input_file).exists():
        print(f"❌ Input file not found: {input_file}")
        sys.exit(1)
    
    convert_extended_to_training_format(input_file, output_file)
