"""
Data Adapter Module
===================
Adapts multiple data sources to the project's unified schema.

Supported sources:
  1. PhysioNet Challenge 2019 (primary ? longitudinal ICU time-series)
  2. human_vital_signs_dataset_2024.csv (static baseline comparison)

PhysioNet 2019 download instructions:
  1. Register at https://physionet.org/register/
  2. Go to https://physionet.org/content/challenge-2019/1.0.0/
  3. Download training_setA.zip and/or training_setB.zip
  4. Extract into:  data/raw/training_setA/  (contains *.psv files, one per patient)
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, List, Optional

VITAL_COLUMNS = [
    'heart_rate', 'systolic_bp', 'diastolic_bp',
    'spo2', 'respiratory_rate', 'temperature'
]

# PhysioNet 2019 column ? project schema
_PHYSIONET_MAP = {
    'HR':          'heart_rate',
    'O2Sat':       'spo2',
    'Temp':        'temperature',
    'SBP':         'systolic_bp',
    'DBP':         'diastolic_bp',
    'Resp':        'respiratory_rate',
    # SOFA-score lab markers — sparse but highly predictive for sepsis
    'WBC':         'wbc',
    'Lactate':     'lactate',
    'Glucose':     'glucose',
    'Creatinine':  'creatinine',
    'BUN':         'bun',
    # Patient demographics
    'Age':         'age',
    'SepsisLabel': 'label',
}

LAB_COLUMNS = ['wbc', 'lactate', 'glucose', 'creatinine', 'bun']

# human_vital_signs_dataset_2024.csv column ? project schema
_HUMAN_VITALS_MAP = {
    'Patient ID':                 'patient_id',
    'Heart Rate':                 'heart_rate',
    'Respiratory Rate':           'respiratory_rate',
    'Body Temperature':           'temperature',
    'Oxygen Saturation':          'spo2',
    'Systolic Blood Pressure':    'systolic_bp',
    'Diastolic Blood Pressure':   'diastolic_bp',
    'Derived_HRV':                'hrv',
    'Derived_MAP':                'map',
    'Derived_Pulse_Pressure':     'pulse_pressure',
    'Derived_BMI':                'bmi',
    'Risk Category':              'label',
}


# ---------------------------------------------------------------------------
# PhysioNet 2019 loader
# ---------------------------------------------------------------------------

def load_physionet_2019(data_dir: str, max_patients: Optional[int] = None) -> pd.DataFrame:
    """
    Load PhysioNet Challenge 2019 sepsis prediction dataset.

    Parameters:
        data_dir: Directory containing training_setA/ and/or training_setB/ folders
        max_patients: Cap on patients to load (None = all); useful for quick tests

    Returns:
        DataFrame in project schema with columns:
        patient_id, timestamp, heart_rate, systolic_bp, diastolic_bp,
        spo2, respiratory_rate, temperature, label
    """
    data_dir = Path(data_dir)
    all_files: List[Path] = []

    for subset in ['training_setA', 'training_setB', 'training']:
        subset_dir = data_dir / subset
        if subset_dir.exists():
            all_files.extend(sorted(subset_dir.glob('*.psv')))

    # Also search recursively ? handles wget output structure:
    # data/raw/physionet.org/files/challenge-2019/1.0.0/training/training_setA/*.psv
    if not all_files:
        all_files = sorted(data_dir.rglob("*.psv"))

    if not all_files:
        raise FileNotFoundError(
            f"\nPhysioNet 2019 data not found in: {data_dir}\n\n"
            "HOW TO DOWNLOAD (run from project root):\n"
            "  python download_physionet.py          # first 2000 patients (~5 min)\n"
            "  python download_physionet.py --all    # all 20,336 patients\n\n"
            "Falling back to synthetic data."
        )

    if max_patients is not None:
        all_files = all_files[:max_patients]

    records = []
    for filepath in all_files:
        try:
            patient_df = pd.read_csv(filepath, sep='|')
            patient_df['patient_id'] = filepath.stem
            # Each row = 1 hour; build synthetic timestamps from ICU admission
            patient_df['timestamp'] = pd.date_range(
                start='2024-01-01 00:00:00',
                periods=len(patient_df),
                freq='1h'
            )
            records.append(patient_df)
        except Exception:
            continue  # skip malformed files

    if not records:
        raise ValueError("All PhysioNet patient files failed to load.")

    df = pd.concat(records, ignore_index=True)
    df = _apply_physionet_mapping(df)

    print(f"? PhysioNet 2019: loaded {df['patient_id'].nunique()} patients, "
          f"{len(df)} records")
    return df


def _apply_physionet_mapping(df: pd.DataFrame) -> pd.DataFrame:
    """Rename PhysioNet columns, drop unnecessary ones, handle missing values."""
    df = df.rename(columns=_PHYSIONET_MAP)

    # Create binary missingness flags for lab columns BEFORE forward-fill.
    # Whether a lab was drawn is clinically informative: sicker patients get
    # more frequent measurements. Labs measured = 1, not measured = 0.
    for lab in LAB_COLUMNS:
        if lab in df.columns:
            df[f'{lab}_measured'] = (~df[lab].isna()).astype(np.float32)

    required = (['patient_id', 'timestamp'] + VITAL_COLUMNS + LAB_COLUMNS
                + [f'{l}_measured' for l in LAB_COLUMNS]
                + ['age', 'label'])
    df = df[[c for c in required if c in df.columns]].copy()

    # Drop rows where ALL vitals are missing (common in PhysioNet sparse records)
    df = df.dropna(subset=VITAL_COLUMNS, how='all')

    # Forward-fill then backward-fill within each patient
    df = df.sort_values(['patient_id', 'timestamp']).reset_index(drop=True)
    df[VITAL_COLUMNS] = (
        df.groupby('patient_id')[VITAL_COLUMNS]
        .transform(lambda s: s.ffill().bfill())
    )

    # Forward-fill age (constant per patient, may appear only on first row)
    if 'age' in df.columns:
        df['age'] = df.groupby('patient_id')['age'].transform(lambda s: s.ffill().bfill())
        df['age'] = df['age'].fillna(df['age'].median())

    # Drop rows that still have any vital NaN after fill
    df = df.dropna(subset=VITAL_COLUMNS).reset_index(drop=True)

    # Label: 0/1 int, default 0 if missing
    if 'label' in df.columns:
        df['label'] = df['label'].fillna(0).astype(int)
    else:
        df['label'] = 0

    return df


# ---------------------------------------------------------------------------
# Static snapshot loader (human_vital_signs_dataset_2024.csv)
# ---------------------------------------------------------------------------

def load_human_vitals_static(filepath: str) -> pd.DataFrame:
    """
    Load human_vital_signs_dataset_2024.csv as a static snapshot dataset.

    This dataset has 1 row per patient (no time-series).
    Use ONLY for snapshot-based baseline model comparison in IEEE paper,
    NOT for trajectory analysis.

    Returns:
        DataFrame with vitals + derived features + binary label
    """
    df = pd.read_csv(filepath)
    df = df.rename(columns=_HUMAN_VITALS_MAP)

    # Binary label: High Risk = 1, Low Risk = 0
    df['label'] = (df['label'] == 'High Risk').astype(int)

    keep = ['patient_id'] + VITAL_COLUMNS + ['label', 'hrv', 'map', 'pulse_pressure', 'bmi']
    df = df[[c for c in keep if c in df.columns]].copy()

    print(f"? Static snapshot dataset: {len(df)} patients, "
          f"High Risk: {df['label'].sum()} ({100*df['label'].mean():.1f}%)")
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Early-warning label extension
# ---------------------------------------------------------------------------

def apply_early_warning_labels(df: pd.DataFrame, horizon_hours: int = 12) -> pd.DataFrame:
    """
    Extend positive labels backward to create an early-warning horizon.

    PhysioNet 2019 ships with SepsisLabel=1 only for the 6 hours immediately
    before/at sepsis onset (~2.5% positive windows). This gives the model very
    little training signal and skews evaluation toward the trivial all-negative
    predictor.

    This function relabels: if a patient's sepsis onset is at hour T, any window
    from hour (T - horizon_hours) onward is marked positive. The result is a
    larger positive class (~5-8%) representing "patient will develop sepsis
    within the next horizon_hours hours" — a clinically meaningful early-warning
    task that mirrors real deployment.

    Parameters:
        df: DataFrame with 'patient_id', 'timestamp', 'label' columns
        horizon_hours: How many hours before first onset to extend labels.
                       Default 12 = 12-hour early warning window (includes the
                       existing 6h PhysioNet window + 6h additional look-ahead).

    Returns:
        DataFrame with extended binary labels
    """
    df = df.copy().sort_values(['patient_id', 'timestamp']).reset_index(drop=True)

    new_labels = df['label'].copy()

    for patient_id, group in df.groupby('patient_id'):
        pos_idx = group.index[group['label'] == 1]
        if len(pos_idx) == 0:
            continue  # non-sepsis patient: labels stay 0

        # First onset timestamp
        first_onset_ts = df.loc[pos_idx[0], 'timestamp']

        # Mark every row from (onset - horizon_hours) onward as positive
        cutoff_ts = first_onset_ts - pd.Timedelta(hours=horizon_hours)
        extend_mask = (df['patient_id'] == patient_id) & (df['timestamp'] >= cutoff_ts)
        new_labels[extend_mask] = 1

    before_pos = int(df['label'].sum())
    df['label'] = new_labels.astype(int)
    after_pos = int(df['label'].sum())

    print(f"Early-warning labels extended ({horizon_hours}h horizon): "
          f"{before_pos} -> {after_pos} positive records "
          f"({100*df['label'].mean():.1f}% positive)")
    return df


# ---------------------------------------------------------------------------
# Vital-sign based physiological deterioration labels
# ---------------------------------------------------------------------------

def apply_deterioration_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replace sepsis labels with vital-sign based physiological deterioration labels.

    A row is labelled deteriorating (1) if ANY of the following clinical
    thresholds are breached — these are standard ICU early-warning criteria:

        Heart Rate      : < 40 or > 120 bpm
        Systolic BP     : < 90 mmHg
        SpO2            : < 94 %
        Respiratory Rate: > 24 or < 8 /min
        Temperature     : > 38.5 or < 35.5 °C

    This produces ~15-25% positive rate vs 5% for SepsisLabel, giving the
    model far more training signal and producing clinically meaningful results
    that generalise beyond sepsis to any acute deterioration.
    """
    df = df.copy()

    hr   = pd.to_numeric(df['heart_rate'],       errors='coerce')
    sbp  = pd.to_numeric(df['systolic_bp'],       errors='coerce')
    spo2 = pd.to_numeric(df['spo2'],              errors='coerce')
    rr   = pd.to_numeric(df['respiratory_rate'],  errors='coerce')
    temp = pd.to_numeric(df['temperature'],        errors='coerce')

    deteriorating = (
        (hr   <  40) | (hr   > 120) |
        (sbp  <  90)                |
        (spo2 <  94)                |
        (rr   >  24) | (rr   <   8) |
        (temp > 38.5) | (temp < 35.5)
    )

    before_pos = int(df['label'].sum())
    df['label'] = deteriorating.astype(int)
    after_pos  = int(df['label'].sum())

    print(f"Deterioration labels applied (vital-sign thresholds): "
          f"{before_pos} sepsis -> {after_pos} deterioration records "
          f"({100 * df['label'].mean():.1f}% positive)")
    return df


# ---------------------------------------------------------------------------
# Patient-level train / val / test split
# ---------------------------------------------------------------------------

def patient_level_split(
    df: pd.DataFrame,
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_seed: int = 42,
    stratify_by_outcome: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split dataset by patient_id ? NOT by individual records or windows.

    Why this matters:
        Splitting windows randomly means windows from the same patient appear
        in both train and test, causing the model to memorize patient-specific
        patterns. This inflates all metrics significantly.

    Parameters:
        df: Full dataset with 'patient_id' and 'label' columns
        test_size: Fraction of patients for test set
        val_size: Fraction of patients for validation set
        random_seed: For reproducibility
        stratify_by_outcome: If True, preserve label ratio across splits
                             (each patient gets 1 if ANY record is positive)

    Returns:
        (df_train, df_val, df_test)
    """
    np.random.seed(random_seed)

    # One label per patient: positive if any record is deteriorating
    patient_labels = df.groupby('patient_id')['label'].max().reset_index()
    patient_labels.columns = ['patient_id', 'patient_label']

    if stratify_by_outcome:
        pos_patients = patient_labels[patient_labels['patient_label'] == 1]['patient_id'].values
        neg_patients = patient_labels[patient_labels['patient_label'] == 0]['patient_id'].values
        np.random.shuffle(pos_patients)
        np.random.shuffle(neg_patients)

        def _split_group(ids):
            n = len(ids)
            n_test = max(1, int(n * test_size))
            n_val  = max(1, int(n * val_size))
            return ids[:n_test], ids[n_test:n_test + n_val], ids[n_test + n_val:]

        pos_test, pos_val, pos_train = _split_group(pos_patients)
        neg_test, neg_val, neg_train = _split_group(neg_patients)

        test_ids  = set(pos_test)  | set(neg_test)
        val_ids   = set(pos_val)   | set(neg_val)
        train_ids = set(pos_train) | set(neg_train)
    else:
        all_ids = patient_labels['patient_id'].values.copy()
        np.random.shuffle(all_ids)
        n = len(all_ids)
        n_test = int(n * test_size)
        n_val  = int(n * val_size)
        test_ids  = set(all_ids[:n_test])
        val_ids   = set(all_ids[n_test:n_test + n_val])
        train_ids = set(all_ids[n_test + n_val:])

    df_test  = df[df['patient_id'].isin(test_ids)].reset_index(drop=True)
    df_val   = df[df['patient_id'].isin(val_ids)].reset_index(drop=True)
    df_train = df[df['patient_id'].isin(train_ids)].reset_index(drop=True)

    print(f"\nPatient-level split (stratified={stratify_by_outcome}):")
    print(f"  Train: {len(train_ids):>5} patients | {len(df_train):>7} records | "
          f"pos rate {100*df_train['label'].mean():.1f}%")
    print(f"  Val:   {len(val_ids):>5} patients | {len(df_val):>7} records | "
          f"pos rate {100*df_val['label'].mean():.1f}%")
    print(f"  Test:  {len(test_ids):>5} patients | {len(df_test):>7} records | "
          f"pos rate {100*df_test['label'].mean():.1f}%")

    return df_train, df_val, df_test
