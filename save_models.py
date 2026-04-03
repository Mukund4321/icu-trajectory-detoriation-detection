"""
Re-trains ML models quickly and saves them to disk for backend use.
Run once after main.py has completed: python save_models.py
This is much faster than re-running main.py — loads processed data from disk.
"""
import numpy as np
import joblib
from pathlib import Path

print("Loading processed data (sample for speed)...")
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv("data/processed/icu_data_processed.csv")
df['timestamp'] = pd.to_datetime(df['timestamp'])

from src.feature_engineering import TrajectoryFeatureEngineer

fe = TrajectoryFeatureEngineer(window_size=12, window_step=1)

# Use _split column if present, else split by patient
if '_split' in df.columns:
    df_train = df[df['_split'] == 'train']
    df_test  = df[df['_split'] == 'test']
else:
    all_pids = df['patient_id'].unique()
    np.random.seed(42)
    np.random.shuffle(all_pids)
    n_test = int(len(all_pids) * 0.2)
    test_pids = set(all_pids[:n_test])
    df_train = df[~df['patient_id'].isin(test_pids)]
    df_test  = df[df['patient_id'].isin(test_pids)]

# Sample up to 3000 train patients and 1000 test patients for speed
# (features are identical to full run — just faster to save models)
TRAIN_CAP = 3000
TEST_CAP  = 1000
train_pids = df_train['patient_id'].unique()
test_pids_arr  = df_test['patient_id'].unique()
if len(train_pids) > TRAIN_CAP:
    train_pids = np.random.choice(train_pids, TRAIN_CAP, replace=False)
if len(test_pids_arr) > TEST_CAP:
    test_pids_arr = np.random.choice(test_pids_arr, TEST_CAP, replace=False)

df_train = df_train[df_train['patient_id'].isin(train_pids)]
df_test  = df_test[df_test['patient_id'].isin(test_pids_arr)]

print(f"Train patients: {df_train['patient_id'].nunique()} | Test: {df_test['patient_id'].nunique()}")
X_train, y_train, _ = fe.create_sliding_windows(df_train)
X_test,  y_test,  _ = fe.create_sliding_windows(df_test)
feature_names = fe.get_feature_names()
print(f"Feature dimension: {len(feature_names)}")

from src.ml_models import MLBaselines
ml = MLBaselines(random_seed=42)

# Scale
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

print("\nTraining RandomForest...")
rf = ml.train_random_forest(X_train_s, y_train)

print("Training LogisticRegression...")
lr = ml.train_logistic_regression(X_train_s, y_train)

print("Training XGBoost...")
xgb = ml.train_xgboost(X_train_s, y_train)

print("Training LightGBM...")
lgbm = ml.train_lightgbm(X_train_s, y_train)

# Save
models_dir = Path("models")
models_dir.mkdir(exist_ok=True)

joblib.dump(rf,   models_dir / "randomforest_model.pkl")
joblib.dump(lr,   models_dir / "logisticregression_model.pkl")
joblib.dump(xgb,  models_dir / "xgboost_model.pkl")
joblib.dump(lgbm, models_dir / "lightgbm_model.pkl")
joblib.dump(scaler, models_dir / "ml_scaler.pkl")
joblib.dump(feature_names, models_dir / "feature_names.pkl")

print("\nAll models saved to models/")
print("  randomforest_model.pkl")
print("  logisticregression_model.pkl")
print("  xgboost_model.pkl")
print("  lightgbm_model.pkl")
print("  ml_scaler.pkl")
print("  feature_names.pkl")

# Quick eval
from sklearn.metrics import roc_auc_score
for name, model in [("RF", rf), ("LR", lr), ("XGB", xgb), ("LGB", lgbm)]:
    proba = model.predict_proba(X_test_s)[:, 1]
    auc = roc_auc_score(y_test, proba)
    print(f"  {name} test AUC: {auc:.4f}")
