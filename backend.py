"""
Minimal FastAPI Backend for ICU Trajectory Detection
=====================================================
In-memory backend — no database required.
Loads trained ML models (RF, LR, XGBoost) from models/ if available.
Falls back to trajectory logic when models are not yet trained.

Start with:  python -m uvicorn backend:app --reload --port 8000
"""

from __future__ import annotations

import json
import pickle
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import torch
from src.trajectory_logic import TrajectoryBasedDetector
from src.dl_models import LSTMTrajectoryModel

app = FastAPI(title="ICU Trajectory Backend", version="2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

# ---------------------------------------------------------------------------
# Load trained ML models from disk (produced by main.py)
# ---------------------------------------------------------------------------
MODELS_DIR = Path("models")
_ml_models: Dict[str, Any] = {}
_ml_scaler = None
_feature_names: List[str] = []

def _load_ml_models():
    global _ml_scaler, _feature_names
    try:
        import joblib
        for name in ["randomforest", "logisticregression", "xgboost", "lightgbm"]:
            path = MODELS_DIR / f"{name}_model.pkl"
            if path.exists():
                _ml_models[name] = joblib.load(path)
        scaler_path = MODELS_DIR / "ml_scaler.pkl"
        if scaler_path.exists():
            _ml_scaler = joblib.load(scaler_path)
        fn_path = MODELS_DIR / "feature_names.pkl"
        if fn_path.exists():
            _feature_names = joblib.load(fn_path)
        if _ml_models:
            print(f"[backend] Loaded ML models: {list(_ml_models.keys())}")
        else:
            print("[backend] No ML models found — using trajectory logic only")
    except Exception as e:
        print(f"[backend] Could not load ML models: {e}")

_load_ml_models()

# ---------------------------------------------------------------------------
# Load LSTM model
# ---------------------------------------------------------------------------
_lstm_model = None

def _load_lstm_model():
    global _lstm_model
    try:
        lstm_path = MODELS_DIR / "lstm_model.pt"
        if lstm_path.exists():
            _lstm_model = LSTMTrajectoryModel(input_size=12, hidden_size=64, num_layers=2, dropout=0.3)
            _lstm_model.load_state_dict(torch.load(str(lstm_path), map_location="cpu"))
            _lstm_model.eval()
            print("[backend] LSTM model loaded")
        else:
            print("[backend] No LSTM model found")
    except Exception as e:
        print(f"[backend] Could not load LSTM model: {e}")

_load_lstm_model()

# ---------------------------------------------------------------------------
# Persistent JSON storage — survives backend restarts
# ---------------------------------------------------------------------------
_DB_DIR  = Path("data")
_DB_DIR.mkdir(parents=True, exist_ok=True)
_PATIENTS_FILE = _DB_DIR / "backend_patients.json"
_READINGS_FILE = _DB_DIR / "backend_readings.json"

def _load_store() -> tuple[dict, dict]:
    patients, readings = {}, {}
    try:
        if _PATIENTS_FILE.exists():
            patients = json.loads(_PATIENTS_FILE.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"[backend] Could not load patients: {e}")
    try:
        if _READINGS_FILE.exists():
            readings = json.loads(_READINGS_FILE.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"[backend] Could not load readings: {e}")
    print(f"[backend] Loaded {len(patients)} patients, {sum(len(v) for v in readings.values())} readings from disk")
    return patients, readings

def _save_store():
    try:
        _PATIENTS_FILE.write_text(json.dumps(_patients, default=str), encoding="utf-8")
        _READINGS_FILE.write_text(json.dumps(_readings, default=str), encoding="utf-8")
    except Exception as e:
        print(f"[backend] Could not save store: {e}")

_patients, _readings = _load_store()

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------
class PatientCreate(BaseModel):
    id: str
    name: str = ""
    age: int = 50
    sex: str = "Unknown"
    notes: str = ""
    baseline: Dict[str, float] = {}
    sim_config: Dict[str, Any] = {}
    initial_readings: List[Dict] = []   # NEW: accept bulk readings on create

# ---------------------------------------------------------------------------
# Vital sign constants
# ---------------------------------------------------------------------------
VITALS = ["heart_rate", "systolic_bp", "diastolic_bp",
          "spo2", "respiratory_rate", "temperature"]

_TREND = {"heart_rate": +15, "systolic_bp": -20, "diastolic_bp": -12,
          "spo2": -4, "respiratory_rate": +6, "temperature": +1.5}
_NOISE = {"heart_rate": 1.0, "systolic_bp": 1.0, "diastolic_bp": 1.0,
          "spo2": 0.3, "respiratory_rate": 0.4, "temperature": 0.1}
_CLIP  = {"heart_rate": (35, 190), "systolic_bp": (70, 200), "diastolic_bp": (40, 130),
          "spo2": (80, 100), "respiratory_rate": (6, 45), "temperature": (35, 41)}
_DEFAULTS = {"heart_rate": 75, "systolic_bp": 120, "diastolic_bp": 78,
             "spo2": 97, "respiratory_rate": 16, "temperature": 37.0}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _generate_row(patient_id, timestamp, baseline, trend_severity,
                  noise_level, deterioration_onset, step_index, total_steps, seed):
    rng = np.random.default_rng(seed + step_index)
    t = step_index / max(1, total_steps - 1)
    onset = float(np.clip(deterioration_onset, 0, 1))
    trend_curve = float(np.clip((t - onset) / max(1e-6, 1 - onset), 0, 1))
    trend = trend_severity * trend_curve

    row: Dict[str, Any] = {"patient_id": patient_id, "timestamp": str(timestamp)}
    for v in VITALS:
        base = baseline.get(v, _DEFAULTS[v])
        val = base + _TREND[v] * trend + noise_level * rng.normal(0, _NOISE[v])
        row[v] = float(np.clip(val, *_CLIP[v]))
    row["label"] = int(trend > 0.6) if trend_severity >= 0.3 else 0
    row["step_index"] = step_index
    return row


def _run_detection(rows: List[Dict]) -> List[Dict]:
    if len(rows) < 6:
        for r in rows:
            r.setdefault("instability_score", 0.0)
            r.setdefault("predicted_deterioration", 0)
        return rows

    df = pd.DataFrame(rows)
    for col in VITALS:
        if col not in df.columns:
            df[col] = _DEFAULTS[col]

    detector = TrajectoryBasedDetector(
        baseline_window=min(12, max(4, len(df) // 4)),
        instability_window=3,
        persistence_threshold=2,
    )
    df_flagged = detector.flag_deterioration(df)
    df_flagged["predicted_deterioration"] = (
        pd.to_numeric(df_flagged.get("predicted_deterioration", 0), errors="coerce")
        .fillna(0).astype(int)
    )
    df_flagged["instability_score"] = (
        pd.to_numeric(df_flagged.get("instability_score", 0.0), errors="coerce")
        .fillna(0.0)
    )
    return df_flagged.to_dict(orient="records")


def _build_ml_features(rows: List[Dict]) -> Optional[np.ndarray]:
    """
    Build a single feature vector from recent readings for ML prediction.
    Uses last 12 readings (1 window). Returns None if models not loaded.
    """
    if not _ml_models or _ml_scaler is None:
        return None

    window = rows[-12:]
    if len(window) < 6:
        return None

    df = pd.DataFrame(window)
    features = []

    for v in VITALS:
        vals = df[v].values.astype(float) if v in df.columns else np.zeros(len(df))
        # slope
        x = np.arange(len(vals))
        try:
            slope = float(np.polyfit(x, vals, 1)[0])
        except Exception:
            slope = 0.0
        features.append(slope)                      # slope (z)
        features.append(float(np.std(vals)))        # volatility
        diffs = np.diff(vals)
        denoms = vals[:-1]
        nz = denoms != 0
        roc = float(np.mean(diffs[nz] / denoms[nz])) if np.any(nz) else 0.0
        features.append(roc)                        # rate of change
        features.append(float(np.mean(vals)))       # mean (raw)
        features.append(float(np.min(vals)))        # min
        features.append(float(np.max(vals)))        # max

    # HR-BP correlation
    if 'heart_rate' in df.columns and 'systolic_bp' in df.columns:
        corr = df['heart_rate'].corr(df['systolic_bp'])
        features.append(float(corr) if not np.isnan(corr) else 0.0)
    else:
        features.append(0.0)

    # baseline deviation
    baseline = df[VITALS].iloc[0].values
    nz_mask = baseline != 0
    if np.any(nz_mask):
        diffs_b = df[VITALS].values[:, nz_mask] - baseline[nz_mask]
        dev = float(np.mean(np.abs(diffs_b / baseline[nz_mask])))
    else:
        dev = 0.0
    features.append(dev)

    # Pad remaining features with 0 to match training feature count
    target_len = len(_feature_names) if _feature_names else 54
    while len(features) < target_len:
        features.append(0.0)
    features = features[:target_len]

    return np.array(features, dtype=np.float32).reshape(1, -1)


# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------

@app.get("/")
def root():
    models_loaded = list(_ml_models.keys())
    return {"status": "ok", "patients": len(_patients), "ml_models": models_loaded}


@app.post("/patients")
def create_patient(body: PatientCreate):
    now = datetime.now().isoformat()
    _patients[body.id] = {
        "id": body.id, "name": body.name, "age": body.age,
        "sex": body.sex, "notes": body.notes,
        "created_at": now, "is_active": True,
        "baseline": body.baseline, "sim_config": body.sim_config,
    }
    # Store initial bulk readings sent from the Streamlit simulation
    if body.initial_readings:
        _readings[body.id] = [
            {k: (v.isoformat() if hasattr(v, 'isoformat') else v)
             for k, v in row.items()}
            for row in body.initial_readings
        ]
    else:
        # Auto-generate 12 baseline readings so LSTM has enough history immediately
        cfg = body.sim_config or {}
        baseline = body.baseline or _DEFAULTS.copy()
        noise_level = float(cfg.get("noise_level", 0.3))
        seed = int(cfg.get("random_seed", 42))
        now_dt = datetime.now()
        auto_rows = []
        for i in range(12):
            ts = now_dt - timedelta(minutes=5 * (12 - i))
            row = _generate_row(body.id, ts, baseline,
                                trend_severity=0.0,
                                noise_level=noise_level,
                                deterioration_onset=1.0,
                                step_index=i, total_steps=12, seed=seed)
            row["timestamp"] = ts.isoformat()
            auto_rows.append(row)
        _readings[body.id] = auto_rows
    _save_store()
    return {"status": "created", "patient_id": body.id}


@app.get("/patients")
def list_patients():
    return list(_patients.values())


@app.delete("/patients/{patient_id}")
def delete_patient(patient_id: str):
    if patient_id not in _patients:
        raise HTTPException(404, "Patient not found")
    del _patients[patient_id]
    _readings.pop(patient_id, None)
    _save_store()
    return {"status": "deleted"}


@app.get("/simulations/{patient_id}")
def get_readings(patient_id: str, limit: int = 200):
    rows = _readings.get(patient_id, [])
    result = rows[-limit:] if limit else rows
    return _run_detection(result)


@app.post("/simulations/{patient_id}/step")
def simulate_steps(patient_id: str, steps: int = 1):
    if patient_id not in _patients:
        raise HTTPException(404, "Patient not found")

    patient = _patients[patient_id]
    cfg = patient.get("sim_config", {})
    baseline = patient.get("baseline", _DEFAULTS.copy())
    trend_severity     = float(cfg.get("trend_severity", 0.2))
    noise_level        = float(cfg.get("noise_level", 0.6))
    deterioration_onset = float(cfg.get("deterioration_onset", 0.5))
    interval_minutes   = int(cfg.get("interval_minutes", 5))
    total_steps        = int(cfg.get("total_steps", 500))
    seed               = int(cfg.get("random_seed", 42))

    existing = _readings.get(patient_id, [])
    if existing:
        last_ts = datetime.fromisoformat(str(existing[-1]["timestamp"]).replace("Z", ""))
        step_offset = int(existing[-1].get("step_index", len(existing) - 1)) + 1
    else:
        start_str = cfg.get("start_time", datetime.now().isoformat())
        last_ts = datetime.fromisoformat(start_str) - timedelta(minutes=interval_minutes)
        step_offset = 0

    for i in range(steps):
        ts = last_ts + timedelta(minutes=interval_minutes * (i + 1))
        row = _generate_row(patient_id, ts, baseline, trend_severity,
                            noise_level, deterioration_onset,
                            step_offset + i, total_steps, seed)
        existing.append(row)

    _readings[patient_id] = existing
    _save_store()
    return {"status": "ok", "steps_written": steps, "total_readings": len(existing)}


@app.post("/predict/{patient_id}")
def predict(patient_id: str):
    rows = _readings.get(patient_id, [])
    if not rows:
        return {"risk_level": "Unknown", "ensemble_probability": 0.0}

    # --- ML model prediction (if models loaded) ---
    ml_prob = None
    model_used = "trajectory_logic"
    feat = _build_ml_features(rows)
    if feat is not None and _ml_scaler is not None:
        try:
            feat_scaled = _ml_scaler.transform(feat)
            probas = []
            for name in ["randomforest", "logisticregression", "xgboost"]:
                if name in _ml_models:
                    p = float(_ml_models[name].predict_proba(feat_scaled)[0][1])
                    probas.append(p)
            if probas:
                weights = [0.45, 0.35, 0.20][:len(probas)]
                ml_prob = float(np.average(probas, weights=weights[:len(probas)]))
                model_used = "ml_ensemble"
        except Exception as e:
            print(f"[backend] ML predict error: {e}")

    # --- LSTM prediction ---
    lstm_prob = None
    if _lstm_model is not None and len(rows) >= 12:
        try:
            window = rows[-12:]
            df_w = pd.DataFrame(window)
            seq = np.zeros((12, 12), dtype=np.float32)
            for i, v in enumerate(VITALS):
                if v in df_w.columns:
                    vals = df_w[v].values.astype(np.float32)
                    mean, std = float(np.mean(vals)), float(np.std(vals)) + 1e-6
                    seq[:, i] = (vals - mean) / std          # z-scored
                    seq[:, i + 6] = vals                     # raw
            x = torch.FloatTensor(seq).unsqueeze(0)          # (1, 12, 12)
            with torch.no_grad():
                logit = _lstm_model(x)
                lstm_prob = float(torch.sigmoid(logit).item())
            print(f"[backend] LSTM prob: {lstm_prob:.4f}")
        except Exception as e:
            print(f"[backend] LSTM inference error: {e}")

    # --- Trajectory logic fallback ---
    flagged = _run_detection(rows[-50:])
    det_rate = float(np.mean([r.get("predicted_deterioration", 0) for r in flagged]))
    avg_score = float(np.mean([r.get("instability_score", 0.0) for r in flagged]))
    traj_prob = float(np.clip(0.6 * det_rate + 0.4 * avg_score, 0, 1))

    # Blend: LSTM 40% + ML 35% + trajectory 25% if all available
    if ml_prob is not None and lstm_prob is not None:
        prob = 0.40 * lstm_prob + 0.35 * ml_prob + 0.25 * traj_prob
        model_used = "full_ensemble"
    elif lstm_prob is not None:
        prob = 0.60 * lstm_prob + 0.40 * traj_prob
        model_used = "lstm+trajectory"
    elif ml_prob is not None:
        prob = 0.70 * ml_prob + 0.30 * traj_prob
        model_used = "ml_ensemble"
    else:
        prob = traj_prob

    risk = "High" if prob > 0.5 else "Medium" if prob > 0.25 else "Low"

    return {
        "patient_id": patient_id,
        "risk_level": risk,
        "ensemble_probability": round(prob, 4),
        "ml_probability": round(ml_prob, 4) if ml_prob is not None else None,
        "lstm_probability": round(lstm_prob, 4) if lstm_prob is not None else None,
        "trajectory_probability": round(traj_prob, 4),
        "deterioration_rate": round(det_rate, 4),
        "avg_instability_score": round(avg_score, 4),
        "model_used": model_used,
    }


@app.get("/prognosis/{patient_id}")
def get_prognosis(patient_id: str):
    """
    Project the patient's trajectory forward at multiple time horizons using
    the simulation model parameters + trained ML models.
    Returns risk probability + projected vitals for: 24h, 3d, 1w, 2w, 1m, 3m.
    """
    patient = _patients.get(patient_id)
    if not patient:
        raise HTTPException(404, "Patient not found")

    cfg                 = patient.get("sim_config", {})
    baseline            = patient.get("baseline", _DEFAULTS.copy())
    existing            = _readings.get(patient_id, [])
    interval_minutes    = int(cfg.get("interval_minutes", 5))
    total_steps         = int(cfg.get("total_steps", 500))
    trend_severity      = float(cfg.get("trend_severity", 0.2))
    noise_level         = float(cfg.get("noise_level", 0.6))
    deterioration_onset = float(cfg.get("deterioration_onset", 0.5))
    seed                = int(cfg.get("random_seed", 42))

    current_step = int(existing[-1].get("step_index", len(existing) - 1)) if existing else 0
    if existing:
        last_ts = datetime.fromisoformat(str(existing[-1]["timestamp"]).replace("Z", ""))
    else:
        last_ts = datetime.now()

    steps_per_day  = max(1, int(round(24 * 60 / interval_minutes)))
    # Sample every 6 hours worth of steps to keep memory lean
    sample_interval = max(1, steps_per_day // 4)

    HORIZONS = [
        ("24h",  1,   "24 Hours",  "Immediate risk — sepsis onset, acute cardiac event"),
        ("3d",   3,   "3 Days",    "Short-term — response to treatment, organ function"),
        ("1w",   7,   "1 Week",    "Medium-term — ICU discharge readiness"),
        ("2w",   14,  "2 Weeks",   "Extended stay — infection control, recovery"),
        ("1m",   30,  "1 Month",   "Long-term — rehabilitation potential, mortality risk"),
        ("3m",   90,  "3 Months",  "Prognosis — full recovery vs chronic deterioration"),
    ]

    # Clinical interpretation lookup
    _INTERP = {
        "High":   "Significant deterioration expected. Immediate clinical intervention recommended.",
        "Medium": "Moderate risk. Increased monitoring frequency and early intervention advised.",
        "Low":    "Trajectory appears stable. Maintain standard monitoring protocol.",
    }

    results = []
    # Build projected rows incrementally so each horizon builds on previous
    projected_rows = []

    for key, days, label, context in HORIZONS:
        target_steps = days * steps_per_day
        current_len  = len(projected_rows)

        # Generate rows up to this horizon (from what we have so far)
        for s in range(current_len + sample_interval,
                       target_steps + sample_interval,
                       sample_interval):
            ts  = last_ts + timedelta(minutes=interval_minutes * s)
            row = _generate_row(patient_id, ts, baseline, trend_severity,
                                noise_level, deterioration_onset,
                                current_step + s, total_steps, seed)
            projected_rows.append(row)

        all_rows = existing + projected_rows

        # ML prediction on window ending at this horizon
        ml_prob = None
        feat = _build_ml_features(all_rows)
        if feat is not None and _ml_scaler is not None:
            try:
                feat_scaled = _ml_scaler.transform(feat)
                probas, wts = [], []
                for name, w in [("randomforest", 0.45), ("logisticregression", 0.35), ("xgboost", 0.20)]:
                    if name in _ml_models:
                        probas.append(float(_ml_models[name].predict_proba(feat_scaled)[0][1]))
                        wts.append(w)
                if probas:
                    ml_prob = float(np.average(probas, weights=wts))
            except Exception as exc:
                print(f"[prognosis] ML error at {key}: {exc}")

        # Trajectory probability
        det_rows  = _run_detection(all_rows[-50:])
        det_rate  = float(np.mean([r.get("predicted_deterioration", 0) for r in det_rows]))
        avg_score = float(np.mean([r.get("instability_score", 0.0) for r in det_rows]))
        traj_prob = float(np.clip(0.6 * det_rate + 0.4 * avg_score, 0, 1))

        prob = (0.70 * ml_prob + 0.30 * traj_prob) if ml_prob is not None else traj_prob
        risk = "High" if prob > 0.5 else "Medium" if prob > 0.25 else "Low"

        # Projected vitals at the end of this horizon
        proj_vitals = {}
        if projected_rows:
            last_row = projected_rows[-1]
            proj_vitals = {v: round(float(last_row.get(v, _DEFAULTS[v])), 1) for v in VITALS}

        results.append({
            "horizon_key":               key,
            "horizon_label":             label,
            "clinical_context":          context,
            "days":                      days,
            "deterioration_probability": round(prob, 4),
            "ml_probability":            round(ml_prob, 4) if ml_prob is not None else None,
            "trajectory_probability":    round(traj_prob, 4),
            "risk_level":                risk,
            "interpretation":            _INTERP[risk],
            "projected_vitals":          proj_vitals,
        })

    return {
        "patient_id":       patient_id,
        "current_step":     current_step,
        "interval_minutes": interval_minutes,
        "prognosis":        results,
    }


@app.get("/export/patients-extended")
def export_all():
    rows = []
    for pid, readings in _readings.items():
        patient = _patients.get(pid, {})
        for r in readings:
            rc = dict(r)
            rc.update({"patient_name": patient.get("name", ""),
                       "patient_age": patient.get("age", ""),
                       "patient_sex": patient.get("sex", "")})
            rows.append(rc)
    if not rows:
        return {"data": ""}
    return {"data": pd.DataFrame(rows).to_csv(sep="\t", index=False)}


@app.get("/export/patients-extended/{patient_id}")
def export_patient(patient_id: str):
    rows = _readings.get(patient_id, [])
    if not rows:
        return {"data": ""}
    df = pd.DataFrame(rows)
    patient = _patients.get(patient_id, {})
    df["patient_name"] = patient.get("name", "")
    df["patient_age"]  = patient.get("age", "")
    df["patient_sex"]  = patient.get("sex", "")
    return {"data": df.to_csv(sep="\t", index=False)}
