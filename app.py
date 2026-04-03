"""
ICU Trajectory Detection - Interactive Web Application
=======================================================
Streamlit interface for patient simulation and stability detection.

Run with: streamlit run app.py
"""

import json
from pathlib import Path
import requests

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Dict, Optional

from src.trajectory_logic import TrajectoryBasedDetector

# Backend API configuration
BACKEND_URL = "http://localhost:8000"

# Backend API wrapper functions
def create_patient_api(patient_id: str, name: str, age: int, sex: str, notes: str, baseline: Dict, sim_config: Dict) -> bool:
    """Create patient via backend API"""
    try:
        payload = {
            "id": patient_id,
            "name": name,
            "age": age,
            "sex": sex,
            "notes": notes,
            "baseline": baseline,
            "sim_config": sim_config
        }
        resp = requests.post(f"{BACKEND_URL}/patients", json=payload, timeout=5)
        if resp.status_code == 200:
            st.success(f"Patient {patient_id} created in backend!")
            return True
        else:
            st.error(f"Backend error: {resp.status_code} - {resp.text}")
            return False
    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to backend. Is it running on port 8000?")
        return False
    except Exception as e:
        st.error(f"Error creating patient: {e}")
        return False

def get_all_patients_api() -> Optional[list]:
    """Get all patients from backend API"""
    try:
        resp = requests.get(f"{BACKEND_URL}/patients", timeout=15)
        if resp.status_code == 200:
            return resp.json()
        else:
            st.warning(f"Could not fetch patients: {resp.status_code}")
            return None
    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to backend. Make sure it's running on port 8000")
        return None
    except Exception as e:
        st.error(f"Error fetching patients: {e}")
        return None

def get_patient_readings_api(patient_id: str, limit: int = 200) -> Optional[pd.DataFrame]:
    """Get patient readings from backend API"""
    try:
        resp = requests.get(f"{BACKEND_URL}/simulations/{patient_id}?limit={limit}", timeout=5)
        if resp.status_code == 200:
            readings = resp.json()
            if not readings:
                return None
            df = pd.DataFrame(readings)
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df = df.sort_values('timestamp').reset_index(drop=True)

            needs_detection = (
                'predicted_deterioration' not in df.columns
                or 'instability_score' not in df.columns
                or df['predicted_deterioration'].isna().any()
                or df['instability_score'].isna().any()
            )
            if needs_detection:
                df = run_stability_detection(df)

            if 'predicted_deterioration' in df.columns:
                df['predicted_deterioration'] = pd.to_numeric(df['predicted_deterioration'], errors='coerce').fillna(0).astype(int)
            if 'instability_score' in df.columns:
                df['instability_score'] = pd.to_numeric(df['instability_score'], errors='coerce').fillna(0.0)
            return df
        else:
            return None
    except Exception as e:
        st.warning(f"Error fetching readings: {e}")
        return None


def simulate_patient_step_api(patient_id: str, steps: int = 1, show_success: bool = True) -> bool:
    """Generate simulation points in backend for a patient"""
    try:
        resp = requests.post(f"{BACKEND_URL}/simulations/{patient_id}/step?steps={steps}", timeout=8)
        if resp.status_code == 200:
            data = resp.json()
            if show_success:
                st.success(f"Added {data.get('steps_written', steps)} new readings for {patient_id}.")
            return True
        st.error(f"Simulation failed: {resp.status_code} - {resp.text}")
        return False
    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to backend. Make sure it's running on port 8000")
        return False
    except Exception as e:
        st.error(f"Error simulating patient: {e}")
        return False


def delete_patient_api(patient_id: str) -> bool:
    """Delete patient and associated readings via backend API"""
    try:
        resp = requests.delete(f"{BACKEND_URL}/patients/{patient_id}", timeout=5)
        if resp.status_code == 200:
            st.success(f"Patient {patient_id} deleted from backend.")
            return True
        st.error(f"Delete failed: {resp.status_code} - {resp.text}")
        return False
    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to backend. Make sure it's running on port 8000")
        return False
    except Exception as e:
        st.error(f"Error deleting patient: {e}")
        return False


def export_patient_extended_format_api(patient_id: Optional[str] = None) -> Optional[str]:
    """Export patient data in extended format with derived fields"""
    try:
        if patient_id:
            resp = requests.get(f"{BACKEND_URL}/export/patients-extended/{patient_id}", timeout=10)
        else:
            resp = requests.get(f"{BACKEND_URL}/export/patients-extended", timeout=10)
        
        if resp.status_code == 200:
            data = resp.json()
            return data.get('data')
        else:
            st.error(f"Export failed: {resp.status_code}")
            return None
    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to backend. Make sure it's running on port 8000")
        return None
    except Exception as e:
        st.error(f"Error exporting data: {e}")
        return None


def predict_deterioration_api(patient_id: str) -> Optional[Dict]:
    """Get ML model predictions for patient deterioration"""
    try:
        resp = requests.post(f"{BACKEND_URL}/predict/{patient_id}", timeout=8)
        if resp.status_code == 200:
            return resp.json()
        return None
    except:
        return None


def get_prognosis_api(patient_id: str) -> Optional[Dict]:
    """Fetch multi-horizon prognosis from backend."""
    try:
        resp = requests.get(f"{BACKEND_URL}/prognosis/{patient_id}", timeout=15)
        if resp.status_code == 200:
            return resp.json()
        st.warning(f"Prognosis endpoint returned {resp.status_code}: {resp.text[:200]}")
        return None
    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to backend on port 8000.")
        return None
    except Exception as e:
        st.warning(f"Prognosis error: {e}")
        return None


st.set_page_config(
    page_title="ICU Trajectory Detection",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    .stApp {
        background: #ffffff;
        color: #1a1a1a;
        font-family: 'Inter', sans-serif;
    }

    body, p, div, span, li {
        color: #1a1a1a !important;
    }

    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        background: #ffffff;
        max-width: 1200px;
        color: #1a1a1a;
    }

    .main-header {
        font-family: 'Inter', sans-serif;
        font-size: 42px;
        font-weight: 600;
        color: #111827;
        margin-bottom: 8px;
    }

    .sub-header {
        font-size: 16px;
        color: #374151;
        margin-bottom: 32px;
    }

    .card {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 14px rgba(0,0,0,0.06);
    }

    .metric-card {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        padding: 16px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.06);
        text-align: center;
    }

    .metric-label {
        font-size: 12px;
        text-transform: uppercase;
        letter-spacing: 0.7px;
        color: #6b7280;
        font-weight: 600;
    }

    .metric-value {
        font-size: 26px;
        font-weight: 700;
        color: #111827;
        margin-top: 6px;
    }

    .info-box {
        background: #f0f9ff;
        border: 1px solid #bae6fd;
        border-radius: 12px;
        padding: 16px 20px;
        color: #0c2340;
    }

    .warning-box {
        background: #fef3c7;
        border: 1px solid #fcd34d;
        border-radius: 12px;
        padding: 16px 20px;
        color: #54210e;
    }

    .section-divider {
        margin: 28px 0;
        border-bottom: 1px solid #e5e7eb;
    }

    .stButton > button {
        background: #2563eb !important;
        color: #FFFFFF !important;
        border: 2px solid #1e40af !important;
        border-radius: 8px;
        padding: 16px 32px;
        font-size: 18px;
        font-weight: 1000 !important;
        font-family: 'Inter', sans-serif;
        transition: all 0.2s ease;
        box-shadow: 0 6px 16px rgba(37, 99, 235, 0.4);
        letter-spacing: 1px;
    }

    .stButton > button:hover {
        background: #1e40af !important;
        color: #FFFFFF !important;
        box-shadow: 0 8px 24px rgba(37, 99, 235, 0.6);
        transform: translateY(-2px);
    }

    .stButton > button span,
    .stButton > button p,
    .stButton > button div,
    .stButton > button * {
        color: #FFFFFF !important;
        font-weight: 1000 !important;
        font-size: 18px !important;
    }

    [data-testid="stButton"],
    [data-testid="stButton"] * {
        color: #FFFFFF !important;
        font-weight: 1000 !important;
    }

    [data-testid="stSidebar"] {
        background: #f9fafb;
        border-right: 1px solid #e5e7eb;
    }

    /* ── Suppress Streamlit's translucent loading overlay on every rerun ── */
    [data-testid="stStatusWidget"]  { display: none !important; }
    [data-testid="stDecoration"]    { display: none !important; }
    [data-testid="stToolbar"]       { display: none !important; }
    .stApp                          { opacity: 1 !important; transition: none !important; }
    .stMainBlockContainer::after    { display: none !important; }
    div[class*="overlayOverlay"]    { display: none !important; }
    /* Keep all blocks fully visible even while script is running */
    [data-testid="stVerticalBlock"],
    [data-testid="stHorizontalBlock"],
    section[data-testid="stMain"]   { opacity: 1 !important; transition: none !important; }
    </style>
""", unsafe_allow_html=True)

# JS: mutation observer that forcibly keeps opacity=1 on every DOM change Streamlit makes
st.markdown("""
<script>
(function(){
  function killOverlay(){
    document.querySelectorAll('[data-testid="stStatusWidget"],[data-testid="stDecoration"]')
      .forEach(el => el.style.display = 'none');
    // Remove any semi-transparent overlay Streamlit adds while running
    document.querySelectorAll('[class*="overlayOverlay"],[class*="StatusWidget"]')
      .forEach(el => el.style.display = 'none');
    // Ensure the main container never fades
    const main = document.querySelector('section[data-testid="stMain"]');
    if(main) main.style.opacity = '1';
  }
  const obs = new MutationObserver(killOverlay);
  obs.observe(document.body, {subtree: true, childList: true, attributes: true});
  killOverlay();
})();
</script>
""", unsafe_allow_html=True)


DATA_DIR = Path("data")
SIM_DIR = DATA_DIR / "simulations"
PATIENTS_FILE = DATA_DIR / "patients.json"


def ensure_storage() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    SIM_DIR.mkdir(parents=True, exist_ok=True)


def load_patient_registry() -> Dict[str, Dict]:
    if not PATIENTS_FILE.exists():
        return {}
    try:
        with PATIENTS_FILE.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
            return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def save_patient_registry(registry: Dict[str, Dict]) -> None:
    ensure_storage()
    with PATIENTS_FILE.open("w", encoding="utf-8") as handle:
        json.dump(registry, handle, indent=2)


def simulation_path(patient_id: str) -> Path:
    safe_id = patient_id.replace(" ", "_")
    return SIM_DIR / f"{safe_id}.csv"


def load_simulation(patient_id: str) -> pd.DataFrame | None:
    path = simulation_path(patient_id)
    if not path.exists():
        return None
    return pd.read_csv(path, parse_dates=['timestamp'])


def save_simulation(patient_id: str, df: pd.DataFrame) -> None:
    ensure_storage()
    path = simulation_path(patient_id)
    df.to_csv(path, index=False)


def simulate_patient_timeseries(
    patient_id: str,
    start_time: datetime,
    duration_hours: int,
    interval_minutes: int,
    baseline: Dict[str, float],
    trend_severity: float,
    noise_level: float,
    deterioration_onset: float,
    random_seed: int
) -> pd.DataFrame:
    rng = np.random.default_rng(random_seed)
    periods = int((duration_hours * 60) / interval_minutes) + 1
    timestamps = pd.date_range(start=start_time, periods=periods, freq=f"{interval_minutes}min")
    t = np.linspace(0, 1, periods)

    onset = np.clip(deterioration_onset, 0.0, 1.0)
    trend_curve = np.clip((t - onset) / max(1e-6, 1 - onset), 0, 1)
    trend = trend_severity * trend_curve

    heart_rate = baseline['heart_rate'] + 15 * trend + noise_level * rng.normal(0, 1, periods)
    systolic_bp = baseline['systolic_bp'] - 20 * trend + noise_level * rng.normal(0, 1, periods)
    diastolic_bp = baseline['diastolic_bp'] - 12 * trend + noise_level * rng.normal(0, 1, periods)
    spo2 = baseline['spo2'] - 4 * trend + noise_level * rng.normal(0, 0.3, periods)
    respiratory_rate = baseline['respiratory_rate'] + 6 * trend + noise_level * rng.normal(0, 0.4, periods)
    temperature = baseline['temperature'] + 1.5 * trend + noise_level * rng.normal(0, 0.1, periods)

    heart_rate = np.clip(heart_rate, 35, 190)
    systolic_bp = np.clip(systolic_bp, 70, 200)
    diastolic_bp = np.clip(diastolic_bp, 40, 130)
    spo2 = np.clip(spo2, 80, 100)
    respiratory_rate = np.clip(respiratory_rate, 6, 45)
    temperature = np.clip(temperature, 35, 41)

    label = (trend > 0.6).astype(int) if trend_severity >= 0.3 else np.zeros_like(trend, dtype=int)

    return pd.DataFrame({
        'patient_id': patient_id,
        'timestamp': timestamps,
        'heart_rate': heart_rate,
        'systolic_bp': systolic_bp,
        'diastolic_bp': diastolic_bp,
        'spo2': spo2,
        'respiratory_rate': respiratory_rate,
        'temperature': temperature,
        'label': label
    })


def generate_live_row(
    patient_id: str,
    timestamp: pd.Timestamp,
    baseline: Dict[str, float],
    trend_severity: float,
    noise_level: float,
    deterioration_onset: float,
    step_index: int,
    total_steps: int,
    random_seed: int
) -> Dict[str, float]:
    rng = np.random.default_rng(random_seed + step_index)
    t = step_index / max(1, total_steps - 1)
    onset = np.clip(deterioration_onset, 0.0, 1.0)
    trend_curve = np.clip((t - onset) / max(1e-6, 1 - onset), 0, 1)
    trend = trend_severity * trend_curve

    heart_rate = baseline['heart_rate'] + 15 * trend + noise_level * rng.normal(0, 1)
    systolic_bp = baseline['systolic_bp'] - 20 * trend + noise_level * rng.normal(0, 1)
    diastolic_bp = baseline['diastolic_bp'] - 12 * trend + noise_level * rng.normal(0, 1)
    spo2 = baseline['spo2'] - 4 * trend + noise_level * rng.normal(0, 0.3)
    respiratory_rate = baseline['respiratory_rate'] + 6 * trend + noise_level * rng.normal(0, 0.4)
    temperature = baseline['temperature'] + 1.5 * trend + noise_level * rng.normal(0, 0.1)

    heart_rate = float(np.clip(heart_rate, 35, 190))
    systolic_bp = float(np.clip(systolic_bp, 70, 200))
    diastolic_bp = float(np.clip(diastolic_bp, 40, 130))
    spo2 = float(np.clip(spo2, 80, 100))
    respiratory_rate = float(np.clip(respiratory_rate, 6, 45))
    temperature = float(np.clip(temperature, 35, 41))

    label = int(trend > 0.6) if trend_severity >= 0.3 else 0

    return {
        'patient_id': patient_id,
        'timestamp': timestamp,
        'heart_rate': heart_rate,
        'systolic_bp': systolic_bp,
        'diastolic_bp': diastolic_bp,
        'spo2': spo2,
        'respiratory_rate': respiratory_rate,
        'temperature': temperature,
        'label': label
    }


def run_stability_detection(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if len(df) < 6:
        df['predicted_deterioration'] = 0
        df['instability_score'] = 0.0
        return df

    baseline_window = min(12, max(4, len(df) // 4))
    detector = TrajectoryBasedDetector(
        baseline_window=baseline_window,
        instability_window=3,
        persistence_threshold=2
    )
    return detector.flag_deterioration(df)


def summarize_stability(df: pd.DataFrame) -> Dict[str, float]:
    if 'predicted_deterioration' not in df.columns:
        return {
            'status': 'Unknown',
            'stability_score': 0.0,
            'deterioration_rate': 0.0,
            'instability_avg': 0.0
        }

    pred_series = pd.to_numeric(df.get('predicted_deterioration', 0), errors='coerce').fillna(0)
    inst_series = pd.to_numeric(df.get('instability_score', 0.0), errors='coerce').fillna(0.0)
    deterioration_rate = float(pred_series.mean()) if len(pred_series) else 0.0
    instability_avg = float(inst_series.mean()) if len(inst_series) else 0.0
    stability_score = max(0.0, 1.0 - deterioration_rate)
    status = 'Stable' if deterioration_rate < 0.2 else 'Unstable'
    return {
        'status': status,
        'stability_score': stability_score,
        'deterioration_rate': deterioration_rate,
        'instability_avg': instability_avg
    }


# Clinical reference ranges
_NORMS = {
    'heart_rate':       (60,  100,  'bpm',         'Tachycardia (>100) or Bradycardia (<60)'),
    'systolic_bp':      (90,  140,  'mmHg',        'Hypotension (<90) or Hypertension (>140)'),
    'diastolic_bp':     (60,  90,   'mmHg',        'Hypotension (<60) or Hypertension (>90)'),
    'spo2':             (95,  100,  '%',            'Hypoxia — SpO2 below 95%'),
    'respiratory_rate': (12,  20,   'breaths/min', 'Bradypnea (<12) or Tachypnea (>20)'),
    'temperature':      (36.5, 38.0,'°C',          'Hypothermia (<36.5) or Fever (>38.0)'),
}

def analyze_deterioration_reasons(df: pd.DataFrame) -> list:
    """Return list of dicts describing each abnormal vital and trend direction."""
    if df.empty:
        return []

    reasons = []
    recent = df.tail(5)
    latest = df.iloc[-1]

    for col, (lo, hi, unit, label) in _NORMS.items():
        if col not in df.columns:
            continue
        val = pd.to_numeric(latest.get(col), errors='coerce')
        if pd.isna(val):
            continue

        abnormal = val < lo or val > hi
        # Trend: compare latest vs 5-step-ago mean
        trend_vals = pd.to_numeric(recent[col], errors='coerce').dropna()
        if len(trend_vals) >= 2:
            delta = float(trend_vals.iloc[-1] - trend_vals.iloc[0])
            if abs(delta) < 0.5:
                trend = 'stable'
            elif delta > 0:
                trend = 'rising'
            else:
                trend = 'falling'
        else:
            trend = 'stable'

        severity = 'critical' if (val < lo * 0.85 or val > hi * 1.15) else ('warning' if abnormal else 'normal')

        reasons.append({
            'vital':    col.replace('_', ' ').title(),
            'value':    val,
            'unit':     unit,
            'low':      lo,
            'high':     hi,
            'abnormal': abnormal,
            'trend':    trend,
            'severity': severity,
            'label':    label,
        })

    return reasons


def render_deterioration_panel(df: pd.DataFrame) -> None:
    """Show a colour-coded vital-signs status panel with trend arrows and deterioration reasons."""
    reasons = analyze_deterioration_reasons(df)
    if not reasons:
        return

    st.markdown("### Clinical Status & Deterioration Analysis")

    # Top risk banner
    abnormal_vitals = [r for r in reasons if r['abnormal']]
    if abnormal_vitals:
        flags = ', '.join(r['vital'] for r in abnormal_vitals)
        st.markdown(f"""
        <div style='background:#fef2f2;border-left:4px solid #ef4444;border-radius:8px;
                    padding:12px 16px;margin-bottom:12px;'>
          <strong style='color:#991b1b;font-size:15px;'>⚠ Deterioration Indicators Detected</strong><br>
          <span style='color:#7f1d1d;font-size:13px;'>Abnormal vitals: {flags}</span>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style='background:#f0fdf4;border-left:4px solid #10b981;border-radius:8px;
                    padding:12px 16px;margin-bottom:12px;'>
          <strong style='color:#065f46;font-size:15px;'>✓ All Vitals Within Normal Range</strong>
        </div>""", unsafe_allow_html=True)

    # Vital cards
    trend_icon = {'rising': '↑', 'falling': '↓', 'stable': '→'}
    sev_colors = {
        'critical': ('#fef2f2', '#ef4444', '#991b1b'),
        'warning':  ('#fffbeb', '#f59e0b', '#78350f'),
        'normal':   ('#f0fdf4', '#10b981', '#065f46'),
    }

    cols = st.columns(3)
    for i, r in enumerate(reasons):
        bg, border, text = sev_colors[r['severity']]
        icon = trend_icon[r['trend']]
        with cols[i % 3]:
            st.markdown(f"""
            <div style='background:{bg};border:1px solid {border};border-radius:8px;
                        padding:10px 12px;margin-bottom:8px;'>
              <div style='color:{text};font-size:11px;font-weight:600;text-transform:uppercase;
                          letter-spacing:0.5px;'>{r['vital']}</div>
              <div style='font-size:22px;font-weight:700;color:#111;'>
                {r['value']:.1f} <span style='font-size:12px;color:#6b7280;'>{r['unit']}</span>
                <span style='font-size:16px;color:{border};'>{icon}</span>
              </div>
              <div style='font-size:11px;color:#6b7280;'>Normal: {r['low']}–{r['high']} {r['unit']}</div>
              {f"<div style='font-size:11px;color:{text};margin-top:4px;'>⚠ {r['label']}</div>" if r['abnormal'] else ""}
            </div>""", unsafe_allow_html=True)

    # Root-cause explanation
    if abnormal_vitals:
        st.markdown("#### Root Cause Analysis")
        for r in abnormal_vitals:
            direction = "elevated above" if r['value'] > r['high'] else "below"
            bound     = r['high'] if r['value'] > r['high'] else r['low']
            deviation = abs(r['value'] - bound)
            trend_str = f"and is **{r['trend']}**" if r['trend'] != 'stable' else "(currently stable)"
            sev_label = "**CRITICAL**" if r['severity'] == 'critical' else "**Warning**"
            st.markdown(
                f"- {sev_label} — **{r['vital']}** is {direction} normal "
                f"({r['value']:.1f} vs limit {bound} {r['unit']}, Δ={deviation:.1f}) {trend_str}. "
                f"*{r['label']}*"
            )


def _build_prognosis_html(prognosis_data: Optional[dict]) -> str:
    """Build the prognosis section for the HTML report."""
    if not prognosis_data:
        return ""
    items = prognosis_data.get('prognosis', [])
    if not items:
        return ""

    risk_colors = {'High': '#ef4444', 'Medium': '#f59e0b', 'Low': '#10b981'}
    risk_bg     = {'High': '#fef2f2', 'Medium': '#fffbeb', 'Low': '#f0fdf4'}

    rows = ''
    for h in items:
        rc    = risk_colors.get(h['risk_level'], '#6b7280')
        bg    = risk_bg.get(h['risk_level'], '#fff')
        pv    = h.get('projected_vitals', {})
        ml_td = f"{h['ml_probability']*100:.1f}%" if h.get('ml_probability') is not None else 'N/A'
        rows += (
            f"<tr style='background:{bg};'>"
            f"<td style='padding:6px 10px;font-weight:600;'>{h['horizon_label']}</td>"
            f"<td style='padding:6px 10px;color:{rc};font-weight:700;'>{h['risk_level']}</td>"
            f"<td style='padding:6px 10px;font-weight:700;'>{h['deterioration_probability']*100:.1f}%</td>"
            f"<td style='padding:6px 10px;'>{ml_td}</td>"
            f"<td style='padding:6px 10px;'>{pv.get('heart_rate','—')}</td>"
            f"<td style='padding:6px 10px;'>{pv.get('systolic_bp','—')}/{pv.get('diastolic_bp','—')}</td>"
            f"<td style='padding:6px 10px;'>{pv.get('spo2','—')}%</td>"
            f"<td style='padding:6px 10px;'>{pv.get('respiratory_rate','—')}</td>"
            f"<td style='padding:6px 10px;color:#6b7280;font-size:11px;'>{h['interpretation']}</td>"
            f"</tr>"
        )

    return f"""
<div class='card'>
  <h2 style='margin-top:0;color:#1e40af;'>Future Course Prediction (ML Model)</h2>
  <p style='color:#6b7280;font-size:13px;'>
    Projected using RF + LR + XGBoost ensemble applied to forward-simulated vitals.
    For clinical decision support only.
  </p>
  <table style='width:100%;border-collapse:collapse;'>
    <thead style='background:#eff6ff;'>
      <tr>
        <th style='padding:8px 10px;text-align:left;color:#1e40af;'>Horizon</th>
        <th style='padding:8px 10px;text-align:left;color:#1e40af;'>Risk</th>
        <th style='padding:8px 10px;text-align:left;color:#1e40af;'>Det. Prob.</th>
        <th style='padding:8px 10px;text-align:left;color:#1e40af;'>ML Prob.</th>
        <th style='padding:8px 10px;text-align:left;color:#1e40af;'>HR (bpm)</th>
        <th style='padding:8px 10px;text-align:left;color:#1e40af;'>BP (mmHg)</th>
        <th style='padding:8px 10px;text-align:left;color:#1e40af;'>SpO₂</th>
        <th style='padding:8px 10px;text-align:left;color:#1e40af;'>RR</th>
        <th style='padding:8px 10px;text-align:left;color:#1e40af;'>Clinical Note</th>
      </tr>
    </thead>
    <tbody>{rows}</tbody>
  </table>
</div>"""


def generate_patient_report(patient_info: dict, df: pd.DataFrame, predictions: Optional[dict],
                            prognosis_data: Optional[dict] = None) -> str:
    """Generate a detailed HTML patient report as a string."""
    from datetime import datetime as _dt
    now_str    = _dt.now().strftime('%Y-%m-%d %H:%M:%S')
    summary    = summarize_stability(df)
    reasons    = analyze_deterioration_reasons(df)
    abnormal   = [r for r in reasons if r['abnormal']]
    risk_level = predictions.get('risk_level', 'Unknown') if predictions else 'N/A'
    prob       = predictions.get('ensemble_probability', 0.0) if predictions else 0.0
    model_used = predictions.get('model_used', 'N/A') if predictions else 'N/A'
    risk_color = '#ef4444' if risk_level == 'High' else '#f59e0b' if risk_level == 'Medium' else '#10b981'

    # Latest vitals
    latest = df.iloc[-1] if not df.empty else {}

    def vrow(label, col, unit=''):
        val = latest.get(col, 'N/A')
        try:
            val = f"{float(val):.1f} {unit}".strip()
        except Exception:
            pass
        norm = _NORMS.get(col)
        lo, hi = (norm[0], norm[1]) if norm else (None, None)
        try:
            fv = float(latest.get(col, 'N/A'))
            flag = '' if (lo is None or lo <= fv <= hi) else ' ⚠'
            color = '#ef4444' if flag else '#111827'
        except Exception:
            flag, color = '', '#111827'
        return f"<tr><td style='padding:6px 12px;color:#6b7280;'>{label}</td><td style='padding:6px 12px;color:{color};font-weight:600;'>{val}{flag}</td></tr>"

    vital_rows = (
        vrow('Heart Rate',        'heart_rate',       'bpm')
        + vrow('Systolic BP',     'systolic_bp',      'mmHg')
        + vrow('Diastolic BP',    'diastolic_bp',     'mmHg')
        + vrow('SpO₂',           'spo2',             '%')
        + vrow('Respiratory Rate','respiratory_rate', 'br/min')
        + vrow('Temperature',     'temperature',      '°C')
    )

    abnormal_section = ''
    if abnormal:
        rows = ''.join(
            f"<tr><td style='padding:6px 12px;'>{r['vital']}</td>"
            f"<td style='padding:6px 12px;color:#ef4444;font-weight:600;'>{r['value']:.1f} {r['unit']}</td>"
            f"<td style='padding:6px 12px;color:#6b7280;'>{r['low']}–{r['high']} {r['unit']}</td>"
            f"<td style='padding:6px 12px;color:#f59e0b;'>{r['trend'].capitalize()}</td>"
            f"<td style='padding:6px 12px;color:#7f1d1d;font-size:12px;'>{r['label']}</td></tr>"
            for r in abnormal
        )
        abnormal_section = f"""
        <h2 style='color:#991b1b;'>Deterioration Indicators</h2>
        <table style='width:100%;border-collapse:collapse;border:1px solid #fca5a5;border-radius:8px;overflow:hidden;'>
          <thead style='background:#fef2f2;'>
            <tr>
              <th style='padding:8px 12px;text-align:left;color:#991b1b;'>Vital</th>
              <th style='padding:8px 12px;text-align:left;color:#991b1b;'>Value</th>
              <th style='padding:8px 12px;text-align:left;color:#991b1b;'>Normal Range</th>
              <th style='padding:8px 12px;text-align:left;color:#991b1b;'>Trend</th>
              <th style='padding:8px 12px;text-align:left;color:#991b1b;'>Clinical Note</th>
            </tr>
          </thead>
          <tbody>{rows}</tbody>
        </table>"""

    history_rows = ''
    for _, row in df.tail(20).iterrows():
        det = int(row.get('predicted_deterioration', 0))
        row_color = '#fef2f2' if det else '#ffffff'
        ts = str(row.get('timestamp', ''))[:16]
        history_rows += (
            f"<tr style='background:{row_color};'>"
            f"<td style='padding:5px 10px;'>{ts}</td>"
            f"<td style='padding:5px 10px;'>{_safe_float(row.get('heart_rate'))}</td>"
            f"<td style='padding:5px 10px;'>{_safe_float(row.get('systolic_bp'))}/{_safe_float(row.get('diastolic_bp'))}</td>"
            f"<td style='padding:5px 10px;'>{_safe_float(row.get('spo2'))}</td>"
            f"<td style='padding:5px 10px;'>{_safe_float(row.get('respiratory_rate'))}</td>"
            f"<td style='padding:5px 10px;'>{_safe_float(row.get('temperature'))}</td>"
            f"<td style='padding:5px 10px;color:{'#ef4444' if det else '#10b981'};font-weight:600;'>{'⚠ Unstable' if det else '✓ Stable'}</td>"
            f"</tr>"
        )

    html = f"""<!DOCTYPE html>
<html lang='en'><head><meta charset='UTF-8'>
<title>ICU Patient Report — {patient_info.get('id','')}</title>
<style>
  body{{font-family:'Segoe UI',Arial,sans-serif;background:#f9fafb;color:#111827;margin:0;padding:32px;}}
  h1{{color:#1e40af;border-bottom:2px solid #1e40af;padding-bottom:8px;}}
  h2{{color:#374151;margin-top:28px;}}
  .card{{background:#fff;border:1px solid #e5e7eb;border-radius:10px;padding:20px;margin-bottom:20px;box-shadow:0 2px 8px rgba(0,0,0,0.05);}}
  table{{width:100%;border-collapse:collapse;}}
  th,td{{text-align:left;border-bottom:1px solid #e5e7eb;}}
  .risk-badge{{display:inline-block;padding:4px 14px;border-radius:20px;font-weight:700;font-size:16px;color:#fff;background:{risk_color};}}
  footer{{margin-top:40px;color:#9ca3af;font-size:12px;border-top:1px solid #e5e7eb;padding-top:12px;}}
</style></head>
<body>
<h1>ICU Patient Deterioration Report</h1>
<p style='color:#6b7280;font-size:13px;'>Generated: {now_str} &nbsp;|&nbsp; System: ICU Trajectory Detection v1.0</p>

<div class='card'>
  <h2 style='margin-top:0;'>Patient Demographics</h2>
  <table>
    <tr><td style='padding:6px 12px;color:#6b7280;'>Patient ID</td><td style='padding:6px 12px;font-weight:600;'>{patient_info.get('id','N/A')}</td></tr>
    <tr><td style='padding:6px 12px;color:#6b7280;'>Name</td><td style='padding:6px 12px;font-weight:600;'>{patient_info.get('name','N/A')}</td></tr>
    <tr><td style='padding:6px 12px;color:#6b7280;'>Age</td><td style='padding:6px 12px;font-weight:600;'>{patient_info.get('age','N/A')}</td></tr>
    <tr><td style='padding:6px 12px;color:#6b7280;'>Sex</td><td style='padding:6px 12px;font-weight:600;'>{patient_info.get('sex','N/A')}</td></tr>
    <tr><td style='padding:6px 12px;color:#6b7280;'>Notes</td><td style='padding:6px 12px;'>{patient_info.get('notes','—')}</td></tr>
    <tr><td style='padding:6px 12px;color:#6b7280;'>Admitted</td><td style='padding:6px 12px;'>{patient_info.get('created_at','N/A')}</td></tr>
    <tr><td style='padding:6px 12px;color:#6b7280;'>Status</td><td style='padding:6px 12px;'>{"Active" if patient_info.get('is_active') else "Inactive"}</td></tr>
  </table>
</div>

<div class='card'>
  <h2 style='margin-top:0;'>AI Risk Assessment</h2>
  <p>Risk Level: <span class='risk-badge'>{risk_level}</span></p>
  <table>
    <tr><td style='padding:6px 12px;color:#6b7280;'>Deterioration Probability</td><td style='padding:6px 12px;font-weight:700;font-size:18px;'>{prob*100:.1f}%</td></tr>
    <tr><td style='padding:6px 12px;color:#6b7280;'>Stability Status</td><td style='padding:6px 12px;'>{summary['status']}</td></tr>
    <tr><td style='padding:6px 12px;color:#6b7280;'>Stability Score</td><td style='padding:6px 12px;'>{summary['stability_score']:.2f}</td></tr>
    <tr><td style='padding:6px 12px;color:#6b7280;'>Deterioration Rate</td><td style='padding:6px 12px;'>{summary['deterioration_rate']*100:.1f}%</td></tr>
    <tr><td style='padding:6px 12px;color:#6b7280;'>Avg Instability Score</td><td style='padding:6px 12px;'>{summary['instability_avg']:.3f}</td></tr>
    <tr><td style='padding:6px 12px;color:#6b7280;'>Model Used</td><td style='padding:6px 12px;'>{model_used}</td></tr>
  </table>
</div>

<div class='card'>
  <h2 style='margin-top:0;'>Latest Vital Signs</h2>
  <table>{vital_rows}</table>
</div>

{f"<div class='card'>{abnormal_section}</div>" if abnormal_section else ""}

<div class='card'>
  <h2 style='margin-top:0;'>Vital Signs History (Last 20 Readings)</h2>
  <table>
    <thead style='background:#f3f4f6;'>
      <tr>
        <th style='padding:8px 10px;'>Time</th>
        <th style='padding:8px 10px;'>HR (bpm)</th>
        <th style='padding:8px 10px;'>BP (mmHg)</th>
        <th style='padding:8px 10px;'>SpO₂ (%)</th>
        <th style='padding:8px 10px;'>RR (br/min)</th>
        <th style='padding:8px 10px;'>Temp (°C)</th>
        <th style='padding:8px 10px;'>Status</th>
      </tr>
    </thead>
    <tbody>{history_rows}</tbody>
  </table>
</div>

{_build_prognosis_html(prognosis_data)}

<footer>
  ICU Trajectory Detection System &nbsp;|&nbsp; AI-powered early warning &nbsp;|&nbsp;
  <strong>For clinical decision support only — not a substitute for physician judgement.</strong>
</footer>
</body></html>"""
    return html


def _safe_float(v, fmt='.1f') -> str:
    try:
        return format(float(v), fmt)
    except Exception:
        return 'N/A'


def render_prognosis_panel(prognosis_data: dict) -> None:
    """Render multi-horizon future course timeline fetched from ML backend."""
    items = prognosis_data.get('prognosis', [])
    if not items:
        st.info("No prognosis data available.")
        return

    st.markdown("### Future Course Prediction")
    st.caption(
        "Projected using trained Random Forest + Logistic Regression + XGBoost ensemble "
        "applied to forward-simulated vitals at each time horizon."
    )

    # ── Probability trend chart ──────────────────────────────────────────────
    labels = [h['horizon_label'] for h in items]
    probs  = [round(h['deterioration_probability'] * 100, 1) for h in items]
    risks  = [h['risk_level'] for h in items]
    colors = ['#ef4444' if r == 'High' else '#f59e0b' if r == 'Medium' else '#10b981' for r in risks]

    fig, ax = plt.subplots(figsize=(12, 3.5))
    bars = ax.bar(labels, probs, color=colors, edgecolor='white', linewidth=0.8)
    ax.set_ylabel('Deterioration Probability (%)', fontsize=11)
    ax.set_title('Predicted Deterioration Risk Over Time', fontsize=13, fontweight='bold')
    ax.set_ylim(0, 100)
    ax.axhline(50, color='#ef4444', linestyle='--', linewidth=0.8, alpha=0.6, label='High risk threshold (50%)')
    ax.axhline(25, color='#f59e0b', linestyle='--', linewidth=0.8, alpha=0.6, label='Medium risk threshold (25%)')
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    for bar, prob in zip(bars, probs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f'{prob:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # ── Horizon cards ────────────────────────────────────────────────────────
    risk_styles = {
        'High':   ('background:#fef2f2;border-left:4px solid #ef4444;', '#991b1b', '⚠'),
        'Medium': ('background:#fffbeb;border-left:4px solid #f59e0b;', '#78350f', '⚡'),
        'Low':    ('background:#f0fdf4;border-left:4px solid #10b981;', '#065f46', '✓'),
    }

    cols = st.columns(3)
    for i, h in enumerate(items):
        style, text_color, icon = risk_styles[h['risk_level']]
        prob_pct = h['deterioration_probability'] * 100
        ml_line  = (f"<div style='font-size:11px;color:#6b7280;'>ML: {h['ml_probability']*100:.1f}%"
                    f" | Traj: {h['trajectory_probability']*100:.1f}%</div>"
                    if h.get('ml_probability') is not None else "")
        pv = h.get('projected_vitals', {})
        vital_preview = (
            f"HR {pv.get('heart_rate','—')} · "
            f"BP {pv.get('systolic_bp','—')}/{pv.get('diastolic_bp','—')} · "
            f"SpO₂ {pv.get('spo2','—')}%"
        ) if pv else ""

        with cols[i % 3]:
            st.markdown(f"""
            <div style='{style}border-radius:8px;padding:12px 14px;margin-bottom:10px;'>
              <div style='color:{text_color};font-size:12px;font-weight:700;
                          text-transform:uppercase;letter-spacing:0.5px;'>
                {icon} {h['horizon_label']}
              </div>
              <div style='font-size:26px;font-weight:800;color:#111;margin:4px 0;'>
                {prob_pct:.1f}%
                <span style='font-size:13px;color:{text_color};font-weight:600;'>{h['risk_level']}</span>
              </div>
              {ml_line}
              <div style='font-size:11px;color:#6b7280;margin-top:4px;'>{vital_preview}</div>
              <div style='font-size:11px;color:{text_color};margin-top:6px;font-style:italic;'>
                {h['interpretation']}
              </div>
            </div>""", unsafe_allow_html=True)

    # ── Projected vitals table ────────────────────────────────────────────────
    with st.expander("Projected Vital Signs at Each Horizon"):
        rows = []
        for h in items:
            pv = h.get('projected_vitals', {})
            rows.append({
                'Horizon':          h['horizon_label'],
                'Risk':             h['risk_level'],
                'Prob (%)':         f"{h['deterioration_probability']*100:.1f}",
                'HR (bpm)':         _safe_float(pv.get('heart_rate')),
                'SBP (mmHg)':       _safe_float(pv.get('systolic_bp')),
                'DBP (mmHg)':       _safe_float(pv.get('diastolic_bp')),
                'SpO₂ (%)':        _safe_float(pv.get('spo2')),
                'RR (br/min)':      _safe_float(pv.get('respiratory_rate')),
                'Temp (°C)':        _safe_float(pv.get('temperature')),
                'Clinical Context': h['clinical_context'],
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def render_live_ecg(df: pd.DataFrame, patient_id: str) -> None:
    """Scrolling ECG that plays through the actual HR trajectory from simulation data."""
    if df.empty or 'heart_rate' not in df.columns:
        return

    hr_series = pd.to_numeric(df['heart_rate'], errors='coerce').dropna()
    if hr_series.empty:
        return

    # Pass up to last 80 HR values so animation tracks the real trajectory
    hr_vals   = [round(float(v), 1) for v in hr_series.tail(80).tolist()]
    latest_hr = hr_vals[-1]
    hr_json   = json.dumps(hr_vals)

    # Unique nonce forces iframe to rebuild every call (so new HR data is picked up)
    import time as _t
    nonce = int(_t.time() * 1000)

    ecg_html = f"""<!DOCTYPE html><html><head><style>
      body{{margin:0;background:transparent;}}
      .ecg-wrap{{background:#0d1117;border-radius:8px;padding:10px 12px;border:1px solid #21262d;}}
      .ecg-title{{color:#58a6ff;font-family:'Courier New',monospace;font-size:13px;
                  display:flex;justify-content:space-between;margin-bottom:6px;}}
      .hr-pill{{background:#388bfd26;color:#58a6ff;padding:2px 10px;border-radius:12px;font-weight:700;}}
      canvas{{display:block;width:100%;}}
    </style></head><body>
    <div class="ecg-wrap">
      <div class="ecg-title">
        <span>&#9679; LIVE ECG &mdash; {patient_id}</span>
        <span class="hr-pill" id="hrLabel">HR: {latest_hr:.0f} bpm</span>
      </div>
      <canvas id="ecg" height="110"></canvas>
    </div>
    <script>/* {nonce} */
    (function(){{
      const canvas = document.getElementById('ecg');
      const ctx    = canvas.getContext('2d');
      const PPS    = 160;   // pixels per second scroll speed
      const H      = canvas.height;
      const mid    = H * 0.50;
      const amp    = H * 0.40;

      // Full HR trajectory from Python simulation data
      const hrValues   = {hr_json};
      let   hrIdx      = 0;
      let   currentHR  = hrValues[0];
      let   beat       = 60.0 / currentHR;
      // Advance one HR sample every HR_STEP seconds of animation
      const HR_STEP    = 0.6;
      let   hrTimer    = 0;

      function resize(){{ canvas.width = canvas.offsetWidth || 900; }}
      resize();
      window.addEventListener('resize', resize);

      function ecgAt(t){{
        const ph = (t % beat) / beat;
        const p  =  0.08 * Math.exp(-0.5*Math.pow((ph-0.18)/0.040,2));
        const q  = -0.12 * Math.exp(-0.5*Math.pow((ph-0.38)/0.012,2));
        const r  =  1.00 * Math.exp(-0.5*Math.pow((ph-0.40)/0.010,2));
        const s  = -0.20 * Math.exp(-0.5*Math.pow((ph-0.43)/0.014,2));
        const tw =  0.25 * Math.exp(-0.5*Math.pow((ph-0.68)/0.070,2));
        const bw =  0.02 * Math.sin(2*Math.PI*0.25*t);
        const n  = (Math.random()-0.5)*0.015;
        return p+q+r+s+tw+bw+n;
      }}

      let buf = new Float32Array(canvas.width);
      let t   = 0;
      for(let i=0;i<buf.length;i++){{ t+=1/PPS; buf[i]=ecgAt(t); }}

      let last = null;
      function frame(ts){{
        if(!last) last=ts;
        const dt      = Math.min((ts-last)/1000, 0.1);
        last = ts;
        const advance = Math.max(1, Math.round(dt*PPS));
        const W = canvas.width;

        // Advance HR along trajectory
        hrTimer += dt;
        if(hrTimer >= HR_STEP){{
          hrTimer = 0;
          if(hrIdx < hrValues.length - 1) hrIdx++;
          currentHR = hrValues[hrIdx];
          beat = 60.0 / currentHR;
          document.getElementById('hrLabel').textContent = 'HR: ' + Math.round(currentHR) + ' bpm';
          // Color the pill red if HR is abnormal (>100 or <60)
          const pill = document.getElementById('hrLabel');
          pill.style.color = (currentHR > 100 || currentHR < 60) ? '#ef4444' : '#58a6ff';
          pill.style.background = (currentHR > 100 || currentHR < 60) ? '#ef444426' : '#388bfd26';
        }}

        if(buf.length !== W){{ buf = new Float32Array(W); t=0; for(let i=0;i<W;i++){{t+=1/PPS;buf[i]=ecgAt(t);}} }}

        for(let i=0;i<advance;i++){{ t+=1/PPS; buf.copyWithin(0,1); buf[W-1]=ecgAt(t); }}

        ctx.fillStyle='#0d1117'; ctx.fillRect(0,0,W,H);

        ctx.strokeStyle='rgba(0,255,65,0.07)'; ctx.lineWidth=0.5;
        for(let gx=0;gx<W;gx+=50){{ ctx.beginPath();ctx.moveTo(gx,0);ctx.lineTo(gx,H);ctx.stroke(); }}
        for(let gy=0;gy<H;gy+=22){{ ctx.beginPath();ctx.moveTo(0,gy);ctx.lineTo(W,gy);ctx.stroke(); }}

        ctx.strokeStyle='rgba(0,255,65,0.3)'; ctx.lineWidth=2;
        ctx.beginPath();ctx.moveTo(W-1,0);ctx.lineTo(W-1,H);ctx.stroke();

        const abnormal = (currentHR > 100 || currentHR < 60);
        ctx.strokeStyle = abnormal ? '#ef4444' : '#00ff41';
        ctx.lineWidth=1.8;
        ctx.shadowColor = ctx.strokeStyle; ctx.shadowBlur=4;
        ctx.beginPath();
        for(let i=0;i<W;i++){{
          const y=mid-buf[i]*amp;
          if(i===0) ctx.moveTo(i,y); else ctx.lineTo(i,y);
        }}
        ctx.stroke(); ctx.shadowBlur=0;
        requestAnimationFrame(frame);
      }}
      requestAnimationFrame(frame);
    }})();
    </script></body></html>"""

    components.html(ecg_html, height=165)


def render_patient_vitals(df: pd.DataFrame, title: str) -> None:
    st.markdown(f"### {title}")
    summary = summarize_stability(df)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Stability Status", summary['status'])
    col2.metric("Stability Score", f"{summary['stability_score']:.2f}")
    col3.metric("Deterioration Rate", f"{summary['deterioration_rate']*100:.1f}%")
    col4.metric("Avg Instability", f"{summary['instability_avg']:.2f}")

    vitals = ['heart_rate', 'systolic_bp', 'diastolic_bp', 'spo2', 'respiratory_rate', 'temperature']
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    axes = axes.flatten()

    deterioration_mask = df.get('predicted_deterioration', pd.Series([0] * len(df))) == 1

    for idx, vital in enumerate(vitals):
        axes[idx].plot(
            range(len(df)),
            df[vital].values,
            marker='o',
            linewidth=2,
            markersize=4,
            color='steelblue'
        )
        if deterioration_mask.any():
            axes[idx].scatter(
                np.where(deterioration_mask)[0],
                df[vital].values[deterioration_mask],
                color='red',
                s=70,
                marker='x',
                linewidth=2,
                label='Predicted Deterioration',
                zorder=5
            )
        axes[idx].set_title(vital.replace('_', ' ').title(), fontweight='bold', fontsize=12)
        axes[idx].set_xlabel('Time Index')
        axes[idx].set_ylabel('Value')
        axes[idx].grid(True, alpha=0.3)
        if deterioration_mask.any():
            axes[idx].legend()

    plt.tight_layout()
    st.pyplot(fig)


def append_live_row(patient_id: str, registry: Dict[str, Dict], df: pd.DataFrame) -> pd.DataFrame:
    sim_config = registry.get(patient_id, {}).get('simulation', {})
    if not sim_config:
        return df

    interval_minutes = int(sim_config.get('interval_minutes', 5))
    trend_severity = float(sim_config.get('trend_severity', 0.2))
    noise_level = float(sim_config.get('noise_level', 0.6))
    deterioration_onset = float(sim_config.get('deterioration_onset', 0.5))
    baseline = sim_config.get('baseline', {})
    total_steps = int(sim_config.get('total_steps', 500))
    step_index = int(sim_config.get('step_index', len(df)))
    random_seed = int(sim_config.get('random_seed', 42))

    if df.empty:
        last_timestamp = pd.to_datetime(sim_config.get('start_time', datetime.now()))
    else:
        last_timestamp = pd.to_datetime(df['timestamp'].iloc[-1])

    next_timestamp = last_timestamp + pd.Timedelta(minutes=interval_minutes)
    row = generate_live_row(
        patient_id=patient_id,
        timestamp=next_timestamp,
        baseline=baseline,
        trend_severity=trend_severity,
        noise_level=noise_level,
        deterioration_onset=deterioration_onset,
        step_index=step_index,
        total_steps=total_steps,
        random_seed=random_seed
    )

    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df = run_stability_detection(df)

    sim_config['step_index'] = step_index + 1
    registry[patient_id]['simulation'] = sim_config
    return df


ensure_storage()

if 'patient_registry' not in st.session_state:
    st.session_state.patient_registry = load_patient_registry()
if 'patient_data_store' not in st.session_state:
    st.session_state.patient_data_store = {}
if 'last_simulation_id' not in st.session_state:
    st.session_state.last_simulation_id = None
if 'live_sim_running' not in st.session_state:
    st.session_state.live_sim_running = False
if 'live_sim_state' not in st.session_state:
    st.session_state.live_sim_state = {}
if 'last_live_update' not in st.session_state:
    st.session_state.last_live_update = 0.0

st.markdown('<div class="main-header">ICU Trajectory Detection System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">AI-Powered Early Warning System for Patient Monitoring</div>', unsafe_allow_html=True)

st.markdown("<div class='info-box'><strong>Navigation:</strong> Use the tabs below if the sidebar is hidden.</div>", unsafe_allow_html=True)
page = st.radio(
    "Main Navigation",
    ["Home", "Patient Simulation", "Patient Records"],
    horizontal=True,
    label_visibility="collapsed"
)

with st.sidebar:
    st.markdown("""
        <div style='text-align: center; padding: 20px 0;'>
            <h2 style='color: #1a1a1a; font-family: Inter; font-weight: 600; margin: 0;'>ICU Monitor</h2>
            <p style='color: #374151; font-size: 14px; margin-top: 8px;'>Patient Analysis System</p>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
    st.markdown("<h3 style='color: #1a1a1a; font-family: Inter; font-weight: 600; font-size: 14px; text-transform: uppercase; letter-spacing: 0.5px;'>Settings</h3>", unsafe_allow_html=True)

    random_seed = st.number_input("Random Seed", value=42, min_value=0)

    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
    st.markdown("<h3 style='color: #1a1a1a; font-family: Inter; font-weight: 600; font-size: 14px; text-transform: uppercase; letter-spacing: 0.5px;'>Session</h3>", unsafe_allow_html=True)

    if st.button("Reset Session", use_container_width=True):
        st.session_state.clear()
        st.rerun()


if page == "Home":
    st.markdown("## Patient Simulation & Stability Detection")

    st.markdown("""
        <div class='info-box'>
            <h3 style='margin-top: 0;'>Frontend Workflow</h3>
            <p style='margin-bottom: 0;'><strong>1.</strong> Add a patient record</p>
            <p style='margin-bottom: 0;'><strong>2.</strong> Simulate pulse and vital trajectories</p>
            <p style='margin-bottom: 0;'><strong>3.</strong> Detect stability in real time</p>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
            <div class='card'>
                <h3 style='text-align: center; color: #1a1a1a;'>Patient Records</h3>
                <ul style='color: #374151; font-weight: 500; line-height: 1.8;'>
                    <li>Manual patient entry</li>
                    <li>Session registry</li>
                    <li>Editable demographics</li>
                    <li>Simulation history</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
            <div class='card'>
                <h3 style='text-align: center; color: #1a1a1a;'>Simulation</h3>
                <ul style='color: #374151; font-weight: 500; line-height: 1.8;'>
                    <li>Pulse + vital trends</li>
                    <li>Noise control</li>
                    <li>Deterioration onset</li>
                    <li>Clinical ranges</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
            <div class='card'>
                <h3 style='text-align: center; color: #1a1a1a;'>Detection</h3>
                <ul style='color: #374151; font-weight: 500; line-height: 1.8;'>
                    <li>Stability status</li>
                    <li>Instability score</li>
                    <li>Deterioration rate</li>
                    <li>Trajectory flags</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
    st.markdown("## Quick Start")

    steps = [
        ("1", "Add Patient", "Use 'Patient Simulation' to add a new patient record"),
        ("2", "Simulate", "Generate pulse and vital trajectories with severity settings"),
        ("3", "Detect", "Review stability status and instability scores"),
        ("4", "Review", "Open 'Patient Records' to compare simulations")
    ]

    for num, title, desc in steps:
        st.markdown(f"""
            <div class='card'>
                <h3 style='color: #1a1a1a; margin-bottom: 8px;'>Step {num}: {title}</h3>
                <p style='color: #374151; font-size: 15px; line-height: 1.6; margin: 0;'>{desc}</p>
            </div>
        """, unsafe_allow_html=True)

elif page == "Patient Simulation":
    st.markdown("## Add Patient Record & Simulate Vitals")

    st.markdown("""
        <div class='info-box'>
            <h4 style='margin-top: 0;'>Create a patient record, then simulate pulse and vitals.</h4>
            <p style='margin-bottom: 0;'>All training happens in the backend. The frontend focuses on simulation and stability detection.</p>
        </div>
    """, unsafe_allow_html=True)

    default_patient_id = f"PT-{len(st.session_state.patient_registry) + 1:03d}"

    with st.form("patient_simulation_form"):
        col1, col2 = st.columns(2)

        with col1:
            patient_id = st.text_input("Patient ID", value=default_patient_id)
            patient_name = st.text_input("Patient Name", value="")
            age = st.number_input("Age", min_value=0, max_value=120, value=55)
            sex = st.selectbox("Sex", ["Female", "Male", "Other"])
            notes = st.text_input("Notes", value="")

        with col2:
            start_date = st.date_input("Simulation Start Date", value=datetime.now().date())
            start_time = st.time_input("Simulation Start Time", value=datetime.now().time())
            duration_hours = st.slider("Duration (hours)", 1, 24, 6)
            interval_minutes = st.selectbox("Interval (minutes)", [5, 10, 15], index=0)

        st.markdown("### Baseline Vitals")
        b1, b2, b3 = st.columns(3)
        with b1:
            heart_rate = st.number_input("Heart Rate (bpm)", 40, 140, 78)
            systolic_bp = st.number_input("Systolic BP", 80, 180, 120)
        with b2:
            diastolic_bp = st.number_input("Diastolic BP", 50, 120, 78)
            spo2 = st.number_input("SpO2 (%)", 85, 100, 97)
        with b3:
            respiratory_rate = st.number_input("Respiratory Rate", 8, 30, 16)
            temperature = st.number_input("Temperature (C)", 35, 40, 37)

        st.markdown("### Simulation Dynamics")
        d1, d2, d3 = st.columns(3)
        with d1:
            trend_severity = st.slider("Trend Severity", 0.0, 1.0, 0.2, 0.05)
        with d2:
            noise_level = st.slider("Noise Level", 0.0, 2.0, 0.6, 0.1)
        with d3:
            deterioration_onset = st.slider("Deterioration Onset", 0.0, 0.9, 0.5, 0.05)

        submitted = st.form_submit_button("Save Patient & Simulate")

    if submitted:
        if not patient_id.strip():
            st.warning("Please provide a Patient ID.")
        else:
            start_dt = datetime.combine(start_date, start_time)
            sim_config = {
                'start_time': start_dt.isoformat(),
                'interval_minutes': int(interval_minutes),
                'trend_severity': float(trend_severity),
                'noise_level': float(noise_level),
                'deterioration_onset': float(deterioration_onset),
                'random_seed': int(random_seed)
            }
            
            baseline = {
                'heart_rate': float(heart_rate),
                'systolic_bp': float(systolic_bp),
                'diastolic_bp': float(diastolic_bp),
                'spo2': float(spo2),
                'respiratory_rate': float(respiratory_rate),
                'temperature': float(temperature)
            }

            # Generate simulation first (so we can send readings with the patient)
            sim_config['baseline'] = baseline
            sim_config['total_steps'] = max(500, int(duration_hours * 60 / interval_minutes))
            sim_config['step_index'] = 0

            df_sim = simulate_patient_timeseries(
                patient_id=patient_id,
                start_time=start_dt,
                duration_hours=duration_hours,
                interval_minutes=interval_minutes,
                baseline=baseline,
                trend_severity=trend_severity,
                noise_level=noise_level,
                deterioration_onset=deterioration_onset,
                random_seed=int(random_seed)
            )
            df_flags = run_stability_detection(df_sim)

            # Convert readings to JSON-serialisable list and send to backend
            readings_payload = []
            for _, row in df_flags.iterrows():
                r = {}
                for col in df_flags.columns:
                    val = row[col]
                    if hasattr(val, 'isoformat'):
                        val = val.isoformat()
                    elif hasattr(val, 'item'):
                        val = val.item()
                    r[col] = val
                readings_payload.append(r)

            # Save patient + initial readings to backend in one call
            try:
                payload = {
                    "id": patient_id, "name": patient_name,
                    "age": age, "sex": sex, "notes": notes,
                    "baseline": baseline, "sim_config": sim_config,
                    "initial_readings": readings_payload
                }
                resp = requests.post(f"{BACKEND_URL}/patients", json=payload, timeout=10)
                if resp.status_code != 200:
                    st.warning(f"Backend save issue: {resp.status_code} - continuing anyway")
            except requests.exceptions.ConnectionError:
                st.warning("Backend not reachable - data saved locally only")

            st.session_state.patient_registry[patient_id] = {
                'patient_id': patient_id,
                'age': age,
                'sex': sex,
                'notes': notes,
                'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'simulation': sim_config
            }
            st.session_state.patient_data_store[patient_id] = df_flags
            save_patient_registry(st.session_state.patient_registry)
            save_simulation(patient_id, df_flags)
            st.session_state.last_simulation_id = patient_id
            st.success(f"Simulation complete for {patient_id}. Saved to backend!")

    if st.session_state.last_simulation_id in st.session_state.patient_data_store:
        patient_id = st.session_state.last_simulation_id
        df_flags = st.session_state.patient_data_store[patient_id]

        st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
        render_patient_vitals(df_flags, f"Simulation Results - {patient_id}")

        st.markdown("### Detailed Simulation Data")
        st.dataframe(df_flags, use_container_width=True)

        csv = df_flags.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Simulation CSV",
            data=csv,
            file_name=f"simulation_{patient_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime='text/csv'
        )

elif page == "Patient Records":
    st.markdown("## Patient Records & Simulations")
    
    # Load patients from backend
    backend_patients = get_all_patients_api()
    
    # Export all patients button
    if backend_patients:
        col1, col2, col3 = st.columns([2, 1, 1])
        with col3:
            if st.button("? Export All (Extended)"):
                all_data = export_patient_extended_format_api()
                if all_data:
                    st.download_button(
                        label="Download All Data TSV",
                        data=all_data,
                        file_name=f"all_patients_{datetime.now().strftime('%Y%m%d_%H%M%S')}.tsv",
                        mime='text/tab-separated-values'
                    )

    if not backend_patients:
        st.markdown("""
            <div class='warning-box'>
                <h3>No Patients Yet</h3>
                <p>Add a patient and simulate vitals on the 'Patient Simulation' page.</p>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("### All Patients")
        patients_df = pd.DataFrame(backend_patients)
        if not patients_df.empty:
            display_cols = [
                col for col in ["id", "name", "age", "sex", "created_at", "is_active"]
                if col in patients_df.columns
            ]
            st.dataframe(patients_df[display_cols], use_container_width=True)
        else:
            st.info("No patients found in backend.")

        patient_ids = [p['id'] for p in backend_patients]
        selected_patient_id = st.selectbox("Select Patient", patient_ids)
        selected_patient = next((p for p in backend_patients if p['id'] == selected_patient_id), {})

        st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
        st.markdown("### Patient Details")
        st.markdown(f"""
            <div class='card'>
                <p><strong>Patient ID:</strong> {selected_patient.get('id', selected_patient_id)}</p>
                <p><strong>Name:</strong> {selected_patient.get('name', 'N/A')}</p>
                <p><strong>Age:</strong> {selected_patient.get('age', 'N/A')}</p>
                <p><strong>Sex:</strong> {selected_patient.get('sex', 'N/A')}</p>
                <p><strong>Notes:</strong> {selected_patient.get('notes', '')}</p>
                <p><strong>Created:</strong> {selected_patient.get('created_at', '')}</p>
                <p><strong>Status:</strong> {"Active" if selected_patient.get('is_active') else "Inactive"}</p>
            </div>
        """, unsafe_allow_html=True)

        # Get ML predictions from backend
        predictions = predict_deterioration_api(selected_patient_id)
        if predictions:
            risk = predictions.get('risk_level', 'Unknown')
            prob = predictions.get('ensemble_probability', 0.0)
            ml_prob = predictions.get('ml_probability')
            traj_prob = predictions.get('trajectory_probability', 0.0)
            model_used = predictions.get('model_used', 'trajectory_logic')
            risk_color = '#ef4444' if risk == 'High' else '#f59e0b' if risk == 'Medium' else '#10b981'
            ml_line = f"<p><strong>ML Ensemble:</strong> {ml_prob*100:.1f}%</p>" if ml_prob is not None else ""
            st.markdown(f"""
                <div class='card' style='border-left: 4px solid {risk_color};'>
                    <p><strong>AI Risk Level:</strong> <span style='color: {risk_color}; font-weight: bold; font-size:18px;'>{risk}</span></p>
                    <p><strong>Combined Probability:</strong> {prob*100:.1f}%</p>
                    {ml_line}
                    <p><strong>Trajectory Score:</strong> {traj_prob*100:.1f}%</p>
                    <p style='color:#6b7280; font-size:12px;'>Model: {model_used}</p>
                </div>
            """, unsafe_allow_html=True)

        c1, c2 = st.columns([2, 1])
        with c1:
            step_count = st.number_input("Simulation steps", min_value=1, max_value=500, value=20, step=1)
        with c2:
            st.write("")
            st.write("")
            if st.button("Simulate Selected Patient"):
                if simulate_patient_step_api(selected_patient_id, int(step_count)):
                    st.rerun()

        live_c1, live_c2, live_c3 = st.columns([2, 1, 2])
        with live_c1:
            live_ecg_enabled = st.checkbox("Live ECG mode", value=False, key=f"live_ecg_{selected_patient_id}")
        with live_c2:
            auto_steps = st.number_input("Auto steps", min_value=1, max_value=20, value=1, step=1, key=f"auto_steps_{selected_patient_id}")
        with live_c3:
            refresh_seconds = st.slider("Refresh (sec)", min_value=1, max_value=10, value=2, step=1, key=f"refresh_sec_{selected_patient_id}")

        if st.button("Delete Selected Patient", type="secondary"):
            if delete_patient_api(selected_patient_id):
                st.session_state.patient_registry.pop(selected_patient_id, None)
                st.session_state.patient_data_store.pop(selected_patient_id, None)
                if st.session_state.last_simulation_id == selected_patient_id:
                    st.session_state.last_simulation_id = None
                st.rerun()

        # Get readings — backend first, fall back to local session state
        df_readings = get_patient_readings_api(selected_patient_id)
        if (df_readings is None or df_readings.empty) and selected_patient_id in st.session_state.patient_data_store:
            df_readings = st.session_state.patient_data_store[selected_patient_id]

        if df_readings is None or df_readings.empty:
            # Try loading from local CSV file (persists across Streamlit restarts)
            local_csv = simulation_path(selected_patient_id)
            if local_csv.exists():
                df_readings = pd.read_csv(local_csv, parse_dates=['timestamp'])
                df_readings = run_stability_detection(df_readings)

        if df_readings is None or df_readings.empty:
            st.info("No simulation data found for this patient. Create one in Patient Simulation.")
        else:
            st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
            st.markdown(f"### Vitals & Stability Data - {selected_patient_id}")

            # ECG animates client-side at 60fps — no Streamlit reruns needed for animation.
            # Fragment only advances simulation data in the background when Live ECG is on.
            _pid   = selected_patient_id
            _steps = int(auto_steps)
            _every = int(refresh_seconds) if live_ecg_enabled else None

            @st.fragment(run_every=_every)
            def _sim_advance():
                if live_ecg_enabled:
                    simulate_patient_step_api(_pid, _steps, show_success=False)
                    df_live = get_patient_readings_api(_pid)
                    df_show = df_live if (df_live is not None and not df_live.empty) else df_readings
                else:
                    df_show = df_readings
                render_live_ecg(df_show, _pid)

            _sim_advance()

            # Deterioration reasons panel
            st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
            render_deterioration_panel(df_readings)

            # Future course prognosis
            st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
            with st.spinner("Computing future course from ML models…"):
                prognosis_data = get_prognosis_api(selected_patient_id)
            if prognosis_data:
                render_prognosis_panel(prognosis_data)
            else:
                st.warning("Prognosis unavailable — ensure the backend is running and models are loaded.")

            # Show vitals visualization
            st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
            render_patient_vitals(df_readings, f"Vitals - {selected_patient_id}")

            st.markdown("### Detailed Reading Data")
            st.dataframe(df_readings.tail(200), use_container_width=True)

            # Download options
            st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
            st.markdown("### Export & Reports")
            col1, col2, col3 = st.columns(3)
            with col1:
                csv = df_readings.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name=f"readings_{selected_patient_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime='text/csv'
                )

            with col2:
                tsv_data = export_patient_extended_format_api(selected_patient_id)
                if tsv_data:
                    st.download_button(
                        label="Download TSV (Extended)",
                        data=tsv_data,
                        file_name=f"patient_{selected_patient_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.tsv",
                        mime='text/tab-separated-values'
                    )

            with col3:
                report_html = generate_patient_report(selected_patient, df_readings, predictions, prognosis_data)
                st.download_button(
                    label="Download Detailed Report",
                    data=report_html.encode('utf-8'),
                    file_name=f"report_{selected_patient_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
                    mime='text/html'
                )


st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
st.markdown("""
<div style='text-align: center; background: #f9fafb; padding: 32px; border-radius: 12px; margin-top: 48px; border: 1px solid #e5e7eb;'>
    <h3 style='color: #1a1a1a; font-family: Inter; margin-bottom: 8px; font-weight: 600;'>ICU Trajectory Detection System</h3>
    <p style='color: #374151; font-size: 14px; margin: 8px 0;'>AI-Powered Early Warning System for Patient Monitoring</p>
    <p style='color: #9ca3af; font-size: 13px; margin-top: 16px;'>Version 1.0 | Built with Streamlit | ? 2026</p>
</div>
""", unsafe_allow_html=True)
