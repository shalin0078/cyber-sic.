try:
    import streamlit as st
except ImportError:
    import sys
    msg = (
        "Streamlit is not installed in the active Python environment.\n"
        "To fix: activate your venv and install requirements, for example:\n"
        "  .\\.venv\\Scripts\\Activate.ps1  # PowerShell activate\n"
        "  python -m pip install -r requirements.txt\n\n"
        "Or install directly: `python -m pip install streamlit`\n\n"
        "You can also run the app without activating the venv by:\n"
        "  python -m streamlit run .\\new1.py\n"
    )
    print(msg)
    sys.exit(1)
import pandas as pd
import pickle
import numpy as np
import hashlib
import time
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.metrics import recall_score, accuracy_score, f1_score
from typing import List, Dict

# Optional imports for fallback training
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

# Try to import shap (optional). If unavailable, we will gracefully disable XAI.
try:
    import shap
    HAS_SHAP = True
except Exception:
    HAS_SHAP = False

# Page configuration for a professional, cyber-themed UI
st.set_page_config(
    page_title="CyberSecure: Real-Time Intrusion Triage Dashboard",
    page_icon="🔒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for a dynamic, cool, dark cyber theme
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
        color: #ffffff;
    }
    .stApp {
        background-color: #0e1117;
    }
    .metric-card {
        background-color: #1a1d2e;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #00ff41;
    }
    .threat-log {
        background-color: #16213e;
        border: 1px solid #0f3460;
        border-radius: 0.5rem;
        padding: 1rem;
    }
    .header {
        color: #00ff41;
        font-family: 'Courier New', monospace;
        font-size: 2rem;
    }
    .subheader {
        color: #b8b8b8;
        font-family: 'Courier New', monospace;
    }
    .alert-danger {
        color: #ff4757;
    }
    .alert-success {
        color: #2ed573;
    }
    .chain-block {
        background-color: #0f3460;
        padding: 0.5rem;
        border-radius: 0.25rem;
        font-family: 'Courier New', monospace;
        font-size: 0.8rem;
    }
    </style>
""", unsafe_allow_html=True)

# Load dataset (try several filenames; fall back to sample)
@st.cache_data
def load_dataset():
    candidates = ['dataset.csv', 'nsl_kdd_dataset.csv', 'nsl_kdd.csv', 'nsl_kdd_dataset.csv']
    for fn in candidates:
        try:
            df = pd.read_csv(fn)
            st.info(f"Loaded dataset from '{fn}'")
            # Normalize label column names
            if 'label' in df.columns:
                df['binary_label'] = df['label'].apply(lambda x: 0 if str(x).lower() in ('normal','benign') else 1)
            elif 'attack' in df.columns:
                df['binary_label'] = df['attack'].apply(lambda x: 0 if str(x).lower() in ('normal','benign') else 1)
            else:
                # If no label, synthesize a weak demo label
                np.random.seed(42)
                df['binary_label'] = np.random.choice([0,1], size=len(df), p=[0.85,0.15])
            return df
        except FileNotFoundError:
            continue
        except pd.errors.ParserError:
            continue
    # If none found, create sample dataframe for demo
    st.warning("No dataset file found. Using synthetic demo data.")
    np.random.seed(42)
    features = ['duration','src_bytes','dst_bytes','wrong_fragment','hot','num_failed_logins']
    sample_data = pd.DataFrame(np.random.rand(1000, len(features)), columns=features)
    sample_data['binary_label'] = np.random.choice([0,1], 1000, p=[0.8, 0.2])
    return sample_data

# Feature engineering (auto-detect numeric + simple encoding for categoricals)
def preprocess_flow(flow_data: pd.DataFrame, feature_columns: List[str]):
    processed = flow_data[feature_columns].copy()
    # For demo, convert categorical/object columns via one-hot encoding
    obj_cols = processed.select_dtypes(include=['object', 'category']).columns.tolist()
    if obj_cols:
        processed = pd.get_dummies(processed, columns=obj_cols, drop_first=True)
    return processed

# Load or train model: prefer an existing 'model.pkl'; fallback to training a simple RandomForest
@st.cache_resource
def load_or_train_model(df: pd.DataFrame, target_col: str = 'binary_label'):
    # Try known model filenames
    for fn in ('model.pkl', 'cybersecure_model.pkl', 'cybersecure_model.joblib'):
        try:
            with open(fn, 'rb') as f:
                model = pickle.load(f)
            st.success(f"Loaded model from '{fn}'")
            return model
        except Exception:
            continue

    # If no model found, attempt to train a simple fallback model (only if dataset has enough rows)
    if df is None or len(df) < 50:
        st.error("No model found and dataset too small to train a fallback model.")
        st.stop()

    st.info("No pre-trained model found. Training a fallback RandomForest model (this may take a few seconds)...")

    # Prepare X, y
    if target_col not in df.columns:
        st.error(f"Target column '{target_col}' not found in dataset; cannot train fallback model.")
        st.stop()

    X = df.drop(columns=[target_col], errors='ignore')
    y = df[target_col]

    # Auto-detect categorical columns (object/category)
    categorical_cols = X.select_dtypes(include=['object','category']).columns.tolist()
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()

    # If there are non-numeric columns, build a preprocessor
    transformers = []
    if categorical_cols:
        transformers.append(('cat', OneHotEncoder(handle_unknown='ignore', sparse=False), categorical_cols))
    if numeric_cols:
        transformers.append(('num', 'passthrough', numeric_cols))

    if transformers:
        preprocessor = ColumnTransformer(transformers=transformers)
        model = Pipeline(steps=[('preproc', preprocessor), ('rf', RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42))])
    else:
        model = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42)

    # Train/test split small portion for speed
    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    except Exception:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model.fit(X_train, y_train)

    # Save model for future runs
    try:
        with open('model.pkl', 'wb') as f:
            pickle.dump(model, f)
        st.success("Fallback model trained and saved as 'model.pkl'.")
    except Exception:
        st.warning("Fallback model trained but could not be saved to disk.")

    return model

# Security Action Logic
def generate_security_action(confidence: float, intrusion_type: str = None) -> str:
    if confidence < 0.7:
        return "Monitor: Low confidence alert – Log for review."
    elif confidence < 0.9:
        return "Throttle: Limit bandwidth on suspicious flow."
    else:
        if intrusion_type and 'dos' in str(intrusion_type).lower():
            return "Block Source IP: Immediate denial of service mitigation."
        elif intrusion_type and 'probe' in str(intrusion_type).lower():
            return "Quarantine Endpoint: Isolate scanning activity."
        elif intrusion_type and 'u2r' in str(intrusion_type).lower():
            return "Alert Admin: Potential privilege escalation detected."
        else:
            return "Block Port Traffic: Default block on HTTP/HTTPS ports."
    return "Escalate: High-risk intrusion – Manual intervention required."

# Simple immutable threat ledger (SHA-256 chaining)
class ThreatLedger:
    def __init__(self):
        self.chain: List[Dict] = []
        self.genesis_hash = hashlib.sha256(b"Genesis Block: CyberSecure Initialized").hexdigest()

    def add_entry(self, intrusion_data: Dict) -> str:
        timestamp = datetime.now().isoformat()
        entry = {**intrusion_data, 'timestamp': timestamp}
        prev_hash = self.chain[-1]['hash'] if self.chain else self.genesis_hash
        entry_str = str(entry) + prev_hash
        entry_hash = hashlib.sha256(entry_str.encode()).hexdigest()
        entry['hash'] = entry_hash
        entry['prev_hash'] = prev_hash
        self.chain.append(entry)
        return entry_hash

# Explainable AI helper
@st.cache_resource
def load_shap_explainer(model, X_sample):
    if not HAS_SHAP:
        raise RuntimeError('SHAP library is unavailable')
    explainer = shap.TreeExplainer(model) if 'Tree' in str(type(model).__name__) else shap.Explainer(model)
    shap_values = explainer(X_sample)
    return explainer, shap_values

# Main Dashboard
def main():
    st.markdown('<h1 class="header">🔒 CyberSecure: Real-Time Intrusion Triage</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subheader">Automated SOC Threat Feed – Prioritizing Recall for Zero Missed Attacks</p>', unsafe_allow_html=True)

    # Load resources
    df = load_dataset()
    model = load_or_train_model(df)

    # Determine feature columns automatically (prefer numeric + a few common fields)
    candidate_defaults = ['duration', 'src_bytes', 'dst_bytes', 'wrong_fragment', 'hot', 'num_failed_logins']
    feature_columns = [c for c in candidate_defaults if c in df.columns]
    if not feature_columns:
        # pick numeric columns excluding the label
        feature_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        feature_columns = [c for c in feature_columns if c != 'binary_label']

    if not feature_columns:
        st.error('No usable feature columns detected in dataset. Please supply numeric features or update the app.')
        st.stop()

    X = preprocess_flow(df, feature_columns)
    y_true = df['binary_label'].values if 'binary_label' in df.columns else np.random.choice([0,1], len(df))

    # Predictions (handle models wrapped in pipelines)
    try:
        y_pred = model.predict(X)
    except Exception as e:
        st.error(f"Model prediction failed: {e}")
        st.stop()

    try:
        y_proba = model.predict_proba(X)[:, 1] if hasattr(model, 'predict_proba') else np.full(len(X), 0.5)
    except Exception:
        # Some sklearn pipelines / wrappers may require transform; fallback
        y_proba = np.full(len(X), 0.5)

    # Evaluation Metrics (Focus on Recall)
    recall = recall_score(y_true, y_pred) if len(np.unique(y_true)) > 1 else 0.0
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred) if len(np.unique(y_true)) > 1 else 0.0

    # Sidebar: Controls and Metrics
    with st.sidebar:
        st.header("📊 System Metrics")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Recall (Sensitivity)", f"{recall:.3f}", delta="Maximized")
        with col2:
            st.metric("Accuracy", f"{accuracy:.3f}")
        with col3:
            st.metric("F1-Score", f"{f1:.3f}")

        st.header("⚙️ Controls")
        num_samples = st.slider("Sample Size for Live Feed", 10, min(len(df), 500), 50)
        refresh_rate = st.slider("Refresh Rate (seconds)", 1, 10, 3)
        enable_xai = st.checkbox("Enable XAI Explanations (SHAP)")
        if enable_xai and not HAS_SHAP:
            st.warning("SHAP library not available — install `shap` to enable XAI visualizations.")
            enable_xai = False
        enable_blockchain = st.checkbox("Enable Immutable Ledger Simulation")

    # Main Content: Live Threat Feed
    col_left, col_right = st.columns([2, 1])

    with col_left:
        st.subheader("🚨 Live Threat Feed")
        threat_log = []
        ledger = ThreatLedger() if enable_blockchain else None

        # Simulate dynamic feed: Sample and "process" in batches
        sample_indices = np.random.choice(len(df), num_samples, replace=False)
        for i in sample_indices:
            flow = X.iloc[[i]]
            try:
                pred = int(model.predict(flow)[0])
            except Exception:
                pred = int(y_pred[i])
            conf = float(y_proba[i])
            label = 'Intrusion' if pred == 1 else 'Benign'

            if pred == 1:
                intrusion_type = df.iloc[i]['label'] if 'label' in df.columns else 'Unknown'
                action = generate_security_action(conf, intrusion_type)

                log_entry = {
                    'timestamp': datetime.now().strftime("%H:%M:%S"),
                    'confidence': f"{conf:.2f}",
                    'action': action,
                    'type': intrusion_type
                }
                threat_log.append(log_entry)

                if enable_blockchain:
                    entry_hash = ledger.add_entry(log_entry)
                    log_entry['hash'] = entry_hash
                    log_entry['prev_hash'] = ledger.chain[-2]['hash'] if len(ledger.chain) > 1 else ledger.genesis_hash

        # Display threat log as dynamic table (mimics live feed)
        if threat_log:
            log_df = pd.DataFrame(threat_log)
            st.dataframe(log_df, use_container_width=True, height=400)

            # Auto-refresh simulation
            time.sleep(refresh_rate)
            st.experimental_rerun()
        else:
            st.info("No intrusions detected in current sample. Adjust sample size for more activity.")

    with col_right:
        st.subheader("🔍 Recent Intrusion Details")
        if threat_log:
            latest = threat_log[-1]
            st.markdown(f"""
                <div class="threat-log">
                    <p><strong>Time:</strong> {latest['timestamp']}</p>
                    <p><strong>Confidence:</strong> <span class="{ 'alert-danger' if float(latest['confidence']) > 0.8 else '' }">{latest['confidence']}</span></p>
                    <p><strong>Type:</strong> {latest['type']}</p>
                    <p><strong>Action:</strong> <span class="alert-danger">{latest['action']}</span></p>
                    {f'<p><strong>Block Hash:</strong> <span class="chain-block">{latest.get("hash", "N/A")}</span></p>' if enable_blockchain else ''}
                </div>
            """, unsafe_allow_html=True)

        # XAI Visualization (optional)
        if enable_xai and HAS_SHAP:
            st.subheader("🧠 Explainable AI Insights")
            sample_for_shap = X.iloc[sample_indices[:10]] if len(X) >= 10 else X.iloc[:10]
            try:
                explainer, shap_values = load_shap_explainer(model, sample_for_shap)
                shap.summary_plot(shap_values, sample_for_shap, show=False, plot_type="bar")
                st.pyplot(plt.gcf())
                plt.close()
            except Exception as e:
                st.warning(f"SHAP plotting failed: {e}")

    # Footer
    st.markdown("---")
    st.markdown('<p class="subheader">Powered by Streamlit | Model: Binary Classifier (fallback RandomForest) | Dataset: Sanitized Network Flows or demo</p>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
