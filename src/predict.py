"""
Robust predictor for the hybrid LSTM + XGBoost model.

Improvements:
- Quiet TensorFlow/Keras info logs where possible
- Provide non-interactive CLI options (`--area`, `--area-index`)
- Add error handling for model loading and prediction
"""

import os
import sys
import argparse
import logging
import pickle
import warnings
from datetime import datetime

import numpy as np
import pandas as pd

# Suppress all warnings before importing TensorFlow/Keras
warnings.filterwarnings("ignore")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")  # Suppress TensorFlow logs completely
os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")

# Suppress TensorFlow/Keras info logs
import logging as tf_logging
tf_logging.getLogger('tensorflow').setLevel(tf_logging.ERROR)
if os.environ.get('TF_CPP_MIN_LOG_LEVEL') is None:
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# ================= CONFIG =================
MODEL_DIR = "models"
DATA_DIR = "data/processed"
TIME_STEPS = 24

logging.basicConfig(level=logging.ERROR, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_all():
    """Load label encoder and hybrid model artifacts.

    Returns: encoder, lstm_model, xgb_model, scaler, X_train
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        
        # Load encoder
        encoder_path = os.path.join(DATA_DIR, "label_encoder.pkl")
        if not os.path.exists(encoder_path):
            raise FileNotFoundError(f"Label encoder not found: {encoder_path}")
        with open(encoder_path, "rb") as f:
            encoder = pickle.load(f)

        # Load hybrid (should be a dict with keys: 'lstm', 'xgb', 'scaler')
        hybrid_path = os.path.join(MODEL_DIR, "hybrid_model.pkl")
        if not os.path.exists(hybrid_path):
            raise FileNotFoundError(f"Hybrid model not found: {hybrid_path}")
        with open(hybrid_path, "rb") as f:
            hybrid = pickle.load(f)

        # Backwards compatibility: hybrid may store paths instead of objects
        lstm = hybrid.get("lstm")
        xgb = hybrid.get("xgb")
        scaler = hybrid.get("scaler")

        # If `lstm` is a filename, try to load it
        try:
            from keras.models import load_model

            if isinstance(lstm, str) and os.path.exists(lstm):
                lstm = load_model(lstm)
        except Exception:
            # keras may not be available or model already loaded; ignore
            pass

        X_train_path = os.path.join(DATA_DIR, "X_train.csv")
        if not os.path.exists(X_train_path):
            raise FileNotFoundError(f"X_train not found: {X_train_path}")
        X_train = pd.read_csv(X_train_path)

        return encoder, lstm, xgb, scaler, X_train


def select_area(encoder, non_interactive=False, area_arg=None, area_index_arg=None):
    areas = list(encoder.classes_)
    if non_interactive:
        # Prefer explicit name, then index, then default to first
        if area_arg and area_arg in areas:
            return area_arg
        if area_index_arg:
            try:
                idx = int(area_index_arg)
                if 1 <= idx <= len(areas):
                    return areas[idx - 1]
            except Exception:
                pass
        return areas[0]

    print("\nAvailable Areas:")
    for i, a in enumerate(areas, 1):
        print(f"{i}. {a}")

    while True:
        try:
            idx = int(input("\nSelect area number: "))
            if 1 <= idx <= len(areas):
                return areas[idx - 1]
        except (ValueError, KeyboardInterrupt):
            print("❌ Invalid choice, try again.")


def get_lstm_sequence(X_train, scaler):
    df = X_train.copy()
    if len(df) < TIME_STEPS:
        pad = pd.concat([df.iloc[[0]]] * (TIME_STEPS - len(df)))
        df = pd.concat([pad, df])

    seq = df.tail(TIME_STEPS)
    # scaler may expect numeric columns only; handle gracefully
    try:
        seq_scaled = scaler.transform(seq)
    except Exception as e:
        logger.warning(f"Scaler transform failed: {e}; attempting numeric-cast")
        seq_numeric = seq.select_dtypes(include=[np.number]).fillna(0)
        seq_scaled = scaler.transform(seq_numeric)

    return np.expand_dims(seq_scaled, axis=0)


def build_input_row(X_train, encoder, area, ts):
    # Create input row with all columns from X_train + lstm_pred
    all_columns = list(X_train.columns) + ['lstm_pred'] if 'lstm_pred' not in X_train.columns else list(X_train.columns)
    
    # Initialize with float64 to accommodate LSTM float predictions
    input_row = pd.DataFrame(0.0, index=[0], columns=all_columns, dtype=np.float64)

    if "area_name" in input_row.columns:
        try:
            input_row.at[0, "area_name"] = float(encoder.transform([area])[0])
        except Exception:
            input_row.at[0, "area_name"] = 0.0

    if "area_id" in input_row.columns:
        try:
            input_row.at[0, "area_id"] = float(encoder.transform([area])[0])
        except Exception:
            input_row.at[0, "area_id"] = 0.0

    if "hour" in input_row.columns:
        input_row.at[0, "hour"] = float(ts.hour)
    if "day_of_week" in input_row.columns:
        input_row.at[0, "day_of_week"] = float(ts.weekday())
    if "month" in input_row.columns:
        input_row.at[0, "month"] = float(ts.month)
    if "base_fare" in input_row.columns:
        input_row.at[0, "base_fare"] = 100.0
    if "day" in input_row.columns:
        input_row.at[0, "day"] = float(ts.day)
    if "lstm_pred" in input_row.columns:
        input_row.at[0, "lstm_pred"] = 0.0

    return input_row


def predict(args=None):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        
        encoder, lstm_model, xgb_model, scaler, X_train = load_all()

        # choose area
        non_interactive = getattr(args, "non_interactive", False) if args else False
        area = select_area(encoder, non_interactive, getattr(args, "area", None), getattr(args, "area_index", None))
        ts = datetime.now()

        input_row = build_input_row(X_train, encoder, area, ts)

        # generate lstm feature
        lstm_pred_value = 0.0
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                X_seq = get_lstm_sequence(X_train, scaler)
                lstm_pred_value = float(lstm_model.predict(X_seq, verbose=0)[0][0])
        except Exception:
            pass
        
        # Update lstm_pred in input_row
        if "lstm_pred" in input_row.columns:
            input_row.at[0, "lstm_pred"] = lstm_pred_value

        # Ensure numeric dtype for xgb and match expected feature names
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            numeric_row = input_row.copy()
            
            # Convert all columns to numeric
            for c in numeric_row.columns:
                if numeric_row[c].dtype == object:
                    numeric_row[c] = pd.to_numeric(numeric_row[c], errors="coerce").fillna(0)

            # Get expected feature names from XGBoost
            expected = None
            try:
                expected = list(getattr(xgb_model, 'feature_names_in_', []))
            except Exception:
                pass

            if not expected:
                try:
                    expected = list(xgb_model.get_booster().feature_names)
                except Exception:
                    pass

            # If we have expected features, ensure all are present and in correct order
            if expected:
                # Add any missing columns with zeros
                for col in expected:
                    if col not in numeric_row.columns:
                        numeric_row[col] = 0.0
                # Select and reorder columns to match training
                numeric_row = numeric_row[expected]
            else:
                # Fallback: ensure all 8 columns exist
                required_cols = ['area_id', 'base_fare', 'area_name', 'hour', 'day', 'day_of_week', 'month', 'lstm_pred']
                for col in required_cols:
                    if col not in numeric_row.columns:
                        numeric_row[col] = 0.0
                # Only use required columns
                numeric_row = numeric_row[[col for col in required_cols if col in numeric_row.columns]]

            demand = float(xgb_model.predict(numeric_row)[0])

        print("\n📍 Area:", area)
        print("⏰ Time:", ts.strftime("%Y-%m-%d %H:%M"))
        print("🚕 Predicted Ride Demand:", int(round(max(0, demand))))
        sys.stdout.flush()


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Predict ride demand for an area")
    p.add_argument("--area", help="Area name (exact)")
    p.add_argument("--area-index", type=int, help="Area index from list (1-based)")
    p.add_argument("--non-interactive", action="store_true", help="Run non-interactively; defaults to first area")
    return p.parse_args(argv)


if __name__ == "__main__":
    try:
        args = parse_args()
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            predict(args)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"\n❌ Error: {str(e)}", file=sys.stderr)
        sys.exit(1)
