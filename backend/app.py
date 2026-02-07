from flask import render_template
from flask import Flask, request, jsonify
import pickle
import os
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
import threading
import logging

# ---------------- CONFIG ----------------
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data" / "processed"
TIME_STEPS = 24

app = Flask(__name__)
app.config.setdefault("MAX_CONTENT_LENGTH", 10 * 1024)  # 10 KB payload limit for predict

# logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Globals for artifacts (loaded lazily)
encoder = None
hybrid = None
lstm_model = None
xgb_model = None
scaler = None
X_train = None
artifacts_loaded = False
artifacts_errors = []
load_lock = threading.Lock()
predict_lock = threading.Lock()


def init_artifacts():
    global encoder, hybrid, lstm_model, xgb_model, scaler, X_train, artifacts_loaded, artifacts_errors
    with load_lock:
        if artifacts_loaded:
            return
        artifacts_errors = []
        try:
            enc_path = DATA_DIR / "label_encoder.pkl"
            if not enc_path.exists():
                raise FileNotFoundError(f"Missing {enc_path}")
            with open(enc_path, "rb") as f:
                encoder = pickle.load(f)
            logger.info("Label encoder loaded successfully")
        except Exception as e:
            artifacts_errors.append(str(e))
            logger.exception("Failed to load label encoder")

        try:
            xgb_path = MODEL_DIR / "xgb_model.pkl"
            if not xgb_path.exists():
                raise FileNotFoundError(f"Missing {xgb_path}")
            with open(xgb_path, "rb") as f:
                xgb_model = pickle.load(f)
            logger.info("XGBoost model loaded successfully")
        except Exception as e:
            artifacts_errors.append(str(e))
            logger.exception("Failed to load XGBoost model")

        try:
            scaler_path = MODEL_DIR / "scaler.pkl"
            if not scaler_path.exists():
                raise FileNotFoundError(f"Missing {scaler_path}")
            with open(scaler_path, "rb") as f:
                scaler = pickle.load(f)
            logger.info("Scaler loaded successfully")
        except Exception as e:
            artifacts_errors.append(str(e))
            logger.exception("Failed to load scaler")

        try:
            import tensorflow as tf
            lstm_path = MODEL_DIR / "lstm_model.h5"
            if not lstm_path.exists():
                raise FileNotFoundError(f"Missing {lstm_path}")
            # Load without compiling to avoid Keras metric incompatibility
            lstm_model = tf.keras.models.load_model(lstm_path, compile=False)
            logger.info("LSTM model loaded successfully (without recompile)")
        except ImportError:
            artifacts_errors.append("TensorFlow not installed")
            logger.exception("TensorFlow not available; LSTM loading skipped")
        except Exception as e:
            artifacts_errors.append(str(e))
            logger.exception("Failed to load LSTM model")

        try:
            xtrain_path = DATA_DIR / "X_train.csv"
            if not xtrain_path.exists():
                raise FileNotFoundError(f"Missing {xtrain_path}")
            X_train = pd.read_csv(xtrain_path)
            logger.info("X_train.csv loaded successfully")
        except Exception as e:
            artifacts_errors.append(str(e))
            logger.exception("Failed to load X_train.csv")

        # Mark as loaded if critical models are available (encoder, xgb_model, scaler, X_train, lstm_model)
        artifacts_loaded = (encoder is not None and xgb_model is not None and 
                           scaler is not None and X_train is not None and lstm_model is not None)
        if artifacts_loaded:
            logger.info("All artifacts loaded successfully (hybrid model ready)")
        else:
            logger.warning("Critical artifacts missing for hybrid model: %s", artifacts_errors)


def get_lstm_sequence():
    if X_train is None:
        raise RuntimeError("Training data not loaded")

    if len(X_train) < TIME_STEPS:
        pad = pd.concat([X_train.iloc[[0]]] * (TIME_STEPS - len(X_train)))
        seq = pd.concat([pad, X_train])
    else:
        seq = X_train.tail(TIME_STEPS)

    # ensure numeric and no NaNs for scaler
    seq = seq.fillna(0).astype(float)
    seq_scaled = scaler.transform(seq)
    return np.expand_dims(seq_scaled, axis=0)


def predict_demand(area_name, base_fare=100, ts=None):
    if not artifacts_loaded:
        raise RuntimeError("Model artifacts not loaded")
    if lstm_model is None:
        raise RuntimeError("LSTM model not loaded")

    if ts is None:
        ts = datetime.now()

    # empty row with training columns
    input_row = pd.DataFrame(columns=X_train.columns)
    input_row.loc[0] = 0

    # fill values safely
    if "area_id" in input_row.columns:
        try:
            area_id = encoder.transform([area_name])[0]
            input_row.at[0, "area_id"] = int(area_id)
        except Exception as e:
            raise ValueError(f"Unknown area: {area_name}") from e

    if "area_name" in input_row.columns:
        try:
            encoded = encoder.transform([area_name])[0]
            input_row.at[0, "area_name"] = int(encoded)
        except Exception as e:
            raise ValueError(f"Unknown area: {area_name}") from e

    if "hour" in input_row.columns:
        input_row.at[0, "hour"] = int(ts.hour)
    if "day" in input_row.columns:
        input_row.at[0, "day"] = int(ts.day)
    if "day_of_week" in input_row.columns:
        input_row.at[0, "day_of_week"] = int(ts.weekday())
    if "month" in input_row.columns:
        input_row.at[0, "month"] = int(ts.month)
    if "base_fare" in input_row.columns:
        input_row.at[0, "base_fare"] = float(base_fare)

    # LSTM prediction (mandatory for hybrid)
    X_seq = get_lstm_sequence()
    with predict_lock:
        lstm_pred = float(lstm_model.predict(X_seq, verbose=0)[0][0])

    # Add LSTM prediction to input row (always, even if not in X_train.columns)
    input_row.at[0, "lstm_pred"] = lstm_pred

    # Ensure correct column order and types - include lstm_pred in expected features
    expected_columns = list(X_train.columns) + ["lstm_pred"]
    input_row = input_row.reindex(columns=expected_columns, fill_value=0).fillna(0).astype(float)

    # Final prediction (XGBoost + LSTM hybrid)
    with predict_lock:
        demand = float(xgb_model.predict(input_row)[0])

    return int(round(max(0, demand))), ts


# ---------------- ROUTES ----------------
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/areas", methods=["GET"])
def get_areas():
    init_artifacts()
    if encoder is None:
        return jsonify({"error": "Encoder not loaded"}), 503
    return jsonify({"areas": list(encoder.classes_)})


@app.route("/predict", methods=["POST"])
def predict():
    data = request.json or {}
    area = data.get("area")
    base_fare = data.get("base_fare", 100)

    init_artifacts()

    if not area:
        return jsonify({"error": "Area is required"}), 400

    try:
        # sanitize base_fare
        base_fare = float(base_fare)
    except Exception:
        return jsonify({"error": "base_fare must be numeric"}), 400

    try:
        ts = datetime.now()
        demand, used_ts = predict_demand(area, base_fare, ts=ts)
        logger.info("Prediction result area=%s demand=%s time=%s", area, demand, used_ts.strftime("%Y-%m-%d %H:%M"))
        return jsonify({"area": area, "predicted_demand": demand, "time": used_ts.strftime("%Y-%m-%d %H:%M")})
    except Exception as e:
        logger.exception("Prediction failed for area=%s", area)
        if isinstance(e, ValueError):
            return jsonify({"error": str(e)}), 400
        if isinstance(e, RuntimeError):
            return jsonify({"error": "Model not available"}), 503
        return jsonify({"error": "Internal server error"}), 500


@app.route("/health", methods=["GET"])
def health():
    init_artifacts()
    status = {"artifacts_loaded": artifacts_loaded}
    if not artifacts_loaded:
        status["errors"] = artifacts_errors
    return jsonify(status)


if __name__ == "__main__":
    # Do not run with debug=True in production. Use FLASK_DEBUG env var to control.
    debug_env = os.getenv("FLASK_DEBUG", "false").lower()
    debug_flag = debug_env in ("1", "true", "yes")
    app.run(debug=debug_flag)
