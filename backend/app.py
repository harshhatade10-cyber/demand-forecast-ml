from flask import Flask, request, jsonify, render_template
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
app.config.setdefault("MAX_CONTENT_LENGTH", 10 * 1024)

# logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------- GLOBALS ----------------
encoder = None
xgb_model = None
scaler = None
lstm_model = None
X_train = None

load_lock = threading.Lock()
predict_lock = threading.Lock()

# ---------------- LOADERS ----------------
def init_artifacts():
    global encoder, xgb_model, scaler, lstm_model, X_train

    with load_lock:
        if encoder is None:
            with open(DATA_DIR / "label_encoder.pkl", "rb") as f:
                encoder = pickle.load(f)
            logger.info("✅ Encoder loaded")

        if xgb_model is None:
            with open(MODEL_DIR / "xgb_model.pkl", "rb") as f:
                xgb_model = pickle.load(f)
            logger.info("✅ XGBoost loaded")

        if scaler is None:
            with open(MODEL_DIR / "scaler.pkl", "rb") as f:
                scaler = pickle.load(f)
            logger.info("✅ Scaler loaded")

        if X_train is None:
            X_train = pd.read_csv(DATA_DIR / "X_train.csv")
            logger.info("✅ X_train loaded")

        # ---- SAFE LSTM LOAD ----
        if lstm_model is None:
            try:
                import tensorflow as tf
                from tensorflow.keras.layers import InputLayer

                lstm_model = tf.keras.models.load_model(
                    MODEL_DIR / "lstm_model.h5",
                    compile=False,
                    custom_objects={"InputLayer": InputLayer}
                )
                logger.info("✅ LSTM loaded safely")

            except Exception as e:
                lstm_model = None
                logger.warning("⚠️ LSTM disabled, fallback to XGB only: %s", e)

# ---------------- HELPERS ----------------
def get_lstm_sequence():
    if len(X_train) < TIME_STEPS:
        pad = pd.concat([X_train.iloc[[0]]] * (TIME_STEPS - len(X_train)))
        seq = pd.concat([pad, X_train])
    else:
        seq = X_train.tail(TIME_STEPS)

    seq = seq.fillna(0).astype(float)
    seq_scaled = scaler.transform(seq)
    return np.expand_dims(seq_scaled, axis=0)

# ---------------- PREDICTION ----------------
def predict_demand(area_name, base_fare=100):
    init_artifacts()
    ts = datetime.now()

    input_row = pd.DataFrame(columns=X_train.columns)
    input_row.loc[0] = 0

    if "area_name" in input_row.columns:
        input_row.at[0, "area_name"] = int(encoder.transform([area_name])[0])

    if "hour" in input_row.columns:
        input_row.at[0, "hour"] = ts.hour
    if "day_of_week" in input_row.columns:
        input_row.at[0, "day_of_week"] = ts.weekday()
    if "month" in input_row.columns:
        input_row.at[0, "month"] = ts.month
    if "base_fare" in input_row.columns:
        input_row.at[0, "base_fare"] = float(base_fare)

    # -------- LSTM PRED (SAFE) --------
    lstm_pred = 0.0
    if lstm_model is not None:
        try:
            X_seq = get_lstm_sequence()
            lstm_pred = float(lstm_model.predict(X_seq, verbose=0)[0][0])
        except Exception as e:
            logger.warning("⚠️ LSTM predict failed, using 0: %s", e)

    input_row["lstm_pred"] = lstm_pred

    expected_cols = list(X_train.columns) + ["lstm_pred"]
    input_row = input_row.reindex(columns=expected_cols, fill_value=0).astype(float)

    with predict_lock:
        demand = float(xgb_model.predict(input_row)[0])

    return int(round(max(0, demand))), ts

# ---------------- ROUTES ----------------
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/areas")
def areas():
    init_artifacts()
    return jsonify({"areas": list(encoder.classes_)})

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json or {}
    area = data.get("area")
    base_fare = data.get("base_fare", 100)

    if not area:
        return jsonify({"error": "Area is required"}), 400

    try:
        demand, ts = predict_demand(area, base_fare)
        return jsonify({
            "area": area,
            "predicted_demand": demand,
            "time": ts.strftime("%Y-%m-%d %H:%M")
        })
    except Exception:
        logger.exception("Prediction failed")
        return jsonify({"error": "Prediction failed"}), 500

@app.route("/health")
def health():
    init_artifacts()
    return jsonify({
        "encoder": encoder is not None,
        "xgb": xgb_model is not None,
        "lstm": lstm_model is not None
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
# ---------------- RUN APP ----------------