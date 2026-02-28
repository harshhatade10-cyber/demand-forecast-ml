from flask import Flask, request, jsonify, render_template
import pickle
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

        if lstm_model is None:
            try:
                import tensorflow as tf
                lstm_model = tf.keras.models.load_model(
                    MODEL_DIR / "lstm_model.keras",
                    compile=False
                )
                logger.info("✅ LSTM loaded safely")
            except Exception as e:
                lstm_model = None
                logger.warning("⚠️ LSTM disabled: %s", e)

# ---------------- HELPERS ----------------
def get_lstm_sequence(location_encoded, ts):
    location_data = X_train[X_train["location"] == location_encoded]

    if len(location_data) < TIME_STEPS:
        pad = pd.concat([location_data.iloc[[0]]] * (TIME_STEPS - len(location_data)))
        seq = pd.concat([pad, location_data])
    else:
        seq = location_data.tail(TIME_STEPS)

    seq = seq.copy()

    seq.loc[seq.index[-1], "hour"] = ts.hour
    seq.loc[seq.index[-1], "day_of_week"] = ts.weekday()
    seq.loc[seq.index[-1], "month"] = ts.month

    seq = seq.fillna(0).astype(float)
    seq_scaled = scaler.transform(seq)

    return np.expand_dims(seq_scaled, axis=0)

# ---------------- PREDICTION ----------------
def predict_demand(city, area, base_fare=100):
    init_artifacts()
    ts = datetime.now()

    location = f"{city}_{area}"

    try:
        location_encoded = encoder.transform([location])[0]
    except Exception:
        raise ValueError("Invalid city or area")

    input_row = pd.DataFrame(columns=X_train.columns)
    input_row.loc[0] = 0

    input_row.at[0, "location"] = location_encoded
    input_row.at[0, "hour"] = ts.hour
    input_row.at[0, "day_of_week"] = ts.weekday()
    input_row.at[0, "month"] = ts.month
    input_row.at[0, "base_fare"] = float(base_fare)

    # LSTM Prediction
    lstm_pred = 0.0
    if lstm_model is not None:
        try:
            X_seq = get_lstm_sequence(location_encoded, ts)
            lstm_pred = float(lstm_model.predict(X_seq, verbose=0)[0][0])
        except Exception as e:
            logger.warning("⚠️ LSTM predict failed: %s", e)

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

@app.route("/locations")
def locations():
    init_artifacts()
    return jsonify({"locations": list(encoder.classes_)})

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json or {}
    city = data.get("city")
    area = data.get("area")
    base_fare = data.get("base_fare", 100)

    if not city or not area:
        return jsonify({"error": "City and Area are required"}), 400

    try:
        demand, ts = predict_demand(city, area, base_fare)
        return jsonify({
            "city": city,
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