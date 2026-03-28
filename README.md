# 🚖 Hybrid Ride Demand Forecasting — LSTM + XGBoost

> A production-grade ML system that predicts ride-sharing demand using a **hybrid LSTM + XGBoost ensemble**, deployed on **Render Cloud** with a full **CI/CD pipeline**, **Docker containerisation**, and **Prometheus monitoring**.

---

## 🎯 Results at a Glance

| Metric | Value |

| Model Accuracy | **90%** |
| Baseline Improvement | **+22 percentage points** over moving-average |
| Preprocessing Time Reduction | **80%** (from ~4 hrs → ~45 min) |
| Deployment | Live on **Render Cloud** |
| Pipeline | Auto-deploy on every `git push` via **GitHub Actions** |


## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                      DATA PIPELINE                       │
│  Historical Ride Data + Weather API → ETL (Python/Bash)  │
└────────────────────────┬────────────────────────────────┘
                         │
          ┌──────────────▼──────────────┐
          │      HYBRID ML MODEL         │
          │                              │
          │  LSTM  ──┐                   │
          │           ├──► Ensemble ──► Prediction │
          │  XGBoost ─┘                  │
          └──────────────────────────────┘
                         │
          ┌──────────────▼──────────────┐
          │     FLASK REST API           │
          │   /predict  /health  /metrics│
          └──────────────────────────────┘
                         │
          ┌──────────────▼──────────────┐
          │         DOCKER              │
          │   Containerised App + Deps  │
          └──────────────────────────────┘
                         │
     ┌───────────────────▼───────────────────┐
     │           CI/CD (GitHub Actions)       │
     │  Test → Build → Push → Deploy (Render) │
     └───────────────────────────────────────┘
                         │
          ┌──────────────▼──────────────┐
          │   MONITORING (Prometheus)    │
          │  Latency │ Errors │ Usage   │
          └──────────────────────────────┘
```

---

## 🧠 Why Hybrid LSTM + XGBoost?

| Model | Strength | Weakness |
|---|---|---|
| **LSTM** | Captures long-term temporal patterns in ride history | Struggles with non-linear tabular features |
| **XGBoost** | Handles weather, hour-of-day, and categorical features with high accuracy | No memory of time-series sequence |
| **Hybrid Ensemble** | Gets the best of both — sequential memory + feature power | 

The final prediction is a **weighted average** of both model outputs, tuned via cross-validation.

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| ML Models | LSTM (TensorFlow/Keras), XGBoost |
| API | Flask (Python) |
| Containerisation | Docker |
| CI/CD | GitHub Actions |
| Cloud Deployment | Render |
| Monitoring | Prometheus |
| Data Processing | Pandas, NumPy, Scikit-learn |

## 🚀 Run Locally

### Option 1 — Docker (Recommended)

```bash
# Clone the repo
git clone https://github.com/harshhatade10-cyber/demand-forecast-ml.git
cd demand-forecast-ml

# Build and run
docker build -t demand-forecast .
docker run -p 5000:5000 demand-forecast
```

API will be live at `http://localhost:5000`

### Option 2 — Python directly

```bash
pip install -r requirements.txt
python backend/app.py
```

---

## 📡 API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/predict` | `POST` | Returns demand forecast for given input |
| `/health` | `GET` | Health check — returns `{"status": "ok"}` |
| `/metrics` | `GET` | Prometheus-compatible metrics endpoint |

### Sample Request

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"hour": 18, "day_of_week": 4, "temperature": 28, "rainfall": 0}'
```

### Sample Response

```json
{
  "predicted_demand": 142,
  "model": "hybrid-lstm-xgboost",
  "confidence": 0.91
}
```

---

## ⚙️ CI/CD Pipeline

Every push to `main` triggers the GitHub Actions workflow:

```
Push to main
    │
    ├── 1. Run Tests (pytest)
    ├── 2. Build Docker Image
    ├── 3. Push to Docker Hub
    └── 4. Auto-deploy to Render Cloud
```

Pipeline config: [`.github/workflows/`](.github/workflows/)

---

## 📊 Monitoring

Prometheus is configured via [`prometheus.yml`](prometheus.yml) to scrape the `/metrics` endpoint.

Tracked metrics:
- Request latency (p50, p95, p99)
- Prediction request count
- Error rate
- Model inference time

---

## 📁 Project Structure

```
demand-forecast-ml/
├── .github/
│   └── workflows/        # GitHub Actions CI/CD pipeline
├── backend/              # Flask API
├── data/                 # Training & test datasets
├── models/               # Saved LSTM + XGBoost model files
├── src/                  # Model training, preprocessing scripts
├── Dockerfile            # Container definition
├── prometheus.yml        # Monitoring config
└── requirements.txt      # Python dependencies
```

## 🧪 Running Tests

```bash
pytest src/tests/ -v
```


## 🌐 Live Demo

🔗 **[View Live on Render →](https://demand-app-v1.onrender.com)** 

---

## 👤 Author

**Harsh Hatade**
Final-year B.E. in AI & Data Science | DevOps & Java Developer


This project is licensed under the MIT License.
