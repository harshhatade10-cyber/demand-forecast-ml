"""
Hybrid Model Training: LSTM + XGBoost for ride demand forecasting.

LSTM captures temporal patterns from sequential data.
XGBoost refines predictions using LSTM output as additional feature. 
"""

import os
import json
import logging
import pickle
import warnings
from datetime import datetime
from typing import Tuple, Dict, Any, Optional

import numpy as np
import pandas as pd

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    mean_absolute_percentage_error
)
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

warnings.filterwarnings('ignore')

# ==================== CONFIGURATION ====================
DATA_DIR = "data/processed"
MODEL_DIR = "models"
LOG_FILE = f"{MODEL_DIR}/training.log"

LSTM_MODEL_PATH = f"{MODEL_DIR}/lstm_model.h5"
SCALER_PATH = f"{MODEL_DIR}/scaler.pkl"
XGB_MODEL_PATH = f"{MODEL_DIR}/xgb_model.pkl"
HYBRID_MODEL_PATH = f"{MODEL_DIR}/hybrid_model.pkl"
METRICS_PATH = f"{MODEL_DIR}/hybrid_metrics.json"
METADATA_PATH = f"{MODEL_DIR}/hybrid_metadata.json"

# LSTM hyperparameters
LSTM_PARAMS = {
    "lstm_units": 64,
    "dropout": 0.2,
    "epochs": 20,
    "batch_size": 32,
    "validation_split": 0.1,
    "early_stopping_patience": 3,
    "time_steps": 24  # 24 hours window
}

# XGBoost hyperparameters
XGB_PARAMS = {
    "n_estimators": 300,
    "learning_rate": 0.05,
    "max_depth": 6,
    "subsample": 0.8,
    "random_state": 42,
    "verbosity": 0
}

RANDOM_STATE = 42
VERBOSE = 1

# Setup logging
os.makedirs(MODEL_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE,encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
# =======================================================


def load_training_data(data_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Load preprocessed training and test datasets.
    
    Args:
        data_dir: Directory containing processed data files
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
        
    Raises:
        FileNotFoundError: If required files are missing
    """
    required_files = [
        f"{data_dir}/X_train.csv",
        f"{data_dir}/X_test.csv",
        f"{data_dir}/y_train.csv",
        f"{data_dir}/y_test.csv"
    ]
    
    missing_files = [f for f in required_files if not os.path.exists(f)]
    if missing_files:
        raise FileNotFoundError(
            f"Missing required files: {missing_files}\n"
            f"Please run preprocess.py first."
        )
    
    try:
        X_train = pd.read_csv(f"{data_dir}/X_train.csv")
        X_test = pd.read_csv(f"{data_dir}/X_test.csv")
        y_train = pd.read_csv(f"{data_dir}/y_train.csv").squeeze()
        y_test = pd.read_csv(f"{data_dir}/y_test.csv").squeeze()
        
        logger.info(f"✓ Loaded X_train: {X_train.shape}")
        logger.info(f"✓ Loaded X_test: {X_test.shape}")
        logger.info(f"✓ Loaded y_train: {y_train.shape}")
        logger.info(f"✓ Loaded y_test: {y_test.shape}")
        
        return X_train, X_test, y_train, y_test
        
    except Exception as e:
        raise ValueError(f"Failed to load data: {e}")


def validate_training_data(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series
) -> None:
    """
    Validate data integrity before training.
    
    Args:
        X_train, X_test, y_train, y_test: Training and test datasets
        
    Raises:
        ValueError: If validation fails
    """
    if X_train.isnull().any().any():
        raise ValueError("X_train contains null values")
    if y_train.isnull().any():
        raise ValueError("y_train contains null values")
    
    if len(X_train) != len(y_train):
        raise ValueError("X_train and y_train have different lengths")
    if len(X_test) != len(y_test):
        raise ValueError("X_test and y_test have different lengths")
    
    logger.info("✓ Data validation passed")


def create_sequences(
    X: np.ndarray,
    y: np.ndarray,
    time_steps: int
) -> Tuple[np.ndarray, np.ndarray, list]:
    """
    Create sequences for LSTM training.
    
    Args:
        X: Feature array (n_samples, n_features)
        y: Target array (n_samples,)
        time_steps: Number of historical steps to use
        
    Returns:
        Tuple of (X_sequences, y_sequences, indices_removed)
    """
    Xs, ys, indices = [], [], []
    for i in range(len(X) - time_steps):
        Xs.append(X[i:i + time_steps])
        ys.append(y[i + time_steps])
        indices.append(i + time_steps)
    
    logger.info(f"✓ Created {len(Xs)} sequences with time_steps={time_steps}")
    
    return np.array(Xs), np.array(ys), indices


def train_lstm_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    lstm_params: Dict[str, Any]
) -> Tuple[Sequential, MinMaxScaler]:
    """
    Train LSTM model for temporal pattern capture.
    
    Args:
        X_train: Training features
        y_train: Training target
        lstm_params: LSTM hyperparameters
        
    Returns:
        Tuple of (trained LSTM model, fitted scaler)
    """
    logger.info("Starting LSTM training...")
    
    # Scale features
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X_train)
    
    # Create sequences
    X_seq, y_seq, _ = create_sequences(
        X_scaled,
        y_train.values,
        lstm_params["time_steps"]
    )
    
    # Build LSTM model
    model = Sequential([
        LSTM(
            units=lstm_params["lstm_units"],
            return_sequences=False,
            input_shape=(X_seq.shape[1], X_seq.shape[2])
        ),
        Dropout(lstm_params["dropout"]),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    
    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    
    # Train with early stopping
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=lstm_params["early_stopping_patience"],
        restore_best_weights=True
    )
    
    history = model.fit(
        X_seq,
        y_seq,
        epochs=lstm_params["epochs"],
        batch_size=lstm_params["batch_size"],
        validation_split=lstm_params["validation_split"],
        callbacks=[early_stop],
        verbose=VERBOSE
    )
    
    logger.info(f"✓ LSTM training completed. Final loss: {history.history['loss'][-1]:.4f}")
    
    return model, scaler


def generate_lstm_predictions(
    model: Sequential,
    X_data: pd.DataFrame,
    scaler: MinMaxScaler,
    y_data: pd.Series,
    time_steps: int,
    set_name: str = "Train"
) -> Tuple[np.ndarray, int]:
    """
    Generate LSTM predictions for given dataset.
    
    Args:
        model: Trained LSTM model
        X_data: Features
        scaler: Fitted scaler
        y_data: Target values
        time_steps: Number of time steps used
        set_name: Name of dataset (for logging)
        
    Returns:
        Tuple of (predictions, number of sequences)
    """
    X_scaled = scaler.transform(X_data)
    X_seq, _, _ = create_sequences(X_scaled, y_data.values, time_steps)
    
    predictions = model.predict(X_seq, verbose=0).flatten()
    
    logger.info(
        f"✓ Generated LSTM predictions for {set_name} set: {predictions.shape[0]} samples"
    )
    
    return predictions, X_seq.shape[0]


def train_xgboost_model(
    X_train_xgb: pd.DataFrame,
    y_train_xgb: pd.Series,
    xgb_params: Dict[str, Any]
) -> XGBRegressor:
    """
    Train XGBoost model using LSTM predictions as features.
    
    Args:
        X_train_xgb: Training features (includes LSTM predictions)
        y_train_xgb: Training target
        xgb_params: XGBoost hyperparameters
        
    Returns:
        Trained XGBoost model
    """
    logger.info("Starting XGBoost training...")
    
    xgb_model = XGBRegressor(**xgb_params)
    xgb_model.fit(X_train_xgb, y_train_xgb)
    
    logger.info("✓ XGBoost training completed")
    
    return xgb_model


def evaluate_hybrid_model(
    model: XGBRegressor,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    set_name: str = "Test"
) -> Dict[str, float]:
    """
    Evaluate hybrid model performance.
    
    Args:
        model: Trained XGBoost model
        X_test: Test features
        y_test: Test target
        set_name: Name of dataset (for logging)
        
    Returns:
        Dictionary of evaluation metrics
    """
    y_pred = model.predict(X_test)
    
    metrics = {
        "mae": float(mean_absolute_error(y_test, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
        "r2": float(r2_score(y_test, y_pred)),
        "mape": float(mean_absolute_percentage_error(y_test, y_pred)),
        "n_samples": len(y_test)
    }
    
    logger.info(f"\n{set_name} Set Performance Metrics:")
    logger.info(f"  MAE  : {metrics['mae']:.2f}")
    logger.info(f"  RMSE : {metrics['rmse']:.2f}")
    logger.info(f"  R²   : {metrics['r2']:.4f}")
    logger.info(f"  MAPE : {metrics['mape']:.2f}%")
    logger.info(f"  Samples: {metrics['n_samples']}")
    
    return metrics


def save_hybrid_models(
    lstm_model: Sequential,
    scaler: MinMaxScaler,
    xgb_model: XGBRegressor,
    metrics: Dict[str, float],
    lstm_params: Dict[str, Any],
    xgb_params: Dict[str, Any]
) -> None:
    """
    Save all models and artifacts to disk.
    
    Args:
        lstm_model: Trained LSTM model
        scaler: Fitted scaler
        xgb_model: Trained XGBoost model
        metrics: Performance metrics
        lstm_params: LSTM hyperparameters
        xgb_params: XGBoost hyperparameters
    """
    try:
        # Save LSTM model
        lstm_model.save(LSTM_MODEL_PATH)
        logger.info(f"✓ LSTM model saved to {LSTM_MODEL_PATH}")
        
        # Save scaler
        with open(SCALER_PATH, "wb") as f:
            pickle.dump(scaler, f)
        logger.info(f"✓ Scaler saved to {SCALER_PATH}")
        
        # Save XGBoost model
        with open(XGB_MODEL_PATH, "wb") as f:
            pickle.dump(xgb_model, f)
        logger.info(f"✓ XGBoost model saved to {XGB_MODEL_PATH}")
        
        # Save hybrid ensemble
        hybrid = {
            "lstm": lstm_model,
            "scaler": scaler,
            "xgb": xgb_model
        }
        with open(HYBRID_MODEL_PATH, "wb") as f:
            pickle.dump(hybrid, f)
        logger.info(f"✓ Hybrid model ensemble saved to {HYBRID_MODEL_PATH}")
        
        # Save metrics
        with open(METRICS_PATH, "w") as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"✓ Metrics saved to {METRICS_PATH}")
        
        # Save metadata
        metadata = {
            "training_date": datetime.now().isoformat(),
            "model_type": "Hybrid LSTM + XGBoost",
            "lstm_params": lstm_params,
            "xgb_params": xgb_params,
            "metrics": metrics
        }
        with open(METADATA_PATH, "w") as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"✓ Metadata saved to {METADATA_PATH}")
        
    except Exception as e:
        logger.error(f"Failed to save models: {e}")
        raise


def main() -> None:
    """Main hybrid model training pipeline."""
    try:
        logger.info("=" * 70)
        logger.info("Starting Hybrid LSTM + XGBoost Training Pipeline")
        logger.info("=" * 70)
        
        # Load data
        X_train, X_test, y_train, y_test = load_training_data(DATA_DIR)
        
        # Validate data
        validate_training_data(X_train, X_test, y_train, y_test)
        
        # ==================== LSTM TRAINING ====================
        lstm_model, scaler = train_lstm_model(X_train, y_train, LSTM_PARAMS)
        
        # Generate LSTM predictions
        lstm_train_pred, n_train_seq = generate_lstm_predictions(
            lstm_model, X_train, scaler, y_train,
            LSTM_PARAMS["time_steps"], "Train"
        )
        
        lstm_test_pred, n_test_seq = generate_lstm_predictions(
            lstm_model, X_test, scaler, y_test,
            LSTM_PARAMS["time_steps"], "Test"
        )
        
        # ==================== DATA ALIGNMENT ====================
        # Align data after sequence creation
        time_steps = LSTM_PARAMS["time_steps"]
        X_train_xgb = X_train.iloc[time_steps:].reset_index(drop=True).copy()
        X_test_xgb = X_test.iloc[time_steps:].reset_index(drop=True).copy()
        
        y_train_xgb = y_train.iloc[time_steps:].reset_index(drop=True)
        y_test_xgb = y_test.iloc[time_steps:].reset_index(drop=True)
        
        # Add LSTM predictions as features
        X_train_xgb["lstm_pred"] = lstm_train_pred[:len(X_train_xgb)]
        X_test_xgb["lstm_pred"] = lstm_test_pred[:len(X_test_xgb)]
        
        logger.info(f"✓ Data aligned for XGBoost training")
        logger.info(f"  X_train_xgb shape: {X_train_xgb.shape}")
        logger.info(f"  X_test_xgb shape: {X_test_xgb.shape}")
        
        # ==================== XGBOOST TRAINING ====================
        xgb_model = train_xgboost_model(X_train_xgb, y_train_xgb, XGB_PARAMS)
        
        # ==================== EVALUATION ====================
        metrics = evaluate_hybrid_model(xgb_model, X_test_xgb, y_test_xgb, "Test")
        
        # ==================== SAVE ARTIFACTS ====================
        save_hybrid_models(
            lstm_model, scaler, xgb_model,
            metrics, LSTM_PARAMS, XGB_PARAMS
        )
        
        logger.info("=" * 70)
        logger.info("✓ Hybrid model training completed successfully!")
        logger.info("=" * 70)
        
    except Exception as e:
        logger.error(f"✗ Training pipeline failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
