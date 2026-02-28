"""
Data preprocessing pipeline for multi-city ride demand forecasting.
"""

import os
import logging
import pickle
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# ================= CONFIG =================

INPUT_PATH = "data/ride_demand_data.csv"
OUTPUT_DIR = "data/processed"

TRAIN_OUTPUT = f"{OUTPUT_DIR}/X_train.csv"
TEST_OUTPUT = f"{OUTPUT_DIR}/X_test.csv"
TRAIN_TARGET_OUTPUT = f"{OUTPUT_DIR}/y_train.csv"
TEST_TARGET_OUTPUT = f"{OUTPUT_DIR}/y_test.csv"

ENCODER_OUTPUT = f"{OUTPUT_DIR}/label_encoder.pkl"

TEST_SIZE = 0.2
RANDOM_STATE = 42

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ================= VALIDATION =================

def validate_data(df: pd.DataFrame) -> None:
    required_columns = [
        "datetime",
        "location",
        "ride_demand"
    ]

    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if df.empty:
        raise ValueError("Dataset is empty")

    if (df["ride_demand"] < 0).any():
        raise ValueError("ride_demand contains negative values")

    logger.info(f"✓ Validation passed. Records: {len(df)}")


# ================= LOAD =================

def load_data(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_csv(path)
    logger.info(f"✓ Loaded dataset. Shape: {df.shape}")
    return df


# ================= PREPROCESS =================

def preprocess_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, LabelEncoder]:

    df = df.copy()

    # Convert datetime
    df["datetime"] = pd.to_datetime(df["datetime"])

    # Extract time features (already present but safe)
    df["hour"] = df["datetime"].dt.hour
    df["day_of_week"] = df["datetime"].dt.dayofweek
    df["month"] = df["datetime"].dt.month

    # Encode location (city + area)
    encoder = LabelEncoder()
    df["location"] = encoder.fit_transform(df["location"])

    logger.info(f"✓ Encoded location ({len(encoder.classes_)} unique values)")

    # Drop unused columns
    df = df.drop(columns=["datetime", "city", "area_name"])

    return df, encoder


# ================= SPLIT =================

def prepare_train_test_split(df: pd.DataFrame):

    X = df.drop(columns=["ride_demand"])
    y = df["ride_demand"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE
    )

    logger.info(f"✓ Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")

    return X_train, X_test, y_train, y_test


# ================= SAVE =================

def save_processed_data(
    X_train,
    X_test,
    y_train,
    y_test,
    encoder
):

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    X_train.to_csv(TRAIN_OUTPUT, index=False)
    X_test.to_csv(TEST_OUTPUT, index=False)
    y_train.to_csv(TRAIN_TARGET_OUTPUT, index=False)
    y_test.to_csv(TEST_TARGET_OUTPUT, index=False)

    with open(ENCODER_OUTPUT, "wb") as f:
        pickle.dump(encoder, f)

    logger.info("✓ Processed data saved successfully")


# ================= MAIN =================

def main():

    df = load_data(INPUT_PATH)
    validate_data(df)

    df, encoder = preprocess_features(df)

    X_train, X_test, y_train, y_test = prepare_train_test_split(df)

    save_processed_data(
        X_train,
        X_test,
        y_train,
        y_test,
        encoder
    )

    logger.info("✓ Preprocessing completed successfully!")


if __name__ == "__main__":
    main()