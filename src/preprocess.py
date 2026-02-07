"""
Data preprocessing pipeline for ride demand forecasting.
Loads raw data, performs feature engineering, encoding, and train-test split.
"""

import os
import logging
import pickle
from pathlib import Path
from typing import Tuple

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Configuration
INPUT_PATH = "data/ride_demand_data.csv"
OUTPUT_DIR = "data/processed"
TRAIN_OUTPUT = f"{OUTPUT_DIR}/X_train.csv"
TEST_OUTPUT = f"{OUTPUT_DIR}/X_test.csv"
TRAIN_TARGET_OUTPUT = f"{OUTPUT_DIR}/y_train.csv"
TEST_TARGET_OUTPUT = f"{OUTPUT_DIR}/y_test.csv"
ENCODER_OUTPUT = f"{OUTPUT_DIR}/label_encoder.pkl"
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def validate_data(df: pd.DataFrame) -> None:
    """
    Validate dataset integrity and quality.
    
    Args:
        df: DataFrame to validate
        
    Raises:
        ValueError: If validation fails
    """
    required_columns = ["datetime", "area_name", "ride_demand"]
    missing_cols = [col for col in required_columns if col not in df.columns]
    
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    if df.empty:
        raise ValueError("Dataset is empty")
    
    if df.isnull().any().any():
        null_cols = df.columns[df.isnull().any()].tolist()
        logger.warning(f"Found null values in columns: {null_cols}")
        df.dropna(inplace=True)
        logger.info(f"Dropped {len(df)} rows with null values")
    
    if (df["ride_demand"] < 0).any():
        raise ValueError("ride_demand contains negative values")
    
    logger.info(f"✓ Data validation passed. Records: {len(df)}")


def load_data(file_path: str) -> pd.DataFrame:
    """
    Load dataset with error handling.
    
    Args:
        file_path: Path to CSV file
        
    Returns:
        pd.DataFrame: Loaded dataset
        
    Raises:
        FileNotFoundError: If file doesn't exist
        pd.errors.ParserError: If file is malformed
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset not found: {file_path}")
    
    try:
        df = pd.read_csv(file_path)
        logger.info(f"✓ Loaded dataset from {file_path}. Shape: {df.shape}")
        return df
    except pd.errors.ParserError as e:
        raise pd.errors.ParserError(f"Failed to parse CSV: {e}")


def preprocess_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, LabelEncoder]:
    """
    Preprocess features and encode categorical variables.
    
    Args:
        df: Raw DataFrame
        
    Returns:
        Tuple of (processed DataFrame, fitted LabelEncoder)
    """
    df = df.copy()
    
    # Feature engineering from datetime
    df["datetime"] = pd.to_datetime(df["datetime"])
    df["hour"] = df["datetime"].dt.hour
    df["day_of_week"] = df["datetime"].dt.dayofweek
    df["month"] = df["datetime"].dt.month
    
    # Drop datetime (replaced with extracted features)
    df = df.drop(columns=["datetime"])
    logger.info("✓ Extracted temporal features from datetime")
    
    # Encode categorical variable (fit on entire data for now - to be improved)
    le = LabelEncoder()
    df["area_name"] = le.fit_transform(df["area_name"])
    logger.info(f"✓ Encoded area_name with {len(le.classes_)} unique values")
    
    return df, le


def validate_columns(df: pd.DataFrame) -> None:
    """Validate that ride_demand target column exists."""
    if "ride_demand" not in df.columns:
        raise ValueError("Target column 'ride_demand' not found in dataset")


def prepare_train_test_split(
    df: pd.DataFrame, 
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split preprocessed data into train and test sets.
    
    Args:
        df: Preprocessed DataFrame with target column
        test_size: Proportion of test data
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    validate_columns(df)
    
    X = df.drop(columns=["ride_demand"])
    y = df["ride_demand"]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    logger.info(
        f"✓ Train-test split completed. "
        f"Train: {X_train.shape[0]} | Test: {X_test.shape[0]}"
    )
    
    return X_train, X_test, y_train, y_test


def save_processed_data(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    encoder: LabelEncoder
) -> None:
    """
    Save preprocessed data and encoder to output directory.
    
    Args:
        X_train, X_test, y_train, y_test: Split datasets
        encoder: Fitted LabelEncoder
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    try:
        X_train.to_csv(TRAIN_OUTPUT, index=False)
        X_test.to_csv(TEST_OUTPUT, index=False)
        y_train.to_csv(TRAIN_TARGET_OUTPUT, index=False)
        y_test.to_csv(TEST_TARGET_OUTPUT, index=False)
        
        with open(ENCODER_OUTPUT, "wb") as f:
            pickle.dump(encoder, f)
        
        logger.info(f"✓ Saved processed data to {OUTPUT_DIR}/")
        logger.info(f"  - {TRAIN_OUTPUT}")
        logger.info(f"  - {TEST_OUTPUT}")
        logger.info(f"  - {ENCODER_OUTPUT}")
        
    except IOError as e:
        logger.error(f"Failed to save processed data: {e}")
        raise


def main() -> None:
    """Main preprocessing pipeline."""
    try:
        # Load and validate
        df = load_data(INPUT_PATH)
        validate_data(df)
        
        # Preprocess
        df, encoder = preprocess_features(df)
        
        # Split
        X_train, X_test, y_train, y_test = prepare_train_test_split(df)
        
        # Save
        save_processed_data(X_train, X_test, y_train, y_test, encoder)
        
        logger.info("=" * 50)
        logger.info("✓ Preprocessing pipeline completed successfully!")
        logger.info("=" * 50)
        
    except Exception as e:
        logger.error(f"✗ Preprocessing failed: {e}")
        raise


if __name__ == "__main__":
    main()
