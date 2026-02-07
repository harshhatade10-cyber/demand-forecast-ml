"""
Generate synthetic ride demand dataset for demand forecasting.
Creates hourly demand data across multiple areas with realistic patterns.
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict

# Configuration constants
RANDOM_SEED = 42
START_DATE = "2023-01-01"
END_DATE = "2023-03-31"
FREQUENCY = "h"  # lowercase 'h' for pandas 2.0+ compatibility (hourly)
BASE_FARE_MIN = 50
BASE_FARE_MAX = 150
BASE_DEMAND = 20
PEAK_MORNING_DEMAND = 30
PEAK_EVENING_DEMAND = 40
AREA_DEMAND_MULTIPLIER = 5
DEMAND_NOISE_MIN = -10
DEMAND_NOISE_MAX = 10
OUTPUT_PATH = "data/ride_demand_data.csv"

# Area mapping
AREAS: Dict[int, str] = {
    1: "Sitabuldi",
    2: "Dharampeth",
    3: "Civi_lines",
    4: "IT_Park",
    5: "Sadar",
    6: "Railway_Station"
}


def generate_dataset() -> pd.DataFrame:
    """
    Generate synthetic ride demand dataset with realistic hourly patterns.
    
    Returns:
        pd.DataFrame: DataFrame with columns: datetime, area_id, base_fare, 
                     area_name, hour, day, ride_demand
    """
    # Set seed for reproducibility
    np.random.seed(RANDOM_SEED)
    
    # Generate date range
    dates = pd.date_range(start=START_DATE, end=END_DATE, freq=FREQUENCY)
    
    # Create base DataFrame
    area_ids = np.random.choice(list(AREAS.keys()), len(dates))
    base_fares = np.random.randint(BASE_FARE_MIN, BASE_FARE_MAX + 1, len(dates))
    
    df = pd.DataFrame({
        "datetime": dates,
        "area_id": area_ids,
        "base_fare": base_fares,
    })
    
    # Add derived features
    df["area_name"] = df["area_id"].map(AREAS)
    df["hour"] = df["datetime"].dt.hour
    df["day"] = df["datetime"].dt.dayofweek
    
    # Calculate ride demand with realistic patterns
    morning_peak = df["hour"].between(8, 10).astype(int)  # 8-10 AM
    evening_peak = df["hour"].between(17, 20).astype(int)  # 5-8 PM
    random_noise = np.random.randint(DEMAND_NOISE_MIN, DEMAND_NOISE_MAX + 1, len(df))
    
    df["ride_demand"] = (
        BASE_DEMAND
        + (morning_peak * PEAK_MORNING_DEMAND)
        + (evening_peak * PEAK_EVENING_DEMAND)
        + (df["area_id"] * AREA_DEMAND_MULTIPLIER)
        + random_noise
    ).astype(int)
    
    return df


def save_dataset(df: pd.DataFrame, output_path: str = OUTPUT_PATH) -> None:
    """
    Save DataFrame to CSV with error handling.
    
    Args:
        df: DataFrame to save
        output_path: Path where CSV will be saved
        
    Raises:
        IOError: If directory doesn't exist or file write fails
    """
    try:
        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
        # Save to CSV
        df.to_csv(output_path, index=False)
        print(f"✓ Dataset successfully generated and saved to: {output_path}")
        print(f"✓ Dataset shape: {df.shape[0]} rows × {df.shape[1]} columns")
        
    except IOError as e:
        print(f"✗ Error saving dataset: {e}")
        raise
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        raise


if __name__ == "__main__":
    # Generate and save dataset
    dataset = generate_dataset()
    save_dataset(dataset)
