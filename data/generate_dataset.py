"""
Advanced synthetic ride demand dataset for multi-city forecasting.
Includes city-level, area-level, peak-hour, and weekend behavior.
"""

import os
import pandas as pd
import numpy as np
from typing import Dict, List

# ================= CONFIG =================

RANDOM_SEED = 42
START_DATE = "2023-01-01"
END_DATE = "2023-03-31"
FREQUENCY = "h"

BASE_FARE_MIN = 50
BASE_FARE_MAX = 150

BASE_DEMAND = 20
PEAK_MORNING_DEMAND = 30
PEAK_EVENING_DEMAND = 40

DEMAND_NOISE_MIN = -8
DEMAND_NOISE_MAX = 8

OUTPUT_PATH = "data/ride_demand_data.csv"

# ================= CITY + AREA =================

CITIES: Dict[str, List[str]] = {
    "Nagpur": ["Sitabuldi", "Dharampeth", "Civil_Lines", "IT_Park"],
    "Pune": ["Baner", "Hinjewadi", "Kothrud", "Viman_Nagar"],
    "Mumbai": ["Andheri", "Bandra", "Dadar", "Powai"],
    "Amravati": ["Rajapeth", "Camp", "Badnera", "Irwin"]
}

# City level multiplier
CITY_MULTIPLIER = {
    "Nagpur": 1.0,
    "Pune": 1.6,
    "Mumbai": 2.5,
    "Amravati": 0.8
}

# Area level multiplier
AREA_MULTIPLIER = {
    "Sitabuldi": 10,
    "Dharampeth": 6,
    "Civil_Lines": 8,
    "IT_Park": 18,

    "Baner": 12,
    "Hinjewadi": 22,
    "Kothrud": 9,
    "Viman_Nagar": 14,

    "Andheri": 28,
    "Bandra": 22,
    "Dadar": 18,
    "Powai": 24,

    "Rajapeth": 6,
    "Camp": 4,
    "Badnera": 3,
    "Irwin": 5
}

# Area type behavior
IT_AREAS = ["IT_Park", "Hinjewadi", "Powai"]
BUSINESS_AREAS = ["Sitabuldi", "Andheri", "Bandra", "Baner"]
RESIDENTIAL_AREAS = ["Kothrud", "Civil_Lines", "Rajapeth", "Badnera"]

# ================= GENERATE =================

def generate_dataset() -> pd.DataFrame:

    np.random.seed(RANDOM_SEED)

    dates = pd.date_range(start=START_DATE, end=END_DATE, freq=FREQUENCY)

    rows = []

    for city, areas in CITIES.items():
        for area in areas:
            for dt in dates:

                hour = dt.hour
                day = dt.dayofweek
                month = dt.month
                weekend = 1 if day >= 5 else 0

                base_fare = np.random.randint(BASE_FARE_MIN, BASE_FARE_MAX + 1)

                # Peak hours
                morning_peak = 1 if 8 <= hour <= 10 else 0
                evening_peak = 1 if 17 <= hour <= 20 else 0

                # Weekend boost
                weekend_boost = 15 if weekend else 0

                # IT weekday boost
                it_boost = 20 if (area in IT_AREAS and not weekend and 9 <= hour <= 18) else 0

                # Business morning boost
                business_boost = 15 if (area in BUSINESS_AREAS and 8 <= hour <= 11) else 0

                # Residential evening boost
                residential_boost = 18 if (area in RESIDENTIAL_AREAS and 18 <= hour <= 22) else 0

                random_noise = np.random.randint(DEMAND_NOISE_MIN, DEMAND_NOISE_MAX + 1)

                ride_demand = (
                    BASE_DEMAND
                    + (morning_peak * PEAK_MORNING_DEMAND)
                    + (evening_peak * PEAK_EVENING_DEMAND)
                    + (CITY_MULTIPLIER[city] * 20)
                    + AREA_MULTIPLIER[area]
                    + weekend_boost
                    + it_boost
                    + business_boost
                    + residential_boost
                    + random_noise
                )

                rows.append({
                    "datetime": dt,
                    "city": city,
                    "area_name": area,
                    "location": f"{city}_{area}",
                    "hour": hour,
                    "day_of_week": day,
                    "month": month,
                    "base_fare": base_fare,
                    "ride_demand": int(max(0, ride_demand))
                })

    df = pd.DataFrame(rows)
    return df


def save_dataset(df: pd.DataFrame, output_path: str = OUTPUT_PATH) -> None:

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"✓ Dataset saved to: {output_path}")
    print(f"✓ Shape: {df.shape}")


if __name__ == "__main__":
    dataset = generate_dataset()
    save_dataset(dataset)