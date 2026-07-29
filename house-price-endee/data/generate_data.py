"""
data/generate_data.py
Generates a synthetic house-price dataset and saves it as CSV.
"""

import numpy as np
import pandas as pd
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    LOCATIONS, PROPERTY_TYPES,
    LOCATION_BASE_PRICE, TYPE_MULTIPLIER, DATA_PATH
)


def generate_dataset(n: int = 2000, seed: int = 42) -> pd.DataFrame:
    print(f"\n{'='*52}")
    print(f"  STEP 1 — Generating {n} property records")
    print(f"{'='*52}")

    np.random.seed(seed)

    locations   = np.random.choice(LOCATIONS,       n)
    prop_types  = np.random.choice(PROPERTY_TYPES,  n)
    area        = np.random.randint(400,  5001, n)
    bedrooms    = np.random.randint(1,    7,    n)
    bathrooms   = np.random.randint(1,    6,    n)
    age         = np.random.randint(0,    51,   n)
    parking     = np.random.randint(0,    5,    n)
    condition   = np.random.randint(1,    11,   n)

    prices = []
    for i in range(n):
        base       = LOCATION_BASE_PRICE[locations[i]]
        mult       = TYPE_MULTIPLIER[prop_types[i]]
        age_f      = max(0.6, 1 - age[i] * 0.008)
        cond_f     = 0.75 + (condition[i] / 10) * 0.5
        park_bonus = 1 + parking[i] * 0.03

        price = (
            base * area[i] * mult * age_f * cond_f * park_bonus
            + bedrooms[i]  * 120_000
            + bathrooms[i] * 80_000
        )
        price += np.random.normal(0, price * 0.05)   # ±5 % realistic noise
        prices.append(max(int(price), 500_000))

    df = pd.DataFrame({
        "location"       : locations,
        "property_type"  : prop_types,
        "area_sqft"      : area,
        "bedrooms"       : bedrooms,
        "bathrooms"      : bathrooms,
        "age_years"      : age,
        "parking_spots"  : parking,
        "condition_score": condition,
        "price"          : prices,
    })

    os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
    df.to_csv(DATA_PATH, index=False)
    print(f"  ✔ Saved {len(df)} rows  →  {DATA_PATH}")
    return df


if __name__ == "__main__":
    generate_dataset()
