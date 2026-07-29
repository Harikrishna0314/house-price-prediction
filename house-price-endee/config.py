"""
config.py — shared constants, paths, and hyperparameters
All modules import from here so settings live in one place.
"""

import os

# ── Base paths ────────────────────────────────────────────────
BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
DATA_PATH     = os.path.join(BASE_DIR, "data",      "house_data.csv")
MODEL_PATH    = os.path.join(BASE_DIR, "model",     "xgb_model.pkl")
EMBEDDER_PATH = os.path.join(BASE_DIR, "model",     "embedder.pkl")
ENDEE_DB_PATH = os.path.join(BASE_DIR, "vector_db", "fallback_store.pkl")
ENDEE_DATA_DIR= os.path.join(BASE_DIR, "vector_db", "endee_data")

# ── Domain constants ──────────────────────────────────────────
LOCATIONS = [
    "Downtown Core",
    "Suburban North",
    "East Village",
    "Riverside West",
    "Tech District",
    "Old Town",
]

PROPERTY_TYPES = [
    "Apartment",
    "House",
    "Villa",
    "Townhouse",
    "Penthouse",
]

LOCATION_BASE_PRICE = {
    "Downtown Core":  9500,
    "Suburban North": 5500,
    "East Village":   6800,
    "Riverside West": 7200,
    "Tech District":  8800,
    "Old Town":       6200,
}

TYPE_MULTIPLIER = {
    "Apartment":  1.00,
    "House":      1.15,
    "Villa":      1.45,
    "Townhouse":  1.10,
    "Penthouse":  1.60,
}

# ── Embedding ─────────────────────────────────────────────────
EMBEDDING_DIM = 8   # number of normalised features → vector dimension

# ── XGBoost hyperparameters ───────────────────────────────────
XGB_PARAMS = dict(
    n_estimators     = 400,
    learning_rate    = 0.05,
    max_depth        = 6,
    subsample        = 0.8,
    colsample_bytree = 0.8,
    random_state     = 42,
    n_jobs           = -1,
    verbosity        = 0,
)
