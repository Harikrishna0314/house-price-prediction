"""
model/embeddings.py
Converts raw property features into normalised 8-D float32 vectors.
These vectors are stored in / queried against the Endee vector database.

Feature layout (index → feature):
  0  location        (label-encoded integer)
  1  property_type   (label-encoded integer)
  2  area_sqft
  3  bedrooms
  4  bathrooms
  5  age_years
  6  parking_spots
  7  condition_score

StandardScaler ensures every feature contributes equally to
the L2 distance used by Endee's ANN search.
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import LOCATIONS, PROPERTY_TYPES, EMBEDDER_PATH


class PropertyEmbedder:
    """Fits on training data, then transforms any property DataFrame
    into a normalised float32 vector ready for Endee."""

    def __init__(self):
        from sklearn.preprocessing import StandardScaler, LabelEncoder
        self.scaler   = StandardScaler()
        self.loc_enc  = LabelEncoder().fit(sorted(LOCATIONS))
        self.type_enc = LabelEncoder().fit(sorted(PROPERTY_TYPES))
        self.fitted   = False

    def _to_raw(self, df: pd.DataFrame) -> np.ndarray:
        return np.column_stack([
            self.loc_enc.transform(df["location"]),
            self.type_enc.transform(df["property_type"]),
            df["area_sqft"].values,
            df["bedrooms"].values,
            df["bathrooms"].values,
            df["age_years"].values,
            df["parking_spots"].values,
            df["condition_score"].values,
        ]).astype(np.float64)

    def fit(self, df: pd.DataFrame) -> "PropertyEmbedder":
        self.scaler.fit(self._to_raw(df))
        self.fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("Call fit() or load() before transform().")
        return self.scaler.transform(self._to_raw(df)).astype(np.float32)

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        return self.fit(df).transform(df)

    def save(self, path: str = EMBEDDER_PATH) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self, path)
        print(f"  ✔  Embedder saved  →  {path}")

    @classmethod
    def load(cls, path: str = EMBEDDER_PATH) -> "PropertyEmbedder":
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Embedder not found at '{path}'. Run: python main.py train"
            )
        return joblib.load(path)
