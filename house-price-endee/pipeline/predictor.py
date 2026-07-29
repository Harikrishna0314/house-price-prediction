"""
pipeline/predictor.py
Wraps model + embedder + Endee store into a single PricePredictor class.
"""

import os
import sys
import pandas as pd
import joblib
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import MODEL_PATH, EMBEDDER_PATH, EMBEDDING_DIM
from model.embeddings import PropertyEmbedder
from vector_db.endee_store import EndeeVectorStore


# ──────────────────────────────────────────────────────────────
#  Sample properties for the demo
# ──────────────────────────────────────────────────────────────

SAMPLES = [
    {
        "location": "Tech District", "property_type": "Apartment",
        "area_sqft": 1200, "bedrooms": 3, "bathrooms": 2,
        "age_years": 4,  "parking_spots": 1, "condition_score": 8,
    },
    {
        "location": "Downtown Core", "property_type": "Penthouse",
        "area_sqft": 3200, "bedrooms": 5, "bathrooms": 4,
        "age_years": 2,  "parking_spots": 3, "condition_score": 10,
    },
    {
        "location": "Suburban North", "property_type": "House",
        "area_sqft": 2100, "bedrooms": 4, "bathrooms": 3,
        "age_years": 15, "parking_spots": 2, "condition_score": 6,
    },
    {
        "location": "Old Town", "property_type": "Villa",
        "area_sqft": 2800, "bedrooms": 4, "bathrooms": 3,
        "age_years": 8,  "parking_spots": 2, "condition_score": 9,
    },
]


# ──────────────────────────────────────────────────────────────
#  PricePredictor
# ──────────────────────────────────────────────────────────────

class PricePredictor:
    """
    High-level inference class.

    predict(property_dict, top_k=5)
    --------------------------------
    1. Encodes the input property → 8-D float32 vector
    2. Runs XGBoost to estimate price
    3. Queries Endee for the k most similar properties
    4. Returns predicted price + range + similar homes

    This two-step approach (ML model + vector DB retrieval) is the
    core innovation: the model gives accuracy, Endee gives explainability.
    """

    def __init__(self):
        self.model    = joblib.load(MODEL_PATH)
        self.embedder = PropertyEmbedder.load(EMBEDDER_PATH)
        self.store    = EndeeVectorStore.load(dim=EMBEDDING_DIM)

    def predict(self, prop: dict, top_k: int = 5) -> dict:
        df  = pd.DataFrame([prop])
        emb = self.embedder.transform(df)           # (1, 8) float32

        # ML prediction
        price = float(self.model.predict(emb)[0])
        low   = round(price * 0.92, -4)
        high  = round(price * 1.08, -4)

        # Endee similarity search
        similar = self.store.search(emb, k=top_k)

        return {
            "input"              : prop,
            "predicted_price"   : round(price, -4),
            "price_range"       : {"low": low, "high": high},
            "price_per_sqft"    : round(price / prop["area_sqft"], 0),
            "similar_properties": similar,
        }


# ──────────────────────────────────────────────────────────────
#  Helper — pretty-print a price
# ──────────────────────────────────────────────────────────────

def fmt(n: float) -> str:
    if n >= 10_000_000:
        return f"₹{n / 10_000_000:.2f} Cr"
    if n >= 100_000:
        return f"₹{n / 100_000:.1f} L"
    return f"₹{n:,.0f}"


# ──────────────────────────────────────────────────────────────
#  CLI demo
# ──────────────────────────────────────────────────────────────

def run_demo() -> None:
    print(f"\n{'='*52}")
    print(f"  STEP 3 — Prediction Demo")
    print(f"{'='*52}")

    predictor = PricePredictor()

    for i, prop in enumerate(SAMPLES, 1):
        res = predictor.predict(prop, top_k=3)

        print(f"\n  ┌─ Property #{i} {'─'*36}")
        print(f"  │  Location      : {prop['location']}")
        print(f"  │  Type          : {prop['property_type']}")
        print(f"  │  Area / Beds   : {prop['area_sqft']:,} sqft  |  "
              f"{prop['bedrooms']} bed  |  {prop['bathrooms']} bath")
        print(f"  │  Age / Cond.   : {prop['age_years']} yrs  |  "
              f"score {prop['condition_score']}/10  |  "
              f"{prop['parking_spots']} parking")
        print(f"  ├{'─'*48}")
        print(f"  │  ★ Predicted   : {fmt(res['predicted_price'])}")
        print(f"  │    Range       : {fmt(res['price_range']['low'])} "
              f"– {fmt(res['price_range']['high'])}")
        print(f"  │    Per sq ft   : ₹{res['price_per_sqft']:,.0f}")
        print(f"  ├{'─'*48}")
        print(f"  │  Endee — {len(res['similar_properties'])} similar homes:")
        for j, sim in enumerate(res["similar_properties"], 1):
            print(f"  │    {j}. {sim.get('location','?'):<18} "
                  f"{sim.get('property_type','?'):<12} "
                  f"{sim.get('area_sqft', '?'):>5} sqft  "
                  f"sim={sim['similarity_score']:.3f}")
        print(f"  └{'─'*48}")

    print(f"\n  Demo complete!")
    print(f"  Start REST API:  python main.py server")
    print(f"  API docs:        http://localhost:8000/docs\n")


if __name__ == "__main__":
    run_demo()
