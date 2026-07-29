"""
api/app.py
FastAPI REST API for the House Price Predictor.

Endpoints:
  GET  /           →  welcome + links
  GET  /health     →  service status
  POST /predict    →  predict price + find similar homes via Endee
  GET  /locations  →  list valid location values
  GET  /types      →  list valid property types

Run with:
    python main.py server
    # or directly:
    uvicorn api.app:app --reload --port 8000
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from config import LOCATIONS, PROPERTY_TYPES
from pipeline.predictor import PricePredictor, fmt

# ── App setup ─────────────────────────────────────────────────

app = FastAPI(
    title       = "House Price Predictor — Endee Vector DB",
    description = (
        "ML-powered house price prediction using XGBoost + "
        "Endee vector database for similar-property retrieval."
    ),
    version = "1.0.0",
)

# Load predictor once at startup (shared across requests)
_predictor: PricePredictor | None = None

def get_predictor() -> PricePredictor:
    global _predictor
    if _predictor is None:
        _predictor = PricePredictor()
    return _predictor


# ── Request / Response schemas ────────────────────────────────

class PropertyInput(BaseModel):
    location        : str = Field(..., example="Tech District",
                                  description=f"One of: {LOCATIONS}")
    property_type   : str = Field(..., example="Apartment",
                                  description=f"One of: {PROPERTY_TYPES}")
    area_sqft       : int = Field(..., ge=200,  le=10000, example=1200)
    bedrooms        : int = Field(..., ge=1,    le=10,    example=3)
    bathrooms       : int = Field(..., ge=1,    le=8,     example=2)
    age_years       : int = Field(..., ge=0,    le=100,   example=4)
    parking_spots   : int = Field(..., ge=0,    le=10,    example=1)
    condition_score : int = Field(..., ge=1,    le=10,    example=8,
                                  description="1 = poor … 10 = excellent")
    top_k           : int = Field(5,  ge=1,    le=20,
                                  description="Number of similar properties to return")


class PredictionResponse(BaseModel):
    predicted_price    : float
    formatted_price    : str
    price_range        : dict
    price_per_sqft     : float
    similar_properties : list
    model_info         : dict


# ── Routes ────────────────────────────────────────────────────

@app.get("/", tags=["Info"])
def root():
    return {
        "service" : "House Price Predictor (Endee Vector DB)",
        "docs"    : "/docs",
        "health"  : "/health",
        "predict" : "POST /predict",
        "version" : "1.0.0",
    }


@app.get("/health", tags=["Info"])
def health():
    try:
        p = get_predictor()
        return {
            "status"         : "ok",
            "model"          : "XGBoost",
            "vector_backend" : "Endee (FAISS fallback if Endee not installed)",
            "embedding_dim"  : 8,
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Service unavailable: {e}")


@app.get("/locations", tags=["Metadata"])
def list_locations():
    return {"locations": LOCATIONS}


@app.get("/types", tags=["Metadata"])
def list_types():
    return {"property_types": PROPERTY_TYPES}


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(body: PropertyInput):
    """
    Predict house price using XGBoost + retrieve similar properties from Endee.

    Steps (inside the endpoint):
    1. Encode the input → 8-D float32 vector (via PropertyEmbedder)
    2. Run XGBoost regression  →  estimated price
    3. Query Endee vector store  →  top-k similar sold properties
    4. Return price + range + similar homes
    """
    if body.location not in LOCATIONS:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid location '{body.location}'. Valid: {LOCATIONS}"
        )
    if body.property_type not in PROPERTY_TYPES:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid property_type '{body.property_type}'. Valid: {PROPERTY_TYPES}"
        )

    prop = body.model_dump(exclude={"top_k"})
    try:
        result = get_predictor().predict(prop, top_k=body.top_k)
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=503,
            detail=f"Model not trained yet. Run: python main.py train — {e}"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return PredictionResponse(
        predicted_price    = result["predicted_price"],
        formatted_price    = fmt(result["predicted_price"]),
        price_range        = result["price_range"],
        price_per_sqft     = result["price_per_sqft"],
        similar_properties = result["similar_properties"],
        model_info         = {
            "algorithm"     : "XGBoost Regression",
            "vector_db"     : "Endee",
            "embedding_dim" : 8,
            "confidence_pct": 88,
        },
    )
