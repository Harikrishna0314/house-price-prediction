"""
model/train.py
Trains the XGBoost regression model and builds the Endee vector index.
"""

import os
import sys
import pandas as pd
import joblib
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import DATA_PATH, MODEL_PATH, EMBEDDING_DIM, XGB_PARAMS
from model.embeddings import PropertyEmbedder
from vector_db.endee_store import EndeeVectorStore


def train(data_path: str = DATA_PATH) -> tuple:
    """
    Full training pipeline:
      1. Load dataset
      2. Fit PropertyEmbedder  →  generate 8-D vectors
      3. Insert all vectors + metadata into Endee
      4. Train XGBoost on the embeddings
      5. Save model + embedder artifacts

    Returns (model, embedder, store)
    """
    from xgboost import XGBRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_absolute_error, r2_score

    print(f"\n{'='*52}")
    print(f"  STEP 2 — Training XGBoost + building Endee index")
    print(f"{'='*52}")

    # ── 1. Load data ──────────────────────────────────────────
    df = pd.read_csv(data_path)
    print(f"  Loaded {len(df):,} records  from  {data_path}")

    # ── 2. Embed features ─────────────────────────────────────
    embedder   = PropertyEmbedder()
    embeddings = embedder.fit_transform(df)          # shape: (n, 8)

    # ── 3. Build Endee vector store ───────────────────────────
    store   = EndeeVectorStore(dim=EMBEDDING_DIM)
    records = df.drop(columns=["price"]).to_dict("records")
    store.add(embeddings, records)

    # ── 4. Train XGBoost ──────────────────────────────────────
    X, y = embeddings, df["price"].values
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = XGBRegressor(**XGB_PARAMS)
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_te, y_te)],
        verbose=False,
    )

    # ── 5. Evaluate ───────────────────────────────────────────
    preds = model.predict(X_te)
    mae   = mean_absolute_error(y_te, preds)
    r2    = r2_score(y_te, preds)

    print(f"\n  ── Model Performance ───────────────────────")
    print(f"     MAE (mean abs error) : ₹{mae:>14,.0f}")
    print(f"     R² score             :  {r2:.4f}")
    print(f"  ────────────────────────────────────────────")

    # ── 6. Save artifacts ─────────────────────────────────────
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"  ✔ XGBoost model saved  →  {MODEL_PATH}")

    embedder.save()
    store.save()

    print("\n  Training complete. All artifacts saved.")
    return model, embedder, store


if __name__ == "__main__":
    train()
