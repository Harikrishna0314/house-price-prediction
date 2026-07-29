"""
tests/test_pipeline.py
Basic smoke tests for all pipeline components.
Run: python main.py test
 or: pytest tests/
"""

import os
import sys
import numpy as np
import pandas as pd
import tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ──────────────────────────────────────────────────────────────
#  Test helpers
# ──────────────────────────────────────────────────────────────

PASS = "  ✔"
FAIL = "  ✘"

def check(label: str, condition: bool) -> bool:
    status = PASS if condition else FAIL
    print(f"{status}  {label}")
    return condition


# ──────────────────────────────────────────────────────────────
#  Individual test functions (also usable with pytest)
# ──────────────────────────────────────────────────────────────

def test_data_generation():
    from data.generate_data import generate_dataset
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
        tmp = f.name

    try:
        import config
        original = config.DATA_PATH
        config.DATA_PATH = tmp

        df = generate_dataset(n=50, seed=0)
        assert len(df) == 50,                        "Wrong row count"
        assert "price" in df.columns,                "Missing price column"
        assert df["price"].min() >= 500_000,         "Price below floor"
        assert df["area_sqft"].between(400, 5000).all(), "Area out of range"
        config.DATA_PATH = original
    finally:
        os.unlink(tmp)


def test_embedder():
    from model.embeddings import PropertyEmbedder
    from data.generate_data import generate_dataset

    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
        tmp = f.name

    import config
    original = config.DATA_PATH
    config.DATA_PATH = tmp

    try:
        df = generate_dataset(n=30, seed=1)
        config.DATA_PATH = original

        emb = PropertyEmbedder()
        out = emb.fit_transform(df)

        assert out.shape == (30, 8),          "Wrong embedding shape"
        assert out.dtype == np.float32,       "Wrong dtype"
        assert not np.isnan(out).any(),       "NaN in embeddings"

        # Single-row transform
        single = emb.transform(df.iloc[:1])
        assert single.shape == (1, 8),        "Single row shape wrong"
    finally:
        os.unlink(tmp)


def test_vector_store():
    from vector_db.endee_store import EndeeVectorStore

    dim  = 8
    store = EndeeVectorStore(dim=dim)

    vecs = np.random.rand(10, dim).astype(np.float32)
    meta = [{"id": i, "location": "Test", "price": 1_000_000 * i} for i in range(10)]
    store.add(vecs, meta)

    query   = vecs[0].reshape(1, -1)
    results = store.search(query, k=3)

    assert len(results) == 3,                          "Wrong result count"
    assert "similarity_score" in results[0],           "Missing similarity_score"
    assert results[0]["similarity_score"] <= 1.0,      "Score out of range"
    assert results[0]["location"] == "Test",           "Metadata not returned"


def test_full_prediction():
    """End-to-end test: generate → train → predict (uses temp paths)."""
    import config, joblib, tempfile

    orig_data     = config.DATA_PATH
    orig_model    = config.MODEL_PATH
    orig_embedder = config.EMBEDDER_PATH
    orig_endee    = config.ENDEE_DB_PATH

    with tempfile.TemporaryDirectory() as tmpdir:
        config.DATA_PATH     = os.path.join(tmpdir, "data.csv")
        config.MODEL_PATH    = os.path.join(tmpdir, "model.pkl")
        config.EMBEDDER_PATH = os.path.join(tmpdir, "emb.pkl")
        config.ENDEE_DB_PATH = os.path.join(tmpdir, "endee.pkl")

        os.makedirs(os.path.join(tmpdir), exist_ok=True)

        from data.generate_data import generate_dataset
        from model.train import train

        generate_dataset(n=100, seed=42)
        model, embedder, store = train(data_path=config.DATA_PATH)

        from pipeline.predictor import PricePredictor
        predictor = PricePredictor()

        result = predictor.predict({
            "location": "Tech District", "property_type": "Apartment",
            "area_sqft": 1200, "bedrooms": 3, "bathrooms": 2,
            "age_years": 4, "parking_spots": 1, "condition_score": 8,
        }, top_k=3)

        assert result["predicted_price"] > 0,               "Price not positive"
        assert result["price_range"]["low"] < result["predicted_price"], "Range low wrong"
        assert len(result["similar_properties"]) <= 3,      "Too many similar"

    # Restore original paths
    config.DATA_PATH     = orig_data
    config.MODEL_PATH    = orig_model
    config.EMBEDDER_PATH = orig_embedder
    config.ENDEE_DB_PATH = orig_endee


# ──────────────────────────────────────────────────────────────
#  Runner
# ──────────────────────────────────────────────────────────────

def run_tests():
    print(f"\n{'='*52}")
    print(f"  Running Tests")
    print(f"{'='*52}\n")

    suite = [
        ("Data generation",         test_data_generation),
        ("Feature embedder",         test_embedder),
        ("Vector store (Endee/FAISS)", test_vector_store),
        ("Full prediction pipeline", test_full_prediction),
    ]

    passed = 0
    for name, fn in suite:
        try:
            fn()
            check(name, True)
            passed += 1
        except Exception as e:
            check(name, False)
            print(f"     Error: {e}")

    total = len(suite)
    print(f"\n  {passed}/{total} tests passed\n")
    if passed < total:
        sys.exit(1)


if __name__ == "__main__":
    run_tests()
