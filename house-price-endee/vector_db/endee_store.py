"""
vector_db/endee_store.py
========================
Endee Vector Database integration for the House Price Predictor.

HOW ENDEE IS USED IN THIS PROJECT:
-----------------------------------
1. Each property (2000 records) is converted to an 8-dimensional
   normalised float32 vector using PropertyEmbedder.

2. All 2000 vectors + their metadata (location, type, area, etc.)
   are inserted into an Endee collection called "house_prices".

3. At prediction time, the query property is embedded into the
   same 8-D space and a similarity search (ANN) is run against
   Endee to find the k most similar real properties.

4. These similar properties provide interpretability —
   "your predicted price is ₹1.2 Cr, and here are 5 comparable
   homes that sold near that price."

WHY ENDEE INSTEAD OF A PLAIN FAISS INDEX?
------------------------------------------
• Endee manages persistence natively — no manual pickle/save needed.
• Endee stores metadata alongside vectors, so results come back
  fully annotated without a separate lookup table.
• Endee's collection API abstracts the ANN algorithm, making it
  easy to swap distance metrics (L2, cosine) without changing
  application code.

ENDEE INSTALLATION:
--------------------
    git clone https://github.com/endee-io/endee.git
    cd endee && pip install -e .

If Endee is not yet installed, this module automatically falls back
to a FAISS-based implementation with an identical interface so the
rest of the pipeline continues to work during development.
"""

from __future__ import annotations
import numpy as np
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import ENDEE_DB_PATH


# ══════════════════════════════════════════════════════════════
#  PRIMARY — Endee-native implementation
# ══════════════════════════════════════════════════════════════

class EndeeVectorStore:
    """
    Wraps the Endee vector database with a simple add / search interface.

    Usage
    -----
    store = EndeeVectorStore(collection="house_prices", dim=8)
    store.add(embedding_matrix, list_of_metadata_dicts)
    results = store.search(query_vector, k=5)
    """

    COLLECTION = "house_prices"

    def __init__(self, dim: int, db_path: str = "./endee_data"):
        """
        Parameters
        ----------
        dim     : dimensionality of each vector (must match embedder output)
        db_path : directory where Endee persists its data
        """
        self.dim     = dim
        self.db_path = db_path
        self._client = None
        self._init_client()

    # ── Endee client initialisation ───────────────────────────

    def _init_client(self) -> None:
        """
        Connect to (or create) an Endee database.

        Adjust the import and constructor to match the version of
        Endee you installed from https://github.com/endee-io/endee.
        Common patterns:

            import endee
            self._client = endee.Client(path=self.db_path)

        or

            from endee import Endee
            self._client = Endee(storage_dir=self.db_path)
        """
        try:
            # ── Try Endee import ──────────────────────────────
            import endee                                          # noqa: F401

            # Pattern A — simple Client class
            try:
                self._client = endee.Client(path=self.db_path)
                self._ensure_collection()
                print("  ✔ Connected to Endee vector database")
                return
            except AttributeError:
                pass

            # Pattern B — Endee class
            try:
                from endee import Endee
                self._client = Endee(storage_dir=self.db_path)
                self._ensure_collection()
                print("  ✔ Connected to Endee vector database")
                return
            except (ImportError, AttributeError):
                pass

            # Pattern C — VectorDB class (some forks)
            try:
                from endee import VectorDB
                self._client = VectorDB(dim=self.dim, path=self.db_path)
                print("  ✔ Connected to Endee (VectorDB mode)")
                return
            except (ImportError, AttributeError):
                pass

            print("  ⚠ Endee imported but API unrecognised — using fallback.")
            self._client = None

        except ImportError:
            print("  ⚠ Endee not installed — using FAISS fallback.")
            print("     To install: pip install -e ./endee   (after cloning)")
            self._client = None

    def _ensure_collection(self) -> None:
        """Create the collection if it does not already exist."""
        try:
            # Try common Endee collection-management methods
            if hasattr(self._client, "create_collection"):
                try:
                    self._client.create_collection(
                        name=self.COLLECTION, dim=self.dim
                    )
                except Exception:
                    pass   # collection may already exist — that's fine

            elif hasattr(self._client, "get_or_create_collection"):
                self._client.get_or_create_collection(
                    name=self.COLLECTION, dim=self.dim
                )
        except Exception:
            pass   # some builds auto-create on first insert

    # ── Core API ──────────────────────────────────────────────

    @property
    def _using_endee(self) -> bool:
        return self._client is not None

    def add(self, embeddings: np.ndarray, records: list[dict]) -> None:
        """
        Insert property vectors and their metadata into Endee.

        Parameters
        ----------
        embeddings : float32 array of shape (n, dim)
        records    : list of n dicts with property metadata
        """
        assert embeddings.shape[1] == self.dim, "Dim mismatch"
        assert embeddings.dtype == np.float32

        if self._using_endee:
            self._endee_add(embeddings, records)
        else:
            self._fallback_add(embeddings, records)

        print(f"  ✔ Vector DB: {len(records)} properties indexed  (dim={self.dim})")

    def search(self, query: np.ndarray, k: int = 5) -> list[dict]:
        """
        Find the k most similar properties to query.

        Returns
        -------
        list of dicts, each containing property metadata +
        'similarity_score' (float, higher = more similar).
        """
        if query.ndim == 2:
            query = query[0]
        query = query.astype(np.float32)

        if self._using_endee:
            return self._endee_search(query, k)
        return self._fallback_search(query, k)

    # ── Endee-specific helpers ────────────────────────────────

    def _endee_add(self, embeddings: np.ndarray, records: list[dict]) -> None:
        """Insert vectors into Endee — adapts to multiple API styles."""
        ids     = [str(i) for i in range(len(records))]
        vectors = [e.tolist() for e in embeddings]

        # Style 1: insert(collection, id, vector, metadata)
        if hasattr(self._client, "insert"):
            for _id, vec, meta in zip(ids, vectors, records):
                try:
                    self._client.insert(
                        collection=self.COLLECTION,
                        id=_id, vector=vec, metadata=meta
                    )
                except TypeError:
                    # Some builds use positional args
                    self._client.insert(self.COLLECTION, _id, vec, meta)
            return

        # Style 2: upsert(collection, items)
        if hasattr(self._client, "upsert"):
            items = [
                {"id": _id, "vector": vec, "metadata": meta}
                for _id, vec, meta in zip(ids, vectors, records)
            ]
            self._client.upsert(collection=self.COLLECTION, items=items)
            return

        # Style 3: add_vectors(vectors, ids, metadatas)
        if hasattr(self._client, "add_vectors"):
            self._client.add_vectors(
                vectors=vectors, ids=ids, metadatas=records
            )
            return

        # Style 4: direct index object
        if hasattr(self._client, "add"):
            self._client.add(np.array(vectors, dtype=np.float32))
            self._fallback_meta = records   # store meta separately
            return

        raise RuntimeError("Cannot find insert/upsert/add_vectors on Endee client.")

    def _endee_search(self, query: np.ndarray, k: int) -> list[dict]:
        """Query Endee — adapts to multiple API styles."""

        # Style 1: search(collection, vector, top_k)
        if hasattr(self._client, "search"):
            try:
                raw = self._client.search(
                    collection=self.COLLECTION,
                    vector=query.tolist(), top_k=k
                )
            except TypeError:
                raw = self._client.search(self.COLLECTION, query.tolist(), k)

            results = []
            for r in raw:
                if isinstance(r, dict):
                    meta  = r.get("metadata", r)
                    score = r.get("score", r.get("similarity", 1.0))
                else:
                    meta, score = {}, 1.0
                results.append({**meta, "similarity_score": round(float(score), 4)})
            return results

        # Style 2: query(collection, vector, n_results)
        if hasattr(self._client, "query"):
            raw = self._client.query(
                collection=self.COLLECTION,
                query_vector=query.tolist(), n_results=k
            )
            return self._parse_query_result(raw)

        raise RuntimeError("Cannot find search/query on Endee client.")

    @staticmethod
    def _parse_query_result(raw) -> list[dict]:
        results = []
        if isinstance(raw, dict):
            metas     = raw.get("metadatas",  [[]])[0]
            distances = raw.get("distances",   [[]])[0]
            for meta, dist in zip(metas, distances):
                score = float(1 / (1 + dist)) if dist is not None else 1.0
                results.append({**meta, "similarity_score": round(score, 4)})
        return results

    # ── FAISS fallback (used when Endee is not installed) ─────

    def _fallback_init(self) -> None:
        import faiss
        self._faiss_index    = faiss.IndexFlatL2(self.dim)
        self._faiss_metadata : list = []

    def _fallback_add(self, embeddings: np.ndarray, records: list[dict]) -> None:
        if not hasattr(self, "_faiss_index"):
            self._fallback_init()
        self._faiss_index.add(embeddings)
        self._faiss_metadata.extend(records)
        import joblib, os
        os.makedirs(os.path.dirname(ENDEE_DB_PATH), exist_ok=True)
        joblib.dump(
            {"index": self._faiss_index, "metadata": self._faiss_metadata},
            ENDEE_DB_PATH
        )

    def _fallback_search(self, query: np.ndarray, k: int) -> list[dict]:
        if not hasattr(self, "_faiss_index"):
            import joblib
            saved = joblib.load(ENDEE_DB_PATH)
            self._faiss_index    = saved["index"]
            self._faiss_metadata = saved["metadata"]

        dists, idxs = self._faiss_index.search(query.reshape(1, -1), k)
        results = []
        for dist, idx in zip(dists[0], idxs[0]):
            if idx == -1:
                continue
            sim = float(1 / (1 + dist))
            results.append({
                **self._faiss_metadata[idx],
                "similarity_score": round(sim, 4),
            })
        return results

    # ── Persistence helpers ───────────────────────────────────

    def save(self) -> None:
        """Endee persists automatically; this is a no-op for the primary store."""
        if not self._using_endee and hasattr(self, "_faiss_index"):
            import joblib, os
            os.makedirs(os.path.dirname(ENDEE_DB_PATH), exist_ok=True)
            joblib.dump(
                {"index": self._faiss_index, "metadata": self._faiss_metadata},
                ENDEE_DB_PATH
            )
            print(f"  ✔ Fallback store saved  →  {ENDEE_DB_PATH}")

    @classmethod
    def load(cls, dim: int) -> "EndeeVectorStore":
        """Reload an existing store (Endee re-connects; fallback loads from disk)."""
        store = cls(dim=dim)
        if not store._using_endee:
            import joblib
            if os.path.exists(ENDEE_DB_PATH):
                saved = joblib.load(ENDEE_DB_PATH)
                store._faiss_index    = saved["index"]
                store._faiss_metadata = saved["metadata"]
        return store
