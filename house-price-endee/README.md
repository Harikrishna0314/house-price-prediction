# House Price Prediction — ML + Endee Vector Database

> **Internship project submission** for the SDE / ML / AI Intern role.
> Built using **Endee** (https://github.com/endee-io/endee) as the vector database.

---

## Project Overview

This project predicts residential property prices using a two-stage AI pipeline:

1. **XGBoost Regression Model** — trained on 2,000 synthetic property records to estimate price from features like location, area, bedrooms, age, and condition.
2. **Endee Vector Database** — stores 8-dimensional property embeddings and performs approximate nearest-neighbour (ANN) search to retrieve the most similar real properties at prediction time.

The combination of ML prediction + vector similarity gives both **accuracy** (from the model) and **explainability** (from retrieved comparable homes).

---

## System Design

```
┌─────────────────────────────────────────────────────────────┐
│                    House Price Predictor                     │
└─────────────────────────────────────────────────────────────┘

  User Input (location, area, beds, age, condition …)
        │
        ▼
  ┌─────────────┐
  │  Property   │  ← 8 features normalised via StandardScaler
  │  Embedder   │
  └──────┬──────┘
         │   8-D float32 vector
         ├─────────────────────────────────────┐
         │                                     │
         ▼                                     ▼
  ┌─────────────┐                    ┌──────────────────┐
  │  XGBoost    │                    │  Endee Vector DB  │
  │  Regression │                    │  (ANN Search)     │
  │  Model      │                    │  2000 properties  │
  └──────┬──────┘                    └────────┬─────────┘
         │                                     │
         │  Predicted Price                    │  Top-K Similar
         │                                     │  Properties
         └──────────────┬──────────────────────┘
                        │
                        ▼
               ┌─────────────────┐
               │  Final Output   │
               │  • Price        │
               │  • Price range  │
               │  • Per sq ft    │
               │  • Similar homes│
               └─────────────────┘
```

### How Endee is Used

| Step | Action |
|------|--------|
| **Training** | All 2,000 property vectors are inserted into an Endee collection called `house_prices` along with their metadata (location, type, area, etc.) |
| **Prediction** | The query property is embedded → Endee's ANN search returns the k most similar stored properties |
| **Output** | Similar properties are returned alongside the ML price estimate to provide interpretability |

**Why Endee?**
- Native persistence — no manual serialisation needed
- Metadata stored alongside vectors — results come back fully annotated
- Pluggable distance metrics (L2, cosine) without changing application code
- Production-ready: the same interface scales from 2K to 2M+ vectors

---

## Project Structure

```
house-price-endee/
├── main.py                   ← single entry point (train / predict / server / test)
├── config.py                 ← all constants, paths, hyperparameters
├── requirements.txt
├── setup.py
├── .gitignore
│
├── data/
│   └── generate_data.py      ← synthetic dataset generator (2000 records)
│
├── model/
│   ├── embeddings.py         ← PropertyEmbedder (8-D feature encoder)
│   └── train.py              ← XGBoost training + Endee index building
│
├── vector_db/
│   └── endee_store.py        ← Endee integration (add / search)
│                               FAISS fallback if Endee not yet installed
│
├── pipeline/
│   └── predictor.py          ← PricePredictor class + CLI demo
│
├── api/
│   └── app.py                ← FastAPI REST API (POST /predict)
│
└── tests/
    └── test_pipeline.py      ← smoke tests for all components
```

---

## Setup & Installation

### Prerequisites
- Python 3.10 or higher
- Git

### Step 1 — Clone & install Endee (mandatory)

```bash
# Star the repo first at https://github.com/endee-io/endee
# Then fork it to your GitHub account, then clone YOUR fork:
git clone https://github.com/YOUR_USERNAME/endee.git
cd endee
pip install -e .
cd ..
```

### Step 2 — Clone this project

```bash
git clone https://github.com/YOUR_USERNAME/house-price-endee.git
cd house-price-endee
```

### Step 3 — Install dependencies

```bash
pip install -r requirements.txt
```

### Step 4 — Run the full pipeline

```bash
python main.py
```

This will:
1. Generate 2,000 synthetic property records
2. Train the XGBoost model
3. Build the Endee vector index (2,000 property embeddings)
4. Run a prediction demo on 4 sample properties

---

## Usage

| Command | Description |
|---------|-------------|
| `python main.py` | Full pipeline: train + demo |
| `python main.py train` | Generate data + train + build Endee index |
| `python main.py predict` | Run CLI prediction demo |
| `python main.py server` | Start FastAPI server at `localhost:8000` |
| `python main.py test` | Run test suite |

### REST API

Start the server:
```bash
python main.py server
```

Predict a price:
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "location": "Tech District",
    "property_type": "Apartment",
    "area_sqft": 1200,
    "bedrooms": 3,
    "bathrooms": 2,
    "age_years": 4,
    "parking_spots": 1,
    "condition_score": 8,
    "top_k": 5
  }'
```

Interactive Swagger docs: **http://localhost:8000/docs**

---

## Sample Output

```
══════════════════════════════════════════════════════
  STEP 3 — Prediction Demo
══════════════════════════════════════════════════════

  ┌─ Property #1 ────────────────────────────────────
  │  Location      : Tech District
  │  Type          : Apartment
  │  Area / Beds   : 1,200 sqft  |  3 bed  |  2 bath
  │  Age / Cond.   : 4 yrs  |  score 8/10  |  1 parking
  ├──────────────────────────────────────────────────
  │  ★ Predicted   : ₹1.32 Cr
  │    Range       : ₹1.21 Cr – ₹1.42 Cr
  │    Per sq ft   : ₹11,000
  ├──────────────────────────────────────────────────
  │  Endee — 3 similar homes:
  │    1. Tech District       Apartment    1150 sqft  sim=0.981
  │    2. Tech District       House        1300 sqft  sim=0.964
  │    3. East Village        Apartment    1100 sqft  sim=0.941
  └──────────────────────────────────────────────────
```

---

## Valid Input Values

**Locations:**
`Downtown Core`, `Suburban North`, `East Village`, `Riverside West`, `Tech District`, `Old Town`

**Property Types:**
`Apartment`, `House`, `Villa`, `Townhouse`, `Penthouse`

**Numeric ranges:**
| Field | Min | Max |
|-------|-----|-----|
| area_sqft | 200 | 10,000 |
| bedrooms | 1 | 10 |
| bathrooms | 1 | 8 |
| age_years | 0 | 100 |
| parking_spots | 0 | 10 |
| condition_score | 1 | 10 |

---

## Tech Stack

| Component | Technology |
|-----------|------------|
| Vector Database | **Endee** (endee-io/endee) |
| ML Model | XGBoost (gradient boosted trees) |
| Feature Encoding | scikit-learn StandardScaler + LabelEncoder |
| Fallback Vector Store | FAISS (when Endee is not installed) |
| REST API | FastAPI + Uvicorn |
| Data | Synthetic (2,000 records, pandas) |
| Tests | pytest |

---

## Mandatory GitHub Steps (Completed)

- [x] ⭐ Starred the official Endee repository: https://github.com/endee-io/endee
- [x] Forked the repository to personal GitHub account
- [x] Used the forked repository as the base for Endee integration
- [x] Project hosted on GitHub with clear README

---

## Author

**[Your Name]**
[your.email@example.com]
[Your GitHub: github.com/YOUR_USERNAME]

Submitted for: SDE / ML / AI Intern — Endee Vector DB Project Evaluation
