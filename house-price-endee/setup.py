from setuptools import setup, find_packages

setup(
    name        = "house-price-endee",
    version     = "1.0.0",
    description = "House Price Prediction using ML + Endee Vector Database",
    packages    = find_packages(),
    python_requires = ">=3.10",
    install_requires = [
        "numpy>=1.26",
        "pandas>=2.2",
        "scikit-learn>=1.4",
        "xgboost>=2.0",
        "faiss-cpu>=1.8",
        "fastapi>=0.111",
        "uvicorn>=0.30",
        "pydantic>=2.7",
        "joblib>=1.4",
    ],
)
