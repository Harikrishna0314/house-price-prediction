"""
House Price Prediction — ML + Endee Vector Database
====================================================
Usage:
    python main.py               # train + predict demo (default)
    python main.py train         # generate data + train + build Endee index
    python main.py predict       # run CLI prediction demo
    python main.py server        # start FastAPI server at localhost:8000
    python main.py test          # run test suite
"""

import sys
import os

# Make sure all subpackages are importable regardless of cwd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def need_training():
    from config import MODEL_PATH, EMBEDDER_PATH
    return not (os.path.exists(MODEL_PATH) and os.path.exists(EMBEDDER_PATH))


def do_train():
    from data.generate_data import generate_dataset
    from model.train import train
    generate_dataset()
    train()


def do_predict():
    from pipeline.predictor import run_demo
    run_demo()


def do_server():
    import uvicorn
    print("\n  ╔══════════════════════════════════════════╗")
    print("  ║   House Price Predictor — REST API       ║")
    print("  ╠══════════════════════════════════════════╣")
    print("  ║  Swagger UI →  http://localhost:8000/docs ║")
    print("  ║  Health     →  http://localhost:8000/health║")
    print("  ║  Predict    →  POST /predict              ║")
    print("  ╚══════════════════════════════════════════╝\n")
    uvicorn.run("api.app:app", host="0.0.0.0", port=8000, reload=False)


def do_test():
    from tests.test_pipeline import run_tests
    run_tests()


def main():
    mode = sys.argv[1].lower() if len(sys.argv) > 1 else "all"

    if mode == "train":
        do_train()
        print("\n  Next step:  python main.py predict\n")

    elif mode == "predict":
        if need_training():
            print("  ⚠  No trained model found — running training first...\n")
            do_train()
        do_predict()

    elif mode == "server":
        if need_training():
            print("  ⚠  No trained model found — running training first...\n")
            do_train()
        do_server()

    elif mode == "test":
        do_test()

    else:  # "all" or default
        do_train()
        do_predict()


if __name__ == "__main__":
    main()
