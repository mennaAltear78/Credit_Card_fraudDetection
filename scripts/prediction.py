import os
import logging
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from joblib import load
import json


# load .env content ro env vars
load_dotenv()

# Resolve paths from env but avoid performing I/O at import time.
PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT"))
MODEL_PATH = PROJECT_ROOT / os.getenv("MODEL_DIR") / os.getenv("MODEL_NAME")
LOG_PATH = PROJECT_ROOT / os.getenv("LOG_DIR") / os.getenv("LOG_NAME")
METRICS_PATH = PROJECT_ROOT / os.getenv("MODEL_DIR") / "metrics.json"

# Globals initialized lazily to avoid side-effects during import (prevents reload loops).
model = None
_logging_configured = False

def init_model_and_logging():
    """Create log dir, configure logging, and load the trained model.

    This should be called from the application startup event, not at import time.
    """
    global model, _logging_configured
    if not _logging_configured:
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | %(levelname)s | %(message)s",
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(LOG_PATH)
            ]
        )
        _logging_configured = True

    if model is None:
        logging.info("Loading trained model...")
        model = load(MODEL_PATH)
        logging.info("Model loaded successfully.")

def predict(input_data: dict):
    global model
    if model is None:
        init_model_and_logging()

    df = pd.DataFrame([input_data])
    prediction = model.predict(df)[0]
    propensity = model.predict_proba(df)[0][1]
    with open(METRICS_PATH, "r") as f:
        model_statistics = json.load(f)
    logging.info(f"Prediction: {prediction}, Propensity: {propensity}")
    logging.info("Model provided a prediction")
    return {
        "prediction": int(prediction),
        "propensity": float(propensity),
        "model_statistics": model_statistics
    }
