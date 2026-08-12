import os
import logging
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from joblib import load
import json


# load .env content ro env vars
load_dotenv()

PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT"))
MODEL_PATH = PROJECT_ROOT / os.getenv("MODEL_DIR") / os.getenv("MODEL_NAME")
LOG_PATH = PROJECT_ROOT / os.getenv("LOG_DIR") / os.getenv("LOG_NAME")
METRICS_PATH = PROJECT_ROOT / os.getenv("MODEL_DIR") / "metrics.json"



LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_PATH)
    ]
)

# loading the trained model
# NOTE: model is loaded only once and not for every prediction
logging.info("Loading trained model...")
model = load(MODEL_PATH)
logging.info("Model loaded successfully.")

def predict(input_data: dict):
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
