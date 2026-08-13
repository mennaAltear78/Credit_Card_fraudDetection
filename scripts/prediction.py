import os
import json
import logging

import pandas as pd
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download
from joblib import load

# Hide Hugging Face symlink warning on Windows
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# Load environment variables
load_dotenv()

# Hugging Face configuration
REPO_ID = os.getenv("REPO_ID", "manna78/credit_model")
MODEL_FILENAME = os.getenv(
    "MODEL_FILENAME",
    "ml/model_dir/model.joblib"
)
METRICS_FILENAME = os.getenv(
    "METRICS_FILENAME",
    "ml/model_dir/metrics.json"
)

# Global variables
model = None
model_statistics = None
_logging_configured = False


def init_model_and_logging():
    """Download model and metrics from Hugging Face and configure logging."""

    global model, model_statistics, _logging_configured

    # 1. Configure logging
    if not _logging_configured:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | %(levelname)s | %(message)s",
            handlers=[
                logging.StreamHandler()
            ]
        )
        _logging_configured = True

    # 2. Download model and metrics
    if model is None:
        logging.info("Downloading model and metrics from Hugging Face...")

        downloaded_model_path = hf_hub_download(
            repo_id=REPO_ID,
            filename=MODEL_FILENAME,
            repo_type="space"
        )

        model = load(downloaded_model_path)

        logging.info("Model loaded successfully into memory.")

        downloaded_metrics_path = hf_hub_download(
            repo_id=REPO_ID,
            filename=METRICS_FILENAME,
            repo_type="space"
        )

        with open(downloaded_metrics_path, "r", encoding="utf-8") as f:
            model_statistics = json.load(f)

        logging.info("Model metrics loaded successfully.")


def predict(input_data: dict):
    global model, model_statistics

    if model is None or model_statistics is None:
        init_model_and_logging()

    df = pd.DataFrame([input_data])

    prediction = model.predict(df)[0]
    propensity = model.predict_proba(df)[0][1]

    logging.info(
        f"Prediction: {prediction}, Propensity: {propensity:.4f}"
    )

    return {
        "prediction": int(prediction),
        "propensity": float(propensity),
        "model_statistics": model_statistics
    }