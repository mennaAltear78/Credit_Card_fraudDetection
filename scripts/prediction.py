import os
import json
import logging
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download
from joblib import load

# إخفاء تحذير الـ Symlinks الخاص بـ Windows
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# تحميل متغيرات البيئة من .env
load_dotenv()

# إعداد المتغيرات المباشرة من Hugging Face مع قيم افتراضية
REPO_ID = os.getenv("REPO_ID", "manna78/credit_model")
MODEL_FILENAME = os.getenv("MODEL_FILENAME", "ml/model_dir/model.joblib")
METRICS_FILENAME = os.getenv("METRICS_FILENAME", "ml/model_dir/metrics.json")

# إعداد مسارات الـ Logging فقط
PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT", "."))
LOG_PATH = PROJECT_ROOT / os.getenv("LOG_DIR", "logs") / os.getenv("LOG_NAME", "app.log")

# متغيرات عامة سيتم تحميلها لاحقاً (Lazy Initialization)
model = None
model_statistics = None
_logging_configured = False


def init_model_and_logging():
    """تنزيل الموديل والـ Metrics من Hugging Face وإعداد الـ Logging أثناء الـ Startup."""
    global model, model_statistics, _logging_configured

    # 1. تهيئة الـ Logging
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

    # 2. جلب الموديل والـ Metrics من Hugging Face وتحميلهما في الذاكرة
    if model is None:
        logging.info("⏳ Downloading model and metrics from Hugging Face...")

        # تنزيل ملف الموديل (.joblib)
        downloaded_model_path = hf_hub_download(
            repo_id=REPO_ID,
            filename=MODEL_FILENAME,
            repo_type="space"
        )
        model = load(downloaded_model_path)
        logging.info("✅ Model loaded successfully into memory.")

        # تنزيل ملف الـ Metrics (.json)
        downloaded_metrics_path = hf_hub_download(
            repo_id=REPO_ID,
            filename=METRICS_FILENAME,
            repo_type="space"
        )
        with open(downloaded_metrics_path, "r", encoding="utf-8") as f:
            model_statistics = json.load(f)
        logging.info("✅ Model metrics loaded successfully.")


def predict(input_data: dict):
    global model, model_statistics
    if model is None or model_statistics is None:
        init_model_and_logging()

    df = pd.DataFrame([input_data])
    prediction = model.predict(df)[0]
    propensity = model.predict_proba(df)[0][1]

    logging.info(f"Prediction: {prediction}, Propensity: {propensity:.4f}")
    
    return {
        "prediction": int(prediction),
        "propensity": float(propensity),
        "model_statistics": model_statistics
    }