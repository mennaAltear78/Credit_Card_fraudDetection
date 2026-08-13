import os
import json
import logging
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from joblib import dump
from huggingface_hub import HfApi
 
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    average_precision_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

# إخفاء تحذيرات Windows Symlinks
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"


def train_model():
    try:
        # 1. تحميل متغيرات البيئة
        load_dotenv()

        # مسارات البيانات والـ Logs المحلية
        PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT", "."))
        DATASET_PATH = PROJECT_ROOT / os.getenv("DATASET_NAME", "ml/creditcard.csv")
        LOG_PATH = PROJECT_ROOT / os.getenv("LOG_DIR", "logs") / os.getenv("LOG_NAME", "app.log")

        # معلمات تدريب الموديل
        TARGET_COL = os.getenv("TARGET_COL", "Class")
        TEST_SIZE = float(os.getenv("TEST_SIZE", 0.2))
        RANDOM_STATE = int(os.getenv("RANDOM_STATE", 42))

        # بيانات Hugging Face لرفع الموديل والـ Metrics
        REPO_ID = os.getenv("REPO_ID", "manna78/credit_model")
        HF_TOKEN = os.getenv("HF_TOKEN")  # Write Token الخاص بحسابك

        # مجلد مؤقت للحفظ المحلي قبل الرفع مباشرة
        TEMP_DIR = PROJECT_ROOT / "temp_output"
        TEMP_DIR.mkdir(parents=True, exist_ok=True)
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

        LOCAL_MODEL_PATH = TEMP_DIR / "model.joblib"
        LOCAL_METRICS_PATH = TEMP_DIR / "metrics.json"

        # إعداد الـ Logging
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | %(levelname)s | %(message)s",
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(LOG_PATH)
            ]
        )

        logging.info("🚀 Training script started...")

        # 2. قراءة البيانات محلياً
        df = pd.read_csv(DATASET_PATH)
        logging.info(f"Dataset loaded successfully from local disk. Shape: {df.shape}")

        X = df.drop(columns=[TARGET_COL])
        y = df[TARGET_COL]

        # 3. تقسيم البيانات وإنشاء الـ Pipeline
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=y
        )

        pipeline = Pipeline(
            steps=[
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=200,
                        min_samples_leaf=1,
                        max_features="log2",
                        max_depth=None,
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                        class_weight="balanced_subsample"
                    )
                )
            ]
        )

        # 4. التدريب والتوقع
        pipeline.fit(X_train, y_train)
        logging.info("Model training completed.")

        train_pred = pipeline.predict(X_train)
        THRESHOLD = 0.21

        test_prob = pipeline.predict_proba(X_test)[:, 1]
        test_pred = (test_prob >= THRESHOLD).astype(int)

        # حساب المقاييس (Metrics)
        train_acc = accuracy_score(y_train, train_pred)
        test_acc = accuracy_score(y_test, test_pred)

        train_report = classification_report(y_train, train_pred)
        test_report = classification_report(y_test, test_pred)

        logging.info(f"Train Accuracy: {train_acc:.3f} | Test Accuracy: {test_acc:.3f}")
        logging.info(f"Test Classification Report:\n{test_report}")

        pr_auc = average_precision_score(y_test, test_prob)
        roc_auc = roc_auc_score(y_test, test_prob)

        metrics = {
            "threshold": THRESHOLD,
            "accuracy": round(accuracy_score(y_test, test_pred), 4),
            "precision": round(precision_score(y_test, test_pred), 4),
            "recall": round(recall_score(y_test, test_pred), 4),
            "f1_score": round(f1_score(y_test, test_pred), 4),
            "roc_auc": round(roc_auc, 4),
            "pr_auc": round(pr_auc, 4),
            "confusion_matrix": confusion_matrix(y_test, test_pred).tolist()
        }

        # 5. حفظ الموديل والـ Metrics محلياً في ملف مؤقت
        with open(LOCAL_METRICS_PATH, "w") as f:
            json.dump(metrics, f, indent=4)

        dump(pipeline, LOCAL_MODEL_PATH)
        logging.info("Saved model and metrics locally in temp directory.")

        # -------------------------------------------------------------
        # 6. رفع الموديل والـ Metrics فقط إلى Hugging Face
        # -------------------------------------------------------------
        logging.info(f"Uploading model & metrics to Hugging Face Space ({REPO_ID})...")
        api = HfApi()

        # رفع ملف الموديل
        api.upload_file(
            path_or_fileobj=str(LOCAL_MODEL_PATH),
            path_in_repo="ml/model_dir/model.joblib",
            repo_id=REPO_ID,
            repo_type="space",
            token=HF_TOKEN
        )

        # رفع ملف الـ Metrics
        api.upload_file(
            path_or_fileobj=str(LOCAL_METRICS_PATH),
            path_in_repo="ml/model_dir/metrics.json",
            repo_id=REPO_ID,
            repo_type="space",
            token=HF_TOKEN
        )

        logging.info(f"🎉 Model & Metrics successfully updated on Hugging Face!")

    except Exception as e:
        logging.exception(f"Training failed: {e}")
        raise


if __name__ == "__main__":
    train_model()