import os
import json
import logging
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from joblib import dump

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
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


def train_model():
    try:
        # Load environment variables
        load_dotenv()

        PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT"))

        DATASET_PATH = PROJECT_ROOT / os.getenv("DATASET_NAME")
        MODEL_PATH = PROJECT_ROOT / os.getenv("MODEL_DIR") / os.getenv("MODEL_NAME")
        METRICS_PATH = PROJECT_ROOT / os.getenv("MODEL_DIR") / "metrics.json"
        LOG_PATH = PROJECT_ROOT / os.getenv("LOG_DIR") / os.getenv("LOG_NAME")

        TARGET_COL = os.getenv("TARGET_COL")
        TEST_SIZE = float(os.getenv("TEST_SIZE"))
        RANDOM_STATE = int(os.getenv("RANDOM_STATE"))

        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | %(levelname)s | %(message)s",
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(LOG_PATH)
            ]
        )

        logging.info("Training script started")

        # Load dataset
        df = pd.read_csv(DATASET_PATH)
        logging.info(f"Dataset loaded with shape {df.shape}")

        X = df.drop(columns=[TARGET_COL])
        y = df[TARGET_COL]

        # Split dataset
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=y
        )

        # Create pipeline
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

        # Train model
        pipeline.fit(X_train, y_train)
        logging.info("Model training completed")

        # Predictions
        train_pred = pipeline.predict(X_train)
        THRESHOLD = 0.21

        test_prob = pipeline.predict_proba(X_test)[:, 1]
        test_pred = (test_prob >= THRESHOLD).astype(int)

        # Accuracy
        train_acc = accuracy_score(y_train, train_pred)
        test_acc = accuracy_score(y_test, test_pred)

        # Classification reports
        train_report = classification_report(y_train, train_pred)
        test_report = classification_report(y_test, test_pred)

        logging.info(f"Train Accuracy: {train_acc:.3f}")
        logging.info(f"Test Accuracy: {test_acc:.3f}")

        logging.info(f"Train Classification Report:\n{train_report}")
        logging.info(f"Test Classification Report:\n{test_report}")
        pr_auc = average_precision_score(y_test, test_prob)
        roc_auc = roc_auc_score(y_test, test_prob)

        logging.info(f"ROC-AUC: {roc_auc:.4f}")
        logging.info(f"PR-AUC: {pr_auc:.4f}")
        # Save model statistics
        metrics = {
            "threshold": THRESHOLD,
            "accuracy": round(accuracy_score(y_test, test_pred), 4),
            "precision": round(precision_score(y_test, test_pred), 4),
            "recall": round(recall_score(y_test, test_pred), 4),
            "f1_score": round(f1_score(y_test, test_pred), 4),
            "roc_auc": round(roc_auc_score(y_test, test_prob), 4),
            "pr_auc": round(average_precision_score(y_test, test_prob), 4),
            "confusion_matrix": confusion_matrix(y_test, test_pred).tolist()
        }

        with open(METRICS_PATH, "w") as f:
            json.dump(metrics, f, indent=4)

        logging.info(f"Metrics saved to {METRICS_PATH}")

        # Save trained model
        dump(pipeline, MODEL_PATH)
        logging.info(f"Model saved to {MODEL_PATH}")

        logging.info("Training script finished successfully.")

    except Exception as e:
        logging.exception(f"Training failed: {e}")
        raise


if __name__ == "__main__":
    train_model()