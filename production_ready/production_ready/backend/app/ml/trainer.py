"""Model training, evaluation, and persistence."""
from __future__ import annotations

import json
import os
import asyncio
from datetime import datetime, timezone
from typing import Any, Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from app.core.config import get_settings
from app.ml.data_generator import generate_dataset, load_from_csv

settings = get_settings()

FEATURE_NAMES = ["sgpa", "jee_score", "marks_12", "attendance", "gender"]
VERSION = "v1.0"

# Path where uploaded training CSV is stored
UPLOADED_CSV_PATH = os.path.join("storage", "training_data.csv")


def build_pipeline(scale_pos_weight=1.0) -> Pipeline:
    """Build the ML pipeline with preprocessing and XGBoost classifier."""
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), ["sgpa", "jee_score", "marks_12", "attendance"]),
        ],
        remainder="passthrough",
    )
    preprocessor.set_output(transform="pandas")
    clf = XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        scale_pos_weight=scale_pos_weight,
        random_state=42,
    )
    logger.info(f"✓ Pipeline built with preprocessor and XGBoost classifier")
    return Pipeline([("preprocessor", preprocessor), ("classifier", clf)])


def train_model(n_samples: int = 2000) -> Dict[str, Any]:
    """
    Train XGBoost pipeline.
    Uses uploaded CSV from storage/training_data.csv if it exists,
    otherwise falls back to synthetic data.
    """
    logger.info(f"━━━━━ TRAINING START ━━━━━")
    
    # ── STEP 1: Determine data source
    if os.path.exists(UPLOADED_CSV_PATH):
        logger.info(f"📂 Using uploaded training CSV: {UPLOADED_CSV_PATH}")
        try:
            df = load_from_csv(UPLOADED_CSV_PATH)
            data_source = f"csv:{len(df)}_records"
            logger.info(f"✓ CSV loaded | {len(df)} records")
        except Exception as e:
            logger.warning(f"⚠️ CSV load failed ({e}), falling back to synthetic data")
            df = generate_dataset(n_samples=n_samples)
            data_source = f"synthetic:{n_samples}"
    else:
        logger.info(f"📊 No uploaded CSV found — generating synthetic training data")
        df = generate_dataset(n_samples=n_samples)
        data_source = f"synthetic:{n_samples}"
        logger.info(f"✓ Synthetic data generated | {n_samples} records")

    # ── STEP 2: Prepare training data
    X = df[FEATURE_NAMES]
    y = df["eligible"]
    
    logger.info(f"📋 Data shape: X={X.shape} | y={y.shape}")
    logger.info(f"📊 Class distribution: {dict(y.value_counts())}")

    if y.nunique() < 2:
        logger.error("✗ Training labels have only one class!")
        raise ValueError(
            "Training labels have only one class. "
            "Check your 'eligible' column or upload a more balanced dataset."
        )

    # Compute class imbalance weight
    num_neg = (y == 0).sum()
    num_pos = (y == 1).sum()
    sp_weight = num_neg / num_pos if num_pos > 0 else 1.0

    # ── STEP 3: Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    logger.info(f"✓ Train/test split | train={X_train.shape[0]} | test={X_test.shape[0]}")

    # ── STEP 4: Build and train pipeline
    logger.info(f"🏗️ Building pipeline (scale_pos_weight={sp_weight:.2f})...")
    pipeline = build_pipeline(scale_pos_weight=sp_weight)
    
    logger.info(f"🔧 Training XGBoost model on {len(X_train)} samples...")
    pipeline.fit(X_train, y_train)
    logger.info(f"✓ Training complete")

    # ── STEP 5: Evaluate on test set
    logger.info(f"📊 Evaluating on test set...")
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    f1 = f1_score(y_test, y_pred)

    logger.info(f"✓ Test Metrics | Accuracy={acc:.4f} | AUC={auc:.4f} | F1={f1:.4f}")

    # ── STEP 6: Cross-validation
    logger.info(f"📈 Running 5-fold cross-validation...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(pipeline, X, y, cv=cv, scoring="roc_auc")
    cv_mean = float(cv_scores.mean())
    cv_std = float(cv_scores.std())
    logger.info(f"✓ CV Scores | Mean={cv_mean:.4f} ± {cv_std:.4f}")

    # ── STEP 7: Save model
    logger.info(f"💾 Saving model...")
    os.makedirs(settings.MODEL_DIR, exist_ok=True)
    os.makedirs(settings.REPORTS_DIR, exist_ok=True)
    joblib.dump(pipeline, settings.model_path)
    logger.info(f"✓ Model saved to {settings.model_path}")

    # Dynamically generate version so frontend and load balancers properly detect new models
    from datetime import datetime, timezone
    dynamic_version = f"v{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"

    # ── STEP 8: Save metadata
    metadata = {
        "model_version": dynamic_version,
        "model_type": "XGBoost",
        "features": FEATURE_NAMES,
        "protected_attribute": "gender",
        "data_source": data_source,
        "training_samples": len(X_train),
        "test_samples": len(y_test),
        "accuracy": round(acc, 4),
        "auc_roc": round(auc, 4),
        "f1_score": round(f1, 4),
        "cross_val_mean": round(cv_mean, 4),
        "cross_val_std": round(cv_std, 4),
        "classification_report": classification_report(y_test, y_pred, output_dict=True),
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }

    # ── STEP 9: Save test data for fairness evaluation
    X_test_copy = X_test.copy()
    X_test_copy["eligible"] = y_test.values
    X_test_copy["predicted"] = y_pred
    X_test_copy["probability"] = y_prob
    test_data_path = os.path.join(settings.MODEL_DIR, "test_data.csv")
    X_test_copy.to_csv(test_data_path, index=False)
    metadata["test_data_path"] = test_data_path
    logger.info(f"✓ Test data saved to {test_data_path}")

    # ── STEP 10: Save metadata JSON
    with open(settings.metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"✓ Metadata saved to {settings.metadata_path}")

    logger.info(f"━━━━━ TRAINING END ━━━━━\n")
    return metadata


async def async_train_model(n_samples: int = 2000) -> Dict[str, Any]:
    """Non-blocking wrapper for async FastAPI handlers."""
    return await asyncio.to_thread(train_model, n_samples)


def load_model() -> Tuple[Pipeline, Dict[str, Any]]:
    """Load the trained model pipeline and metadata with validation."""
    logger.info(f"📁 Loading model from {settings.model_path}")
    
    if not os.path.exists(settings.model_path):
        logger.error(f"✗ Model file not found: {settings.model_path}")
        raise FileNotFoundError(
            f"No trained model at {settings.model_path}. "
            "Call POST /v1/retrain to train first."
        )
    
    try:
        pipeline = joblib.load(settings.model_path)
        logger.debug(f"✓ Pipeline loaded from joblib")
    except Exception as e:
        logger.error(f"✗ Failed to load pipeline: {e}")
        raise RuntimeError(f"Pipeline load failed: {e}")
    
    if not os.path.exists(settings.metadata_path):
        logger.error(f"✗ Metadata file not found: {settings.metadata_path}")
        raise FileNotFoundError(f"Metadata file not found at {settings.metadata_path}")
    
    try:
        with open(settings.metadata_path) as f:
            metadata = json.load(f)
        logger.info(
            f"✓ Model loaded successfully | version={metadata.get('model_version')} | "
            f"accuracy={metadata.get('accuracy')} | auc={metadata.get('auc_roc')} | "
            f"samples={metadata.get('training_samples')}"
        )
    except Exception as e:
        logger.error(f"✗ Failed to load metadata: {e}")
        raise RuntimeError(f"Metadata load failed: {e}")
    
    # ── Validate model structure
    if "preprocessor" not in pipeline.named_steps:
        logger.error("✗ Pipeline missing preprocessor step")
        raise RuntimeError("Invalid pipeline: missing preprocessor")
    if "classifier" not in pipeline.named_steps:
        logger.error("✗ Pipeline missing classifier step")
        raise RuntimeError("Invalid pipeline: missing classifier")
    
    logger.debug(f"✓ Pipeline validated | steps={list(pipeline.named_steps.keys())}")
    
    return pipeline, metadata
