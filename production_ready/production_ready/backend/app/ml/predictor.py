"""Inference service for scholarship eligibility prediction."""
from __future__ import annotations

import asyncio
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from loguru import logger

from app.ml.trainer import load_model, FEATURE_NAMES

import os
from app.core.config import get_settings

# Module-level cache
_cached_pipeline = None
_cached_metadata: Optional[Dict] = None
_last_model_mtime = 0.0

def _get_pipeline():
    global _cached_pipeline, _cached_metadata, _last_model_mtime
    settings = get_settings()
    
    current_mtime = 0.0
    if os.path.exists(settings.model_path):
        current_mtime = os.path.getmtime(settings.model_path)
    else:
        raise FileNotFoundError(f"No trained model at {settings.model_path}. Call POST /v1/retrain to train first.")

    # Reload if cache is empty OR the model file has been updated
    if _cached_pipeline is None or current_mtime > _last_model_mtime:
        _cached_pipeline, _cached_metadata = load_model()
        _last_model_mtime = current_mtime
        logger.info(f"✓ Model loaded and cached | Version: {_cached_metadata.get('model_version')} | Features: {FEATURE_NAMES}")
    else:
        logger.debug(f"✓ Using cached model | Version: {_cached_metadata.get('model_version')}")
    
    return _cached_pipeline, _cached_metadata

def invalidate_cache():
    """Manually clear cache (fallback)."""
    global _cached_pipeline, _cached_metadata, _last_model_mtime
    logger.warning("⚠️ Model cache invalidated — forcing reload on next prediction")
    _cached_pipeline = None
    _cached_metadata = None
    _last_model_mtime = 0.0


def predict(
    sgpa: float,
    jee_score: int,
    marks_12: float,
    attendance: float,
    gender: str,
) -> Dict[str, Any]:
    """Run inference on a single student record."""
    
    # ── STEP 1: Log input values
    logger.info(
        f"━━━ PREDICTION START ━━━"
    )
    logger.info(
        f"📥 INPUT RECEIVED | sgpa={sgpa} | jee={jee_score} | marks_12={marks_12} | "
        f"attendance={attendance} | gender={gender}"
    )
    
    # ── STEP 2: Get cached pipeline
    pipeline, metadata = _get_pipeline()
    
    # ── STEP 3: Validate inputs
    try:
        if not (0 <= sgpa <= 10):
            raise ValueError(f"SGPA {sgpa} out of range [0, 10]")
        if not (0 <= jee_score <= 360):
            raise ValueError(f"JEE Score {jee_score} out of range [0, 360]")
        if not (0 <= marks_12 <= 100):
            raise ValueError(f"Marks 12 {marks_12} out of range [0, 100]")
        if not (0 <= attendance <= 100):
            raise ValueError(f"Attendance {attendance} out of range [0, 100]")
        if gender.lower() not in ["female", "male", "other"]:
            raise ValueError(f"Gender '{gender}' not in [female, male, other]")
        logger.debug(f"✓ Input validation passed")
    except ValueError as e:
        logger.error(f"✗ Input validation failed: {e}")
        raise

    # ── STEP 4: Encode gender
    gender_encoded = 0 if gender.lower() == "female" else 1
    logger.debug(f"🔤 Gender encoding: {gender} → {gender_encoded} (0=female, 1=male)")

    # ── STEP 5: Create DataFrame with correct column order
    input_df = pd.DataFrame([{
        "sgpa": sgpa,
        "jee_score": jee_score,
        "marks_12": marks_12,
        "attendance": attendance,
        "gender": gender_encoded,
    }], columns=FEATURE_NAMES)
    
    logger.debug(f"📊 Input DataFrame:\n{input_df.to_string()}")
    
    # ── STEP 6: Get preprocessor and classifier steps
    preprocessor = pipeline.named_steps.get("preprocessor")
    classifier = pipeline.named_steps.get("classifier")
    
    if preprocessor is None or classifier is None:
        logger.error("✗ Pipeline missing preprocessor or classifier")
        raise RuntimeError("Invalid pipeline structure")
    
    # ── STEP 7: Apply preprocessing
    try:
        X_transformed = preprocessor.transform(input_df)
        logger.debug(f"🔧 Preprocessed features shape: {X_transformed.shape}")
        logger.debug(f"📈 Preprocessed values:\n{X_transformed}")
    except Exception as e:
        logger.error(f"✗ Preprocessing failed: {e}")
        raise
    
    # ── STEP 8: Get raw probabilities
    try:
        proba_raw = pipeline.predict_proba(input_df)
        logger.debug(f"🎯 Raw probabilities (both classes): {proba_raw}")
        prob = float(proba_raw[0, 1])  # class 1 (eligible) probability
        logger.info(f"✓ Probability computed: {prob:.4f} ({prob*100:.2f}%)")
    except Exception as e:
        logger.error(f"✗ Prediction failed: {e}")
        raise
    
    # ── STEP 9: Determine eligibility
    eligible = prob >= 0.5
    logger.info(f"✓ Eligibility: {'ELIGIBLE' if eligible else 'NOT ELIGIBLE'} (threshold=0.5)")

    # ── STEP 10: Assign confidence
    if prob >= 0.75:
        confidence = "High"
    elif prob >= 0.55:
        confidence = "Medium"
    else:
        confidence = "Low"
    logger.info(f"✓ Confidence level: {confidence} (prob={prob:.4f})")

    # ── STEP 11: Prepare and log result
    result = {
        "eligible": eligible,
        "probability": round(prob, 4),
        "confidence": confidence,
        "model_version": metadata.get("model_version", "v1.0"),
    }
    
    logger.info(
        f"📤 PREDICTION OUTPUT | eligible={eligible} | prob={result['probability']} | "
        f"confidence={confidence} | model={result['model_version']}"
    )
    logger.info(f"━━━ PREDICTION END ━━━\n")
    
    return result


async def async_predict(
    sgpa: float,
    jee_score: int,
    marks_12: float,
    attendance: float,
    gender: str,
) -> Dict[str, Any]:
    """Async wrapper for predict function."""
    return await asyncio.to_thread(predict, sgpa, jee_score, marks_12, attendance, gender)
