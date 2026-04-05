"""Retrain route: POST /retrain."""
from fastapi import APIRouter, Depends
from loguru import logger

from app.core.security import require_role
from app.ml.trainer import async_train_model, UPLOADED_CSV_PATH
from app.ml.predictor import invalidate_cache
from app.schemas.schemas import RetrainResponse

import os

router = APIRouter(tags=["Model Management"])


@router.post("/retrain", response_model=RetrainResponse)
async def trigger_retrain(
    n_samples: int = 2000,
    user: dict = Depends(require_role("admin", "analyst")),
):
    """
    Retrain the ML model.

    - If a CSV was uploaded via **POST /v1/upload-training-data**, it will be used automatically.
    - Otherwise, a synthetic dataset of `n_samples` records is generated.
    """
    csv_exists = os.path.exists(UPLOADED_CSV_PATH)
    logger.info(
        f"🔄 RETRAIN TRIGGERED | user={user.get('sub')} | "
        f"csv_uploaded={csv_exists} | n_samples={n_samples}"
    )

    try:
        logger.info(f"📊 Starting model training...")
        metadata = await async_train_model(n_samples=n_samples)
        logger.info(f"✓ Model training complete | accuracy={metadata['accuracy']:.4f} | auc={metadata['auc_roc']:.4f}")

        logger.info(f"♻️ Invalidating prediction cache...")
        invalidate_cache()
        logger.info(f"✓ Cache invalidated — next prediction will use new model")

    except Exception as e:
        logger.error(f"✗ Retrain failed: {e}")
        raise

    data_source = metadata.get("data_source", "unknown")
    msg = (
        f"✓ Model trained on {data_source} | "
        f"Accuracy={metadata['accuracy']:.4f}, AUC={metadata['auc_roc']:.4f}, F1={metadata['f1_score']:.4f}"
    )
    
    logger.info(f"📤 Retrain response: {msg}")
    
    return RetrainResponse(
        status="success",
        model_version=metadata["model_version"],
        accuracy=metadata["accuracy"],
        auc_roc=metadata["auc_roc"],
        f1_score=metadata["f1_score"],
        cross_val_mean=metadata["cross_val_mean"],
        training_samples=metadata["training_samples"],
        message=msg,
    )
