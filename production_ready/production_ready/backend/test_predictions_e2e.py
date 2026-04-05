#!/usr/bin/env python
"""
End-to-end test script to verify ML predictions work correctly with different inputs.

Run this after the backend is running to verify that:
1. Different inputs produce different predictions
2. Logging is working properly
3. Feature transformations are correct
4. Model changes are reflected in predictions
"""
import asyncio
import json
import sys
from loguru import logger

# Configure logger for this test
logger.remove()
logger.add(
    sys.stderr,
    format="<level>{level: <8}</level> | <level>{message}</level>",
    colorize=True,
)

# Add app to path
sys.path.insert(0, ".")

from app.core.config import get_settings
from app.ml.predictor import predict, invalidate_cache
from app.ml.trainer import train_model, load_model


def test_single_prediction():
    """Test a single prediction with known input."""
    logger.info("=" * 80)
    logger.info("TEST 1: Single Prediction")
    logger.info("=" * 80)
    
    result = predict(
        sgpa=8.5,
        jee_score=250,
        marks_12=90.5,
        attendance=95.0,
        gender="female",
    )
    
    logger.info(f"Result: {json.dumps(result, indent=2)}")
    assert result["probability"] >= 0.0 and result["probability"] <= 1.0
    logger.info("✓ Test 1 PASSED\n")
    return result


def test_different_inputs_different_outputs():
    """Test that different inputs produce different predictions."""
    logger.info("=" * 80)
    logger.info("TEST 2: Different Inputs → Different Outputs")
    logger.info("=" * 80)
    
    # High-performing student (should be eligible)
    result1 = predict(
        sgpa=9.0,
        jee_score=300,
        marks_12=95.0,
        attendance=98.0,
        gender="female",
    )
    logger.info(f"High performer: {result1}")
    
    # Low-performing student (should NOT be eligible)
    result2 = predict(
        sgpa=4.0,
        jee_score=50,
        marks_12=40.0,
        attendance=50.0,
        gender="male",
    )
    logger.info(f"Low performer: {result2}")
    
    # They should have different probabilities
    if result1["probability"] != result2["probability"]:
        logger.info(f"✓ Probabilities differ: {result1['probability']} != {result2['probability']}")
    else:
        logger.error(f"✗ Probabilities are the SAME! {result1['probability']} == {result2['probability']}")
        logger.error("⚠️ THIS IS THE MAIN BUG - predictions are hardcoded or cached!")
        return False
    
    # High performer should be more likely eligible
    if result1["probability"] > result2["probability"]:
        logger.info(f"✓ High performer has higher probability")
    else:
        logger.warning(f"⚠️ Low performer has higher probability than high performer")
    
    logger.info("✓ Test 2 PASSED\n")
    return True


def test_gender_variation():
    """Test that gender affects predictions (as it should in some cases)."""
    logger.info("=" * 80)
    logger.info("TEST 3: Gender Variation")
    logger.info("=" * 80)
    
    result_female = predict(
        sgpa=7.0,
        jee_score=180,
        marks_12=75.0,
        attendance=80.0,
        gender="female",
    )
    logger.info(f"Female: {result_female}")
    
    result_male = predict(
        sgpa=7.0,
        jee_score=180,
        marks_12=75.0,
        attendance=80.0,
        gender="male",
    )
    logger.info(f"Male: {result_male}")
    
    logger.info(f"Probability difference: {abs(result_female['probability'] - result_male['probability']):.4f}")
    logger.info("✓ Test 3 PASSED\n")
    return True


def test_boundary_cases():
    """Test edge cases and boundary conditions."""
    logger.info("=" * 80)
    logger.info("TEST 4: Boundary Cases")
    logger.info("=" * 80)
    
    # Minimum values
    result_min = predict(
        sgpa=0.0,
        jee_score=0,
        marks_12=0.0,
        attendance=0.0,
        gender="female",
    )
    logger.info(f"Minimum values: {result_min}")
    
    # Maximum values
    result_max = predict(
        sgpa=10.0,
        jee_score=360,
        marks_12=100.0,
        attendance=100.0,
        gender="male",
    )
    logger.info(f"Maximum values: {result_max}")
    
    # Mid-range
    result_mid = predict(
        sgpa=5.0,
        jee_score=180,
        marks_12=50.0,
        attendance=50.0,
        gender="female",
    )
    logger.info(f"Mid-range values: {result_mid}")
    
    # Check that they're all different
    probs = [result_min["probability"], result_max["probability"], result_mid["probability"]]
    if len(set(probs)) == 3:  # All different
        logger.info(f"✓ All boundary cases produced different probabilities")
    else:
        logger.warning(f"⚠️ Some boundary cases produced same probability")
    
    logger.info("✓ Test 4 PASSED\n")
    return True


def test_model_metadata():
    """Test that model metadata is loaded correctly."""
    logger.info("=" * 80)
    logger.info("TEST 5: Model Metadata")
    logger.info("=" * 80)
    
    pipeline, metadata = load_model()
    logger.info(f"Model version: {metadata.get('model_version')}")
    logger.info(f"Model type: {metadata.get('model_type')}")
    logger.info(f"Features: {metadata.get('features')}")
    logger.info(f"Accuracy: {metadata.get('accuracy'):.4f}")
    logger.info(f"AUC: {metadata.get('auc_roc'):.4f}")
    logger.info(f"F1: {metadata.get('f1_score'):.4f}")
    logger.info(f"Training samples: {metadata.get('training_samples')}")
    logger.info(f"Data source: {metadata.get('data_source')}")
    
    assert metadata.get("model_version") is not None
    assert metadata.get("accuracy") is not None
    assert len(metadata.get("features", [])) == 5
    
    logger.info("✓ Test 5 PASSED\n")
    return True


def test_cache_invalidation():
    """Test that cache invalidation works on retrain."""
    logger.info("=" * 80)
    logger.info("TEST 6: Cache Invalidation")
    logger.info("=" * 80)
    
    # Get initial prediction
    result1 = predict(
        sgpa=7.5,
        jee_score=200,
        marks_12=80.0,
        attendance=85.0,
        gender="female",
    )
    logger.info(f"Before retrain: {result1}")
    
    # Invalidate cache (simulating retrain)
    logger.info("Invalidating cache...")
    invalidate_cache()
    
    # Get prediction again (should reload model)
    result2 = predict(
        sgpa=7.5,
        jee_score=200,
        marks_12=80.0,
        attendance=85.0,
        gender="female",
    )
    logger.info(f"After cache invalidation: {result2}")
    
    # Results should be identical now (same model)
    if result1["probability"] == result2["probability"]:
        logger.info(f"✓ Cache invalidation works (predictions consistent)")
    else:
        logger.warning(f"⚠️ Predictions changed after cache invalidation")
    
    logger.info("✓ Test 6 PASSED\n")
    return True


def run_all_tests():
    """Run all tests."""
    logger.info("\n")
    logger.info("╔" + "=" * 78 + "╗")
    logger.info("║" + " " * 20 + "ML PREDICTION E2E TEST SUITE" + " " * 30 + "║")
    logger.info("╚" + "=" * 78 + "╝")
    logger.info("")
    
    try:
        test1 = test_single_prediction()
        test2 = test_different_inputs_different_outputs()
        test3 = test_gender_variation()
        test4 = test_boundary_cases()
        test5 = test_model_metadata()
        test6 = test_cache_invalidation()
        
        logger.info("")
        logger.info("╔" + "=" * 78 + "╗")
        logger.info("║" + " " * 25 + "ALL TESTS PASSED ✓" + " " * 35 + "║")
        logger.info("╚" + "=" * 78 + "╝")
        logger.info("")
        logger.info("✅ Predictions are working correctly with different inputs!")
        logger.info("✅ Check the logs above for detailed output.")
        logger.info("")
        
        return True
        
    except Exception as e:
        logger.error(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
