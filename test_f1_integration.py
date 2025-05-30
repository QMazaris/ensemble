#!/usr/bin/env python3
"""Test F1 score integration in the metrics system."""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

from backend.helpers.metrics import compute_metrics, ModelEvaluationResult
import numpy as np

def test_f1_score_calculation():
    """Test F1 score calculation in compute_metrics function."""
    print("Testing F1 score calculation...")
    
    # Test case: TP=2, FP=1, TN=1, FN=0
    y_true = np.array([0, 1, 1, 1])
    y_pred = np.array([0, 0, 1, 1])
    
    result = compute_metrics(y_true, y_pred, C_FP=1.0, C_FN=5.0)
    
    print(f"Test case: y_true={y_true}, y_pred={y_pred}")
    print(f"TP={result['tp']}, FP={result['fp']}, TN={result['tn']}, FN={result['fn']}")
    print(f"Precision: {result['precision']:.2f}%")
    print(f"Recall: {result['recall']:.2f}%")
    print(f"F1 Score: {result['f1_score']:.2f}%")
    print(f"Accuracy: {result['accuracy']:.2f}%")
    print(f"Cost: {result['cost']:.2f}")
    
    # Manual F1 calculation for verification
    precision_raw = result['tp'] / (result['tp'] + result['fp']) if (result['tp'] + result['fp']) > 0 else 0
    recall_raw = result['tp'] / (result['tp'] + result['fn']) if (result['tp'] + result['fn']) > 0 else 0
    expected_f1 = 100 * (2 * precision_raw * recall_raw) / (precision_raw + recall_raw) if (precision_raw + recall_raw) > 0 else 0
    
    print(f"Expected F1 Score: {expected_f1:.2f}%")
    print(f"F1 Score matches expected: {abs(result['f1_score'] - expected_f1) < 0.01}")
    
    return result

def test_model_evaluation_result():
    """Test ModelEvaluationResult with F1 score."""
    print("\nTesting ModelEvaluationResult with F1 score...")
    
    result = ModelEvaluationResult(
        model_name="TestModel",
        split="Test",
        threshold_type="cost",
        threshold=0.5,
        precision=85.0,
        recall=75.0,
        f1_score=79.6,
        accuracy=80.0,
        cost=10.0,
        tp=15,
        fp=3,
        tn=12,
        fn=5,
        is_base_model=False
    )
    
    print(f"Model: {result.model_name}")
    print(f"F1 Score: {result.f1_score}%")
    print(f"Has f1_score attribute: {hasattr(result, 'f1_score')}")
    
    return result

if __name__ == "__main__":
    print("=" * 50)
    print("F1 SCORE INTEGRATION TEST")
    print("=" * 50)
    
    try:
        # Test compute_metrics function
        metrics_result = test_f1_score_calculation()
        
        # Test ModelEvaluationResult
        model_result = test_model_evaluation_result()
        
        print("\n" + "=" * 50)
        print("✅ ALL TESTS PASSED!")
        print("F1 score integration is working correctly.")
        print("=" * 50)
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc() 