#!/usr/bin/env python3
"""
Test script for bitwise logic functionality
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_imports():
    """Test that all required modules can be imported."""
    print("🔍 Testing imports...")
    
    try:
        from backend.helpers.stacked_logic import generate_combined_runs
        print("✅ stacked_logic imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import stacked_logic: {e}")
        return False
    
    try:
        from shared.config_manager import config_manager
        print("✅ config_manager imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import config_manager: {e}")
        return False
        
    try:
        from backend import config_adapter as config
        print("✅ config_adapter imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import config_adapter: {e}")
        return False
        
    return True

def test_config_access():
    """Test that we can access the bitwise logic configuration."""
    print("\n🔍 Testing config access...")
    
    try:
        from shared.config_manager import config_manager
        
        # Test reading bitwise logic config
        bitwise_config = config_manager.get('models.bitwise_logic', {
            'rules': [],
            'enabled': False
        })
        
        print(f"✅ Bitwise logic config: {bitwise_config}")
        return True
        
    except Exception as e:
        print(f"❌ Failed to access config: {e}")
        return False

def test_bitwise_logic_function():
    """Test the bitwise logic function with sample data."""
    print("\n🔍 Testing bitwise logic function...")
    
    try:
        import numpy as np
        from backend.helpers.stacked_logic import generate_combined_runs
        from backend.helpers.metrics import ModelEvaluationRun, ModelEvaluationResult
        
        # Create sample data
        n_samples = 100
        y_true = np.random.randint(0, 2, n_samples)
        
        # Create mock model runs
        ad_decisions = np.random.randint(0, 2, n_samples)
        cl_decisions = np.random.randint(0, 2, n_samples)
        
        # Create sample ModelEvaluationRun objects
        ad_result = ModelEvaluationResult(
            model_name="AD_Decision",
            split='Full',
            threshold_type='base',
            threshold=None,
            precision=50.0,
            recall=50.0,
            f1_score=50.0,
            accuracy=50.0,
            cost=100.0,
            tp=25, fp=25, tn=25, fn=25,
            is_base_model=True
        )
        
        cl_result = ModelEvaluationResult(
            model_name="CL_Decision",
            split='Full',
            threshold_type='base',
            threshold=None,
            precision=50.0,
            recall=50.0,
            f1_score=50.0,
            accuracy=50.0,
            cost=100.0,
            tp=25, fp=25, tn=25, fn=25,
            is_base_model=True
        )
        
        ad_run = ModelEvaluationRun(
            model_name="AD_Decision",
            results=[ad_result],
            probabilities={'Full': ad_decisions}
        )
        
        cl_run = ModelEvaluationRun(
            model_name="CL_Decision",
            results=[cl_result],
            probabilities={'Full': cl_decisions}
        )
        
        runs = [ad_run, cl_run]
        
        # Define combined logic rules
        combined_logic = {
            'Combined_OR': {
                'columns': ['AD_Decision', 'CL_Decision'],
                'logic': 'OR'
            },
            'Combined_AND': {
                'columns': ['AD_Decision', 'CL_Decision'],
                'logic': 'AND'
            }
        }
        
        # Test the function
        combined_runs = generate_combined_runs(
            runs=runs,
            combined_logic=combined_logic,
            y_true=y_true,
            C_FP=1.0,
            C_FN=30.0,
            N_SPLITS=5
        )
        
        print(f"✅ Generated {len(combined_runs)} combined models:")
        for run in combined_runs:
            print(f"  • {run.model_name}")
            for result in run.results:
                print(f"    - Accuracy: {result.accuracy:.1f}%, Cost: {result.cost:.1f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to test bitwise logic function: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("🚀 Testing Bitwise Logic Implementation")
    print("="*50)
    
    tests = [
        test_imports,
        test_config_access,
        test_bitwise_logic_function
    ]
    
    passed = 0
    for test in tests:
        if test():
            passed += 1
        else:
            print(f"❌ Test {test.__name__} failed")
    
    print(f"\n📊 Results: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("🎉 All tests passed! Bitwise logic implementation is working.")
        return True
    else:
        print("❌ Some tests failed. Please check the implementation.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 