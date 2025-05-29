#!/usr/bin/env python3
"""
Test script to demonstrate the improved logging functionality for the Ensemble Pipeline.
This script simulates API calls and data operations to show the logging in action.
"""

import time
import sys
from pathlib import Path

# Add the project root to the path so we can import our modules
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from shared.logging_config import get_api_logger, get_frontend_logger, log_api_call, log_data_operation

def test_api_logging():
    """Test API logging functionality."""
    print("🧪 Testing API Logging...")
    
    api_logger = get_api_logger()
    frontend_logger = get_frontend_logger()
    
    # Simulate various API calls
    print("\n1. Successful API calls:")
    log_api_call(api_logger, "GET", "/results/metrics", status=200, time=0.123, data_info="3 models, 15 metrics")
    log_api_call(api_logger, "GET", "/results/predictions", status=200, time=0.087, data_info="1000 predictions")
    log_api_call(api_logger, "GET", "/results/threshold-sweep", status=200, time=0.234, params={"model_name": "XGBoost", "data_type": "costs"})
    
    print("\n2. Failed API calls:")
    log_api_call(api_logger, "GET", "/results/metrics", status=404, time=0.045)
    log_api_call(frontend_logger, "GET", "/results/predictions", status=500, time=0.156)
    
    print("\n3. Frontend API calls:")
    log_api_call(frontend_logger, "GET", "http://localhost:8000/results/metrics", status=200, time=0.089, data_info="Successfully loaded 3 models")

def test_data_logging():
    """Test data operation logging functionality."""
    print("\n🧪 Testing Data Operation Logging...")
    
    api_logger = get_api_logger()
    frontend_logger = get_frontend_logger()
    
    # Simulate data operations
    print("\n1. Data loading operations:")
    log_data_operation(api_logger, "fetch", "metrics_data", count=15, source="data_service", success=True)
    log_data_operation(frontend_logger, "load", "predictions_data", count=1000, source="API", success=True)
    log_data_operation(frontend_logger, "load", "sweep_data", source="backup_file", success=True)
    
    print("\n2. Failed data operations:")
    log_data_operation(frontend_logger, "load", "metrics_data", source="API", success=False)
    log_data_operation(api_logger, "fetch", "predictions_data", source="data_service", success=False)

def test_detailed_logging():
    """Test detailed logging scenarios."""
    print("\n🧪 Testing Detailed Logging Scenarios...")
    
    api_logger = get_api_logger()
    frontend_logger = get_frontend_logger()
    
    # Simulate realistic scenarios
    api_logger.info("🚀 Starting pipeline run...")
    api_logger.info("📥 API CALL: POST /pipeline/run")
    time.sleep(0.1)  # Simulate processing time
    api_logger.info("📤 API RESPONSE: POST /pipeline/run - Status: 200 - Time: 0.098s")
    
    frontend_logger.info("🔄 Starting metrics data loading process...")
    frontend_logger.info("📡 Attempting to load metrics data from API...")
    frontend_logger.info("🚀 FRONTEND API CALL: GET http://localhost:8000/results/metrics")
    time.sleep(0.05)
    frontend_logger.info("✅ FRONTEND API SUCCESS: /results/metrics - 0.089s - Metrics: 3, Summaries: 3, CM: 3")
    
    api_logger.info("🔍 Fetching model metrics from data service...")
    api_logger.info("✅ Returning metrics data: 3 model metrics, 3 summaries, 3 confusion matrices")
    
    frontend_logger.info("📊 Converted to DataFrames - Metrics: 3, Summary: 3, CM: 3")
    frontend_logger.info("🔍 Attempting to load sweep data from data service...")
    frontend_logger.info("✅ Got sweep data from data service")

def main():
    """Main test function."""
    print("🧪 Testing Ensemble Pipeline Logging System")
    print("=" * 50)
    
    # Create logs directory if it doesn't exist
    logs_dir = Path("output/logs")
    logs_dir.mkdir(parents=True, exist_ok=True)
    
    # Run tests
    test_api_logging()
    test_data_logging()
    test_detailed_logging()
    
    print(f"\n✅ Logging test completed!")
    print(f"📁 Check the following log files:")
    print(f"   - {logs_dir}/api.log")
    print(f"   - {logs_dir}/frontend.log")
    print(f"\n💡 These logs will also appear in your console when running the actual application.")

if __name__ == "__main__":
    main() 