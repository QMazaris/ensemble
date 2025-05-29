#!/usr/bin/env python3
"""
Test runner for the ensemble pipeline project.
Runs all tests and generates a coverage report.
"""

import pytest
import sys
from pathlib import Path

def main():
    """Run all tests with coverage reporting."""
    # Add project root to path
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))
    
    # Test discovery and execution
    test_args = [
        str(Path(__file__).parent),  # Test directory
        "-v",  # Verbose output
        "--tb=short",  # Short traceback format
        "--strict-markers",  # Strict marker checking
        "-x",  # Stop on first failure (optional)
    ]
    
    # Add coverage if available
    try:
        import coverage
        test_args.extend([
            "--cov=backend",
            "--cov=shared", 
            "--cov=frontend",
            "--cov-report=term-missing",
            "--cov-report=html:htmlcov"
        ])
        print("Running tests with coverage reporting...")
    except ImportError:
        print("Coverage not available. Install with: pip install pytest-cov")
        print("Running tests without coverage...")
    
    # Run tests
    exit_code = pytest.main(test_args)
    
    if exit_code == 0:
        print("\n✅ All tests passed!")
    else:
        print(f"\n❌ Tests failed with exit code {exit_code}")
    
    return exit_code

if __name__ == "__main__":
    sys.exit(main()) 