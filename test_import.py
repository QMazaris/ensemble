#!/usr/bin/env python3
"""Test script to check imports."""

import sys
from pathlib import Path

# Add frontend/streamlit to path
frontend_path = Path("frontend/streamlit")
sys.path.insert(0, str(frontend_path))

try:
    from utils import _save_base_model_columns_config_helper
    print("✅ Import successful: _save_base_model_columns_config_helper")
except ImportError as e:
    print(f"❌ Import failed: {e}")
except Exception as e:
    print(f"❌ Error: {e}")

try:
    import streamlit as st
    print("✅ Streamlit import successful")
except ImportError as e:
    print(f"❌ Streamlit import failed: {e}")

print("Test completed.") 