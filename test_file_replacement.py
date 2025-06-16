#!/usr/bin/env python3
"""
Test script to verify file replacement works in Docker container.
Run this inside your Docker container to test the fix.
"""

import os
import pandas as pd
import time
from pathlib import Path

def test_file_replacement():
    """Test that file replacement is detected correctly."""
    test_file = "data/test_replacement.csv"
    
    print("🧪 Testing file replacement detection...")
    
    # Create initial test file
    df1 = pd.DataFrame({"col1": [1, 2, 3], "col2": ["a", "b", "c"]})
    df1.to_csv(test_file, index=False)
    print(f"✅ Created initial file: {test_file}")
    
    # Load using our robust function
    try:
        from backend.helpers.data import load_csv_robust
        loaded_df1 = load_csv_robust(test_file)
        print(f"✅ Initial load successful: {loaded_df1.shape}")
    except Exception as e:
        print(f"❌ Initial load failed: {e}")
        return False
    
    # Replace the file with different content
    print("🔄 Replacing file with new content...")
    df2 = pd.DataFrame({"col1": [4, 5, 6, 7], "col2": ["d", "e", "f", "g"], "col3": [10, 20, 30, 40]})
    df2.to_csv(test_file, index=False)
    
    # Small delay to ensure file system sync
    time.sleep(0.1)
    
    # Try to load the replaced file
    try:
        loaded_df2 = load_csv_robust(test_file)
        print(f"✅ Replacement load successful: {loaded_df2.shape}")
        
        # Verify we got the new content
        if loaded_df2.shape[0] == 4 and "col3" in loaded_df2.columns:
            print("✅ File replacement detected correctly!")
            success = True
        else:
            print("❌ File replacement not detected - got old content")
            success = False
            
    except Exception as e:
        print(f"❌ Replacement load failed: {e}")
        success = False
    
    # Cleanup
    try:
        os.remove(test_file)
        print("🧹 Cleaned up test file")
    except:
        pass
    
    return success

def test_file_info():
    """Show file system information for debugging."""
    print("\n📊 File System Information:")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Data directory exists: {os.path.exists('data')}")
    
    if os.path.exists("data"):
        data_files = list(Path("data").glob("*.csv"))
        print(f"CSV files in data/: {[f.name for f in data_files]}")
        
        for file in data_files[:3]:  # Show first 3 files
            stat = file.stat()
            print(f"  {file.name}: {stat.st_size} bytes, modified {time.ctime(stat.st_mtime)}")

if __name__ == "__main__":
    print("🔍 Docker File Replacement Test")
    print("=" * 40)
    
    test_file_info()
    
    print("\n" + "=" * 40)
    success = test_file_replacement()
    
    print("\n" + "=" * 40)
    if success:
        print("🎉 All tests passed! File replacement should work correctly.")
    else:
        print("⚠️  Tests failed. File replacement issue may persist.") 