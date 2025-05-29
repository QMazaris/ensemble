import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys
from unittest.mock import Mock

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from backend.helpers.data import (
    prepare_data, apply_variance_filter, apply_correlation_filter,
    Regular_Split, get_cv_splitter
)


class MockConfig:
    """Mock configuration for testing."""
    def __init__(self):
        self.TARGET = 'GT_Label'
        self.EXCLUDE_COLS = ['Image', 'ID']
        self.SUMMARY = False
        self.N_SPLITS = 5
        self.RANDOM_STATE = 42


class TestDataHelpers:
    """Test suite for data helper functions."""
    
    def setup_method(self):
        """Setup test data for each test method."""
        # Create sample data
        self.sample_data = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [0.1, 0.2, 0.3, 0.4, 0.5],
            'categorical': ['A', 'B', 'A', 'B', 'A'],
            'GT_Label': ['Good', 'Bad', 'Good', 'Bad', 'Good'],
            'Image': ['img1.jpg', 'img2.jpg', 'img3.jpg', 'img4.jpg', 'img5.jpg'],
            'ID': [1, 2, 3, 4, 5]
        })
        self.config = MockConfig()
    
    def test_prepare_data_basic(self):
        """Test basic data preparation functionality."""
        X, y, numeric_cols, encoded_cols = prepare_data(self.sample_data, self.config)
        
        # Check that target is properly encoded
        assert list(y) == [0, 1, 0, 1, 0]  # Good=0, Bad=1
        
        # Check that excluded columns are removed
        assert 'Image' not in X.columns
        assert 'ID' not in X.columns
        assert 'GT_Label' not in X.columns
        
        # Check that categorical variables are encoded
        assert 'categorical_B' in X.columns  # One-hot encoded
        
        # Check numeric and encoded column tracking
        assert 'feature1' in numeric_cols
        assert 'feature2' in numeric_cols
        assert 'categorical_B' in encoded_cols
    
    def test_prepare_data_invalid_target(self):
        """Test data preparation with invalid target values."""
        invalid_data = self.sample_data.copy()
        invalid_data.loc[0, 'GT_Label'] = 'Invalid'
        
        with pytest.raises(ValueError, match="Found unmapped values"):
            prepare_data(invalid_data, self.config)
    
    def test_apply_variance_filter(self):
        """Test variance filtering functionality."""
        # Create data with low variance feature
        test_data = pd.DataFrame({
            'high_var': [1, 2, 3, 4, 5],
            'low_var': [1, 1, 1, 1, 1],  # No variance
            'medium_var': [1, 1, 2, 2, 2]
        })
        
        filtered_data = apply_variance_filter(test_data, threshold=0.1, SUMMARY=False)
        
        # Low variance column should be removed
        assert 'low_var' not in filtered_data.columns
        assert 'high_var' in filtered_data.columns
        assert 'medium_var' in filtered_data.columns
    
    def test_apply_correlation_filter(self):
        """Test correlation filtering functionality."""
        # Create highly correlated features
        test_data = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [1.1, 2.1, 3.1, 4.1, 5.1],  # Highly correlated with feature1
            'feature3': [5, 4, 3, 2, 1]  # Not correlated
        })
        
        filtered_data = apply_correlation_filter(test_data, threshold=0.9, SUMMARY=False)
        
        # One of the highly correlated features should be removed
        assert len(filtered_data.columns) == 2
        assert 'feature1' in filtered_data.columns
        assert 'feature3' in filtered_data.columns
    
    def test_regular_split(self):
        """Test regular train/test split functionality."""
        X = pd.DataFrame({'feature1': range(100), 'feature2': range(100, 200)})
        y = pd.Series([0, 1] * 50)  # Balanced classes
        
        X_train, y_train, train_idx, test_idx, single_splits = Regular_Split(
            self.config, X, y
        )
        
        # Check split proportions
        assert len(X_train) == 80  # 80% train
        assert len(X_train) + len(test_idx) == 100  # Total samples
        
        # Check that splits dictionary is properly formed
        assert 'Train' in single_splits
        assert 'Test' in single_splits
        assert 'Full' in single_splits
        
        # Check data integrity
        assert len(X_train) == len(y_train)
        assert len(train_idx) == len(X_train)
    
    def test_get_cv_splitter(self):
        """Test cross-validation splitter creation."""
        splitter = get_cv_splitter(self.config)
        
        # Check that it's the right type and configuration
        assert splitter.n_splits == 5
        assert splitter.shuffle == True
        assert splitter.random_state == 42
        
        # Test that it produces the expected number of splits
        X = pd.DataFrame({'feature': range(50)})
        y = pd.Series([0, 1] * 25)
        
        splits = list(splitter.split(X, y))
        assert len(splits) == 5
        
        # Check that each split has train and test indices
        for train_idx, test_idx in splits:
            assert len(train_idx) > 0
            assert len(test_idx) > 0
            assert len(set(train_idx) & set(test_idx)) == 0  # No overlap


if __name__ == "__main__":
    pytest.main([__file__]) 