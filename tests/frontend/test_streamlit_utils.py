"""
Tests for frontend streamlit utilities.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from frontend.streamlit.utils import (
    load_metrics_data, load_predictions_data, plot_confusion_matrix,
    plot_roc_curve, plot_precision_recall_curve, plot_threshold_sweep
)


class TestDataLoading:
    """Test data loading functions."""
    
    @patch('frontend.streamlit.utils.data_service')
    def test_load_metrics_data_success(self, mock_data_service):
        """Test successful metrics data loading."""
        # Mock data service responses
        mock_metrics = {
            'model_metrics': [{'model_name': 'XGBoost', 'accuracy': 0.85}],
            'model_summary': [{'model_name': 'XGBoost', 'split': 'Full'}],
            'confusion_matrices': [{'model_name': 'XGBoost', 'tp': 10}]
        }
        mock_sweep = {'XGBoost': {'thresholds': [0.5], 'costs': [1.0]}}
        
        mock_data_service.get_metrics_data.return_value = mock_metrics
        mock_data_service.get_sweep_data.return_value = mock_sweep
        
        metrics_df, summary_df, cm_df, sweep_data = load_metrics_data()
        
        assert metrics_df is not None
        assert summary_df is not None
        assert cm_df is not None
        assert sweep_data is not None
        assert len(metrics_df) == 1
        assert metrics_df.iloc[0]['model_name'] == 'XGBoost'
    
    @patch('frontend.streamlit.utils.data_service')
    def test_load_metrics_data_no_data(self, mock_data_service):
        """Test metrics data loading when no data available."""
        mock_data_service.get_metrics_data.return_value = None
        mock_data_service.get_sweep_data.return_value = None
        
        result = load_metrics_data()
        
        assert result == (None, None, None, None)
    
    @patch('frontend.streamlit.utils.data_service')
    def test_load_predictions_data_success(self, mock_data_service):
        """Test successful predictions data loading."""
        mock_predictions = [
            {'GT': 1, 'XGBoost': 0.8, 'RandomForest': 0.7},
            {'GT': 0, 'XGBoost': 0.3, 'RandomForest': 0.4}
        ]
        
        mock_data_service.get_predictions_data.return_value = mock_predictions
        
        predictions_df = load_predictions_data()
        
        assert predictions_df is not None
        assert len(predictions_df) == 2
        assert 'GT' in predictions_df.columns
        assert 'XGBoost' in predictions_df.columns
    
    @patch('frontend.streamlit.utils.data_service')
    def test_load_predictions_data_no_data(self, mock_data_service):
        """Test predictions data loading when no data available."""
        mock_data_service.get_predictions_data.return_value = None
        
        result = load_predictions_data()
        
        assert result is None


class TestPlottingFunctions:
    """Test plotting functions."""
    
    def test_plot_confusion_matrix(self):
        """Test confusion matrix plotting."""
        cm_data = {
            'tp': 10, 'fp': 5, 'tn': 15, 'fn': 3,
            'threshold_type': 'cost'
        }
        
        fig = plot_confusion_matrix(cm_data)
        
        assert fig is not None
        assert 'Confusion Matrix' in fig.layout.title.text
    
    def test_plot_roc_curve(self):
        """Test ROC curve plotting."""
        y_true = np.array([0, 0, 1, 1])
        y_scores = np.array([0.1, 0.4, 0.35, 0.8])
        
        fig = plot_roc_curve(y_true, y_scores)
        
        assert fig is not None
        assert 'ROC Curve' in fig.layout.title
        assert len(fig.data) == 2  # ROC curve + diagonal line
    
    def test_plot_precision_recall_curve(self):
        """Test Precision-Recall curve plotting."""
        y_true = np.array([0, 0, 1, 1])
        y_scores = np.array([0.1, 0.4, 0.35, 0.8])
        
        fig = plot_precision_recall_curve(y_true, y_scores)
        
        assert fig is not None
        assert 'Precision-Recall Curve' in fig.layout.title
        assert len(fig.data) == 1  # PR curve
    
    def test_plot_threshold_sweep(self):
        """Test threshold sweep plotting."""
        sweep_data = {
            'XGBoost': {
                'thresholds': [0.1, 0.5, 0.9],
                'costs': [10.0, 5.0, 8.0],
                'accuracies': [0.7, 0.85, 0.75]
            }
        }
        
        fig = plot_threshold_sweep(sweep_data, 'XGBoost')
        
        assert fig is not None
        assert 'Threshold Sweep - XGBoost' in fig.layout.title
        assert len(fig.data) == 2  # Cost and accuracy curves
    
    def test_plot_threshold_sweep_missing_model(self):
        """Test threshold sweep plotting with missing model."""
        sweep_data = {'RandomForest': {}}
        
        fig = plot_threshold_sweep(sweep_data, 'XGBoost')
        
        assert fig is None


class TestUtilityFunctions:
    """Test utility functions."""
    
    def test_ensure_directories(self):
        """Test directory creation."""
        from frontend.streamlit.utils import ensure_directories
        
        # This should not raise an exception
        ensure_directories()
    
    def test_get_plot_groups(self):
        """Test plot file grouping."""
        from frontend.streamlit.utils import get_plot_groups
        from pathlib import Path
        import tempfile
        import os
        
        # Create temporary directory with test files
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create test plot files
            test_files = [
                'XGBoost_roc_curve.png',
                'RandomForest_confusion_matrix.png',
                'threshold_sweep_comparison.png',
                'model_comparison.png',
                'other_plot.png'
            ]
            
            for file_name in test_files:
                (temp_path / file_name).touch()
            
            groups = get_plot_groups(temp_path)
            
            assert 'XGBoost' in groups
            assert 'RandomForest' in groups
            assert 'Comparison' in groups
            assert 'Other' in groups


if __name__ == "__main__":
    pytest.main([__file__]) 