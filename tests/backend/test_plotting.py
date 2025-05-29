import pytest
import numpy as np
from pathlib import Path
import sys
from unittest.mock import Mock

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from backend.helpers.plotting import (
    plot_threshold_sweep, plot_runs_at_threshold, plot_class_balance
)
from backend.helpers.metrics import ModelEvaluationResult, ModelEvaluationRun


class TestPlottingHelpers:
    """Test suite for plotting helper functions."""
    
    def setup_method(self):
        """Setup test data for each test method."""
        # Create sample sweep results
        self.sweep_results = {
            0.3: {'precision': 0.8, 'recall': 0.7, 'accuracy': 0.75, 'cost': 10},
            0.5: {'precision': 0.85, 'recall': 0.65, 'accuracy': 0.80, 'cost': 8},
            0.7: {'precision': 0.9, 'recall': 0.6, 'accuracy': 0.78, 'cost': 12}
        }
        
        # Create sample model runs
        self.sample_runs = [
            ModelEvaluationRun(
                model_name='Model1',
                results=[
                    ModelEvaluationResult(
                        model_name='Model1',
                        split='Test',
                        threshold_type='cost',
                        threshold=0.5,
                        precision=0.85,
                        recall=0.75,
                        accuracy=0.80,
                        cost=8.0,
                        tp=15, fp=3, tn=12, fn=5,
                        is_base_model=False
                    )
                ],
                probabilities={'Test': [0.1, 0.9, 0.3, 0.7]},
                sweep_data=None
            ),
            ModelEvaluationRun(
                model_name='Model2',
                results=[
                    ModelEvaluationResult(
                        model_name='Model2',
                        split='Test',
                        threshold_type='cost',
                        threshold=0.6,
                        precision=0.82,
                        recall=0.78,
                        accuracy=0.79,
                        cost=9.0,
                        tp=14, fp=4, tn=11, fn=6,
                        is_base_model=False
                    )
                ],
                probabilities={'Test': [0.2, 0.8, 0.4, 0.6]},
                sweep_data=None
            )
        ]
    
    def test_plot_threshold_sweep_basic(self):
        """Test basic threshold sweep plotting functionality."""
        plot_data = plot_threshold_sweep(
            self.sweep_results, 
            C_FP=1.0, 
            C_FN=5.0,
            cost_optimal_thr=0.5,
            accuracy_optimal_thr=0.3,
            SUMMARY=False
        )
        
        # Check that all required data is present
        assert 'thresholds' in plot_data
        assert 'precision' in plot_data
        assert 'recall' in plot_data
        assert 'accuracy' in plot_data
        assert 'cost' in plot_data
        
        # Check data integrity
        assert len(plot_data['thresholds']) == 3
        assert plot_data['thresholds'] == [0.3, 0.5, 0.7]
        assert plot_data['precision'] == [0.8, 0.85, 0.9]
        assert plot_data['accuracy'] == [0.75, 0.80, 0.78]
        
        # Check optimal thresholds are stored
        assert plot_data['cost_optimal_threshold'] == 0.5
        assert plot_data['accuracy_optimal_threshold'] == 0.3
        
        # Check cost parameters
        assert plot_data['C_FP'] == 1.0
        assert plot_data['C_FN'] == 5.0
    
    def test_plot_threshold_sweep_empty(self):
        """Test threshold sweep with empty results."""
        empty_results = {}
        plot_data = plot_threshold_sweep(empty_results, C_FP=1.0, C_FN=5.0)
        
        # Should return empty lists
        assert plot_data['thresholds'] == []
        assert plot_data['precision'] == []
        assert plot_data['recall'] == []
        assert plot_data['accuracy'] == []
        assert plot_data['cost'] == []
    
    def test_plot_runs_at_threshold_basic(self):
        """Test basic model comparison plotting functionality."""
        comparison_data = plot_runs_at_threshold(
            self.sample_runs,
            threshold_type='cost',
            split_name='Test',
            C_FP=1.0,
            C_FN=5.0
        )
        
        # Check that all required data is present
        assert 'model_names' in comparison_data
        assert 'precision' in comparison_data
        assert 'recall' in comparison_data
        assert 'accuracy' in comparison_data
        assert 'cost' in comparison_data
        assert 'thresholds' in comparison_data
        
        # Check data integrity
        assert len(comparison_data['model_names']) == 2
        assert comparison_data['model_names'] == ['Model1', 'Model2']
        assert comparison_data['precision'] == [0.85, 0.82]
        assert comparison_data['recall'] == [0.75, 0.78]
        assert comparison_data['accuracy'] == [0.80, 0.79]
        assert comparison_data['cost'] == [8.0, 9.0]
        
        # Check metadata
        assert comparison_data['split_name'] == 'Test'
        assert comparison_data['threshold_type'] == 'cost'
    
    def test_plot_runs_at_threshold_invalid_type(self):
        """Test model comparison with invalid threshold type."""
        with pytest.raises(ValueError, match="threshold_type must be 'cost' or 'accuracy'"):
            plot_runs_at_threshold(
                self.sample_runs,
                threshold_type='invalid',
                split_name='Test'
            )
    
    def test_plot_runs_at_threshold_no_matches(self):
        """Test model comparison with no matching results."""
        with pytest.raises(RuntimeError, match="No results for split=Nonexistent, threshold_type=cost"):
            plot_runs_at_threshold(
                self.sample_runs,
                threshold_type='cost',
                split_name='NonExistent'
            )
    
    def test_plot_class_balance_basic(self):
        """Test basic class balance plotting functionality."""
        y = np.array([0, 0, 0, 1, 1])  # 3 Good, 2 Bad
        
        balance_data = plot_class_balance(y, SUMMARY=False)
        
        # Check that all required data is present
        assert 'labels' in balance_data
        assert 'values' in balance_data
        assert 'percentages' in balance_data
        assert 'total' in balance_data
        
        # Check data integrity
        assert balance_data['labels'] == ['Good (0)', 'Bad (1)']
        assert balance_data['values'] == [3, 2]
        assert balance_data['total'] == 5
        
        # Check percentages (should sum to 100)
        assert abs(sum(balance_data['percentages']) - 100.0) < 0.001
        assert abs(balance_data['percentages'][0] - 60.0) < 0.001  # 3/5 = 60%
        assert abs(balance_data['percentages'][1] - 40.0) < 0.001  # 2/5 = 40%
    
    def test_plot_class_balance_imbalanced(self):
        """Test class balance with highly imbalanced data."""
        y = np.array([0] * 90 + [1] * 10)  # 90% Good, 10% Bad
        
        balance_data = plot_class_balance(y, SUMMARY=False)
        
        assert balance_data['values'] == [90, 10]
        assert balance_data['total'] == 100
        assert abs(balance_data['percentages'][0] - 90.0) < 0.001
        assert abs(balance_data['percentages'][1] - 10.0) < 0.001
    
    def test_plot_class_balance_single_class(self):
        """Test class balance with only one class present."""
        y = np.array([0, 0, 0, 0])  # Only Good class
        
        balance_data = plot_class_balance(y, SUMMARY=False)
        
        assert balance_data['values'] == [4, 0]  # 4 Good, 0 Bad
        assert balance_data['total'] == 4
        assert balance_data['percentages'][0] == 100.0
        assert balance_data['percentages'][1] == 0.0


if __name__ == "__main__":
    pytest.main([__file__]) 