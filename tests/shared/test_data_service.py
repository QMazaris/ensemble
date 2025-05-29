import pytest
import tempfile
import json
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from shared.data_service import DataService, data_service


class TestDataService:
    """Test suite for the DataService class."""
    
    def setup_method(self):
        """Setup for each test method."""
        # Clear any existing data
        data_service.clear_all_data()
    
    def test_singleton_pattern(self):
        """Test that DataService follows singleton pattern."""
        service1 = DataService()
        service2 = DataService()
        assert service1 is service2
        assert service1 is data_service
    
    def test_set_and_get_metrics_data(self):
        """Test setting and getting metrics data."""
        test_metrics = {
            'model_metrics': [{'model': 'test', 'accuracy': 0.95}],
            'confusion_matrices': [{'tp': 10, 'fp': 2, 'tn': 8, 'fn': 1}],
            'model_summary': [{'model': 'test', 'total_samples': 21}]
        }
        
        data_service.set_metrics_data(test_metrics)
        retrieved_metrics = data_service.get_metrics_data()
        
        assert retrieved_metrics == test_metrics
        assert retrieved_metrics['model_metrics'][0]['accuracy'] == 0.95
    
    def test_set_and_get_predictions_data(self):
        """Test setting and getting predictions data."""
        test_predictions = {
            'GT': [0, 1, 0, 1],
            'index': [0, 1, 2, 3],
            'model1': [0.1, 0.9, 0.2, 0.8],
            'model2': [0.3, 0.7, 0.4, 0.6]
        }
        
        data_service.set_predictions_data(test_predictions)
        retrieved_predictions = data_service.get_predictions_data()
        
        assert retrieved_predictions == test_predictions
        assert len(retrieved_predictions['GT']) == 4
    
    def test_set_and_get_sweep_data(self):
        """Test setting and getting sweep data."""
        test_sweep = {
            'model1': {
                'probabilities': [0.1, 0.2, 0.3],
                'thresholds': [0.3, 0.5, 0.7],
                'costs': [10, 8, 12],
                'accuracies': [0.8, 0.85, 0.82]
            }
        }
        
        data_service.set_sweep_data(test_sweep)
        retrieved_sweep = data_service.get_sweep_data()
        
        assert retrieved_sweep == test_sweep
        assert retrieved_sweep['model1']['thresholds'] == [0.3, 0.5, 0.7]
    
    def test_has_data(self):
        """Test the has_data method."""
        assert not data_service.has_data()
        
        data_service.set_metrics_data({'test': 'data'})
        assert data_service.has_data()
        
        data_service.clear_all_data()
        assert not data_service.has_data()
    
    def test_clear_all_data(self):
        """Test clearing all data."""
        # Set some data
        data_service.set_metrics_data({'test': 'metrics'})
        data_service.set_predictions_data({'test': 'predictions'})
        data_service.set_sweep_data({'test': 'sweep'})
        
        assert data_service.has_data()
        
        # Clear all data
        data_service.clear_all_data()
        
        assert not data_service.has_data()
        assert data_service.get_metrics_data() is None
        assert data_service.get_predictions_data() is None
        assert data_service.get_sweep_data() is None
    
    def test_save_to_files(self):
        """Test saving data to files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Set test data
            test_metrics = {
                'model_metrics': [{'model': 'test', 'accuracy': 0.95}],
                'confusion_matrices': [{'tp': 10, 'fp': 2}],
                'model_summary': [{'model': 'test', 'samples': 100}]
            }
            test_sweep = {'model1': {'thresholds': [0.5], 'costs': [10]}}
            test_predictions = {'GT': [0, 1], 'model1': [0.1, 0.9]}
            
            data_service.set_metrics_data(test_metrics)
            data_service.set_sweep_data(test_sweep)
            data_service.set_predictions_data(test_predictions)
            
            # Save to files
            data_service.save_to_files(temp_path)
            
            # Check that files were created
            assert (temp_path / 'model_metrics.csv').exists()
            assert (temp_path / 'confusion_matrices.csv').exists()
            assert (temp_path / 'model_summary.csv').exists()
            assert (temp_path / 'threshold_sweep_data.json').exists()
            assert (temp_path / 'all_model_predictions.csv').exists()
            
            # Check JSON content
            with open(temp_path / 'threshold_sweep_data.json', 'r') as f:
                saved_sweep = json.load(f)
            assert saved_sweep == test_sweep
    
    def test_get_nonexistent_data(self):
        """Test getting data that doesn't exist."""
        data_service.clear_all_data()
        
        assert data_service.get_metrics_data() is None
        assert data_service.get_predictions_data() is None
        assert data_service.get_sweep_data() is None


if __name__ == "__main__":
    pytest.main([__file__]) 