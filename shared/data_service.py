"""
Simplified data service for storing pipeline results in memory.
This service acts as a singleton to store data across different modules.
Simple in-memory storage - users re-run pipeline for fresh data.
"""

from typing import Dict, Any, Optional

class DataService:
    """Singleton service for storing pipeline data in memory."""
    
    _instance = None
    _data = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DataService, cls).__new__(cls)
            cls._instance._data = {}
            print("📊 DataService ready (pure in-memory storage)")
        return cls._instance
    
    def set_metrics_data(self, data: Dict[str, Any]):
        """Store metrics data in memory."""
        self._data['metrics'] = data
        print("💾 Metrics data stored in memory")
    
    def get_metrics_data(self) -> Optional[Dict[str, Any]]:
        """Get metrics data from memory."""
        return self._data.get('metrics')
    
    def set_predictions_data(self, data: Any):
        """Store predictions data in memory."""
        self._data['predictions'] = data
        print("💾 Predictions data stored in memory")
    
    def get_predictions_data(self) -> Optional[Any]:
        """Get predictions data from memory."""
        return self._data.get('predictions')
    
    def set_sweep_data(self, data: Dict[str, Any]):
        """Store threshold sweep data in memory."""
        self._data['sweep'] = data
        print("💾 Sweep data stored in memory")
    
    def get_sweep_data(self) -> Optional[Dict[str, Any]]:
        """Get threshold sweep data from memory."""
        return self._data.get('sweep')
    
    def clear_all_data(self):
        """Clear all stored data from memory."""
        self._data.clear()
        print("🗑️ All data cleared from memory")
    
    def has_data(self) -> bool:
        """Check if service has any data."""
        return len(self._data) > 0
    
    def get_data_summary(self) -> Dict[str, Any]:
        """Get a summary of stored data."""
        summary = {
            "has_metrics": "metrics" in self._data,
            "has_predictions": "predictions" in self._data,
            "has_sweep": "sweep" in self._data,
            "total_datasets": len(self._data)
        }
        
        if "predictions" in self._data:
            predictions = self._data["predictions"]
            if isinstance(predictions, list):
                summary["prediction_count"] = len(predictions)
                if predictions:
                    summary["model_count"] = len([k for k in predictions[0].keys() if k not in ['GT', 'index']])
            elif isinstance(predictions, dict):
                summary["prediction_count"] = len(predictions.get('GT', []))
                summary["model_count"] = len([k for k in predictions.keys() if k not in ['GT', 'index']])
        
        if "metrics" in self._data:
            metrics = self._data["metrics"]
            summary["metrics_count"] = len(metrics.get('model_metrics', []))
        
        return summary

# Create singleton instance
data_service = DataService() 