"""
Data service for in-memory data sharing between backend and frontend.
This replaces CSV file intermediaries with direct data structures.
"""

import json
import threading
from typing import Dict, Any, Optional
from pathlib import Path


class DataService:
    """Singleton service for sharing data between backend and frontend."""
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self._data = {}
            self._initialized = True
    
    def set_metrics_data(self, metrics_data: Dict[str, Any]):
        """Store metrics data in memory."""
        self._data['metrics'] = metrics_data
    
    def get_metrics_data(self) -> Optional[Dict[str, Any]]:
        """Retrieve metrics data from memory."""
        return self._data.get('metrics')
    
    def set_predictions_data(self, predictions_data: Dict[str, Any]):
        """Store predictions data in memory."""
        self._data['predictions'] = predictions_data
    
    def get_predictions_data(self) -> Optional[Dict[str, Any]]:
        """Retrieve predictions data from memory."""
        return self._data.get('predictions')
    
    def set_sweep_data(self, sweep_data: Dict[str, Any]):
        """Store threshold sweep data in memory."""
        self._data['sweep'] = sweep_data
    
    def get_sweep_data(self) -> Optional[Dict[str, Any]]:
        """Retrieve threshold sweep data from memory."""
        return self._data.get('sweep')
    
    def clear_all_data(self):
        """Clear all stored data."""
        self._data.clear()
    
    def has_data(self) -> bool:
        """Check if any data is available."""
        return bool(self._data)
    
    def save_to_files(self, output_dir: Path):
        """Save current data to files as backup (optional)."""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        if 'metrics' in self._data:
            metrics = self._data['metrics']
            
            # Save individual CSV files for backward compatibility
            if 'model_metrics' in metrics:
                import pandas as pd
                pd.DataFrame(metrics['model_metrics']).to_csv(
                    output_dir / 'model_metrics.csv', index=False
                )
            
            if 'confusion_matrices' in metrics:
                import pandas as pd
                pd.DataFrame(metrics['confusion_matrices']).to_csv(
                    output_dir / 'confusion_matrices.csv', index=False
                )
            
            if 'model_summary' in metrics:
                import pandas as pd
                pd.DataFrame(metrics['model_summary']).to_csv(
                    output_dir / 'model_summary.csv', index=False
                )
        
        if 'sweep' in self._data:
            with open(output_dir / 'threshold_sweep_data.json', 'w') as f:
                json.dump(self._data['sweep'], f)
        
        if 'predictions' in self._data:
            import pandas as pd
            pd.DataFrame(self._data['predictions']).to_csv(
                output_dir / 'all_model_predictions.csv', index=False
            )


# Global instance
data_service = DataService() 