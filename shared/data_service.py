"""
Shared data service for storing pipeline results in memory.
This service acts as a singleton to store data across different modules.
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional

class DataService:
    """Singleton service for storing pipeline data in memory with JSON backup."""
    
    _instance = None
    _data = {}
    _backup_dir = Path("output/data_service_backup")
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DataService, cls).__new__(cls)
            cls._instance._data = {}
            cls._instance._backup_dir.mkdir(parents=True, exist_ok=True)
        return cls._instance
    
    def _save_backup(self, key: str, data: Any):
        """Save data to JSON backup file."""
        try:
            backup_file = self._backup_dir / f"{key}.json"
            with open(backup_file, 'w') as f:
                json.dump(data, f, indent=2, default=str)
        except Exception as e:
            print(f"Warning: Failed to save backup for {key}: {e}")
    
    def _load_backup(self, key: str) -> Optional[Any]:
        """Load data from JSON backup file."""
        try:
            backup_file = self._backup_dir / f"{key}.json"
            if backup_file.exists():
                with open(backup_file, 'r') as f:
                    data = json.load(f)
                    # Fix data types after loading from JSON
                    return self._fix_data_types(data, key)
        except Exception as e:
            print(f"Warning: Failed to load backup for {key}: {e}")
        return None
    
    def _fix_data_types(self, data, key: str):
        """Fix data types that may have been corrupted during JSON serialization."""
        if key == 'metrics' and isinstance(data, dict):
            # Fix metrics data types
            for section_name in ['model_metrics', 'model_summary', 'confusion_matrices']:
                if section_name in data and isinstance(data[section_name], list):
                    for item in data[section_name]:
                        if isinstance(item, dict):
                            # Convert numeric fields to proper types
                            numeric_fields = ['accuracy', 'precision', 'recall', 'cost', 'threshold', 
                                            'tp', 'fp', 'tn', 'fn', 'total_samples']
                            for field in numeric_fields:
                                if field in item and item[field] is not None:
                                    try:
                                        if field in ['tp', 'fp', 'tn', 'fn', 'total_samples']:
                                            # These should be integers
                                            item[field] = int(float(str(item[field])))
                                        else:
                                            # These should be floats
                                            item[field] = float(str(item[field]))
                                    except (ValueError, TypeError):
                                        # Keep original value if conversion fails
                                        pass
        elif key == 'predictions' and isinstance(data, list):
            # Fix predictions data types - ensure all numeric values are floats
            for item in data:
                if isinstance(item, dict):
                    for field, value in item.items():
                        if field not in ['GT', 'index'] and value is not None:  # GT and index are kept as-is
                            try:
                                item[field] = float(str(value))
                            except (ValueError, TypeError):
                                # Keep original value if conversion fails
                                pass
        elif key == 'sweep' and isinstance(data, dict):
            # Fix sweep data types
            for model_name, model_data in data.items():
                if isinstance(model_data, dict):
                    for field in ['probabilities', 'thresholds', 'costs', 'accuracies']:
                        if field in model_data and isinstance(model_data[field], list):
                            try:
                                model_data[field] = [float(str(x)) for x in model_data[field]]
                            except (ValueError, TypeError):
                                # Keep original values if conversion fails
                                pass
        
        return data
    
    def set_metrics_data(self, data: Dict[str, Any]):
        """Store metrics data in memory and backup."""
        self._data['metrics'] = data
        self._save_backup('metrics', data)
    
    def get_metrics_data(self) -> Optional[Dict[str, Any]]:
        """Get metrics data from memory, fallback to backup."""
        if 'metrics' in self._data:
            return self._data['metrics']
        
        # Try to load from backup
        backup_data = self._load_backup('metrics')
        if backup_data:
            self._data['metrics'] = backup_data
            return backup_data
        
        return None
    
    def set_predictions_data(self, data: Any):
        """Store predictions data in memory and backup."""
        self._data['predictions'] = data
        self._save_backup('predictions', data)
    
    def get_predictions_data(self) -> Optional[Any]:
        """Get predictions data from memory, fallback to backup."""
        if 'predictions' in self._data:
            return self._data['predictions']
        
        # Try to load from backup
        backup_data = self._load_backup('predictions')
        if backup_data:
            self._data['predictions'] = backup_data
            return backup_data
        
        return None
    
    def set_sweep_data(self, data: Dict[str, Any]):
        """Store threshold sweep data in memory and backup."""
        self._data['sweep'] = data
        self._save_backup('sweep', data)
    
    def get_sweep_data(self) -> Optional[Dict[str, Any]]:
        """Get threshold sweep data from memory, fallback to backup."""
        if 'sweep' in self._data:
            return self._data['sweep']
        
        # Try to load from backup
        backup_data = self._load_backup('sweep')
        if backup_data:
            self._data['sweep'] = backup_data
            return backup_data
        
        return None
    
    def clear_all_data(self):
        """Clear all stored data from memory and backup files."""
        self._data.clear()
        
        # Remove backup files
        try:
            for backup_file in self._backup_dir.glob("*.json"):
                backup_file.unlink()
        except Exception as e:
            print(f"Warning: Failed to clear backup files: {e}")
    
    def save_to_files(self, output_dir: Path, save_csv_backup: bool = False):
        """Save current data to files as backup (optional).
        
        Args:
            output_dir: Directory to save files to
            save_csv_backup: Whether to save CSV files for backup/debugging (default: False)
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # Always save JSON files for API compatibility
        for key in ['metrics', 'predictions', 'sweep']:
            data = getattr(self, f'get_{key}_data')()
            if data:
                json_file = output_dir / f"{key}.json"
                with open(json_file, 'w') as f:
                    json.dump(data, f, indent=2, default=str)
        
        # Only save CSV files if explicitly requested (for backup/debugging)
        if save_csv_backup and 'metrics' in self._data:
            metrics = self._data['metrics']
            
            # Save individual CSV files for backward compatibility
            if 'model_metrics' in metrics:
                import pandas as pd
                pd.DataFrame(metrics['model_metrics']).to_csv(
                    output_dir / "model_metrics.csv", index=False
                )
            
            if 'confusion_matrices' in metrics:
                import pandas as pd
                pd.DataFrame(metrics['confusion_matrices']).to_csv(
                    output_dir / "confusion_matrices.csv", index=False
                )
    
    def load_from_backup(self, key: str) -> bool:
        """Explicitly load data from backup file if needed."""
        backup_data = self._load_backup(key)
        if backup_data:
            self._data[key] = backup_data
            return True
        return False
    
    def load_all_from_backup(self) -> bool:
        """Load all available data from backup files."""
        success = False
        for key in ['metrics', 'predictions', 'sweep']:
            if self.load_from_backup(key):
                success = True
                print(f"Loaded {key} data from backup")
        return success
    
    def has_backup_data(self) -> bool:
        """Check if any backup files exist."""
        backup_files = list(self._backup_dir.glob("*.json"))
        return len(backup_files) > 0

# Create singleton instance
data_service = DataService() 