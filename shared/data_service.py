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
        """Store predictions data in memory and optionally backup."""
        # Store in memory first (this is the primary storage)
        self._data['predictions'] = data
        
        # Optionally save backup file (but don't fail if this doesn't work)
        try:
            self._save_backup('predictions', data)
            print("💾 Predictions data stored in memory and backed up to file")
        except Exception as e:
            print(f"⚠️ Predictions data stored in memory, but backup failed: {e}")
            # Don't raise exception - memory storage is what matters
    
    def recover_predictions_data(self, force_recreate=False):
        """
        Recover predictions data if missing by regenerating from saved models.
        
        Args:
            force_recreate: If True, recreate even if backup exists
            
        Returns:
            bool: True if recovery was successful, False otherwise
        """
        # Check if predictions backup already exists and we're not forcing recreate
        if 'predictions' in self._data and not force_recreate:
            return True
            
        print("⚠️ Predictions data missing. Attempting to recover from saved models...")
        
        try:
            # Import necessary modules
            import os
            import pandas as pd
            import numpy as np
            from pathlib import Path
            import joblib
            import sys
            import yaml
            
            # Check if we have saved models
            model_dir = Path("output/models")
            if not model_dir.exists():
                print("❌ No model directory found. Cannot recover predictions.")
                return False
                
            # Load feature info to get training data structure
            feature_info_path = model_dir / "exact_training_features.pkl"
            if not feature_info_path.exists():
                print("❌ No feature info found. Cannot recover predictions.")
                return False
                
            with open(feature_info_path, 'rb') as f:
                feature_info = joblib.load(f)
            
            # Load config
            with open("config.yaml", 'r') as f:
                config = yaml.safe_load(f)
            
            # Load the original data
            data_path = config.get("data", {}).get("path", "data/IC Weld.csv")
            if not os.path.exists(data_path):
                print(f"❌ Original data file not found at {data_path}. Cannot recover predictions.")
                return False
                
            print(f"📂 Loading data from: {data_path}")
            df = pd.read_csv(data_path)
            
            # Prepare data the same way as in the original pipeline
            target_column = config.get("data", {}).get("target_column", "GT_Label")
            exclude_columns = config.get("data", {}).get("exclude_columns", [])
            
            # Remove excluded columns
            if exclude_columns:
                df = df.drop(columns=[col for col in exclude_columns if col in df.columns])
                print(f"🗑️ Excluded columns: {exclude_columns}")
            
            # Separate features and target
            if target_column not in df.columns:
                print(f"❌ Target column '{target_column}' not found in data.")
                return False
                
            y = df[target_column]
            X = df.drop(columns=[target_column])
            
            # Convert target to binary if needed
            good_tag = config.get("data", {}).get("good_tag", "Good")
            bad_tag = config.get("data", {}).get("bad_tag", "Bad")
            
            if y.dtype == 'object':
                y = (y == good_tag).astype(int)
                print(f"🔄 Converted target to binary: {good_tag} -> 1, {bad_tag} -> 0")
            
            # Apply filtering if configured
            filter_data = config.get("features", {}).get("filter_data", False)
            if filter_data:
                # Apply variance filter
                variance_threshold = config.get("features", {}).get("variance_threshold", 0.0)
                if variance_threshold > 0:
                    from sklearn.feature_selection import VarianceThreshold
                    selector = VarianceThreshold(threshold=variance_threshold)
                    X_filtered = selector.fit_transform(X)
                    selected_features = X.columns[selector.get_support()]
                    X = pd.DataFrame(X_filtered, columns=selected_features, index=X.index)
                    print(f"🔍 Applied variance filter: {len(selected_features)} features selected")
                
                # Apply correlation filter
                correlation_threshold = config.get("features", {}).get("correlation_threshold", 1.0)
                if correlation_threshold < 1.0:
                    corr_matrix = X.corr().abs()
                    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
                    to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > correlation_threshold)]
                    X = X.drop(columns=to_drop)
                    print(f"🔍 Applied correlation filter: removed {len(to_drop)} highly correlated features")
            
            # Ensure we use the exact same features as during training
            expected_features = feature_info['feature_columns']
            
            # Check if we have all expected features
            missing_features = [f for f in expected_features if f not in X.columns]
            if missing_features:
                print(f"❌ Missing expected features: {missing_features}")
                return False
            
            # Reorder columns to match training exactly
            X = X[expected_features]
            print(f"✅ Using exact training features: {len(expected_features)} features")
            
            # Find available model files
            model_files = list(model_dir.glob("*_production.pkl"))
            
            if not model_files:
                print("❌ No production model files found. Cannot recover predictions.")
                return False
            
            print(f"📊 Found {len(model_files)} model files. Generating predictions...")
            
            # Initialize predictions data structure
            predictions_data = {
                'GT': y.tolist(),
                'index': df.index.tolist()
            }
            
            successful_models = 0
            
            # Load each model and generate predictions
            for model_file in model_files:
                try:
                    model_name = model_file.stem.replace('_production', '')
                    print(f"🔄 Processing {model_name}...")
                    
                    # Load the model
                    model = joblib.load(model_file)
                    
                    # Generate predictions
                    if hasattr(model, 'predict_proba'):
                        probas = model.predict_proba(X)[:, 1]  # Get probability of positive class
                    else:
                        # Fallback for models without predict_proba
                        probas = model.decision_function(X)
                        # Normalize to [0,1] range
                        probas = (probas - probas.min()) / (probas.max() - probas.min())
                    
                    predictions_data[model_name] = probas.tolist()
                    successful_models += 1
                    print(f"✅ Generated predictions for {model_name}: {len(probas)} samples")
                    
                except Exception as e:
                    print(f"⚠️ Failed to generate predictions for {model_file.name}: {e}")
                    continue
            
            if successful_models == 0:
                print("❌ Failed to generate predictions from any model.")
                return False
            
            # Store the recovered predictions data directly in memory
            self._data['predictions'] = predictions_data
            print(f"✅ Successfully recovered predictions data with {successful_models} models!")
            print(f"📊 Data summary: {len(predictions_data['GT'])} samples, {successful_models} models")
            return True
            
        except Exception as e:
            print(f"❌ Failed to recover predictions data: {e}")
            import traceback
            print(f"Full traceback: {traceback.format_exc()}")
            return False
    
    def get_predictions_data(self) -> Optional[Any]:
        """Get predictions data from memory, fallback to backup, or recover if missing."""
        # First check if we have data in memory
        if 'predictions' in self._data:
            return self._data['predictions']
        
        # Try to load from backup file if it exists
        backup_data = self._load_backup('predictions')
        if backup_data:
            self._data['predictions'] = backup_data
            print("📂 Loaded predictions data from backup file")
            return backup_data
        
        # If no backup exists and no memory data, try to recover from models
        print("🔄 No predictions data found. Attempting automatic recovery...")
        if self.recover_predictions_data():
            return self._data.get('predictions')
        
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