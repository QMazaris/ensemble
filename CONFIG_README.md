# 🔧 Configuration System

This project now uses a modern YAML-based configuration system that is easy to use and maintain.

## 📁 Configuration Files

### `config.yaml` - Main Configuration File
The main configuration file in YAML format that contains all settings for the AI pipeline.

### `shared/config_manager.py` - Configuration Manager
A powerful configuration manager that provides:
- Easy loading and saving of YAML configurations
- Dot notation access (e.g., `config.get('data.path')`)
- Type validation and default values
- Clean API for both backend and frontend

### `backend/config_adapter.py` - Backward Compatibility
An adapter that provides backward compatibility with the old `config.py` interface while using the new YAML system.

## 🚀 How to Use

### Frontend (Streamlit)
```python
from shared.config_manager import get_config

# Get the configuration manager
config = get_config()

# Access configuration values
data_path = config.get('data.path')
cost_fp = config.cost_fp  # Property access
use_kfold = config.use_kfold

# Update configuration
config.set('costs.false_positive', 5)
config.save()
```

### Backend (Existing Code)
The backend continues to work as before thanks to the config adapter:
```python
from backend import config_adapter as config

# All existing code works unchanged
print(config.DATA_PATH)
print(config.C_FP)
print(config.USE_KFOLD)
```

## ⚙️ Configuration Sections

### Data Configuration
```yaml
data:
  path: 'data/training_data.csv'
  target_column: 'GT_Label'
  test_size: 0.2
  random_state: 42
```

### Model Configuration
```yaml
models:
  enabled:
    - 'XGBoost'
    - 'RandomForest'
```

### Training Configuration
```yaml
training:
  use_kfold: true
  n_splits: 5
  use_smote: true
```

### Cost Configuration
```yaml
costs:
  false_positive: 1
  false_negative: 30
```

### Export Configuration
```yaml
export:
  save_models: true
  save_plots: true
  export_onnx: true
```

## 🎯 Benefits

1. **Easy to Edit**: YAML is human-readable and easy to modify
2. **Frontend Friendly**: No more Python execution for config loading
3. **Type Safe**: Proper type handling and validation
4. **Backward Compatible**: Existing backend code works unchanged
5. **Version Control Friendly**: YAML diffs are clean and readable
6. **Real-time Updates**: Configuration can be updated through the web interface

## 🔄 Migration from config.py

The old `backend/config.py` file has been replaced with:
1. `config.yaml` - Contains all configuration values
2. `shared/config_manager.py` - Provides the configuration API
3. `backend/config_adapter.py` - Ensures backward compatibility

No changes are needed to existing backend code - it continues to work as before.

## 📝 Adding New Configuration Options

1. Add the new option to `config.yaml`:
```yaml
new_section:
  new_option: default_value
```

2. Access it in your code:
```python
config = get_config()
value = config.get('new_section.new_option', default_value)
```

3. If needed for backward compatibility, add a property to `ConfigAdapter` in `backend/config_adapter.py`.

## 🧪 Testing Configuration

You can test configuration loading:
```python
from shared.config_manager import load_config

# Load configuration
config = load_config('config.yaml')

# Validate
assert config.validate(), "Configuration is invalid"

# Print current settings
print(config.to_dict())
``` 