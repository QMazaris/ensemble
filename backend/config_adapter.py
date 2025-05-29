"""
Configuration adapter for backward compatibility.

This module provides a bridge between the old config.py interface
and the new YAML-based configuration system.
"""

import os
import sys
from pathlib import Path
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier

# Add the root directory to the path
root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))

from shared.config_manager import get_config

class ConfigAdapter:
    """Adapter to provide backward compatibility with the old config.py interface."""
    
    def __init__(self):
        self.config = get_config()
        self._setup_derived_properties()
    
    def _setup_derived_properties(self):
        """Setup derived properties and paths."""
        # Create output directories if they don't exist
        for subdir in ['models', 'plots', 'predictions', 'logs', 'streamlit_data']:
            dir_path = os.path.join(self.OUTPUT_DIR, subdir)
            os.makedirs(dir_path, exist_ok=True)
    
    # ===== Data Configuration =====
    @property
    def DATA_PATH(self):
        return self.config.data_path
    
    @property
    def TARGET(self):
        return self.config.target_column
    
    @property
    def EXCLUDE_COLS(self):
        return self.config.get('data.exclude_columns', [])
    
    @property
    def TEST_SIZE(self):
        return self.config.get('data.test_size', 0.2)
    
    @property
    def RANDOM_STATE(self):
        return self.config.get('data.random_state', 42)
    
    # ===== Feature Engineering =====
    @property
    def FilterData(self):
        return self.config.get('features.filter_data', False)
    
    @property
    def VARIANCE_THRESH(self):
        return self.config.get('features.variance_threshold', 0.01)
    
    @property
    def CORRELATION_THRESH(self):
        return self.config.get('features.correlation_threshold', 0.95)
    
    # ===== Model Configuration =====
    @property
    def BASE_MODEL_OUTPUT_COLUMNS(self):
        return self.config.get('models.base_model_columns', {
            'AD': 'AnomalyScore',
            'CL': 'CL_ConfMax'
        })
    
    @property
    def BASE_MODEL_DECISIONS(self):
        """Get the base model decision configuration."""
        return self.config.get('models.base_model_decisions', {
            'enabled_columns': ['AD_Decision', 'CL_Decision'],
            'good_tag': 'Good',
            'bad_tag': 'Bad',
            'combined_failure_model': 'AD_or_CL_Fail'
        })
    
    @property
    def BASE_MODEL_DECISION_COLUMNS(self):
        """Get list of decision columns that should be processed as base models."""
        return self.BASE_MODEL_DECISIONS.get('enabled_columns', ['AD_Decision', 'CL_Decision'])
    
    @property
    def GOOD_TAG(self):
        """Get the tag used to represent 'good' classification in decision columns."""
        return self.BASE_MODEL_DECISIONS.get('good_tag', 'Good')
    
    @property
    def BAD_TAG(self):
        """Get the tag used to represent 'bad' classification in decision columns."""
        return self.BASE_MODEL_DECISIONS.get('bad_tag', 'Bad')
    
    @property
    def COMBINED_FAILURE_MODEL_NAME(self):
        """Get the name for the combined failure model (e.g., 'AD_or_CL_Fail')."""
        return self.BASE_MODEL_DECISIONS.get('combined_failure_model', 'AD_or_CL_Fail')
    
    @property
    def MODELS(self):
        """Get the models dictionary with instantiated sklearn/xgboost objects."""
        models = {}
        enabled_models = self.config.get('models.enabled', ['XGBoost', 'RandomForest'])
        
        if 'XGBoost' in enabled_models:
            xgb_params = self.config.get('model_params.XGBoost', {})
            models['XGBoost'] = xgb.XGBClassifier(**xgb_params)
        
        if 'RandomForest' in enabled_models:
            rf_params = self.config.get('model_params.RandomForest', {})
            models['RandomForest'] = RandomForestClassifier(**rf_params)
        
        return models
    
    # ===== Training Configuration =====
    @property
    def USE_KFOLD(self):
        return self.config.use_kfold
    
    @property
    def N_SPLITS(self):
        return self.config.n_splits
    
    @property
    def USE_SMOTE(self):
        return self.config.get('training.use_smote', True)
    
    @property
    def SMOTE_RATIO(self):
        return self.config.get('training.smote_ratio', 0.5)
    
    # ===== Cost Configuration =====
    @property
    def C_FP(self):
        return self.config.cost_fp
    
    @property
    def C_FN(self):
        return self.config.cost_fn
    
    # ===== Optimization Configuration =====
    @property
    def OPTIMIZE_HYPERPARAMS(self):
        return self.config.get('optimization.enabled', True)
    
    @property
    def HYPERPARAM_ITER(self):
        return self.config.get('optimization.iterations', 25)
    
    @property
    def OPTIMIZE_FINAL_MODEL(self):
        return self.config.get('optimization.optimize_final_model', True)
    
    @property
    def N_JOBS(self):
        return self.config.get('optimization.n_jobs', -1)
    
    @property
    def HYPERPARAM_SPACE(self):
        """Get hyperparameter search spaces."""
        spaces = self.config.get('hyperparam_spaces', {})
        # Add shared parameters
        shared_params = {'random_state': [42], 'n_jobs': [-1]}
        
        result = {'shared': shared_params}
        result.update(spaces)
        return result
    
    # ===== Export Configuration =====
    @property
    def SAVE_MODEL(self):
        return self.config.get('export.save_models', True)
    
    @property
    def SAVE_PLOTS(self):
        return self.config.get('export.save_plots', True)
    
    @property
    def SAVE_PREDICTIONS(self):
        return self.config.get('export.save_predictions', True)
    
    @property
    def EXPORT_ONNX(self):
        return self.config.get('export.export_onnx', True)
    
    @property
    def ONNX_OPSET_VERSION(self):
        return self.config.get('export.onnx_opset_version', 12)
    
    # ===== Output Configuration =====
    @property
    def OUTPUT_DIR(self):
        return self.config.output_dir
    
    @property
    def MODEL_DIR(self):
        return self.config.model_dir
    
    @property
    def PLOT_DIR(self):
        return self.config.plots_dir
    
    @property
    def PREDICTIONS_DIR(self):
        return self.config.predictions_dir
    
    @property
    def LOG_DIR(self):
        base_dir = self.config.output_dir
        logs_subdir = self.config.get('output.subdirs.logs', 'logs')
        return os.path.join(base_dir, logs_subdir)
    
    # ===== Logging Configuration =====
    @property
    def SUMMARY(self):
        return self.config.get('logging.summary', True)
    
    # ===== Backward Compatibility =====
    @property
    def dirs_to_create(self):
        """List of directories to create."""
        return [
            self.OUTPUT_DIR,
            self.MODEL_DIR,
            self.PLOT_DIR,
            self.PREDICTIONS_DIR,
            self.LOG_DIR
        ]

# Create a global instance that mimics the old config module
_adapter = ConfigAdapter()

# Export all properties as module-level attributes for backward compatibility
for attr_name in dir(_adapter):
    if not attr_name.startswith('_') and not callable(getattr(_adapter, attr_name)):
        globals()[attr_name] = getattr(_adapter, attr_name)

# Also make properties available as module attributes
DATA_PATH = _adapter.DATA_PATH
TARGET = _adapter.TARGET
EXCLUDE_COLS = _adapter.EXCLUDE_COLS
TEST_SIZE = _adapter.TEST_SIZE
RANDOM_STATE = _adapter.RANDOM_STATE
FilterData = _adapter.FilterData
VARIANCE_THRESH = _adapter.VARIANCE_THRESH
CORRELATION_THRESH = _adapter.CORRELATION_THRESH
BASE_MODEL_OUTPUT_COLUMNS = _adapter.BASE_MODEL_OUTPUT_COLUMNS
BASE_MODEL_DECISIONS = _adapter.BASE_MODEL_DECISIONS
BASE_MODEL_DECISION_COLUMNS = _adapter.BASE_MODEL_DECISION_COLUMNS
GOOD_TAG = _adapter.GOOD_TAG
BAD_TAG = _adapter.BAD_TAG
COMBINED_FAILURE_MODEL_NAME = _adapter.COMBINED_FAILURE_MODEL_NAME
MODELS = _adapter.MODELS
USE_KFOLD = _adapter.USE_KFOLD
N_SPLITS = _adapter.N_SPLITS
USE_SMOTE = _adapter.USE_SMOTE
SMOTE_RATIO = _adapter.SMOTE_RATIO
C_FP = _adapter.C_FP
C_FN = _adapter.C_FN
OPTIMIZE_HYPERPARAMS = _adapter.OPTIMIZE_HYPERPARAMS
HYPERPARAM_ITER = _adapter.HYPERPARAM_ITER
OPTIMIZE_FINAL_MODEL = _adapter.OPTIMIZE_FINAL_MODEL
N_JOBS = _adapter.N_JOBS
HYPERPARAM_SPACE = _adapter.HYPERPARAM_SPACE
SAVE_MODEL = _adapter.SAVE_MODEL
SAVE_PLOTS = _adapter.SAVE_PLOTS
SAVE_PREDICTIONS = _adapter.SAVE_PREDICTIONS
EXPORT_ONNX = _adapter.EXPORT_ONNX
ONNX_OPSET_VERSION = _adapter.ONNX_OPSET_VERSION
OUTPUT_DIR = _adapter.OUTPUT_DIR
MODEL_DIR = _adapter.MODEL_DIR
PLOT_DIR = _adapter.PLOT_DIR
PREDICTIONS_DIR = _adapter.PREDICTIONS_DIR
LOG_DIR = _adapter.LOG_DIR
SUMMARY = _adapter.SUMMARY
dirs_to_create = _adapter.dirs_to_create 