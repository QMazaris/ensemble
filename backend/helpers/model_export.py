"""
Helper functions for exporting models in different formats (pickle and ONNX).
"""

import os
import joblib
import numpy as np
from pathlib import Path

# Try to import ONNX dependencies, but make them optional
ONNX_AVAILABLE = False
XGBOOST_ONNX_AVAILABLE = False
try:
    import onnx
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType
    ONNX_AVAILABLE = True
    print("ONNX dependencies loaded successfully")
    
    # Try to import XGBoost ONNX conversion tools
    try:
        import onnxmltools
        from onnxmltools.convert import convert_xgboost
        from onnxmltools.convert.common.data_types import FloatTensorType as XGBFloatTensorType
        XGBOOST_ONNX_AVAILABLE = True
        print("XGBoost ONNX conversion tools loaded successfully")
    except ImportError:
        print("XGBoost ONNX conversion tools not available - XGBoost models will use pickle export")
        XGBOOST_ONNX_AVAILABLE = False
        
except ImportError as e:
    print(f"Warning: ONNX dependencies not available: {e}")
    print("ONNX export will not be available. Models will be exported as pickle files.")
    ONNX_AVAILABLE = False
    XGBOOST_ONNX_AVAILABLE = False



def export_model(model, model_name, model_dir, config=None, feature_names=None):
    """
    Export a model in either pickle or ONNX format based on configuration.
    
    Args:
        model: The trained model to export
        model_name (str): Name of the model
        model_dir (str): Directory to save the model
        config: Configuration dictionary containing export settings (optional)
        feature_names: List of feature names (columns) to use for ONNX export
    
    Returns:
        str: Path to the exported model file
    """
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        
    # Handle case where config is None
    if config is None:
        print(f"No config provided for {model_name}, using pickle export")
        return export_model_pickle(model, model_name, model_dir)
        
    export_onnx = config.get("export", {}).get("export_onnx", False)
    print(f"Export ONNX setting for {model_name}: {export_onnx}")
  
    
    if export_onnx and ONNX_AVAILABLE:
        print(f"Attempting ONNX export for {model_name}")
        return export_model_onnx(model, model_name, model_dir, config, feature_names)
    else:
        if export_onnx and not ONNX_AVAILABLE:
            print(f"Warning: ONNX export requested but ONNX not available. Falling back to pickle export for {model_name}")
        else:
            print(f"ONNX export not requested for {model_name}, using pickle export")
        return export_model_pickle(model, model_name, model_dir)

def export_model_pickle(model, model_name, model_dir):
    """
    Export a model in pickle format.
    
    Args:
        model: The trained model to export
        model_name (str): Name of the model
        model_dir (str): Directory to save the model
    
    Returns:
        str: Path to the exported model file
    """
    model_path = os.path.join(model_dir, f"{model_name}.pkl")
    joblib.dump(model, model_path)
    print(f"Model exported as pickle: {model_path}")
    return model_path

def export_model_onnx(model, model_name, model_dir, config, feature_names=None):
    """
    Export a model in ONNX format.
    
    Args:
        model: The trained model to export
        model_name (str): Name of the model
        model_dir (str): Directory to save the model
        config: Configuration dictionary containing ONNX settings
        feature_names: List of feature names (columns)
    
    Returns:
        str: Path to the exported ONNX model file
    """
    
    if not ONNX_AVAILABLE:
        print(f"❌ ONNX not available for {model_name}. Falling back to pickle export.")
        return export_model_pickle(model, model_name, model_dir)
    
    print(f"Starting ONNX export for {model_name}")
    print(f"Model type: {type(model)}")
    
    # Determine the number of features
    if hasattr(model, 'n_features_in_'):
        n_features = model.n_features_in_
        print(f"Features from model.n_features_in_: {n_features}")
    elif feature_names is not None:
        n_features = len(feature_names)
        print(f"Features from feature_names: {n_features}")
    else:
        raise ValueError("Cannot determine number of features. Please provide feature_names.")
    
    # Get ONNX opset version from config
    opset_version = config.get("export", {}).get("onnx_opset_version", 12)
    print(f"Using ONNX opset version: {opset_version}")
    
    try:
        # Check if this is an XGBoost model
        if 'xgboost' in str(type(model)).lower():
            return _export_xgboost_onnx(model, model_name, model_dir, n_features, opset_version)
        else:
            return _export_sklearn_onnx(model, model_name, model_dir, n_features, opset_version)
            
    except Exception as e:
        print(f"❌ Error exporting {model_name} to ONNX: {str(e)}")
        print(f"Exception type: {type(e)}")
        import traceback
        print(f"Full traceback: {traceback.format_exc()}")
        print(f"Falling back to pickle export for {model_name}")
        return export_model_pickle(model, model_name, model_dir)

def _export_sklearn_onnx(model, model_name, model_dir, n_features, opset_version):
    """Export sklearn-compatible models to ONNX"""
    # Define the input type for ONNX conversion
    initial_type = [('float_input', FloatTensorType([None, n_features]))]
    print(f"Initial type for ONNX: {initial_type}")
    
    # Convert the model to ONNX format
    print(f"Converting {model_name} to ONNX using sklearn-onnx...")
    onnx_model = convert_sklearn(
        model,
        initial_types=initial_type,
        target_opset=opset_version
    )
    
    # Save the ONNX model
    onnx_path = os.path.join(model_dir, f"{model_name}.onnx")
    print(f"Saving ONNX model to: {onnx_path}")
    with open(onnx_path, "wb") as f:
        f.write(onnx_model.SerializeToString())
    
    print(f"✅ Model '{model_name}' exported to ONNX format: {onnx_path}")
    return onnx_path

def _export_xgboost_onnx(model, model_name, model_dir, n_features, opset_version):
    """Export XGBoost models to ONNX using simplified approach with feature name conversion"""
    if not XGBOOST_ONNX_AVAILABLE:
        raise ValueError(f"XGBoost ONNX conversion not available. Install onnxmltools: pip install onnxmltools")
    
    print(f"Converting {model_name} to ONNX using onnxmltools...")
    
    # Define the input type for ONNX conversion
    initial_type = [('input', XGBFloatTensorType([None, n_features]))]
    print(f"Initial type for ONNX: {initial_type}")
    
    # Create ONNX-compatible feature names (f0, f1, f2, ...)
    onnx_feature_names = [f'f{i}' for i in range(n_features)]
    print(f"ONNX-compatible feature names: {onnx_feature_names}")
    
    try:
        # Create a new model with ONNX-compatible feature names
        import xgboost as xgb
        import numpy as np
        
        # Create dummy data with ONNX feature names to initialize the model
        X_dummy = np.random.random((10, n_features))
        y_dummy = np.random.randint(0, 2, 10)
        
        # Create new model with same parameters
        onnx_model_instance = xgb.XGBClassifier(**model.get_params())
        onnx_model_instance.fit(X_dummy, y_dummy)
        
        # Copy the trained booster from original model to new model
        # This preserves all learned parameters while using new feature names
        original_booster = model.get_booster()
        new_booster = onnx_model_instance.get_booster()
        
        # ---------------------------------------------
        # Ensure ONNX-compatible feature names (f0, f1, …)
        # Conversion fails if feature names are arbitrary strings
        # like "Porosity". We overwrite the booster feature names
        # *after* loading the real model to guarantee the JSON dump
        # used by onnxmltools only contains names following the
        # pattern `f%d`.
        # ---------------------------------------------
        # Save original model to temporary buffer and load into new model
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.json') as tmp_file:
            original_booster.save_model(tmp_file.name)
            new_booster.load_model(tmp_file.name)
            
            # After loading, overwrite feature names again to preserve fN mapping
            new_booster.feature_names = onnx_feature_names  # type: ignore[attr-defined]
            # If the booster already tracks feature types, reset them to 'q'
            if hasattr(new_booster, "feature_types"):
                # XGBoost expects 'i' (indicator) or 'q' (quantitative) strings per feature.
                # Default everything to quantitative.
                new_booster.feature_types = ['q'] * n_features  # type: ignore[attr-defined]
        
        print("✅ Successfully created XGBoost model with ONNX-compatible feature names")
        
        # Convert to ONNX using the model with proper feature names
        onnx_model = convert_xgboost(onnx_model_instance, initial_types=initial_type)
        print("✅ XGBoost ONNX conversion successful!")
        
    except Exception as e:
        print(f"XGBoost ONNX conversion failed: {str(e)}")
        print(f"Exception type: {type(e)}")
        import traceback
        print(f"Full traceback: {traceback.format_exc()}")
        raise RuntimeError(f"XGBoost ONNX conversion failed: {str(e)}")
    
    # Save the ONNX model
    onnx_path = os.path.join(model_dir, f"{model_name}.onnx")
    print(f"Saving ONNX model to: {onnx_path}")
    with open(onnx_path, "wb") as f:
        f.write(onnx_model.SerializeToString())
    
    # Also save feature name mapping for inference
    feature_mapping_path = os.path.join(model_dir, f"{model_name}_feature_mapping.json")
    import json
    
    # Get original feature names if available
    original_feature_names = []
    if hasattr(model, 'feature_names_in_') and model.feature_names_in_ is not None:
        original_feature_names = list(model.feature_names_in_)
    elif hasattr(model, 'get_booster'):
        # Try to get feature names from booster
        try:
            booster = model.get_booster()
            original_feature_names = booster.feature_names or []
        except:
            pass
    
    # Create mapping
    feature_mapping = {
        'onnx_features': onnx_feature_names,
        'original_features': original_feature_names,
        'mapping': {original: f'f{i}' for i, original in enumerate(original_feature_names)} if original_feature_names else {}
    }
    
    with open(feature_mapping_path, 'w') as f:
        json.dump(feature_mapping, f, indent=2)
    
    print(f"💾 Feature mapping saved to: {feature_mapping_path}")
    print(f"✅ Model '{model_name}' exported to ONNX format: {onnx_path}")
    return onnx_path

def get_model_file_extension(config=None):
    """
    Get the appropriate file extension based on export configuration.
    
    Args:
        config: Configuration dictionary (optional)
    
    Returns:
        str: File extension ('.onnx' or '.pkl')
    """
    if config is not None and config.get("export", {}).get("export_onnx", False) and ONNX_AVAILABLE:
        return '.onnx'
    else:
        return '.pkl' 