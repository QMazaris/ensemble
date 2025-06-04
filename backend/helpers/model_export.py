"""
Helper functions for exporting models in different formats (pickle and ONNX).
"""

import os
import joblib
import numpy as np
from pathlib import Path

# Try to import ONNX dependencies, but make them optional
ONNX_AVAILABLE = False
try:
    import onnx
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType
    ONNX_AVAILABLE = True
    print("ONNX dependencies loaded successfully")
except ImportError as e:
    print(f"Warning: ONNX dependencies not available: {e}")
    print("ONNX export will not be available. Models will be exported as pickle files.")
    ONNX_AVAILABLE = False



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
    
    # Define the input type for ONNX conversion
    initial_type = [('float_input', FloatTensorType([None, n_features]))]
    print(f"Initial type for ONNX: {initial_type}")
    
    # Get ONNX opset version from config
    opset_version = config.get("export", {}).get("onnx_opset_version", 12)
    print(f"Using ONNX opset version: {opset_version}")
    
    try:
        # Convert the model to ONNX format
        print(f"Converting {model_name} to ONNX...")
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
        
    except Exception as e:
        print(f"❌ Error exporting {model_name} to ONNX: {str(e)}")
        print(f"Exception type: {type(e)}")
        import traceback
        print(f"Full traceback: {traceback.format_exc()}")
        print(f"Falling back to pickle export for {model_name}")
        return export_model_pickle(model, model_name, model_dir)

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