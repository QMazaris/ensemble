"""
Helper functions for exporting models in different formats (pickle and ONNX).
"""

import os
import joblib
import numpy as np
from pathlib import Path

# Try to import ONNX dependencies, but make them optional
try:
    import onnx
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    print("Warning: ONNX dependencies not available. ONNX export will be disabled.")

def export_model(model, model_name, model_dir, config=None, feature_names=None):
    """
    Export a model in either pickle or ONNX format based on configuration.
    
    Args:
        model: The trained model to export
        model_name (str): Name of the model
        model_dir (str): Directory to save the model
        config: Configuration object containing export settings (optional)
        feature_names: List of feature names (columns) to use for ONNX export
    
    Returns:
        str: Path to the exported model file
    """
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        
    # Handle case where config is None
    if config is None:
        return export_model_pickle(model, model_name, model_dir)
        
    if getattr(config, 'EXPORT_ONNX', False) and ONNX_AVAILABLE:
        return export_model_onnx(model, model_name, model_dir, config, feature_names)
    else:
        if getattr(config, 'EXPORT_ONNX', False) and not ONNX_AVAILABLE:
            print(f"Warning: ONNX export requested but ONNX not available. Falling back to pickle export for {model_name}")
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

def export_model_onnx(model, model_name, model_dir, config, feature_names):
    """
    Export a model in ONNX format.
    
    Args:
        model: The trained model to export
        model_name (str): Name of the model
        model_dir (str): Directory to save the model
        config: Configuration object containing ONNX settings
        feature_names: List of feature names (columns) to use for ONNX export
    
    Returns:
        str: Path to the exported model file
    """
    if not ONNX_AVAILABLE:
        raise ImportError("ONNX dependencies are not available. Please install onnx and skl2onnx.")
    
    try:
        if feature_names is None:
            raise ValueError("feature_names must be provided for ONNX export to ensure correct input shape.")
        n_features = len(feature_names)
        initial_type = [('float_input', FloatTensorType([None, n_features]))]
        opset_version = getattr(config, 'ONNX_OPSET_VERSION', 12)
        onnx_model = convert_sklearn(
            model, 
            initial_types=initial_type,
            target_opset=opset_version
        )
        model_path = os.path.join(model_dir, f"{model_name}.onnx")
        onnx.save_model(onnx_model, model_path)
        print(f"Model exported as ONNX: {model_path}")
        return model_path
        
    except Exception as e:
        print(f"Error exporting {model_name} to ONNX: {str(e)}")
        print(f"Falling back to pickle export for {model_name}")
        return export_model_pickle(model, model_name, model_dir)

def get_model_file_extension(config=None):
    """
    Get the appropriate file extension based on export configuration.
    
    Args:
        config: Configuration object (optional)
    
    Returns:
        str: File extension ('.onnx' or '.pkl')
    """
    if config is not None and getattr(config, 'EXPORT_ONNX', False) and ONNX_AVAILABLE:
        return '.onnx'
    else:
        return '.pkl' 