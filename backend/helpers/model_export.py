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
    """Export XGBoost models to ONNX"""
    if not XGBOOST_ONNX_AVAILABLE:
        raise ValueError(f"XGBoost ONNX conversion not available. Install onnxmltools: pip install onnxmltools")
    
    print(f"Converting {model_name} to ONNX using onnxmltools...")
    
    # Define the input type for ONNX conversion
    initial_type = [('float_input', FloatTensorType([None, n_features]))]
    print(f"Initial type for ONNX: {initial_type}")
    
    onnx_model = None  # Initialize the variable
    
    # Strategy 1: Try direct conversion
    try:
        print("Attempting direct XGBoost->ONNX conversion...")
        onnx_model = convert_xgboost(
            model,
            initial_types=initial_type,
            target_opset=opset_version
        )
        print("✅ Direct conversion successful")
    except Exception as e1:
        print(f"Direct conversion failed: {str(e1)}")
        
        # Strategy 2: Try with workaround for feature names
        if "feature names should follow pattern 'f%d'" in str(e1):
            try:
                print("Attempting conversion with XGBoost native ONNX export...")
                # Use XGBoost's native save_model with ONNX format
                import tempfile
                with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as tmp_file:
                    temp_json_path = tmp_file.name
                
                # Save model in JSON format first
                model.save_model(temp_json_path)
                
                # Load back and try conversion
                import xgboost as xgb
                temp_model = xgb.XGBClassifier()
                temp_model.load_model(temp_json_path)
                os.unlink(temp_json_path)
                
                # Try conversion again
                onnx_model = convert_xgboost(
                    temp_model,
                    initial_types=initial_type,
                    target_opset=opset_version
                )
                print("✅ Workaround conversion successful")
                
            except Exception as e2:
                print(f"Workaround conversion also failed: {str(e2)}")
                
                # Strategy 3: Final fallback - create dummy model with f%d names
                try:
                    print("Attempting final fallback with feature renaming...")
                    # Create a dummy dataset with proper feature names
                    import pandas as pd
                    dummy_feature_names = [f'f{i}' for i in range(n_features)]
                    X_dummy = pd.DataFrame(np.random.random((10, n_features)), columns=dummy_feature_names)
                    y_dummy = np.random.randint(0, 2, 10)
                    
                    # Create and train a new model with proper feature names
                    import xgboost as xgb
                    fallback_model = xgb.XGBClassifier(**model.get_params())
                    fallback_model.fit(X_dummy, y_dummy)
                    
                    # Copy the booster from the original model
                    fallback_model._Booster = model._Booster.copy()
                    
                    onnx_model = convert_xgboost(
                        fallback_model,
                        initial_types=initial_type,
                        target_opset=opset_version
                    )
                    print("✅ Fallback conversion successful")
                    
                except Exception as e3:
                    print(f"All conversion strategies failed. Final error: {str(e3)}")
                    raise RuntimeError(f"XGBoost ONNX conversion failed with all strategies. Last error: {str(e3)}")
        else:
            # Re-raise the original exception if it's not about feature names
            raise e1
    
    # Check if conversion was successful
    if onnx_model is None:
        raise RuntimeError("ONNX model conversion failed - no valid ONNX model was created")
    
    # Save the ONNX model
    onnx_path = os.path.join(model_dir, f"{model_name}.onnx")
    print(f"Saving ONNX model to: {onnx_path}")
    with open(onnx_path, "wb") as f:
        f.write(onnx_model.SerializeToString())
    
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