import os
import sys
import joblib
from pathlib import Path

# Optional ONNX dependencies
ONNX_AVAILABLE = False
try:
    import onnx
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType
    ONNX_AVAILABLE = True
    print("ONNX dependencies loaded successfully")
except ImportError as e:
    print(f"Warning: ONNX dependencies not available: {e}")
    print("ONNX export will not be available. Models will remain in pickle format.")
    ONNX_AVAILABLE = False


def export_model(model, model_name, model_dir, config=None, feature_names=None):
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    export_onnx = config.get("export", {}).get("export_onnx", False) if config else False

    if export_onnx and ONNX_AVAILABLE:
        return export_model_onnx(model, model_name, model_dir, config, feature_names)
    else:
        return export_model_pickle(model, model_name, model_dir)


def export_model_pickle(model, model_name, model_dir):
    model_path = os.path.join(model_dir, f"{model_name}.pkl")
    joblib.dump(model, model_path)
    print(f"✅ Model exported as pickle: {model_path}")
    return model_path


def export_model_onnx(model, model_name, model_dir, config, feature_names=None):
    if not ONNX_AVAILABLE:
        print("❌ ONNX export requested but not available.")
        return export_model_pickle(model, model_name, model_dir)

    if hasattr(model, 'n_features_in_'):
        n_features = model.n_features_in_
    elif feature_names:
        n_features = len(feature_names)
    else:
        raise ValueError("Feature count unknown. Provide feature_names or a model with n_features_in_.")

    initial_type = [('float_input', FloatTensorType([None, n_features]))]
    opset_version = config.get("export", {}).get("onnx_opset_version", 12)

    try:
        onnx_model = convert_sklearn(model, initial_types=initial_type, target_opset=opset_version)
        onnx_path = os.path.join(model_dir, f"{model_name}.onnx")
        with open(onnx_path, "wb") as f:
            f.write(onnx_model.SerializeToString())
        print(f"✅ Model exported to ONNX format: {onnx_path}")
        return onnx_path
    except Exception as e:
        import traceback
        print(f"❌ Error converting to ONNX: {e}")
        print(traceback.format_exc())
        return export_model_pickle(model, model_name, model_dir)


# =============================================================================
# CLI Entry Point
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Convert a .pkl model to ONNX or re-export as pickle.")
    parser.add_argument("pkl_path", type=str, help="Path to the input .pkl model file")
    parser.add_argument("--output_dir", type=str, default="models", help="Directory to save the exported model")
    parser.add_argument("--export_onnx", action="store_true", help="Enable ONNX export")
    parser.add_argument("--opset", type=int, default=12, help="ONNX opset version")
    parser.add_argument("--features", type=int, default=None, help="Number of input features")

    args = parser.parse_args()

    # Load the model
    model_path = Path(args.pkl_path)
    if not model_path.exists():
        print(f"❌ File not found: {model_path}")
        sys.exit(1)

    print(f"📦 Loading model from: {model_path}")
    model = joblib.load(model_path)

    # Construct config
    config = {
        "export": {
            "export_onnx": args.export_onnx,
            "onnx_opset_version": args.opset,
        }
    }

    # Use filename stem as model name
    model_name = model_path.stem

    # Dummy feature names if user supplies feature count
    feature_names = [f"f{i}" for i in range(args.features)] if args.features else None

    # Export the model
    export_model(model, model_name, args.output_dir, config=config, feature_names=feature_names)
