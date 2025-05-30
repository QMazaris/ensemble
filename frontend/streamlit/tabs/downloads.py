import streamlit as st
import pandas as pd
from pathlib import Path
import sys

# Add parent directory to path to access utils
parent_dir = str(Path(__file__).parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Add root directory to path to access shared modules
root_dir = str(Path(__file__).parent.parent.parent.parent)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

# Import utility functions
from utils import get_cached_data, MODEL_DIR

def render_downloads_tab():
    """Render the downloads tab content."""
    st.write("### Download Files")
    
    # Download predictions - now using cached data instead of fresh API call
    try:
        # Use cached predictions data
        predictions_data = get_cached_data(
            cache_key="predictions_data",
            api_endpoint="/results/predictions",
            default_value={"predictions": []},
            force_refresh=False
        )
        
        if predictions_data and predictions_data.get('predictions'):
            predictions_df = pd.DataFrame(predictions_data['predictions'])
            
            # Convert DataFrame to CSV for download
            csv_data = predictions_df.to_csv(index=False)
            st.download_button(
                "Download Predictions CSV",
                csv_data.encode('utf-8'),
                file_name="all_model_predictions.csv",
                mime="text/csv"
            )
        else:
            st.info("No predictions data available. Please run the pipeline first.")
    except Exception as e:
        st.error(f"Error loading predictions data: {str(e)}")
    
    # Download models
    st.write("#### Download Models")
    
    # Get all model files and sort them
    model_files = sorted(MODEL_DIR.glob("*.*"))
    
    # Group files by model name (without extension)
    model_groups = {}
    for file in model_files:
        if file.suffix in ['.pkl', '.onnx']:
            base_name = file.stem
            if base_name not in model_groups:
                model_groups[base_name] = []
            model_groups[base_name].append(file)
    
    # Display models in sorted order
    for model_name in sorted(model_groups.keys()):
        st.write(f"##### {model_name}")
        for model_file in sorted(model_groups[model_name]):
            with open(model_file, 'rb') as f:
                file_type = "ONNX" if model_file.suffix == '.onnx' else "Pickle"
                st.download_button(
                    f"Download {file_type} Model",
                    f.read(),
                    file_name=model_file.name,
                    mime="application/octet-stream"
                )