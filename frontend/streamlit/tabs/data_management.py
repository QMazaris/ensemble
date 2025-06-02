import streamlit as st
import pandas as pd
from pathlib import Path
import requests
from datetime import datetime

# Import from parent utils module
import sys
from pathlib import Path
root_dir = str(Path(__file__).parent.parent.parent.parent)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

def render_data_management_tab(config_settings):
    """Render the data management tab content."""
    st.write("### Data Management")
    
    # File Upload Section
    st.write("#### Upload New Training Data")
    uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])
    
    if uploaded_file is not None:
        try:
            # Read the data with explicit type checking
            df = pd.read_csv(uploaded_file, low_memory=False)
            
            # Data validation section
            st.write("#### Data Validation")
            
            problematic_cols = []
            categorical_cols = {}
            for col in df.columns:
                # Check for object dtype columns
                if df[col].dtype == 'object':
                    # Try to convert to numeric if possible
                    try:
                        pd.to_numeric(df[col], errors='raise')
                        df[col] = pd.to_numeric(df[col])
                    except:
                        # If conversion fails, check if it's a date
                        try:
                            # Use a more robust date parsing approach if needed
                            df[col] = pd.to_datetime(df[col], errors='raise')
                        except:
                            # If still object, check if it's likely categorical
                            unique_vals = df[col].dropna().unique()
                            if len(unique_vals) > 0 and len(unique_vals) <= 50: # Heuristic for categorical
                                categorical_cols[col] = unique_vals.tolist()
                            else:
                                problematic_cols.append(col)
            
            if problematic_cols or categorical_cols:
                st.warning("⚠️ Potential data issues detected. Please review the details below.")
                st.info("Please review the 'Data Preview' and 'Column Information' sections below carefully to ensure structure is acceptable before saving.")
                
                if problematic_cols:
                    st.write("**Columns with potentially unhandled data types:**")
                    st.write("These columns could contain mixed types or formats that couldn't be automatically identified as numeric, datetime, or simple categorical. They might require cleaning.")
                    for col in problematic_cols:
                        st.write(f"- **{col}**: `{df[col].dtype}`")
                        # Optionally display sample values for problematic columns as well
                        sample_vals = df[col].dropna().unique()
                        if len(sample_vals) <= 10: # Show a few samples
                             st.write(f"  Sample values: {', '.join(str(x) for x in sample_vals[:10])}")
                        elif len(sample_vals) > 10:
                             st.write(f"  Contains {len(sample_vals)} unique values (showing first 10 sample values: {', '.join(str(x) for x in sample_vals[:10])})")
                    
                if categorical_cols:
                    st.write("**Columns identified as potentially categorical:**")
                    st.write("These columns have a limited number of unique string values and have been identified as potentially categorical.")
                    for col, unique_vals in categorical_cols.items():
                         st.write(f"- **{col}**: `{df[col].dtype}`")
                         st.write(f"  Unique values ({len(unique_vals)}): {', '.join(str(x) for x in unique_vals)}")
            
            # Display data preview with type information
            st.write("#### Data Preview")
            
            st.dataframe(df.head())
            
            st.write("##### Column Information")
            for col in df.columns:
                st.write(f"**{col}**")
                st.write(f"- Type: {df[col].dtype}")
                st.write(f"- Non-null values: {df[col].count()}")
                if df[col].dtype in ['int64', 'float64']:
                    st.write(f"- Range: {df[col].min():.2f} to {df[col].max():.2f}")
            
            # Basic data validation
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("##### Data Shape")
                st.write(f"Rows: {df.shape[0]}")
                st.write(f"Columns: {df.shape[1]}")
                
                # Check for missing values
                missing_values = df.isnull().sum()
                if missing_values.any():
                    st.warning("Missing values detected:")
                    st.write(missing_values[missing_values > 0])
                else:
                    st.success("No missing values detected")
            
            with col2:
                st.write("##### Data Quality Checks")
                # Check for duplicate rows
                duplicates = df.duplicated().sum()
                if duplicates > 0:
                    st.warning(f"Found {duplicates} duplicate rows")
                else:
                    st.success("No duplicate rows found")
                
                # Check for constant columns
                constant_cols = [col for col in df.columns if df[col].nunique() <= 1]
                if constant_cols:
                    st.warning(f"Found {len(constant_cols)} constant columns: {', '.join(constant_cols)}")
            
            # Save options
            st.write("#### Save Options")
            if problematic_cols:
                st.warning("⚠️ Some columns have non-standard data types. Consider cleaning the data before saving.")
            
            save_name = st.text_input("Save as (without .csv extension)", "training_data")
            if st.button("Save Dataset"):
                if not save_name:
                    st.error("Please provide a name for the dataset")
                    return
                    
                # Create data directory if it doesn't exist
                data_dir = Path("data")
                data_dir.mkdir(exist_ok=True)
                
                # Save the file
                file_path = data_dir / f"{save_name}.csv"
                try:
                    df.to_csv(file_path, index=False)
                    st.success(f"Dataset saved successfully to {file_path}")
                    
                    # Update config to use new dataset
                    st.info("Please update the config file to use the new dataset path if needed")
                except Exception as e:
                    st.error(f"Error saving file: {str(e)}")
        
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
            st.write("Please check that your CSV file is properly formatted and try again.")
    
    # Current Dataset Information
    st.write("#### Current Dataset Information")
    data_dir = Path("data")
    if data_dir.exists():
        training_files = list(data_dir.glob("*.csv"))
        if training_files:
            st.write("Available datasets:")
            for file in training_files:
                with st.expander(f"📊 {file.name}"):
                    try:
                        df = pd.read_csv(file, low_memory=False)
                        st.write(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
                        st.write(f"Last modified: {file.stat().st_mtime}")
                        
                        # Show column types
                        st.write("Column types:")
                        for col, dtype in df.dtypes.items():
                            st.write(f"- {col}: {dtype}")
                            
                        # Quick preview
                        if st.button(f"Preview {file.name}", key=f"preview_{file.name}"):
                            st.dataframe(df.head())
                            
                        # Delete option
                        if st.button(f"Delete {file.name}", key=f"delete_{file.name}"):
                            try:
                                file.unlink()
                                st.success(f"Deleted {file.name}")
                                st.rerun()
                            except Exception as e:
                                st.error(f"Error deleting file: {str(e)}")
                    except Exception as e:
                        st.error(f"Error reading {file.name}: {str(e)}")
        else:
            st.info("No datasets found in the data directory")
    else:
        st.info("Data directory not found")
