import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from app.utils import (
    load_metrics_data, plot_confusion_matrix, plot_roc_curve,
    plot_precision_recall_curve, plot_threshold_sweep, get_plot_groups,
    MODEL_DIR, PLOT_DIR, PREDICTIONS_DIR
)
from pathlib import Path
import inspect
import re
import ast
import importlib.util

def render_overview_tab():
    """Render the overview tab content."""
    st.write("### Latest Metrics")
    
    metrics_df, summary_df, cm_df, sweep_data = load_metrics_data()
    
    if summary_df is not None and not summary_df.empty:
        # Filter for 'Full' split and cost-optimal threshold for overview
        overview_data = summary_df[
            (summary_df['split'] == 'Full') &
            (summary_df['threshold_type'] == 'cost')
        ].copy()

        # Filter out decision-based models
        overview_data = overview_data[
            ~overview_data['model_name'].isin(['AD_Decision', 'CL_Decision', 'AD_or_CL_Fail'])
        ]
        
        if not overview_data.empty:
            # Model Performance Overview
            st.write("#### Model Performance Overview (Cost-Optimal Threshold on Full Data)")
            fig = px.bar(overview_data, 
                        x='model_name', 
                        y=['accuracy', 'precision', 'recall'],
                        title='Model Performance Metrics',
                        barmode='group',
                        labels={'value': 'Score (%)', 'model_name': 'Model'})
            fig.update_layout(yaxis_range=[0, 100])
            st.plotly_chart(fig, use_container_width=True)
            
            # Cost Analysis
            st.write("#### Cost Analysis (Cost-Optimal Threshold on Full Data)")
            fig = px.bar(overview_data,
                        x='model_name',
                        y='cost',
                        title='Model Costs',
                        labels={'cost': 'Cost', 'model_name': 'Model'})
            st.plotly_chart(fig, use_container_width=True)

        # Display detailed metrics table
        st.write("#### Detailed Metrics Table (All Models, Splits, Thresholds)")
        st.dataframe(summary_df.style.format({
            'accuracy': '{:.1f}%',
            'precision': '{:.1f}%',
            'recall': '{:.1f}%',
            'cost': '{:.1f}',
            'threshold': '{:.3f}'
        }))
    else:
        st.info("Run the pipeline to see metrics here.")

def render_model_analysis_tab():
    """Render the model analysis tab content."""
    st.write("### Model Analysis")
    
    metrics_df, summary_df, cm_df, sweep_data = load_metrics_data()
    
    if summary_df is not None and not summary_df.empty:
        # Model Selection
        model_options = summary_df['model_name'].unique()
        model = st.selectbox("Select Model", model_options)
        
        # Filter data for selected model
        model_data = summary_df[summary_df['model_name'] == model]
        model_cm = cm_df[cm_df['model_name'] == model]

        # Split Selection
        split_options = model_data['split'].unique()
        selected_split = st.selectbox("Select Split", split_options)

        # Filter data for selected split
        model_data_split = model_data[model_data['split'] == selected_split]
        model_cm_split = model_cm[model_cm['split'] == selected_split]
        
        if not model_data_split.empty:
            # Create two columns for metrics and confusion matrix
            col1, col2 = st.columns(2)
            
            with col1:
                st.write(f"#### Performance Metrics ({selected_split} Split)")
                metrics_display = model_data_split[['threshold_type', 'accuracy', 'precision', 'recall', 'cost', 'threshold']]
                st.dataframe(metrics_display.style.format({
                    'accuracy': '{:.1f}%',
                    'precision': '{:.1f}%',
                    'recall': '{:.1f}%',
                    'cost': '{:.1f}',
                    'threshold': '{:.3f}'
                }))
            
            with col2:
                st.write(f"#### Confusion Matrix ({selected_split} Split)")
                cm_threshold_options = model_cm_split['threshold_type'].unique()
                selected_cm_threshold = (st.selectbox("Select Threshold Type for CM", cm_threshold_options) 
                                       if len(cm_threshold_options) > 1 else cm_threshold_options[0])

                cm_data = model_cm_split[model_cm_split['threshold_type'] == selected_cm_threshold].iloc[0]
                st.plotly_chart(plot_confusion_matrix(cm_data), use_container_width=True)

            # Model Curves and Threshold Analysis
            if model in sweep_data:
                render_model_curves(model, sweep_data, model_data_split)
            elif model_data_split['threshold_type'].isin(['base']).any():
                st.info("Threshold sweep data is not available for decision-based models.")
    else:
        st.info("Run the pipeline to see analysis here.")

def render_model_curves(model, sweep_data, model_data_split):
    """Render model curves and threshold analysis."""
    st.write("#### Model Curves and Threshold Analysis")
    
    col3, col4 = st.columns(2)
    
    with col3:
        st.write("##### ROC Curve (Full Data)")
        try:
            pred_df = pd.read_csv(PREDICTIONS_DIR / 'all_model_predictions.csv')
            y_true = pred_df['GT'].values
            probs = np.array(sweep_data[model]['probabilities'])
            st.plotly_chart(plot_roc_curve(probs, y_true), use_container_width=True)
        except FileNotFoundError:
            st.warning("Predictions CSV not found for ROC/PR curves.")
    
    with col4:
        st.write("##### Precision-Recall Curve (Full Data)")
        try:
            pred_df = pd.read_csv(PREDICTIONS_DIR / 'all_model_predictions.csv')
            y_true = pred_df['GT'].values
            probs = np.array(sweep_data[model]['probabilities'])
            st.plotly_chart(plot_precision_recall_curve(probs, y_true), use_container_width=True)
        except FileNotFoundError:
            st.warning("Predictions CSV not found for ROC/PR curves.")

    # Threshold Analysis
    st.write("##### Probability Distribution and Threshold Sweep")
    
    probs = np.array(sweep_data[model]['probabilities'])
    fig_hist = px.histogram(
        x=probs,
        title=f'Probability Distribution - {model}',
        nbins=50,
        labels={'x': 'Probability', 'y': 'Count'}
    )
    st.plotly_chart(fig_hist, use_container_width=True)

    # Get optimal thresholds
    cost_optimal_thr = (model_data_split[model_data_split['threshold_type'] == 'cost']['threshold'].iloc[0] 
                       if not model_data_split[model_data_split['threshold_type'] == 'cost'].empty else None)
    acc_optimal_thr = (model_data_split[model_data_split['threshold_type'] == 'accuracy']['threshold'].iloc[0] 
                      if not model_data_split[model_data_split['threshold_type'] == 'accuracy'].empty else None)

    # Plot threshold sweep
    fig_sweep = plot_threshold_sweep(sweep_data[model], model, cost_optimal_thr, acc_optimal_thr)
    st.plotly_chart(fig_sweep, use_container_width=True)

def render_plots_gallery_tab():
    """Render the plots gallery tab content."""
    st.write("### Plots Gallery")
    
    imgs = sorted(PLOT_DIR.glob("*.png"))
    if not imgs:
        st.info("No static plots available. Run the pipeline to generate plots.")
    else:
        plot_groups = get_plot_groups(PLOT_DIR)
        ordered_groups = ['Comparison', 'Threshold Sweep'] + sorted([g for g in plot_groups.keys() 
                                                                    if g not in ['Comparison', 'Threshold Sweep', 'Other']]) + ['Other']
        
        for group in ordered_groups:
            if group in plot_groups:
                st.write(f"#### {group.replace('_', ' ').title()} Plots")
                group_imgs = plot_groups[group]
                cols_per_row = 3
                rows = (len(group_imgs) + cols_per_row - 1) // cols_per_row
                for r in range(rows):
                    cols = st.columns(cols_per_row)
                    for c in range(cols_per_row):
                        img_index = r * cols_per_row + c
                        if img_index < len(group_imgs):
                            img = group_imgs[img_index]
                            
                            # Add error handling for truncated images
                            try:
                                cols[c].image(str(img), caption=img.name, use_container_width=True)
                            except (OSError, Exception) as e:
                                st.warning(f"Could not load plot {img.name}: {e}")
                                # Optionally, display a placeholder or just skip

def render_downloads_tab():
    """Render the downloads tab content."""
    st.write("### Download Files")
    
    # Download predictions
    pred_file = MODEL_DIR / "all_model_predictions.csv"
    if pred_file.exists():
        with open(pred_file, 'rb') as f:
            st.download_button(
                "Download Predictions CSV",
                f.read(),
                file_name="all_model_predictions.csv",
                mime="text/csv"
            )
    
    # Download models
    st.write("#### Download Models")
    for model_file in MODEL_DIR.glob("*.pkl"):
        with open(model_file, 'rb') as f:
            st.download_button(
                f"Download {model_file.name}",
                f.read(),
                file_name=model_file.name,
                mime="application/octet-stream"
            )

def render_data_management_tab():
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

def update_config_file(updates):
    """Update specific key-value pairs in the config.py file."""
    try:
        config_path = Path("config.py")
        if not config_path.exists():
            st.error("config.py not found.")
            return
            
        lines = config_path.read_text().splitlines()
        new_lines = []
        updated_keys = set()

        for line in lines:
            added = False
            for key, value in updates.items():
                # Check if the line defines this config key
                if line.strip().startswith(f"{key} ="):
                    if isinstance(value, str):
                        new_lines.append(f"{key} = '{value}'")
                    elif isinstance(value, list):
                         # Format list nicely, handle potential strings inside list
                        formatted_list_items = [f"'{item}'" if isinstance(item, str) else str(item) for item in value]
                        new_lines.append(f"{key} = [{', '.join(formatted_list_items)}]")
                    else:
                        new_lines.append(f"{key} = {value}")
                    updated_keys.add(key)
                    added = True
                    break # Move to the next line in original file
            if not added:
                new_lines.append(line)
                
        config_path.write_text("\n".join(new_lines))
        
        # Store success message in session state
        st.session_state.config_update_success = True
        st.rerun()
        
    except Exception as e:
        st.error(f"Error updating config file: {e}")

def render_preprocessing_tab(config_settings):
    """Render the data preprocessing and configuration tab."""
    # Display success message if it exists in session state
    if st.session_state.get('config_update_success', False):
        st.success("Config updated successfully!")
        # Clear the success message after displaying it
        st.session_state.config_update_success = False
    
    st.write("### Data Preprocessing & Configuration")
    
    data_dir = Path("data")
    if not data_dir.exists():
        st.warning("Data directory not found. Please upload data in the Data Management tab.")
        return
    
    available_datasets = list(data_dir.glob("*.csv"))
    if not available_datasets:
        st.info("No CSV datasets found in the data directory. Please upload data first.")
        return

    st.write("#### Dataset Selection")
    # Use file names relative to the data directory for cleaner display and saving
    dataset_options = [f.name for f in available_datasets]
    
    # Attempt to pre-select the currently configured dataset
    current_data_path_name = Path(config_settings.get('DATA_PATH', '')).name
    try:
        default_index = dataset_options.index(current_data_path_name) if current_data_path_name in dataset_options else 0
    except ValueError:
        default_index = 0 # Fallback if config path is invalid or file not found

    selected_dataset_name = st.selectbox(
        "Choose a dataset", 
        dataset_options,
        index=default_index
    )
    
    selected_dataset_path = data_dir / selected_dataset_name
    
    df = None
    try:
        df = pd.read_csv(selected_dataset_path, low_memory=False)
        # Convert object columns to string for consistent handling
        for col in df.columns:
             if df[col].dtype == 'object':
                 df[col] = df[col].astype(str)
                 
        st.write("**Selected Dataset Preview:**")
        st.dataframe(df.head())
        
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        return # Stop if dataset can't be loaded
        
    st.write("#### Column Configuration")
    
    all_columns = df.columns.tolist()
    
    # Target Column Selection
    current_target = config_settings.get('TARGET', '')
    try:
        target_default_index = all_columns.index(current_target) if current_target and current_target in all_columns else 0
    except ValueError:
         target_default_index = 0
         
    selected_target_column = st.selectbox(
        "Select Target Column",
        all_columns,
        index=target_default_index
    )
    
    # Exclude Columns Selection
    current_exclude = config_settings.get('EXCLUDE_COLS', [])
    # Ensure current_exclude are strings for comparison
    current_exclude_str = [str(col) for col in current_exclude]

    selected_exclude_columns = st.multiselect(
        "Select Columns to Exclude",
        all_columns,
        default=[col for col in all_columns if col in current_exclude_str] # Pre-select based on config
    )
    
    # Ensure target column is not in excluded columns
    if selected_target_column in selected_exclude_columns:
        st.warning("Target column cannot be in the list of excluded columns. Removing target from excluded.")
        selected_exclude_columns.remove(selected_target_column)
        st.rerun()

    # Save Config Button
    st.write("#### Save Configuration")
    if st.button("💾 Save Data & Column Config"):
        config_updates = {
            'DATA_PATH': selected_dataset_path.as_posix(), # Use forward slashes
            'TARGET': selected_target_column,
            'EXCLUDE_COLS': selected_exclude_columns
        }
        update_config_file(config_updates)
        
    # --- Preprocessing Previews ---
    st.write("#### Preprocessing Previews")
    
    # Add filtering settings inputs here
    st.write("##### Filtering Settings for Preview")
    col_filter1, col_filter2 = st.columns(2)
    with col_filter1:
        # Use config settings as default values
        preview_variance_threshold = st.number_input(
            "Variance Threshold (>)",
            min_value=0.0,
            max_value=1.0,
            step=0.001,
            value=float(config_settings.get('VARIANCE_THRESH', 0.01)), # Ensure float for number_input
            format="%.3f"
        )
    with col_filter2:
        # Use config settings as default values
        preview_correlation_threshold = st.number_input(
            "Correlation Threshold (<)",
            min_value=0.0,
            max_value=1.0,
            step=0.001,
            value=float(config_settings.get('CORRELATION_THRESH', 0.95)), # Ensure float for number_input
            format="%.3f"
        )
        
    # Button to trigger preview update
    if st.button("🔄 Update Previews"):
        # Streamlit will automatically rerun the script from top,
        # updating the previews based on the current widget values.
        st.rerun()

    # Data for preprocessing (excluding target and explicitly excluded columns)
    cols_to_process = [col for col in all_columns if col != selected_target_column and col not in selected_exclude_columns]
    df_processed = df[cols_to_process].copy()

    # 1. One-Hot Encoding Preview
    st.write("##### After One-Hot Encoding")
    try:
        df_ohe = pd.get_dummies(df_processed, drop_first=True)
        st.dataframe(df_ohe.head())
        st.write(f"Shape after OHE: {df_ohe.shape[0]} rows × {df_ohe.shape[1]} columns")
    except Exception as e:
        st.error(f"Error during One-Hot Encoding preview: {e}")
        
    # 2. Feature Filtering Preview
    st.write("##### After Feature Filtering (Variance & Correlation)")
    try:
        # Feature Filtering logic (similar to your pipeline)
        # Requires config settings for thresholds
        # Use values from the input widgets, not config_settings directly
        variance_threshold = preview_variance_threshold
        correlation_threshold = preview_correlation_threshold
        
        df_filtered = df_ohe.copy() # Start with OHE data
        
        # Remove low variance features
        variances = df_filtered.var()
        cols_to_keep_variance = variances[variances > variance_threshold].index
        df_filtered = df_filtered[cols_to_keep_variance]
        
        # Remove highly correlated features (simple approach, can be improved)
        corr_matrix = df_filtered.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        cols_to_drop_corr = [column for column in upper.columns if any(upper[column] > correlation_threshold)]
        df_filtered = df_filtered.drop(cols_to_drop_corr, axis=1)
        
        st.dataframe(df_filtered.head())
        st.write(f"Shape after Filtering: {df_filtered.shape[0]} rows × {df_filtered.shape[1]} columns")
        st.write(f"Applied Variance Threshold: >{variance_threshold}")
        st.write(f"Applied Correlation Threshold: <{correlation_threshold} (absolute value)")
        
    except Exception as e:
        st.error(f"Error during Feature Filtering preview: {e}")

def render_model_zoo_tab():
    """Render the model zoo tab content."""
    st.write("### Model Zoo")

    # Import required model classes
    import xgboost as xgb
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC

    config_path = Path("config.py")
    if not config_path.exists():
        st.error("config.py not found.")
        return

    # Read and parse the config file
    try:
        config_content = config_path.read_text()
        # Load the actual config module to get current values
        spec = importlib.util.spec_from_file_location("config", config_path)
        config_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config_module)
        current_models = config_module.MODELS

        # --- Use AST to extract source code for MODELS and HYPERPARAM_SPACE ---
        tree = ast.parse(config_content)
        models_code = "# MODELS section not found"
        hyperparams_code = "# HYPERPARAM_SPACE section not found"

        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        if target.id == 'MODELS':
                            # Extract the source code for the value node
                            # Use ast.get_source_segment which handles indentation within the segment
                            models_code = ast.get_source_segment(config_content, node) or "# MODELS section not found"
                        elif target.id == 'HYPERPARAM_SPACE':
                            # Extract the source code for the value node
                            hyperparams_code = ast.get_source_segment(config_content, node) or "# HYPERPARAM_SPACE section not found"

        # The AST approach is more reliable, removing the regex fallback.

    except Exception as e:
        st.error(f"Error loading or parsing config.py: {e}")
        return

    # Define available models and their parameters
    available_models = {
        'XGBoost': {
            'class': xgb.XGBClassifier,
            'params': {
                'objective': {'type': 'select', 'options': ['binary:logistic'], 'default': 'binary:logistic'},
                'max_depth': {'type': 'number', 'min': 1.0, 'max': 20.0, 'step': 1.0, 'default': 4.0},
                'learning_rate': {'type': 'number', 'min': 0.001, 'max': 0.5, 'step': 0.001, 'default': 0.1},
                'n_estimators': {'type': 'number', 'min': 50.0, 'max': 1000.0, 'step': 50.0, 'default': 400.0},
                'subsample': {'type': 'number', 'min': 0.1, 'max': 1.0, 'step': 0.05, 'default': 0.8},
                'colsample_bytree': {'type': 'number', 'min': 0.1, 'max': 1.0, 'step': 0.05, 'default': 0.8},
                'min_child_weight': {'type': 'number', 'min': 1.0, 'max': 20.0, 'step': 1.0, 'default': 5.0},
                'gamma': {'type': 'number', 'min': 0.0, 'max': 1.0, 'step': 0.1, 'default': 0.1},
                'scale_pos_weight': {'type': 'number', 'min': 1.0, 'max': 100.0, 'step': 1.0, 'default': 62.5},
                'random_state': {'type': 'number', 'min': 0.0, 'max': 1000.0, 'step': 1.0, 'default': 42.0},
                'n_jobs': {'type': 'number', 'min': -1.0, 'max': 16.0, 'step': 1.0, 'default': -1.0},
                'eval_metric': {'type': 'select', 'options': ['logloss', 'auc', 'error'], 'default': 'logloss'}
            }
        },
        'RandomForest': {
            'class': RandomForestClassifier,
            'params': {
                'n_estimators': {'type': 'number', 'min': 50.0, 'max': 1000.0, 'step': 50.0, 'default': 300.0},
                'max_depth': {'type': 'number', 'min': 1.0, 'max': 20.0, 'step': 1.0, 'default': 6.0},
                'min_samples_split': {'type': 'number', 'min': 2.0, 'max': 50.0, 'step': 1.0, 'default': 10.0},
                'min_samples_leaf': {'type': 'number', 'min': 1.0, 'max': 50.0, 'step': 1.0, 'default': 5.0},
                'class_weight': {'type': 'select', 'options': ['balanced', 'balanced_subsample', None], 'default': 'balanced_subsample'},
                'max_features': {'type': 'select', 'options': ['sqrt', 'log2', None], 'default': 'sqrt'},
                'bootstrap': {'type': 'select', 'options': [True, False], 'default': True},
                'oob_score': {'type': 'select', 'options': [True, False], 'default': True},
                'n_jobs': {'type': 'number', 'min': -1.0, 'max': 16.0, 'step': 1.0, 'default': -1.0},
                'random_state': {'type': 'number', 'min': 0.0, 'max': 1000.0, 'step': 1.0, 'default': 42.0}
            }
        }
    }

    # Model Selection
    st.write("#### Select and Configure Model")
    selected_model = st.selectbox("Choose a model to configure", list(available_models.keys()))

    if selected_model:
        st.write(f"##### Configure {selected_model} Parameters")
        
        # Get current model configuration from loaded config module
        current_config = {}
        if selected_model in current_models:
            model_instance = current_models[selected_model]
            # Get parameters from the model instance
            current_config = model_instance.get_params()

        # Create parameter input widgets
        edited_params = {}
        for param_name, param_info in available_models[selected_model]['params'].items():
            current_value = current_config.get(param_name, param_info['default'])
            
            if param_info['type'] == 'select':
                edited_params[param_name] = st.selectbox(
                    param_name,
                    options=param_info['options'],
                    index=param_info['options'].index(current_value) if current_value in param_info['options'] else 0
                )
            elif param_info['type'] == 'number':
                # Convert all numeric values to float for consistency
                current_value = float(current_value)
                edited_params[param_name] = st.number_input(
                    param_name,
                    min_value=float(param_info['min']),
                    max_value=float(param_info['max']),
                    step=float(param_info.get('step', 1.0)),
                    value=current_value
                )

        # Display current configuration
        st.write("##### Current Configuration")
        st.json(edited_params)

        # Save button
        if st.button("💾 Save Model Configuration"):
            # Construct new model definition
            model_class = available_models[selected_model]['class']
            param_str = ', '.join(f"{k}={repr(v)}" for k, v in edited_params.items())
            new_model_def = f"'{selected_model}': {model_class.__name__}({param_str})"

            # Update config file
            try:
                # Find and replace the model definition
                model_pattern = rf"'{selected_model}':\s*{model_class.__name__}\(.*?\)"
                new_config_content = re.sub(model_pattern, new_model_def, config_content, flags=re.DOTALL)
                
                # Write updated config
                config_path.write_text(new_config_content)
                st.success(f"Configuration for {selected_model} updated successfully!")
                st.rerun()
            except Exception as e:
                st.error(f"Error updating configuration: {e}")

    # Advanced Configuration Editor
    st.write("#### Advanced Configuration Editor")
    st.info("""
    Edit the model configurations directly below. You can also modify these settings by editing `config.py` directly.
    The MODELS section defines the model instances and their parameters, while HYPERPARAM_SPACE defines the parameter ranges for optimization.
    """)
    
    # Display the cleaned-up sections
    st.write("##### Model Definitions")
    # Now using AST extracted code, no need for cleaning here
    st.code(models_code, language='python')
    edited_models_code = st.text_area("Edit MODELS", models_code, height=300)

    st.write("##### Hyperparameter Search Space")
    # Now using AST extracted code, no need for cleaning here
    st.code(hyperparams_code, language='python')
    edited_hyperparams_code = st.text_area("Edit HYPERPARAM_SPACE", hyperparams_code, height=200)

    if st.button("💾 Save Advanced Configuration"):
        try:
            # Replace the sections in the config file while preserving the rest
            new_config_content = config_content
            if models_code != "# MODELS section not found":
                new_models_section = f"MODELS = {{\n{edited_models_code.split('=', 1)[1].strip()}\n}}"
                new_config_content = re.sub(r"MODELS\s*=\s*\{[^}]*\}", new_models_section, new_config_content, flags=re.DOTALL)
            if hyperparams_code != "# HYPERPARAM_SPACE section not found":
                new_hyperparams_section = f"HYPERPARAM_SPACE = {{\n{edited_hyperparams_code.split('=', 1)[1].strip()}\n}}"
                new_config_content = re.sub(r"HYPERPARAM_SPACE\s*=\s*\{[^}]*\}", new_hyperparams_section, new_config_content, flags=re.DOTALL)
            
            config_path.write_text(new_config_content)
            st.success("Advanced configuration updated successfully! You may need to restart the app to see the changes take effect.")
            st.rerun()
        except Exception as e:
            st.error(f"Error updating advanced configuration: {e}")

# ... rest of the file ... 