from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import os
import sys
from pathlib import Path
import json
import joblib
import pandas as pd
import numpy as np
import logging
import time
from datetime import datetime
import yaml            # PyYAML


#  Constants
CONFIG_PATH = Path("config.yaml")

# Set up logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('output/logs/api.log', encoding='utf-8')
    ]
)

# Create logger for API calls
api_logger = logging.getLogger('ensemble_api')

# Ensure logs directory exists
os.makedirs('output/logs', exist_ok=True)

# Add backend directory to path
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))
sys.path.insert(0, str(backend_dir.parent))  # Add project root too

from backend import config_adapter as config
from backend.helpers import (
    prepare_data, ModelEvaluationResult, ModelEvaluationRun,
    plot_threshold_sweep, plot_runs_at_threshold, plot_class_balance,
    save_all_model_probabilities_from_structure
)
from backend.helpers.stacked_logic import generate_combined_runs

# Import data service for in-memory data storage
from shared import data_service

app = FastAPI(title="Ensemble Pipeline API", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Middleware to log all API calls and responses
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    
    # Log the incoming request
    api_logger.info(f"📥 API CALL: {request.method} {request.url}")
    if request.query_params:
        api_logger.info(f"📥 Query Params: {dict(request.query_params)}")
    
    # Process the request
    response = await call_next(request)
    
    # Calculate processing time
    process_time = time.time() - start_time
    
    # Log the response
    api_logger.info(f"📤 API RESPONSE: {request.method} {request.url} - Status: {response.status_code} - Time: {process_time:.3f}s")
    
    return response

# Pydantic models for API
class PipelineStatus(BaseModel):
    status: str
    message: str
    progress: Optional[float] = None

class ModelPredictionRequest(BaseModel):
    features: List[float]
    model_name: str

class ModelPredictionResponse(BaseModel):
    prediction: float
    probability: float
    model_name: str

class ThresholdSweepResponse(BaseModel):
    thresholds: List[float]
    precision: List[float]
    recall: List[float]
    f1_score: List[float]
    accuracy: List[float]
    cost: List[float]
    cost_optimal_threshold: Optional[float]
    accuracy_optimal_threshold: Optional[float]

class ModelComparisonResponse(BaseModel):
    model_names: List[str]
    precision: List[float]
    recall: List[float]
    f1_score: List[float]
    accuracy: List[float]
    cost: List[float]
    thresholds: List[float]

class BaseModelDecisionConfig(BaseModel):
    enabled_columns: List[str]
    good_tag: str
    bad_tag: str

class BaseModelDecisionResponse(BaseModel):
    config: BaseModelDecisionConfig
    available_columns: List[str]  # All columns in the dataset that could be decision columns

class BaseModelColumnsConfig(BaseModel):
    columns: List[str]  # Simple list of column names (e.g., ["AnomalyScore", "CL_ConfMax"])

class BaseModelColumnsResponse(BaseModel):
    config: BaseModelColumnsConfig
    available_columns: List[str]  # All columns in the dataset that could be base model output columns

class BitwiseLogicRule(BaseModel):
    name: str
    columns: List[str]
    logic: str  # 'OR', 'AND', 'XOR', '|', '&', '^'

class BitwiseLogicConfig(BaseModel):
    rules: List[BitwiseLogicRule]
    enabled: bool = True

class BitwiseLogicConfigResponse(BaseModel):
    config: BitwiseLogicConfig
    available_models: List[str]
    available_logic_ops: List[str]

# Global variables for pipeline status
pipeline_status = {"status": "idle", "message": "Ready to run", "progress": 0.0}

# In-memory storage for training data info to avoid CSV dependency
training_data_info = None

def fix_numeric_types(data):
    """Fix numeric types in API responses to prevent Arrow serialization issues."""
    if isinstance(data, dict):
        if 'results' in data and isinstance(data['results'], dict):
            # Fix metrics data
            for section_name in ['model_metrics', 'model_summary', 'confusion_matrices']:
                if section_name in data['results'] and isinstance(data['results'][section_name], list):
                    for item in data['results'][section_name]:
                        if isinstance(item, dict):
                            # Convert numeric fields to proper types
                            numeric_fields = ['accuracy', 'precision', 'recall', 'f1_score', 'cost', 'threshold', 
                                            'tp', 'fp', 'tn', 'fn', 'total_samples']
                            for field in numeric_fields:
                                if field in item and item[field] is not None:
                                    try:
                                        if field in ['tp', 'fp', 'tn', 'fn', 'total_samples']:
                                            # These should be integers
                                            item[field] = int(float(str(item[field])))
                                        else:
                                            # These should be floats
                                            item[field] = float(str(item[field]))
                                    except (ValueError, TypeError):
                                        # Keep original value if conversion fails
                                        pass
        elif 'predictions' in data and isinstance(data['predictions'], list):
            # Fix predictions data
            for item in data['predictions']:
                if isinstance(item, dict):
                    for field, value in item.items():
                        if field not in ['GT', 'index'] and value is not None:
                            try:
                                item[field] = float(str(value))
                            except (ValueError, TypeError):
                                pass
        elif isinstance(data, list):
            # Handle case where data is directly a list (like predictions)
            for item in data:
                if isinstance(item, dict):
                    for field, value in item.items():
                        if field not in ['GT', 'index'] and value is not None:
                            try:
                                item[field] = float(str(value))
                            except (ValueError, TypeError):
                                pass
    return data

@app.get("/")
def read_root():
    return {"message": "Welcome to the Ensemble Pipeline API", "status": "running"}

@app.get("/health")
def health_check():
    return {"status": "healthy", "api_version": "1.0.0"}

def merge_dicts(base: dict, updates: dict) -> None:
    """
    Recursively merge `updates` into `base`, replacing only the keys present in `updates`.
    If a value is the sentinel {"__delete__": true}, remove that key from base.
    """
    for key, value in updates.items():
        # Check for deletion sentinel
        if isinstance(value, dict) and value == {"__delete__": True}:
            # Remove the key from base if it exists
            if key in base:
                del base[key]
        elif isinstance(value, dict) and key in base and isinstance(base[key], dict):
            # Recursively merge nested dictionaries
            merge_dicts(base[key], value)
        else:
            # Normal assignment
            base[key] = value

def save_config(config: dict) -> None:
    """
    Save the given config dictionary to config.yaml on disk.
    """
    with CONFIG_PATH.open("w") as f:
        yaml.safe_dump(config, f, sort_keys=False)

# New config API functions
@app.get("/config/load")
def load_config() -> dict:
    """
    Read config.yaml from disk and return its contents as a Python dict.
    """
    with CONFIG_PATH.open("r") as f:
        return yaml.safe_load(f)

@app.post("/config/update")
def update_config_partial(partial_conf: Dict[str, Any]):
    """
    Accept a nested dict of changes and merge them into config.yaml.
    Only the keys present in `partial_conf` will be replaced.
    
    To delete a key from the configuration, set its value to {"__delete__": true}.
    Example: {"data": {"target_column": {"__delete__": true}}} will remove the target_column key.
    """
    try:
        conf = load_config()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read config.yaml: {e}")

    try:
        merge_dicts(conf, partial_conf)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to merge config updates: {e}")

    try:
        save_config(conf)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to write config.yaml: {e}")

    return {"message": "Configuration partially updated successfully", "updated": partial_conf}


@app.get("/pipeline/status", response_model=PipelineStatus)
def get_pipeline_status():
    """Get the current status of the pipeline."""
    return PipelineStatus(**pipeline_status)

@app.post("/pipeline/run")
async def run_pipeline_endpoint(background_tasks: BackgroundTasks):
    """Run the ML pipeline in the background."""
    global pipeline_status
    
    if pipeline_status["status"] == "running":
        raise HTTPException(status_code=400, detail="Pipeline is already running")
    
    def run_pipeline_task():
        global pipeline_status
        try:
            pipeline_status = {"status": "running", "message": "Pipeline started", "progress": 0.0}
            
            # Clear any existing data
            data_service.clear_all_data()
            
            # Run the pipeline directly in this process
            pipeline_status["message"] = "Loading data and preparing features"
            pipeline_status["progress"] = 0.2
            
            # Import and run the pipeline function directly
            from backend.run import main as run_pipeline_main
            results = run_pipeline_main(config)
            
            pipeline_status["message"] = "Training models"
            pipeline_status["progress"] = 0.6
            
            pipeline_status["message"] = "Evaluating models and generating reports"
            pipeline_status["progress"] = 0.9
            
            # Data should now be available in data_service since we ran in the same process
            pipeline_status = {"status": "completed", "message": "Pipeline completed successfully", "progress": 1.0}
            
        except Exception as e:
            pipeline_status = {"status": "failed", "message": f"Pipeline failed: {str(e)}", "progress": 0.0}
            # Log the full traceback for debugging
            import traceback
            print(f"Pipeline failed with exception: {e}")
            print(f"Traceback: {traceback.format_exc()}")
    
    background_tasks.add_task(run_pipeline_task)
    return {"message": "Pipeline started in background", "status": "running"}

@app.get("/models/list")
def list_available_models():
    """List all available trained models."""
    model_dir = Path(config.MODEL_DIR)
    if not model_dir.exists():
        return {"models": []}
    
    models = []
    for model_file in model_dir.glob("*.pkl"):
        models.append({
            "name": model_file.stem,
            "path": str(model_file),
            "size": model_file.stat().st_size,
            "created": model_file.stat().st_mtime
        })
    
    return {"models": models}

@app.post("/models/predict", response_model=ModelPredictionResponse)
def predict_with_model(request: ModelPredictionRequest):
    """Make a prediction using a specific model."""
    model_path = Path(config.MODEL_DIR) / f"{request.model_name}.pkl"
    
    if not model_path.exists():
        raise HTTPException(status_code=404, detail=f"Model {request.model_name} not found")
    
    try:
        # Load the model
        model = joblib.load(model_path)
        
        # Make prediction
        features_array = np.array(request.features).reshape(1, -1)
        prediction = model.predict(features_array)[0]
        probability = model.predict_proba(features_array)[0][1]  # Probability of positive class
        
        return ModelPredictionResponse(
            prediction=float(prediction),
            probability=float(probability),
            model_name=request.model_name
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/data/info")
def get_data_info():
    """Get information about the training data."""
    global training_data_info
    
    # Check if data is already loaded in memory
    if training_data_info is not None:
        return training_data_info
    
    try:
        if not os.path.exists(config.DATA_PATH):
            raise HTTPException(status_code=404, detail="Training data not found")
        
        df = pd.read_csv(config.DATA_PATH)
        
        # Store in memory for future requests
        training_data_info = {
            "shape": df.shape,
            "columns": df.columns.tolist(),
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.to_dict().items()},
            "missing_values": df.isnull().sum().to_dict(),
            "target_distribution": df[config.TARGET].value_counts().to_dict() if config.TARGET in df.columns else None
        }
        
        return training_data_info
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load data info: {str(e)}")

@app.get("/results/metrics")
def get_model_metrics():
    """Get the latest model evaluation metrics."""
    try:
        api_logger.info("🔍 Fetching model metrics from data service...")
        metrics_data = data_service.get_metrics_data()
        
        if metrics_data is None:
            api_logger.warning("❌ No pipeline results available")
            raise HTTPException(status_code=404, detail="No pipeline results available. Run the pipeline first.")
        
        # Log details about the data being returned
        num_models = len(metrics_data.get('model_metrics', []))
        num_summaries = len(metrics_data.get('model_summary', []))
        num_cm = len(metrics_data.get('confusion_matrices', []))
        
        api_logger.info(f"✅ Returning metrics data: {num_models} model metrics, {num_summaries} summaries, {num_cm} confusion matrices")
        
        response = {"results": metrics_data}
        return fix_numeric_types(response)
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        error_details = f"Error in get_model_metrics: {str(e)}\nTraceback: {traceback.format_exc()}"
        api_logger.error(f"❌ {error_details}")
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")

@app.get("/results/threshold-sweep")
def get_threshold_sweep_data(model_name: str, data_type: str = "costs"):
    """Get threshold sweep data for a specific model.
    
    Args:
        model_name: Name of the model
        data_type: Type of data to return (costs, accuracies, thresholds, probabilities)
    """
    api_logger.info(f"🔍 Fetching threshold sweep data for model: {model_name}, data_type: {data_type}")
    
    sweep_data = data_service.get_sweep_data()
    
    if sweep_data is None:
        api_logger.warning("❌ No threshold sweep data available")
        raise HTTPException(status_code=404, detail="No threshold sweep data available. Run the pipeline first.")
    
    # Filter data for the requested model
    if model_name not in sweep_data:
        available_models = list(sweep_data.keys())
        api_logger.warning(f"❌ Model '{model_name}' not found. Available models: {available_models}")
        raise HTTPException(status_code=404, detail=f"No data available for model: {model_name}. Available models: {available_models}")
    
    model_data = sweep_data[model_name]
    if data_type not in model_data:
        available_data_types = list(model_data.keys())
        api_logger.warning(f"❌ Data type '{data_type}' not available for model '{model_name}'. Available: {available_data_types}")
        raise HTTPException(status_code=404, detail=f"Data type '{data_type}' not available. Available data types: {available_data_types}")
    
    data_points = len(model_data[data_type]) if isinstance(model_data[data_type], list) else "N/A"
    api_logger.info(f"✅ Returning {data_type} data for {model_name}: {data_points} data points")
    
    return {
        "model_name": model_name,
        "data_type": data_type,
        "data": model_data[data_type],
        "available_data_types": list(model_data.keys())
    }

@app.get("/results/model-comparison")
def get_model_comparison(threshold_type: str = "cost", split: str = "Full"):
    """Get model comparison data."""
    try:
        metrics_data = data_service.get_metrics_data()
        
        if metrics_data is None:
            raise HTTPException(status_code=404, detail="No metrics data available. Run the pipeline first.")
        
        # Extract model comparison data from metrics
        model_metrics = metrics_data.get('model_metrics', [])
        
        # Filter by split and threshold type
        filtered_metrics = [
            metric for metric in model_metrics 
            if metric.get('split') == split and metric.get('threshold_type') == threshold_type
        ]
        
        if not filtered_metrics:
            available_splits = list(set(m.get('split') for m in model_metrics))
            available_thresholds = list(set(m.get('threshold_type') for m in model_metrics))
            raise HTTPException(
                status_code=404, 
                detail=f"No comparison data available for {threshold_type}-optimal thresholds on {split} split. Available splits: {available_splits}, Available threshold types: {available_thresholds}"
            )
        
        # Format data for comparison
        model_names = [metric['model_name'] for metric in filtered_metrics]
        precision = [metric['precision'] for metric in filtered_metrics]
        recall = [metric['recall'] for metric in filtered_metrics]
        f1_score = [metric['f1_score'] for metric in filtered_metrics]
        accuracy = [metric['accuracy'] for metric in filtered_metrics]
        cost = [metric['cost'] for metric in filtered_metrics]
        thresholds = [metric['threshold'] for metric in filtered_metrics]
        
        return ModelComparisonResponse(
            model_names=model_names,
            precision=precision,
            recall=recall,
            f1_score=f1_score,
            accuracy=accuracy,
            cost=cost,
            thresholds=thresholds
        )
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        error_details = f"Error in get_model_comparison: {str(e)}\nTraceback: {traceback.format_exc()}"
        print(error_details)
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")

@app.get("/results/predictions")
def get_predictions_data():
    """Get model predictions data."""
    api_logger.info("🔍 Fetching predictions data from data service...")
    
    predictions_data = data_service.get_predictions_data()
    
    if predictions_data is None:
        api_logger.warning("❌ No predictions data available")
        raise HTTPException(status_code=404, detail="No predictions data available. Run the pipeline first.")
    
    num_predictions = len(predictions_data) if isinstance(predictions_data, list) else "Unknown"
    api_logger.info(f"✅ Returning predictions data: {num_predictions} predictions")
    
    response = {"predictions": predictions_data}
    return fix_numeric_types(response)

@app.post("/data/load")
def load_training_data():
    """Load training data into memory and return basic info."""
    global training_data_info
    
    try:
        if not os.path.exists(config.DATA_PATH):
            raise HTTPException(status_code=404, detail="Training data not found")
        
        df = pd.read_csv(config.DATA_PATH)
        
        # Store in memory for future requests
        training_data_info = {
            "shape": df.shape,
            "columns": df.columns.tolist(),
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.to_dict().items()},
            "missing_values": df.isnull().sum().to_dict(),
            "target_distribution": df[config.TARGET].value_counts().to_dict() if config.TARGET in df.columns else None,
            "sample_data": df.head().to_dict('records')  # Include sample data
        }
        
        return {"message": "Training data loaded successfully", "data_info": training_data_info}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load training data: {str(e)}")

@app.delete("/data/clear")
def clear_cached_data():
    """Clear all cached data from memory."""
    global training_data_info
    training_data_info = None
    data_service.clear_all_data()
    return {"message": "All cached data cleared successfully"}

@app.post("/data/save-csv-backup")
def save_csv_backup():
    """Force save CSV backup files for debugging purposes."""
    try:
        output_dir = Path(config.OUTPUT_DIR) / "streamlit_data"
        data_service.save_to_files(output_dir, save_csv_backup=True)
        return {"message": "CSV backup files saved successfully", "output_dir": str(output_dir)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save CSV backup: {str(e)}")

@app.get("/config/base-models", response_model=BaseModelDecisionResponse)
def get_base_model_config():
    """Get the current base model decision configuration and available decision columns."""
    try:
        # Get current configuration from config manager
        from shared.config_manager import config_manager
        
        # Get base model config (with defaults if not present)
        base_model_config = config_manager.get('models.base_model_decisions', {
            'enabled_columns': [],
            'good_tag': 'Good',
            'bad_tag': 'Bad'
        })
        
        # Dynamically detect available columns from the dataset
        available_columns = []
        try:
            # Try to load the current dataset to get available columns
            data_path = config_manager.get('data.path', 'data/training_data.csv')
            if os.path.exists(data_path):
                df = pd.read_csv(data_path, nrows=5)  # Just read a few rows to get column names
                
                # Exclude target and feature columns, focusing on potential decision columns
                target_column = config_manager.get('data.target_column', 'GT_Label')
                exclude_columns = config_manager.get('data.exclude_columns', [])
                
                # Look for columns that might be decision columns (typically containing Decision, decision, etc.)
                potential_decision_cols = [col for col in df.columns 
                                         if col != target_column 
                                         and col not in exclude_columns
                                         and ('decision' in col.lower() or 'Decision' in col)]
                
                # If no decision columns found, include all categorical columns as potential options
                if not potential_decision_cols:
                    for col in df.columns:
                        if col != target_column and col not in exclude_columns:
                            # Check if it's a categorical column or has few unique values
                            try:
                                unique_values = df[col].nunique()
                                if unique_values <= 10:  # Likely categorical
                                    potential_decision_cols.append(col)
                            except:
                                continue
                
                available_columns = potential_decision_cols
                
        except Exception as e:
            api_logger.warning(f"Could not detect available columns from dataset: {e}")
            # Fallback to any currently configured columns
            available_columns = base_model_config.get('enabled_columns', [])
        
        return BaseModelDecisionResponse(
            config=BaseModelDecisionConfig(**base_model_config),
            available_columns=available_columns
        )
        
    except Exception as e:
        api_logger.error(f"Failed to get base model config: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get base model config: {str(e)}")

@app.post("/config/base-models")
def update_base_model_config(base_model_config: BaseModelDecisionConfig):
    """Update base model decision configuration."""
    try:
        # Update the configuration using the config manager
        from shared.config_manager import config_manager
        
        config_updates = {
            'models.base_model_decisions.enabled_columns': base_model_config.enabled_columns,
            'models.base_model_decisions.good_tag': base_model_config.good_tag,
            'models.base_model_decisions.bad_tag': base_model_config.bad_tag,
        }
        
        config_manager.update(config_updates)
        config_manager.save()
        
        # Reload the config adapter to pick up the changes
        config._adapter.config = config_manager
        
        api_logger.info(f"Updated base model decision configuration: {config_updates}")
        
        return {
            "message": "Base model decision configuration updated successfully",
            "updated_config": base_model_config.dict()
        }
        
    except Exception as e:
        api_logger.error(f"Failed to update base model config: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to update base model config: {str(e)}")

@app.get("/config/base-model-columns", response_model=BaseModelColumnsResponse)
def get_base_model_columns_config():
    """Get the current base model columns configuration and available columns."""
    try:
        # Get current configuration from config manager
        from shared.config_manager import config_manager
        
        # Get base model columns config - now a simple list
        base_model_columns = config_manager.get('models.base_model_columns', [])
        
        # Ensure it's a list (handle backward compatibility)
        if isinstance(base_model_columns, dict):
            # Convert old dict format to list format
            base_model_columns = list(base_model_columns.values())
        elif not isinstance(base_model_columns, list):
            base_model_columns = []
        
        # Dynamically detect available columns from the dataset
        available_columns = []
        try:
            # Try to load the current dataset to get available columns
            data_path = config_manager.get('data.path', 'data/training_data.csv')
            if os.path.exists(data_path):
                df = pd.read_csv(data_path, nrows=5)  # Just read a few rows to get column names
                
                # Exclude target column and decision columns
                target_column = config_manager.get('data.target_column', 'GT_Label')
                exclude_columns = config_manager.get('data.exclude_columns', [])
                decision_columns = config_manager.get('models.base_model_decisions.enabled_columns', [])
                
                # Look for columns that might be base model output columns (typically numeric/continuous)
                potential_output_cols = []
                for col in df.columns:
                    if (col != target_column 
                        and col not in exclude_columns 
                        and col not in decision_columns):
                        try:
                            # Check if it's numeric or can be converted to numeric
                            pd.to_numeric(df[col], errors='raise')
                            potential_output_cols.append(col)
                        except (ValueError, TypeError):
                            # If not numeric, check if it might be a score column based on name
                            if any(keyword in col.lower() for keyword in ['score', 'prob', 'conf', 'output', 'pred']):
                                potential_output_cols.append(col)
                
                available_columns = potential_output_cols
                
        except Exception as e:
            api_logger.warning(f"Could not detect available columns from dataset: {e}")
            # Fallback to currently configured columns
            available_columns = base_model_columns
        
        return BaseModelColumnsResponse(
            config=BaseModelColumnsConfig(columns=base_model_columns),
            available_columns=available_columns
        )
        
    except Exception as e:
        api_logger.error(f"Failed to get base model columns config: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get base model columns config: {str(e)}")

@app.post("/config/base-model-columns")
def update_base_model_columns_config(base_model_columns_config: BaseModelColumnsConfig):
    """Update base model columns configuration."""
    try:
        # Update the configuration using the config manager
        from shared.config_manager import config_manager
        
        # Simple validation - check that columns is a list
        if not isinstance(base_model_columns_config.columns, list):
            raise HTTPException(status_code=400, detail="columns must be a list")
        
        # Filter out empty values
        filtered_columns = [col.strip() for col in base_model_columns_config.columns if col and col.strip()]
        
        # Save the filtered configuration
        config_updates = {
            'models.base_model_columns': filtered_columns
        }
        
        config_manager.update(config_updates)
        config_manager.save()
        
        # Reload the config adapter to pick up the changes
        config._adapter.config = config_manager
        
        api_logger.info(f"Updated base model columns configuration: {filtered_columns}")
        
        return {
            "message": "Base model columns configuration updated successfully",
            "updated_config": {"columns": filtered_columns}
        }
        
    except HTTPException:
        raise
    except Exception as e:
        api_logger.error(f"Failed to update base model columns config: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to update base model columns config: {str(e)}")

@app.get("/config")
def get_config():
    """Get the current configuration."""
    config_dict = {}
    for attr in dir(config):
        if not attr.startswith('_') and not attr.startswith('__'):
            value = getattr(config, attr)
            if not callable(value):
                try:
                    # Handle special cases
                    if attr == 'MODELS':
                        # For models, just return the model names and types
                        config_dict[attr] = {name: str(type(model).__name__) for name, model in value.items()}
                    elif hasattr(value, '__dict__'):
                        # For complex objects, convert to string
                        config_dict[attr] = str(value)
                    else:
                        # Try to serialize the value
                        json.dumps(value, default=str)
                        config_dict[attr] = value
                except:
                    config_dict[attr] = str(value)
    
    return {"config": config_dict}

@app.post("/config/update")
def update_config(updates: Dict[str, Any]):
    """Update configuration parameters."""
    try:
        for key, value in updates.items():
            if hasattr(config, key):
                setattr(config, key, value)
            else:
                raise HTTPException(status_code=400, detail=f"Unknown config parameter: {key}")
        
        return {"message": "Configuration updated successfully", "updated": updates}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update config: {str(e)}")

@app.get("/files/list")
def list_output_files():
    """List all output files generated by the pipeline."""
    output_dir = Path(config.OUTPUT_DIR)
    if not output_dir.exists():
        return {"files": []}
    
    files = []
    for file_path in output_dir.rglob("*"):
        if file_path.is_file():
            files.append({
                "name": file_path.name,
                "path": str(file_path.relative_to(output_dir)),
                "size": file_path.stat().st_size,
                "modified": file_path.stat().st_mtime,
                "type": file_path.suffix
            })
    
    return {"files": files}

@app.delete("/files/cleanup")
def cleanup_output_files():
    """Clean up all output files."""
    try:
        output_dir = Path(config.OUTPUT_DIR)
        if output_dir.exists():
            import shutil
            shutil.rmtree(output_dir)
            output_dir.mkdir(exist_ok=True)
        
        return {"message": "Output files cleaned up successfully"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cleanup failed: {str(e)}")

@app.get("/debug/data-service")
def debug_data_service():
    """Debug endpoint to see what data is stored in data_service."""
    return {
        "has_metrics": data_service.get_metrics_data() is not None,
        "has_predictions": data_service.get_predictions_data() is not None,
        "has_sweep": data_service.get_sweep_data() is not None,
        "metrics_keys": list(data_service.get_metrics_data().keys()) if data_service.get_metrics_data() else None,
        "metrics_sample": str(data_service.get_metrics_data())[:500] if data_service.get_metrics_data() else None,
        "predictions_sample": str(data_service.get_predictions_data())[:500] if data_service.get_predictions_data() else None,
        "sweep_sample": str(data_service.get_sweep_data())[:500] if data_service.get_sweep_data() else None
    }

@app.get("/config/bitwise-logic", response_model=BitwiseLogicConfigResponse)
def get_bitwise_logic_config():
    """Get the current bitwise logic configuration and available models."""
    try:
        # Get current configuration from config manager
        from shared.config_manager import config_manager
        
        # Get bitwise logic config (with defaults if not present)
        bitwise_config = config_manager.get('models.bitwise_logic', {
            'rules': [],
            'enabled': False
        })
        
        # Get available models from current metrics data
        available_models = []
        try:
            metrics_data = data_service.get_metrics_data()
            if metrics_data and 'model_summary' in metrics_data:
                available_models = list(set(
                    metric.get('model_name', '') 
                    for metric in metrics_data['model_summary']
                    if metric.get('model_name')
                ))
        except Exception as e:
            api_logger.warning(f"Could not get available models from metrics: {e}")
        
        # Add base models that should always be available
        base_decision_models = config.BASE_MODEL_DECISION_COLUMNS
        available_models.extend(base_decision_models)
        
        # Remove duplicates and sort
        available_models = sorted(list(set(available_models)))
        
        # Available logic operations
        available_logic_ops = ['OR', 'AND', 'XOR', '|', '&', '^']
        
        return BitwiseLogicConfigResponse(
            config=BitwiseLogicConfig(**bitwise_config),
            available_models=available_models,
            available_logic_ops=available_logic_ops
        )
        
    except Exception as e:
        api_logger.error(f"Failed to get bitwise logic config: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get bitwise logic config: {str(e)}")

@app.post("/config/bitwise-logic")
def update_bitwise_logic_config(bitwise_config: BitwiseLogicConfig):
    """Update bitwise logic configuration."""
    try:
        from shared.config_manager import config_manager
        
        # Validate the configuration
        for rule in bitwise_config.rules:
            if rule.logic not in ['OR', 'AND', 'XOR', '|', '&', '^']:
                raise HTTPException(
                    status_code=400, 
                    detail=f"Invalid logic operation '{rule.logic}'. Must be one of: OR, AND, XOR, |, &, ^"
                )
            
            if len(rule.columns) < 2:
                raise HTTPException(
                    status_code=400,
                    detail=f"Rule '{rule.name}' must have at least 2 columns"
                )
        
        # Update the configuration
        config_updates = {
            'models.bitwise_logic.rules': [rule.dict() for rule in bitwise_config.rules],
            'models.bitwise_logic.enabled': bitwise_config.enabled
        }
        
        config_manager.update(config_updates)
        config_manager.save()
        
        # Reload the config adapter to pick up the changes
        config._adapter.config = config_manager
        
        api_logger.info(f"Updated bitwise logic configuration: {len(bitwise_config.rules)} rules, enabled: {bitwise_config.enabled}")
        
        return {
            "message": "Bitwise logic configuration updated successfully",
            "updated_config": bitwise_config.dict()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        api_logger.error(f"Failed to update bitwise logic config: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to update bitwise logic config: {str(e)}")

@app.post("/config/bitwise-logic/apply")
def apply_bitwise_logic():
    """Apply bitwise logic rules to current results and update the data service."""
    try:
        from shared.config_manager import config_manager
        
        # Get current bitwise logic configuration
        bitwise_config = config_manager.get('models.bitwise_logic', {
            'rules': [],
            'enabled': False
        })
        
        if not bitwise_config.get('enabled', False):
            return {"message": "Bitwise logic is disabled", "combined_models_created": 0}
        
        rules = bitwise_config.get('rules', [])
        if not rules:
            return {"message": "No bitwise logic rules configured", "combined_models_created": 0}
        
        # Get current results from data service
        metrics_data = data_service.get_metrics_data()
        if not metrics_data:
            raise HTTPException(status_code=404, detail="No model results available. Run the pipeline first.")
        
        # Load current predictions to get y_true
        predictions_data = data_service.get_predictions_data()
        if not predictions_data:
            raise HTTPException(status_code=404, detail="No predictions data available. Run the pipeline first.")
        
        # Convert predictions data to DataFrame for easier handling
        predictions_df = pd.DataFrame(predictions_data)
        y_true = predictions_df['GT'].values
        
        # Create ModelEvaluationRun objects from current metrics
        runs = []
        
        # Group metrics by model name
        model_runs = {}
        for metric in metrics_data.get('model_summary', []):
            model_name = metric['model_name']
            if model_name not in model_runs:
                model_runs[model_name] = []
            
            # Create ModelEvaluationResult
            result = ModelEvaluationResult(
                model_name=model_name,
                split=metric['split'],
                threshold_type=metric['threshold_type'],
                threshold=metric.get('threshold'),
                precision=metric['precision'],
                recall=metric['recall'],
                f1_score=metric['f1_score'],
                accuracy=metric['accuracy'],
                cost=metric['cost'],
                tp=metric.get('tp', 0),
                fp=metric.get('fp', 0),
                tn=metric.get('tn', 0),
                fn=metric.get('fn', 0),
                is_base_model=metric.get('threshold_type') in ['base', 'combined']
            )
            model_runs[model_name].append(result)
        
        # Create runs with probabilities
        for model_name, results in model_runs.items():
            probabilities = {}
            if model_name in predictions_df.columns:
                probabilities['Full'] = predictions_df[model_name].values
            
            run = ModelEvaluationRun(
                model_name=model_name,
                results=results,
                probabilities=probabilities
            )
            runs.append(run)
        
        # Apply bitwise logic rules
        combined_logic = {}
        for rule in rules:
            combined_logic[rule['name']] = {
                'columns': rule['columns'],
                'logic': rule['logic']
            }
        
        # Generate combined runs
        new_runs = generate_combined_runs(
            runs=runs,
            combined_logic=combined_logic,
            y_true=y_true,
            C_FP=config.C_FP,
            C_FN=config.C_FN,
            N_SPLITS=config.N_SPLITS
        )
        
        # Add new combined runs to the existing metrics data
        for new_run in new_runs:
            # Add to model_metrics
            for result in new_run.results:
                metric_dict = {
                    'model_name': result.model_name,
                    'split': result.split,
                    'threshold_type': result.threshold_type,
                    'threshold': result.threshold,
                    'accuracy': result.accuracy,
                    'precision': result.precision,
                    'recall': result.recall,
                    'f1_score': result.f1_score,
                    'cost': result.cost,
                    'tp': result.tp,
                    'fp': result.fp,
                    'tn': result.tn,
                    'fn': result.fn
                }
                metrics_data['model_metrics'].append(metric_dict)
                metrics_data['model_summary'].append(metric_dict)
            
            # Add confusion matrix data
            for result in new_run.results:
                cm_dict = {
                    'model_name': result.model_name,
                    'split': result.split,
                    'threshold_type': result.threshold_type,
                    'tp': result.tp,
                    'fp': result.fp,
                    'tn': result.tn,
                    'fn': result.fn
                }
                metrics_data['confusion_matrices'].append(cm_dict)
            
            # Add to predictions
            if 'Full' in new_run.probabilities:
                predictions_df[new_run.model_name] = new_run.probabilities['Full']
        
        # Update data service with new combined data
        data_service.set_metrics_data(metrics_data)
        data_service.set_predictions_data(predictions_df.to_dict('records'))
        
        api_logger.info(f"Applied bitwise logic: created {len(new_runs)} combined models")
        
        return {
            "message": f"Bitwise logic applied successfully. Created {len(new_runs)} combined models.",
            "combined_models_created": len(new_runs),
            "combined_model_names": [run.model_name for run in new_runs]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        error_details = f"Failed to apply bitwise logic: {str(e)}\nTraceback: {traceback.format_exc()}"
        api_logger.error(error_details)
        raise HTTPException(status_code=500, detail=f"Failed to apply bitwise logic: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 