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

from backend import config_adapter as config
from backend.helpers import (
    prepare_data, ModelEvaluationResult, ModelEvaluationRun,
    plot_threshold_sweep, plot_runs_at_threshold, plot_class_balance,
    save_all_model_probabilities_from_structure
)

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
    combined_failure_model: str

class BaseModelDecisionResponse(BaseModel):
    config: BaseModelDecisionConfig
    available_columns: List[str]  # All columns in the dataset that could be decision columns

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
    """Get the current base model decision configuration and available columns."""
    try:
        # Get current configuration
        base_model_decisions = config.BASE_MODEL_DECISIONS
        
        # Get available columns from the dataset if it exists
        available_columns = []
        try:
            if os.path.exists(config.DATA_PATH):
                df = pd.read_csv(config.DATA_PATH)
                # Look for columns that might be decision columns (contain common decision keywords)
                decision_keywords = ['decision', 'class', 'label', 'prediction', 'result']
                available_columns = [col for col in df.columns 
                                   if any(keyword.lower() in col.lower() for keyword in decision_keywords)]
                # Sort for consistent ordering
                available_columns.sort()
        except Exception as e:
            api_logger.warning(f"Could not load dataset to get available columns: {e}")
        
        return BaseModelDecisionResponse(
            config=BaseModelDecisionConfig(**base_model_decisions),
            available_columns=available_columns
        )
    except Exception as e:
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
            'models.base_model_decisions.combined_failure_model': base_model_config.combined_failure_model
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 