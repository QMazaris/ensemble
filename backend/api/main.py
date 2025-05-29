from fastapi import FastAPI, HTTPException, BackgroundTasks
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

# Add backend directory to path
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

from backend import config_adapter as config
from backend.helpers import (
    prepare_data, ModelEvaluationResult, ModelEvaluationRun,
    plot_threshold_sweep, plot_runs_at_threshold, plot_class_balance,
    save_all_model_probabilities_from_structure
)
from backend.run import main as run_pipeline

app = FastAPI(title="Ensemble Pipeline API", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
    accuracy: List[float]
    cost: List[float]
    cost_optimal_threshold: Optional[float]
    accuracy_optimal_threshold: Optional[float]

class ModelComparisonResponse(BaseModel):
    model_names: List[str]
    precision: List[float]
    recall: List[float]
    accuracy: List[float]
    cost: List[float]
    thresholds: List[float]

# Global variables for pipeline status
pipeline_status = {"status": "idle", "message": "Ready to run", "progress": 0.0}
pipeline_results = None

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
        global pipeline_status, pipeline_results
        try:
            pipeline_status = {"status": "running", "message": "Pipeline started", "progress": 0.0}
            
            # Run the pipeline
            pipeline_status["message"] = "Loading data and preparing features"
            pipeline_status["progress"] = 0.2
            
            results = run_pipeline(config)
            
            pipeline_status["message"] = "Training models"
            pipeline_status["progress"] = 0.6
            
            pipeline_status["message"] = "Evaluating models and generating reports"
            pipeline_status["progress"] = 0.9
            
            pipeline_results = results
            pipeline_status = {"status": "completed", "message": "Pipeline completed successfully", "progress": 1.0}
            
        except Exception as e:
            pipeline_status = {"status": "failed", "message": f"Pipeline failed: {str(e)}", "progress": 0.0}
    
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
    try:
        if not os.path.exists(config.DATA_PATH):
            raise HTTPException(status_code=404, detail="Training data not found")
        
        df = pd.read_csv(config.DATA_PATH)
        
        return {
            "shape": df.shape,
            "columns": df.columns.tolist(),
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.to_dict().items()},
            "missing_values": df.isnull().sum().to_dict(),
            "target_distribution": df[config.TARGET].value_counts().to_dict() if config.TARGET in df.columns else None
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load data info: {str(e)}")

@app.get("/results/metrics")
def get_model_metrics():
    """Get the latest model evaluation metrics."""
    global pipeline_results
    
    if pipeline_results is None:
        # Try to load from saved results
        streamlit_data_dir = Path("output/streamlit_data")
        if streamlit_data_dir.exists():
            # Load saved metrics if available
            return {"message": "Results available in output/streamlit_data"}
        else:
            raise HTTPException(status_code=404, detail="No pipeline results available. Run the pipeline first.")
    
    return {"results": pipeline_results}

@app.get("/results/threshold-sweep")
def get_threshold_sweep_data(model_name: str, split: str = "Test"):
    """Get threshold sweep data for a specific model."""
    # This would typically load from saved results
    # For now, return a placeholder response
    return {
        "message": f"Threshold sweep data for {model_name} on {split} split",
        "data": "Available after pipeline completion"
    }

@app.get("/results/model-comparison")
def get_model_comparison(threshold_type: str = "cost", split: str = "Test"):
    """Get model comparison data."""
    # This would typically load from saved results
    return {
        "message": f"Model comparison for {threshold_type}-optimal thresholds on {split} split",
        "data": "Available after pipeline completion"
    }

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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 