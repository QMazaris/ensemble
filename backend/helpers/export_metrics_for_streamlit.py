import pandas as pd
import numpy as np
import json
from pathlib import Path
from .metrics import ModelEvaluationResult, ModelEvaluationRun
from .reporting import print_performance_summary

def export_metrics_for_streamlit(runs, output_dir, meta_model_names=None):
    """Export evaluation results using in-memory data service instead of CSV files.
    
    Args:
        runs: List of ModelEvaluationRun objects from the pipeline
        output_dir: Directory to save backup files (optional)
        meta_model_names: Set of model names that are meta-models (used for performance summary)
    """
    # Import data service
    try:
        # Use consistent import path
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent.parent.parent))
        from shared import data_service
    except ImportError as e:
        print(f"Warning: Could not import data service: {e}")
        print("This may cause data export issues")
        return
    
    # 1. Prepare detailed metrics data
    metrics_data = []
    for run in runs:
        # Handle both list and dict cases for results
        if isinstance(run.results, list):
            for result in run.results:
                metrics_data.append({
                    'model_name': run.model_name,
                    'split': result.split,  # Using direct split attribute
                    'threshold_type': result.threshold_type,
                    'accuracy': result.accuracy,
                    'precision': result.precision,
                    'recall': result.recall,
                    'f1_score': result.f1_score,
                    'cost': result.cost,
                    'threshold': result.threshold,
                    'tp': result.tp,
                    'fp': result.fp,
                    'tn': result.tn,
                    'fn': result.fn
                })
        else:  # dict case
            for split_name, result in run.results.items():
                metrics_data.append({
                    'model_name': run.model_name,
                    'split': split_name,
                    'threshold_type': result.threshold_type,
                    'accuracy': result.accuracy,
                    'precision': result.precision,
                    'recall': result.recall,
                    'f1_score': result.f1_score,
                    'cost': result.cost,
                    'threshold': result.threshold,
                    'tp': result.tp,
                    'fp': result.fp,
                    'tn': result.tn,
                    'fn': result.fn
                })
    
    # 2. Prepare threshold sweep data
    sweep_data = {}
    for run in runs:
        # Only export sweep data for models that have probabilities and sweep data
        if (hasattr(run, 'probabilities') and run.probabilities and 
            hasattr(run, 'sweep_data') and run.sweep_data):
            
            # Use the 'Full' split sweep data if available, otherwise use the first available split
            if 'Full' in run.sweep_data:
                sweep = run.sweep_data['Full']
            elif run.sweep_data:
                # Use the first available split
                first_split = list(run.sweep_data.keys())[0]
                sweep = run.sweep_data[first_split]
            else:
                continue
                
            # Convert sweep data to lists for JSON serialization
            thresholds = [float(t) for t in sweep.keys()]  # Convert to float
            costs = [float(sweep[t]['cost']) for t in sweep.keys()]  # Convert to float
            accuracies = [float(sweep[t]['accuracy']) for t in sweep.keys()]  # Convert to float
            f1_scores = [float(sweep[t]['f1_score']) for t in sweep.keys()]  # Convert to float
            
            # Get probabilities for the same split
            if 'oof' in run.probabilities:
                probs = run.probabilities['oof']
            elif 'Full' in run.probabilities:
                probs = run.probabilities['Full']
            else:
                first_split = list(run.probabilities.keys())[0]
                probs = run.probabilities[first_split]
                
            # Convert to list and ensure all values are Python native types
            if hasattr(probs, 'tolist'):
                probs = [float(p) for p in probs.tolist()]  # Convert to float
            elif not isinstance(probs, list):
                probs = [float(p) for p in list(probs)]  # Convert to float
            else:
                probs = [float(p) for p in probs]  # Convert to float
                
            sweep_data[run.model_name] = {
                'probabilities': probs,
                'thresholds': thresholds,
                'costs': costs,
                'accuracies': accuracies,
                'f1_scores': f1_scores
            }
    
    # 3. Prepare confusion matrices data
    cm_data = []
    for run in runs:
        if isinstance(run.results, list):
            for result in run.results:
                cm_data.append({
                    'model_name': run.model_name,
                    'split': result.split,
                    'threshold_type': result.threshold_type,
                    'tp': result.tp,
                    'fp': result.fp,
                    'tn': result.tn,
                    'fn': result.fn
                })
        else:  # dict case
            for split_name, result in run.results.items():
                cm_data.append({
                    'model_name': run.model_name,
                    'split': split_name,
                    'threshold_type': result.threshold_type,
                    'tp': result.tp,
                    'fp': result.fp,
                    'tn': result.tn,
                    'fn': result.fn
                })
    
    # 4. Prepare summary statistics data
    summary_data = []
    for run in runs:
        if isinstance(run.results, list):
            for result in run.results:
                summary_data.append({
                    'model_name': run.model_name,
                    'split': result.split,
                    'threshold_type': result.threshold_type,
                    'accuracy': result.accuracy,
                    'precision': result.precision,
                    'recall': result.recall,
                    'f1_score': result.f1_score,
                    'cost': result.cost,
                    'threshold': result.threshold,
                    'total_samples': (result.tp + result.fp + result.tn + result.fn)
                })
        else:  # dict case
            for split_name, result in run.results.items():
                summary_data.append({
                    'model_name': run.model_name,
                    'split': split_name,
                    'threshold_type': result.threshold_type,
                    'accuracy': result.accuracy,
                    'precision': result.precision,
                    'recall': result.recall,
                    'f1_score': result.f1_score,
                    'cost': result.cost,
                    'threshold': result.threshold,
                    'total_samples': (result.tp + result.fp + result.tn + result.fn)
                })
    
    # Store all data in the data service
    metrics_package = {
        'model_metrics': metrics_data,
        'confusion_matrices': cm_data,
        'model_summary': summary_data
    }
    
    data_service.set_metrics_data(metrics_package)
    data_service.set_sweep_data(sweep_data)
    
    # Save to files as backup - this is important for API access
    if output_dir:
        output_dir = Path(output_dir)
        # Save JSON files for API compatibility
        data_service.save_to_files(output_dir, save_csv_backup=False)
    
    print(f"Metrics data stored in memory and saved to {output_dir}")
    print("\nPerformance Summary:")
    if meta_model_names is not None:
        print_performance_summary(runs, meta_model_names)
    else:
        print("Skipping performance summary (meta_model_names not provided)")

if __name__ == "__main__":
    # Import the runs directly from the pipeline
    # This script is now intended to be called from run.py after pipeline completion
    # The code below is primarily for standalone testing if needed, but the primary
    # use case is being called by run.py
    pass # Placeholder for potential standalone testing 