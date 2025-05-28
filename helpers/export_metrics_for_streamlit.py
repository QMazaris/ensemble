import pandas as pd
import numpy as np
import json
from pathlib import Path
from helpers.metrics import ModelEvaluationResult, ModelEvaluationRun
from helpers.reporting import print_performance_summary

def export_metrics_for_streamlit(runs, output_dir, meta_model_names=None):
    """Export evaluation results in a format suitable for Streamlit visualization.
    
    Args:
        runs: List of ModelEvaluationRun objects from the pipeline
        output_dir: Directory to save the exported data
        meta_model_names: Set of model names that are meta-models (used for performance summary)
    """
    # Temporary: Print the structure of the received runs object for debugging
    print("\n--- Structure of runs object received by export_metrics_for_streamlit ---")
    for i, run in enumerate(runs):
        print(f"Run {i+1}: Model Name = {run.model_name}")
        print(f"  Results type: {type(run.results)}")
        if isinstance(run.results, list):
            print(f"  Results length: {len(run.results)}")
            if len(run.results) > 0:
                print(f"  First result type: {type(run.results[0])}")
                if hasattr(run.results[0], '__dict__'):
                    print(f"  First result attributes: {dir(run.results[0])}")
        elif isinstance(run.results, dict):
            print(f"  Results Keys: {list(run.results.keys())}")
        if hasattr(run, 'threshold_sweep') and run.threshold_sweep is not None:
             print(f"  Threshold Sweep available: True")
        else:
             print(f"  Threshold Sweep available: False")
    print("--------------------------------------------------------------------")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # 1. Export detailed metrics for each model
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
                    'cost': result.cost,
                    'threshold': result.threshold,
                    'tp': result.tp,
                    'fp': result.fp,
                    'tn': result.tn,
                    'fn': result.fn
                })
    
    metrics_df = pd.DataFrame(metrics_data)
    metrics_df.to_csv(output_dir / 'model_metrics.csv', index=False)
    
    # 2. Export threshold sweep data
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
                'accuracies': accuracies
            }
    
    with open(output_dir / 'threshold_sweep_data.json', 'w') as f:
        json.dump(sweep_data, f)
    
    # 3. Export confusion matrices
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
    
    cm_df = pd.DataFrame(cm_data)
    cm_df.to_csv(output_dir / 'confusion_matrices.csv', index=False)
    
    # 4. Export summary statistics
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
                    'cost': result.cost,
                    'threshold': result.threshold,
                    'total_samples': (result.tp + result.fp + result.tn + result.fn)
                })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(output_dir / 'model_summary.csv', index=False)
    
    print(f"Metrics data exported to {output_dir}")
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
    
    # For standalone testing, you would need to somehow load or generate a sample
    # 'runs' object here.
    
    print("This script is intended to be called from run.py. No action taken when run directly.")
    # Example of how you *could* load if needed for testing, but not the primary flow:
    # import pickle
    # results_file = Path("output") / "latest_results.pkl"
    # if results_file.exists():
    #     try:
    #         with open(results_file, 'rb') as f:
    #             results_total = pickle.load(f)
    #         export_metrics_for_streamlit(results_total, 'output/streamlit_data')
    #     except Exception as e:
    #         print(f"Error loading results for standalone test: {str(e)}")
    # else:
    #     print("No saved results found for standalone test. Run the pipeline first.") 