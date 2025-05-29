import os
import pandas as pd
import numpy as np

def print_performance_summary(runs, meta_model_names, splits=('Train', 'Test', 'Full'), thr_types=('cost', 'accuracy')):
    print("\n" + "="*80)
    print("SUMMARY OF MODEL PERFORMANCE")
    print("="*80)
    def _find(model, split, thr):
        for run in runs:
            if run.model_name != model:
                continue
            for r in run.results:
                if r.split == split and r.threshold_type == thr:
                    return r
        return None
    for model in meta_model_names:
        found_any = False
        for split in splits:
            for thr in thr_types:
                r = _find(model, split, thr)
                if r:
                    found_any = True
                    break
            if found_any:
                break
        if not found_any:
            continue
        print(f"\n{model}:")
        for split in splits:
            for thr in thr_types:
                r = _find(model, split, thr)
                if not r:
                    continue
                print(f"  {split} Split ({thr}-optimal):")
                print(f"    Accuracy:  {r.accuracy:.1f}%  Precision: {r.precision:.1f}%  Recall: {r.recall:.1f}%")
                print(f"    Threshold: {r.threshold:.3f}")
                print(f"    Cost:      {r.cost}")
                print(f"    Confusion Matrix: [[TN={r.tn} FP={r.fp}], [FN={r.fn} TP={r.tp}]]")
    all_models = {run.model_name for run in runs}
    base_models = all_models - set(meta_model_names)
    print("\nBASE MODELS PERFORMANCE (preset confidence threshold):")
    for model in base_models:
        found_any = False
        for split in splits:
            r = _find(model, split, 'base')
            if r:
                found_any = True
                break
        if not found_any:
            continue
        print(f"\n{model}:")
        for split in splits:
            r = _find(model, split, 'base')
            if not r:
                continue
            print(f"  {split} Split:")
            print(f"    Accuracy:  {r.accuracy:.1f}%  Precision: {r.precision:.1f}%  Recall: {r.recall:.1f}%")
            print(f"    Cost:      {r.cost}")
            print(f"    Confusion Matrix: [[TN={r.tn} FP={r.fp}], [FN={r.fn} TP={r.tp}]]")
    # Only print K-FOLD AVERAGED MODEL PERFORMANCE if any kfold_avg_ models exist
    if any(run.model_name.startswith("kfold_avg_") for run in runs):
        print("\nK-FOLD AVERAGED MODEL PERFORMANCE:")
        for run in runs:
            if run.model_name.startswith("kfold_avg_"):
                if not run.results:
                    continue
                print(f"\n{run.model_name}:")
                for r in run.results:
                    print(f"  {r.split} Split ({r.threshold_type}-optimal):")
                    print(f"    Accuracy:  {r.accuracy:.1f}%  Precision: {r.precision:.1f}%  Recall: {r.recall:.1f}%")
                    print(f"    Threshold: {r.threshold:.3f}")
                    print(f"    Cost:      {r.cost}")
                    print(f"    Confusion Matrix: [[TN={r.tn} FP={r.fp}], [FN={r.fn} TP={r.tp}]]")

def save_all_model_probabilities_from_structure(results_total, predictions_dir, index, y_true, SUMMARY=None, save_csv_backup=False):
    """Store all model probabilities in data service instead of saving to CSV."""
    # Import data service
    try:
        from ...shared import data_service
    except ImportError:
        # Fallback for direct execution
        import sys
        from pathlib import Path
        sys.path.append(str(Path(__file__).parent.parent.parent))
        from shared import data_service
    
    if not results_total:
        raise ValueError("results_total is empty")
    
    def pick_oof_or_full_or_longest(probas):
        if not probas:
            return None
        if 'oof' in probas:
            return probas['oof']
        if 'Full' in probas:
            return probas['Full']
        non_empty = [arr for arr in probas.values() if arr is not None and hasattr(arr, 'shape')]
        if not non_empty:
            return None
        return max(non_empty, key=lambda arr: arr.shape[0])
    
    # Build structured data instead of CSV
    predictions_data = {
        'GT': y_true.tolist() if hasattr(y_true, 'tolist') else y_true,
        'index': index.tolist() if hasattr(index, 'tolist') else index
    }
    
    for run in results_total:
        model_col = run.model_name
        probas = pick_oof_or_full_or_longest(run.probabilities)
        if probas is None:
            print(f"[WARN] No probabilities found for model {model_col}, skipping.")
            continue
        
        # Convert to list for JSON serialization
        if hasattr(probas, 'tolist'):
            probas_list = probas.tolist()
        else:
            probas_list = list(probas)
            
        if len(probas_list) == len(index):
            predictions_data[model_col] = probas_list
        else:
            # Pad with None values if shorter
            padded_probas = probas_list + [None] * (len(index) - len(probas_list))
            predictions_data[model_col] = padded_probas
    
    # Store in data service
    data_service.set_predictions_data(predictions_data)
    
    # Optionally save to file as backup only if requested
    if save_csv_backup and predictions_dir:
        import pandas as pd
        from pathlib import Path
        predictions_dir = Path(predictions_dir)
        predictions_dir.mkdir(exist_ok=True)
        pd.DataFrame(predictions_data).to_csv(
            predictions_dir / 'all_model_predictions.csv', index=False
        )
    
    if SUMMARY:
        print(f"Stored predictions data in memory with {len(predictions_data)-2} models")
    
    return predictions_data 