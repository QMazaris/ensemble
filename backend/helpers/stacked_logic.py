import numpy as np
from sklearn.metrics import confusion_matrix
from typing import List, Dict, Any
from .metrics import ModelEvaluationRun, ModelEvaluationResult, compute_metrics


def pick_oof_or_full_or_longest(probas):
    """Pick the best available probability array, prioritizing OOF predictions for k-fold models."""
    if not probas:
        return None
    if 'oof' in probas:
        return np.array(probas['oof'])  # Use OOF predictions for k-fold models
    if 'Full' in probas:
        return np.array(probas['Full'])
    non_empty = [arr for arr in probas.values() if arr is not None and hasattr(arr, 'shape')]
    if not non_empty:
        return None
    return max(non_empty, key=lambda arr: arr.shape[0])


def generate_combined_runs(
    runs: List[ModelEvaluationRun],
    combined_logic: Dict[str, Dict[str, Any]],
    y_true: np.ndarray,
    C_FP: float,
    C_FN: float,
    N_SPLITS: int,
    model_thresholds: Dict[str, float] = None
) -> List[ModelEvaluationRun]:
    """
    Create new ModelEvaluationRun entries by bitwise-combining
    the decisions of existing runs.

    Args:
        runs: existing base-model runs (probabilities are accessed using pick_oof_or_full_or_longest).
        combined_logic: {
            "NewModelName": {"columns": ["AD_Decision","CL_Decision"], "logic": "OR"},
             ...
        }
        y_true: ground-truth labels as a 1D array of 0/1.
        C_FP, C_FN: false-positive/false-negative costs.
        N_SPLITS: number of folds for cost normalization.
        model_thresholds: optional dict mapping column names to custom thresholds.
                         If None or missing for a column, defaults to 0.5.

    Returns:
        List of ModelEvaluationRun for each combined model.
    """
    # Map operator names to numpy functions and custom functions for negated operations
    ops = {
        'OR':  np.bitwise_or, '|':  np.bitwise_or,
        'AND': np.bitwise_and, '&':  np.bitwise_and,
        'XOR': np.bitwise_xor, '^':  np.bitwise_xor,
        'NOR': lambda a, b: ~np.bitwise_or(a, b),
        'NAND': lambda a, b: ~np.bitwise_and(a, b),
        'NXOR': lambda a, b: ~np.bitwise_xor(a, b)
    }

    # Helper: look up a run by model_name with flexible matching
    name_to_run = {}
    for run in runs:
        # Store both the exact name and cleaned name for lookup
        name_to_run[run.model_name] = run
        # Also store with cleaned name (remove kfold_avg_ prefix if present)
        clean_name = run.model_name.replace('kfold_avg_', '')
        name_to_run[clean_name] = run
    
    # Default thresholds if not provided
    if model_thresholds is None:
        model_thresholds = {}

    new_runs = []
    for new_name, cfg in combined_logic.items():
        cols = cfg.get("columns", [])
        logic = cfg.get("logic", "OR")

        # --- validation ---
        if not cols:
            raise ValueError(f"No columns specified for combined model '{new_name}'")
        if logic not in ops:
            raise ValueError(f"Unsupported logic '{logic}' for '{new_name}'")

        # fetch each base-model decision array using the helper function
        try:
            arrays = []
            for col in cols:
                # Try to find the run with flexible name matching
                run = None
                if col in name_to_run:
                    run = name_to_run[col]
                elif f"kfold_avg_{col}" in name_to_run:
                    run = name_to_run[f"kfold_avg_{col}"]
                
                if run is None:
                    available_models = list(set([run.model_name for run in runs] + [run.model_name.replace('kfold_avg_', '') for run in runs]))
                    raise ValueError(f"Model '{col}' not found in available runs. Available models: {available_models}")
                
                # Use the helper function to get the right probability array
                probas = pick_oof_or_full_or_longest(run.probabilities)
                if probas is None:
                    raise ValueError(f"No valid probabilities found for model {col}")
                
                # Convert to binary decisions using custom thresholds
                # Get threshold for this column (default to 0.5 if not specified)
                threshold = model_thresholds.get(col, 0.5)
                
                # For base models like AD_Decision/CL_Decision, these might already be binary
                if np.all(np.isin(probas, [0, 1])):
                    # Already binary decisions - use as is
                    decisions = probas.astype(int)
                else:
                    # Convert probabilities to binary decisions using custom threshold
                    decisions = (probas >= threshold).astype(int)
                
                arrays.append(decisions)
        except KeyError as e:
            available_models = list(set([run.model_name for run in runs] + [run.model_name.replace('kfold_avg_', '') for run in runs]))
            raise ValueError(f"Missing runs for models: {cols}. Available models: {available_models}") from e

        # Ensure all arrays have the same length
        array_lengths = [len(arr) for arr in arrays]
        if not all(length == array_lengths[0] for length in array_lengths):
            raise ValueError(f"Found input variables with inconsistent numbers of samples: {array_lengths}")

        # apply bitwise logic in sequence
        combined = arrays[0]
        op = ops[logic]
        for arr in arrays[1:]:
            combined = op(combined, arr)

        # compute confusion, cost, metrics
        # Ensure binary classification with labels [0, 1]
        cm = confusion_matrix(y_true, combined, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        
        # Validate that we have a 2x2 matrix
        if cm.shape != (2, 2):
            raise ValueError(f"Expected 2x2 confusion matrix for binary classification, got {cm.shape}")
        
        cost = (C_FP * fp + C_FN * fn) / N_SPLITS
        metrics = compute_metrics(y_true, combined, C_FP, C_FN)
        metrics.update({'cost': cost, 'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn})

        # build the single-result list
        result = ModelEvaluationResult(
            model_name=new_name,
            split='Full',
            threshold_type='combined',
            threshold=None,
            precision=metrics['precision'],
            recall=metrics['recall'],
            f1_score=metrics['f1_score'],
            accuracy=metrics['accuracy'],
            cost=cost,
            tp=tp, fp=fp, tn=tn, fn=fn,
            is_base_model=True
        )

        run = ModelEvaluationRun(
            model_name=new_name,
            results=[result],
            probabilities={'Full': combined}
        )
        new_runs.append(run)

    return new_runs
