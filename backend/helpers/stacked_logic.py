import numpy as np
from sklearn.metrics import confusion_matrix
from typing import List, Dict, Any
from .metrics import ModelEvaluationRun, ModelEvaluationResult, compute_metrics


def generate_combined_runs(
    runs: List[ModelEvaluationRun],
    combined_logic: Dict[str, Dict[str, Any]],
    y_true: np.ndarray,
    C_FP: float,
    C_FN: float,
    N_SPLITS: int
) -> List[ModelEvaluationRun]:
    """
    Create new ModelEvaluationRun entries by bitwise-combining
    the 'Full' decisions of existing runs.

    Args:
        runs: existing base-model runs (probabilities['Full'] is 0/1 decisions).
        combined_logic: {
            "NewModelName": {"columns": ["AD_Decision","CL_Decision"], "logic": "OR"},
             ...
        }
        y_true: ground-truth labels as a 1D array of 0/1.
        C_FP, C_FN: false-positive/false-negative costs.
        N_SPLITS: number of folds for cost normalization.

    Returns:
        List of ModelEvaluationRun for each combined model.
    """
    # Map operator names to numpy functions
    ops = {
        'OR':  np.bitwise_or, '|':  np.bitwise_or,
        'AND': np.bitwise_and, '&':  np.bitwise_and,
        'XOR': np.bitwise_xor, '^':  np.bitwise_xor
    }

    # Helper: look up a run by model_name
    name_to_run = {run.model_name: run for run in runs}

    new_runs = []
    for new_name, cfg in combined_logic.items():
        cols = cfg.get("columns", [])
        logic = cfg.get("logic", "OR")

        # --- validation ---
        if not cols:
            raise ValueError(f"No columns specified for combined model '{new_name}'")
        if logic not in ops:
            raise ValueError(f"Unsupported logic '{logic}' for '{new_name}'")

        # fetch each base-model decision array
        try:
            arrays = [
                name_to_run[col].probabilities['Full'].astype(int)
                for col in cols
            ]
        except KeyError as e:
            missing = [c for c in cols if c not in name_to_run]
            raise ValueError(f"Missing runs for models: {missing}") from e

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
