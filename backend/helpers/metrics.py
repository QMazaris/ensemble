from dataclasses import dataclass
from typing import Optional
import numpy as np
from sklearn.metrics import confusion_matrix

@dataclass
class ModelEvaluationResult:
    model_name: str
    split: str
    threshold_type: str       # 'cost', 'accuracy', or 'base' for base models
    threshold: float = None   # None for base models
    precision: float = None
    recall: float = None
    f1_score: float = None    # Add F1 score field
    accuracy: float = None
    cost: float = None
    tp: int = None
    fp: int = None
    tn: int = None
    fn: int = None
    is_base_model: bool = False  # True for base model results

@dataclass
class ModelEvaluationRun:
    model_name: str
    results: list  # List of ModelEvaluationResult (including base and meta models)
    probabilities: Optional[dict] = None  # Dict of model_name -> np.ndarray (including base models if available)
    sweep_data: Optional[dict] = None  # Dict of split_name -> sweep_results from threshold_sweep_with_cost

def compute_metrics(y_true, y_pred, C_FP, C_FN, as_dict=True):
    """Compute precision, recall, F1 score, accuracy, and cost metrics."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    precision = 100 * tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = 100 * tp / (tp + fn) if (tp + fn) > 0 else 0
    # Calculate F1 score using precision and recall values (not percentages)
    precision_raw = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall_raw = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 100 * (2 * precision_raw * recall_raw) / (precision_raw + recall_raw) if (precision_raw + recall_raw) > 0 else 0
    accuracy = 100 * (tp + tn) / (tp + tn + fp + fn)
    cost = C_FP * fp + C_FN * fn
    if as_dict:
        return {
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'accuracy': accuracy,
            'tp': tp,
            'fp': fp,
            'tn': tn,
            'fn': fn,
            'cost': cost
        }
    else:
        return precision, recall, f1_score, accuracy, cost

def threshold_sweep_with_cost(proba, y_true, C_FP, C_FN, thresholds=None, SUMMARY=None, split_name=None):
    """Perform a sweep over thresholds to find optimal cost and accuracy."""
    if thresholds is None:
        thresholds = np.linspace(0, 1, 21)
    best_by_cost = {'threshold': None, 'cost': float('inf')}
    best_by_accuracy = {'threshold': None, 'accuracy': -1}
    sweep_results = {}
    for thr in thresholds:
        y_pred = (proba >= thr).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        cost = C_FP * fp + C_FN * fn
        total_mistakes = fp + fn
        metrics = compute_metrics(y_true, y_pred, C_FP, C_FN)
        metrics.update({
            'threshold': thr,
            'tp': tp,
            'fp': fp,
            'tn': tn,
            'fn': fn,
            'cost': cost,
            'total_mistakes': total_mistakes
        })
        sweep_results[thr] = metrics
        if cost < best_by_cost['cost']:
            best_by_cost = {'threshold': thr, 'cost': cost, **metrics}
        if metrics['accuracy'] > best_by_accuracy['accuracy']:
            best_by_accuracy = {'threshold': thr, 'accuracy': metrics['accuracy'], **metrics}
    if SUMMARY:
        if split_name:
            print(f"\n=== {split_name.upper()} SPLIT ===")
        print(f"Best threshold by cost: {best_by_cost['threshold']} with expected cost {best_by_cost['cost']}")
        print(f"  Confusion Matrix (cost-opt): [[TN={best_by_cost['tn']} FP={best_by_cost['fp']}], [FN={best_by_cost['fn']} TP={best_by_cost['tp']}]]")
        print(f"Best threshold by accuracy: {best_by_accuracy['threshold']} with accuracy {best_by_accuracy['accuracy']:.2f}%")
        print(f"  Confusion Matrix (acc-opt):  [[TN={best_by_accuracy['tn']} FP={best_by_accuracy['fp']}], [FN={best_by_accuracy['fn']} TP={best_by_accuracy['tp']}]]\n")
    return sweep_results, best_by_cost, best_by_accuracy

def _mk_result(model_name, split_name, best, thr_type):
    """Create a ModelEvaluationResult from best metrics."""
    return ModelEvaluationResult(
        model_name=model_name,
        split=split_name,
        threshold_type=thr_type,
        threshold=best['threshold'],
        precision=best['precision'],
        recall=best['recall'],
        f1_score=best['f1_score'],
        accuracy=best['accuracy'],
        cost=best['cost'],
        tp=best['tp'], fp=best['fp'],
        tn=best['tn'], fn=best['fn'],
        is_base_model=False
    )

def _average_results(res_list, model_name):
    """Average results across folds."""
    fields = ['precision', 'recall', 'f1_score', 'accuracy', 'cost', 'tp', 'fp', 'tn', 'fn']
    avg = {f: float(np.mean([getattr(r, f) for r in res_list])) for f in fields}
    return ModelEvaluationResult(
        model_name=f'kfold_avg_{model_name}',
        split='Full',
        threshold_type=res_list[0].threshold_type,
        threshold=float(np.mean([r.threshold for r in res_list])),
        is_base_model=False,
        **avg
    )

def _average_probabilities(prob_arrays):
    """Average probability arrays across folds."""
    if not prob_arrays:
        return None
    min_len = min(map(len, prob_arrays))
    return np.mean([p[:min_len] for p in prob_arrays], axis=0)

def _average_sweep_data(fold_sweeps):
    """Average sweep data across multiple folds.
    
    Args:
        fold_sweeps: List of sweep dictionaries from different folds
        
    Returns:
        dict: Averaged sweep data
    """
    if not fold_sweeps:
        return {}
        
    avg_sweep = {}
    # Get all unique thresholds across all folds
    all_thresholds = sorted(set().union(*[set(s.keys()) for s in fold_sweeps]))
    
    for thr in all_thresholds:
        # For each threshold, average the metrics across folds
        metrics = {}
        for metric in ['cost', 'accuracy', 'precision', 'recall', 'f1_score', 'tp', 'fp', 'tn', 'fn']:
            values = [s[thr][metric] for s in fold_sweeps if thr in s]
            if values:  # Only average if we have values for this threshold
                metrics[metric] = float(np.mean(values))
        if metrics:  # Only add if we have metrics
            avg_sweep[float(thr)] = metrics
            
    return avg_sweep

def calculate_final_production_thresholds(model, X, y, C_FP, C_FN, model_name, SUMMARY=None):
    """Calculate and print final production thresholds using the entire dataset.
    
    Args:
        model: Trained model to evaluate
        X: Full feature matrix
        y: Full target vector
        C_FP: Cost of false positive
        C_FN: Cost of false negative
        model_name: Name of the model for printing
        SUMMARY: Whether to print detailed output
    
    Returns:
        tuple: (cost_optimal_threshold, accuracy_optimal_threshold, sweep_results)
    """
    # Get probabilities on full dataset
    proba = model.predict_proba(X)[:, 1]
    
    # Perform threshold sweep
    sweep_results, best_cost, best_acc = threshold_sweep_with_cost(
        proba, y, C_FP=C_FP, C_FN=C_FN,
        SUMMARY=SUMMARY, split_name="FULL DATASET"
    )
    
    if SUMMARY:
        print(f"\n{'='*80}")
        print(f"FINAL PRODUCTION THRESHOLDS FOR {model_name}")
        print(f"{'='*80}")
        print(f"Cost-optimal threshold: {best_cost['threshold']:.3f}")
        print(f"  • Expected cost: {best_cost['cost']:.1f}")
        print(f"  • Accuracy: {best_cost['accuracy']:.1f}%")
        print(f"  • Precision: {best_cost['precision']:.1f}%")
        print(f"  • Recall: {best_cost['recall']:.1f}%")
        print(f"  • F1 Score: {best_cost['f1_score']:.1f}%")
        print(f"  • Confusion Matrix: [[TN={best_cost['tn']} FP={best_cost['fp']}], [FN={best_cost['fn']} TP={best_cost['tp']}]]")
        print(f"\nAccuracy-optimal threshold: {best_acc['threshold']:.3f}")
        print(f"  • Accuracy: {best_acc['accuracy']:.1f}%")
        print(f"  • Expected cost: {best_acc['cost']:.1f}")
        print(f"  • Precision: {best_acc['precision']:.1f}%")
        print(f"  • Recall: {best_acc['recall']:.1f}%")
        print(f"  • F1 Score: {best_acc['f1_score']:.1f}%")
        print(f"  • Confusion Matrix: [[TN={best_acc['tn']} FP={best_acc['fp']}], [FN={best_acc['fn']} TP={best_acc['tp']}]]")
        print(f"{'='*80}\n")
    
    return best_cost['threshold'], best_acc['threshold'], sweep_results 