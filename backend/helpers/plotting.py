import numpy as np
import os
import pandas as pd
from collections import Counter

def plot_threshold_sweep(sweep_results, C_FP, C_FN, output_path=None, cost_optimal_thr=None, accuracy_optimal_thr=None, SUMMARY = None):
    """Return threshold sweep data for frontend plotting instead of generating matplotlib plots."""
    thresholds = list(sweep_results.keys())
    precision = [sweep_results[t]['precision'] for t in thresholds]
    recall = [sweep_results[t]['recall'] for t in thresholds]
    f1_score = [sweep_results[t]['f1_score'] for t in thresholds]
    accuracy = [sweep_results[t]['accuracy'] for t in thresholds]
    cost = [sweep_results[t].get('cost', 0) for t in thresholds]
    
    # Return data instead of plotting
    plot_data = {
        'thresholds': thresholds,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'accuracy': accuracy,
        'cost': cost,
        'cost_optimal_threshold': cost_optimal_thr,
        'accuracy_optimal_threshold': accuracy_optimal_thr,
        'C_FP': C_FP,
        'C_FN': C_FN
    }
    
    if SUMMARY:
        print(f"Generated threshold sweep data with {len(thresholds)} points")
    
    return plot_data

def plot_runs_at_threshold(runs, threshold_type, split_name='Test', C_FP=1.0, C_FN=1.0, output_path=None):
    """Return model comparison data for frontend plotting instead of generating matplotlib plots."""
    split_name = split_name.capitalize()
    if threshold_type not in ('cost','accuracy'):
        raise ValueError("threshold_type must be 'cost' or 'accuracy'")
    
    labels, precisions, recalls, f1_scores, accuracies, costs, thresholds = [], [], [], [], [], [], []
    for run in runs:
        match = next((r for r in run.results if r.split == split_name and r.threshold_type == threshold_type), None)
        if not match:
            match = next((r for r in run.results if r.split == split_name and r.is_base_model), None)
        if not match:
            continue
        labels.append(run.model_name)
        precisions.append(match.precision or 0.0)
        recalls.append(match.recall or 0.0)
        f1_scores.append(match.f1_score or 0.0)
        accuracies.append(match.accuracy or 0.0)
        costs.append(match.cost or 0.0)
        thresholds.append(match.threshold if match.threshold is not None else np.nan)
    
    if not labels:
        raise RuntimeError(f"No results for split={split_name}, threshold_type={threshold_type}")
    
    # Return data instead of plotting
    comparison_data = {
        'model_names': labels,
        'precision': precisions,
        'recall': recalls,
        'f1_score': f1_scores,
        'accuracy': accuracies,
        'cost': costs,
        'thresholds': thresholds,
        'split_name': split_name,
        'threshold_type': threshold_type,
        'C_FP': C_FP,
        'C_FN': C_FN
    }
    
    return comparison_data

def plot_class_balance(y, output_path=None, SUMMARY=None):
    """Return class balance data for frontend plotting instead of generating matplotlib plots."""
    # Count classes
    counts = Counter(np.array(y))
    labels = ['Good (0)', 'Bad (1)']
    values = [counts.get(0, 0), counts.get(1, 0)]
    total = sum(values)
    percents = [v / total * 100 for v in values]
    
    # Return data instead of plotting
    balance_data = {
        'labels': labels,
        'values': values,
        'percentages': percents,
        'total': total
    }
    
    if SUMMARY:
        print(f"Generated class balance data: {dict(zip(labels, values))}")
    
    return balance_data 