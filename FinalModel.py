#!/usr/bin/env python3
# train_weld_meta_model_v2.1.py
#
# Usage:
#   pip install pandas scikit-learn joblib matplotlib
#   python train_weld_meta_model_v2.1.py "C:/.../ensemble_resultsV2.1.csv"

# Additions that could be made to improve the models:
# 1. use k-fold cross validation
# 2. use grid search to find the best hyperparameters
# 3. use a feature selection method to select the best features
# 4. add more encapsulation


# Standard library imports
import os
import sys

# Third-party imports
import joblib
import matplotlib
matplotlib.use('Agg')  # Use a non-GUI backend to prevent Tkinter errors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_curve
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

import config  # Import the config module
config.create_directories()  # Ensure output folders exist

from dataclasses import dataclass, asdict
from typing import Optional
import numpy as np

@dataclass
class SweepEntry:
    threshold: float
    precision: float
    recall: float
    accuracy: float
    cost: float
    tp: int
    fp: int
    tn: int
    fn: int

@dataclass
class ModelEvaluationResult:
    model_name: str
    split: str
    threshold_type: str       # 'cost', 'accuracy', or 'base' for base models
    threshold: float = None   # None for base models
    precision: float = None
    recall: float = None
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


# ---------- Data Preparation ----------
def prepare_data(df):
    """
    Prepare features & target from df by:
      • dropping config.EXCLUDE_COLS + config.TARGET
      • auto-detecting numeric vs categorical cols
      • one-hot-encoding all categoricals

    Returns:
        X: pd.DataFrame of features
        y: pd.Series of 0/1 target
        feature_cols: list of numeric feature names
        encoded_cols: list of new one-hot column names
    """
    # Copy to avoid side-effects
    df = df.copy()

    # 1) Extract & encode target
    target = config.TARGET
    # adjust mapping if your target labels differ
    y = df[target].map({"Good": 0, "Bad": 1})
    if y.isnull().any():
        raise ValueError(f"Found unmapped values in {target}")

    # 2) Drop excluded cols + target
    to_drop = set(config.EXCLUDE_COLS) | {target}
    missing = to_drop - set(df.columns)
    if missing:
        raise KeyError(f"Columns not found in DataFrame: {missing}")
    df_feats = df.drop(columns=list(to_drop))

    # 3) Split numeric vs categorical
    numeric_cols = df_feats.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = df_feats.select_dtypes(
        include=['object', 'category', 'bool']
    ).columns.tolist()

    # 4) One‑hot encode ALL categoricals
    df_cat = pd.get_dummies(df_feats[categorical_cols], prefix=categorical_cols, drop_first=False)

    # 5) Reassemble X
    X = pd.concat([df_feats[numeric_cols], df_cat], axis=1)

    # 6) Return feature lists
    encoded_cols = df_cat.columns.tolist()
    return X, y, numeric_cols, encoded_cols

def train_and_evaluate_model(
    model,
    X_train, y_train,
    splits: dict,
    model_name: str = None,
    model_dir: str = None,
    save_model: bool = True
):
    """
    Train once with SMOTE, optionally save, then predict on each split.

    Args:
        model: untrained sklearn‐style estimator
        X_train, y_train: training data
        splits: {'Train': (X_train, y_train), 'Test': (X_test, y_test), 'Full': (X, y)}
        model_name: filename (without .pkl) under which to save
        model_dir: directory in which to save the model
        save_model: whether to save the trained model

    Returns:
        split_preds: dict mapping split_name -> (probabilities, y_true)
        trained_model: the fitted model instance
    """
    # 1) Balance & train once
    print("Applying SMOTE & training model...")
    smote = SMOTE(random_state=42)
    X_bal, y_bal = smote.fit_resample(X_train, y_train)
    model.fit(X_bal, y_bal)

    # 2) Save if desired
    if save_model and model_name and model_dir:
        os.makedirs(model_dir, exist_ok=True)
        path = os.path.join(model_dir, f"{model_name}.pkl")
        joblib.dump(model, path)
        print(f"Saved trained model to {path}")

    # 3) Predict on each split
    split_preds = {}
    for split_name, (X_eval, y_eval) in splits.items():
        print(f"Predicting probabilities on {split_name} split...")
        proba = model.predict_proba(X_eval)[:, 1]
        split_preds[split_name] = (proba, y_eval)

    return split_preds, model

# ---------- Metrics Computation ----------
def compute_metrics(y_true, y_pred, C_FP, C_FN, as_dict=True):
    """Compute classification metrics and cost.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        C_FP: Cost of false positive
        C_FN: Cost of false negative
        as_dict: If True, return metrics as a dictionary
    
    Returns:
        tuple or dict: (precision, recall, accuracy, cost) or dictionary of metrics
    """
    # Compute confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    # Calculate metrics
    precision = 100 * tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = 100 * tp / (tp + fn) if (tp + fn) > 0 else 0
    accuracy = 100 * (tp + tn) / (tp + tn + fp + fn)
    # Calculate cost
    cost = C_FP * fp + C_FN * fn
    if as_dict:
        return {
            'precision': precision,
            'recall': recall,
            'accuracy': accuracy,
            'tp': tp,
            'fp': fp,
            'tn': tn,
            'fn': fn,
            'cost': cost
        }
    else:
        return precision, recall, accuracy, cost

def threshold_sweep_with_cost(proba, y_true, C_FP, C_FN, thresholds=None):
    """Perform threshold sweep analysis with cost calculation.
    
    Args:
        proba: array-like, predicted probabilities
        y_true: array-like, ground truth labels (0 or 1)
        C_FP: cost of false positive
        C_FN: cost of false negative
        thresholds: list or array of thresholds to evaluate (default: np.linspace(0, 1, 11))
    
    Returns:
        tuple: (sweep_results, best_by_cost, best_by_accuracy)
            - sweep_results: dict mapping threshold -> metrics (including cost)
            - best_by_cost: dict with metrics for threshold with lowest cost
            - best_by_accuracy: dict with metrics for threshold with highest accuracy
    """
    from sklearn.metrics import confusion_matrix
    if thresholds is None:
        thresholds = np.linspace(0, 1, 11)
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
        
        # Track best threshold by cost
        if cost < best_by_cost['cost']:
            best_by_cost = {'threshold': thr, 'cost': cost, **metrics}
            
        # Track best threshold by accuracy (minimum mistakes)
        if metrics['accuracy'] > best_by_accuracy['accuracy']:
            best_by_accuracy = {'threshold': thr, 'accuracy': metrics['accuracy'], **metrics}
    
    print(f"Best threshold by cost: {best_by_cost['threshold']} with expected cost {best_by_cost['cost']}")
    print(f"Best threshold by accuracy: {best_by_accuracy['threshold']} with accuracy {best_by_accuracy['accuracy']:.2f}%")
    
    return sweep_results, best_by_cost, best_by_accuracy

# ---------- Visualization ----------
def plot_threshold_sweep(sweep_results, C_FP, C_FN, output_path=None, cost_optimal_thr=None, accuracy_optimal_thr=None):
    """Plot threshold sweep results with cost.
    
    Args:
        sweep_results: Results from threshold_sweep function
        output_path: Path to save the plot
        C_FP: Cost of false positive (scrapping a good weld)
        C_FN: Cost of false negative (shipping a bad weld)
    """
    thresholds = list(sweep_results.keys())
    precision = [sweep_results[t]['precision'] for t in thresholds]
    recall = [sweep_results[t]['recall'] for t in thresholds]
    accuracy = [sweep_results[t]['accuracy'] for t in thresholds]
    cost = [sweep_results[t].get('cost', 0) for t in thresholds]  # Get cost if available
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # Plot metrics on primary y-axis
    ax1.plot(thresholds, precision, 'b-', 
             label='Precision: % of predicted bad welds that were actually bad')
    ax1.plot(thresholds, recall, 'r-', 
             label='Recall: % of actual bad welds caught')
    # Plot accuracy with a thicker, more prominent line
    ax1.plot(thresholds, accuracy, color='black', linewidth=2.5, label='Accuracy: % of all welds correctly classified')

    # After saving the figure, always close it to free memory and prevent Tkinter errors
    # (Add this after each plt.savefig(fig, ...))

    
    # Set labels and title
    ax1.set_xlabel('Confidence Threshold (higher = stricter)')
    ax1.set_ylabel('Percentage')
    ax1.set_ylim(0, 105)  # 0-100% for metrics
    ax1.grid(True)
    
    # Create secondary y-axis for cost
    ax2 = ax1.twinx()
    cost_line = ax2.plot(thresholds, cost, 'm--', 
                        label=f'Cost (FP={C_FP}, FN={C_FN})')
    ax2.set_ylabel('Cost', color='m')
    ax2.tick_params(axis='y', labelcolor='m')
    
    # Plot vertical lines for optimal thresholds if provided
    if cost_optimal_thr is not None:
        ax1.axvline(x=cost_optimal_thr, color='red', linestyle='--', alpha=0.7, label='Cost-Optimal Threshold')
    if accuracy_optimal_thr is not None:
        try:
            acc_val = accuracy[thresholds.index(accuracy_optimal_thr)]
            ax1.axvline(x=accuracy_optimal_thr, color='orange', linestyle='--', alpha=0.7, label='Accuracy-Optimal Threshold')
        except ValueError:
            # In case the threshold is not exactly in the list due to float precision
            pass  # Just skip if we can't find the exact threshold

    # Combine legends from both axes
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    
    # Create a single legend and position it outside the plot
    fig.legend(lines1 + lines2, labels1 + labels2, 
              loc='upper center', 
              bbox_to_anchor=(0.5, 0.96),  # Position above the plot
              ncol=2,  # Two columns for better layout
              frameon=False)  # No frame around the legend
    
    # Add title with cost information
    plt.suptitle('Effect of Confidence Threshold on Model Performance and Cost', y=0.99)
    
    # Adjust layout to make room for the legend and title
    plt.subplots_adjust(top=0.85)  # Add more space at the top
    
    # Adjust layout and save if needed
    fig.tight_layout(rect=[0, 0, 1, 0.9])  # Make room for the legend
    
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        print(f"Saved threshold sweep plot to {output_path}")
    
    plt.close(fig)  # Properly close the figure to avoid Tkinter errors



def plot_all_models_at_threshold(output_path, results_df, threshold_type, C_FP, C_FN, split_type='test', model_thresholds=None):
    """
    Plot precision, recall, and accuracy for all models using model-specific thresholds.
    
    This function visualizes model performance metrics across different models. It can show:
    - Precision: Percentage of positive predictions that were correct
    - Recall: Percentage of actual positives that were identified
    - Accuracy: Percentage of all predictions that were correct
    - Cost: Calculated cost based on false positives and false negatives
    
    The function can filter results by data split (train, test, or all) and threshold type (cost or accuracy).
    
    Args:
        output_path: Path to save the plot as an image file (use None to not save)
        results_df: DataFrame containing model evaluation metrics
        threshold_type: Type of threshold to use ('cost' or 'accuracy')
        split_type: Which data split to visualize ('train', 'test', or 'all')
        model_thresholds: Dictionary mapping model names to their thresholds (kept for backward compatibility)
        
    Returns:
        None (displays or saves the plot)
        
    Example:
        plot_all_models_at_threshold(
            output_path='model_comparison.png',
            results_df=my_results,
            threshold_type='cost',
            split_type='test'
        )
    """
    # Convert split_type to lowercase for case-insensitive comparison
    split_type = split_type.lower()
    if split_type not in ['train', 'test', 'all']:
        raise ValueError("split_type must be 'train', 'test', or 'all'")
    
    # Filter results_df based on split_type and threshold_type
    if split_type == 'all':
        # Use only the 'Full' split for the full dataset stats
        metrics_df = results_df[(results_df['split'].str.lower() == 'full') & 
                              (results_df['threshold_type'] == threshold_type)]
        base_models_df = results_df[(results_df['split'].str.lower() == 'full') &
                                 (results_df['model'].isin(['AD_Decision', 'CL_Decision', 'AD_or_CL_Fail']))]
    else:
        split_name = split_type.capitalize()
        metrics_df = results_df[(results_df['split'] == split_name) & 
                              (results_df['threshold_type'] == threshold_type)]
        # Get base models for the specified split
        base_models_df = results_df[
            (results_df['split'] == split_name) &
            (results_df['model'].isin(['AD_Decision', 'CL_Decision', 'AD_or_CL_Fail']))
        ]
    
    # Combine both DataFrames
    metrics_df = pd.concat([metrics_df, base_models_df]).drop_duplicates()
    
    # Prepare data for plotting
    models = metrics_df['model'].unique()
    metrics = {
        'Model': [],
        'Precision': [],
        'Recall': [],
        'Accuracy': [],
        'Cost': [],
        'Threshold': []
    }
    
    for model in models:
        model_data = metrics_df[metrics_df['model'] == model].iloc[0]  # Get first row for this model
        metrics['Model'].append(model)
        metrics['Precision'].append(model_data.get('precision', 0))
        metrics['Recall'].append(model_data.get('recall', 0))
        metrics['Accuracy'].append(model_data.get('accuracy', 0))
        metrics['Cost'].append(model_data.get('cost', 0))
        metrics['Threshold'].append(model_data.get('threshold', None))
    
    # Create the plot
    x = np.arange(len(metrics['Model']))
    width = 0.2
    
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Plot metrics
    precision_bars = ax.bar(x - 1.5*width, metrics['Precision'], width, label='Precision', color='#4F81BD')
    recall_bars = ax.bar(x - 0.5*width, metrics['Recall'], width, label='Recall', color='#C0504D')
    accuracy_bars = ax.bar(x + 0.5*width, metrics['Accuracy'], width, label='Accuracy', color='#9BBB59')
    
    # Add cost on secondary y-axis
    ax2 = ax.twinx()
    cost_bars = ax2.bar(x + 1.5*width, metrics['Cost'], width, label='Cost', color='gray', alpha=0.6)
    
    # Add cost ratio annotation to the title
    cost_ratio = C_FN / C_FP
    split_display = split_type.capitalize()
    if split_type == 'all':
        split_display = 'Full Dataset'
    
    # Function to add value labels on top of bars
    def add_value_labels(bars, ax, is_cost=False, decimals=1):
        for bar in bars:
            height = bar.get_height()
            if is_cost:
                # For cost, use integers since it's a count
                label = f'{int(height)}'
                va = 'bottom' if height >= 0 else 'top'
                y = height + (5 if height >= 0 else -5)
            else:
                # For percentages, show 1 decimal place
                label = f'{height:.{decimals}f}%'
                va = 'bottom' if height >= 0 else 'top'
                y = height + 0.5  # Small offset for percentages
                
            ax.text(bar.get_x() + bar.get_width()/2., y,
                    label,
                    ha='center', va=va, rotation=0, fontsize=8)
    
    # Add value labels to all bars
    add_value_labels(precision_bars, ax)
    add_value_labels(recall_bars, ax)
    add_value_labels(accuracy_bars, ax)
    add_value_labels(cost_bars, ax2, is_cost=True)
    
    # Customize the plot
    ax.set_xticks(x)
    ax.set_xticklabels(metrics['Model'], rotation=45, ha='right')
    ax.set_ylabel('Percentage')
    ax2.set_ylabel('Cost', color='gray')
    ax2.tick_params(axis='y', labelcolor='gray')
    
    # Add threshold annotations
    for i, (model, threshold) in enumerate(zip(metrics['Model'], metrics['Threshold'])):
        if threshold is not None and not np.isnan(threshold):
            ax.text(i, -5, f'Thr: {threshold:.2f}', ha='center', va='top', rotation=45, fontsize=8)
    
    # Add legend and title
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    # Set the title with cost ratio information
    plt.title(f'Model Performance Comparison\n({split_display} Data, {threshold_type.capitalize()}-optimized Thresholds, Cost Ratio: {cost_ratio:.1f})')
    plt.subplots_adjust(top=0.9)  # Add more space for the title
    plt.tight_layout()
    
    # Ensure the MyPlots directory exists
    plots_dir = 'MyPlots'
    os.makedirs(plots_dir, exist_ok=True)
    
    # Create default filename if none provided
    if output_path is None:
        output_path = os.path.join(plots_dir, f'model_comparison_{split_type}_{threshold_type}_{pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")}.png')
    
    # Save the plot if output_path is provided
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {os.path.abspath(output_path)}")
    
    # Always show the plot
    # plt.show()
    
    # Always close the plot to prevent Tkinter errors in scripts
    plt.close(fig)

def plot_runs_at_threshold(
    runs: list[ModelEvaluationRun],
    threshold_type: str,
    split_name: str = 'Test',
    C_FP: float = 1.0,
    C_FN: float = 1.0,
    output_path: str = None
):
    split_name = split_name.capitalize()
    if threshold_type not in ('cost','accuracy'):
        raise ValueError("threshold_type must be 'cost' or 'accuracy'")

    labels, precisions, recalls, accuracies, costs, thresholds = [], [], [], [], [], []

    for run in runs:
        # pick the threshold‐opt result (or base model fallback)
        match = next((r for r in run.results
                      if r.split == split_name and r.threshold_type == threshold_type), None)
        if not match:
            match = next((r for r in run.results
                          if r.split == split_name and r.is_base_model), None)
        if not match:
            continue

        labels.append(run.model_name)
        precisions.append(match.precision or 0.0)
        recalls.append(match.recall or 0.0)
        accuracies.append(match.accuracy or 0.0)
        costs.append(match.cost or 0.0)
        thresholds.append(match.threshold if match.threshold is not None else np.nan)

    if not labels:
        raise RuntimeError(f"No results for split={split_name}, threshold_type={threshold_type}")

    x = np.arange(len(labels))
    w = 0.2

    # make the figure a bit less tall
    fig, ax = plt.subplots(figsize=(14, 6))

    # main bars
    p_b = ax.bar(x - 1.5*w, precisions, w, label='Precision')
    r_b = ax.bar(x - 0.5*w, recalls,    w, label='Recall')
    a_b = ax.bar(x + 0.5*w, accuracies, w, label='Accuracy')

    # twin y‐axis for cost
    ax2 = ax.twinx()
    c_b = ax2.bar(x + 1.5*w, costs, w, label='Cost', alpha=0.6)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha='right')
    ax.set_ylabel('Percentage')
    ax2.set_ylabel('Cost')

        # --------------------------------------------------
    # annotate all four sets of bars on the correct axes
    # --------------------------------------------------

    # helper that takes a bar‐container, the axis to draw on,
    # a formatting function, and an absolute offset in data units
    def _annotate(bars, axis, fmt, offset):
        for bar in bars:
            h = bar.get_height()
            axis.text(
                bar.get_x() + bar.get_width()/2,
                h + offset,
                fmt(h),
                ha='center',
                va='bottom',
                fontsize=8
            )

    # percentage bars: use ax, offset in percentage‐points (0.5%)
    _annotate(p_b, ax, lambda h: f"{h:.1f}%", 0.5)
    _annotate(r_b, ax, lambda h: f"{h:.1f}%", 0.5)
    _annotate(a_b, ax, lambda h: f"{h:.1f}%", 0.5)

    # cost bars: use ax2, offset in cost‐units (e.g. 1% of max cost)
    max_cost = max(costs) if costs else 0
    cost_offset = max_cost * 0.01
    _annotate(c_b, ax2, lambda h: f"{int(h)}", cost_offset)


    # give some extra space at the bottom for threshold labels
    fig.subplots_adjust(bottom=0.25)

    # now drop threshold labels *just* below the x‐axis, in axes‐fraction coords
    for i, thr in enumerate(thresholds):
        if not np.isnan(thr):
            ax.text(x[i], -0.05, f"Thr: {thr:.2f}",
                    ha='center', va='top', rotation=45, fontsize=8,
                    transform=ax.get_xaxis_transform())

    # merge legends
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    
    # after you've collected h1, l1, h2, l2…
    combined_handles = h1 + h2
    combined_labels  = l1 + l2

    # 1) push the legend above the axes and center it
    ax.legend(
        combined_handles,
        combined_labels,
        loc='upper left',            # anchor “point” is the lower‐center of the legend box
        ncol=2,                        # spread items in 4 columns (if you like)
    )

    # 2) give extra room at top so nothing overlaps
    fig.subplots_adjust(top=0.85)

    # After plotting bars
    ax.relim()
    ax.autoscale_view()
    ylim = ax.get_ylim()
    ax.set_ylim(ylim[0], ylim[1] * 1.15)

    # Same for ax2
    ax2.relim()
    ax2.autoscale_view()
    ylim2 = ax2.get_ylim()
    ax2.set_ylim(ylim2[0], ylim2[1] * 1.15)


    ratio = C_FN / C_FP if C_FP else float('inf')
    ax.set_title(
        f"Model Performance Comparison\n"
        f"({split_name} Data, {threshold_type.capitalize()}‑opt Thresholds, Cost Ratio {ratio:.1f})"
    )

    # finalize
    plt.tight_layout()
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {output_path}")
    else:
        plt.show()
    plt.close(fig)

def save_all_model_probabilities_from_structure(
    results_total,
    predictions_dir,
    index,
    y_true
):
    """
    From a list of ModelEvaluationRun, builds a CSV with:
      • 1st column: ground truth (y_true)
      • subsequent columns: one probability column per model
        (using the “Full” split when available)

    Args:
        results_total: list of ModelEvaluationRun
        predictions_dir: directory to write the CSV
        index: full‐dataset index (e.g. df.index)
        y_true: ground‐truth array or pd.Series aligned to index
    """
    os.makedirs(predictions_dir, exist_ok=True)

    # start an empty DF with your index
    wide_df = pd.DataFrame(index=index)

    if not results_total:
        raise ValueError("results_total is empty")

    # insert GT as the very first column
    wide_df['GT'] = y_true

    # helper to pick the "Full" split or fall back to the longest array
    def pick_full_or_longest(probas):
        return probas.get(
            'Full',
            max(probas.values(), key=lambda arr: arr.shape[0])
        )

    # now add exactly one column per model
    for run in results_total:
        model_col = run.model_name
        probas = pick_full_or_longest(run.probabilities)

        # align by length/index
        if len(probas) == len(index):
            wide_df[model_col] = probas
        else:
            # shorter → assume starts at the top
            wide_df[model_col] = pd.Series(probas, index=index[: len(probas)])

    # save to CSV
    out_path = os.path.join(predictions_dir, 'all_model_predictions.csv')
    wide_df.to_csv(out_path, index=False)
    print(f"Saved GT as first column + one prob‐column per model to {out_path}")

# ---------- Main Function ----------
def main(config):
    """Main function to run the entire pipeline.
    
    Args:
        config: Configuration object containing paths and parameters
    """

    # Unpack config
    csv_path = config.DATA_PATH
    model_path = config.MODEL_DIR
    image_path = config.PLOT_DIR

    C_FP = config.C_FP
    C_FN = config.C_FN
    
    # Define model zoo
    MODELS = config.MODELS
    
    # Ensure output directories exist
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    # Load data
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Prepare data
    print("Preparing data...")
    X, y, feature_cols, encoded_cols = prepare_data(df)
    
    # Save feature columns for later use
    # Update the feature_info dictionary to include encoding details
    feature_info = {
        'feature_cols': feature_cols,  # Original numeric features
        'encoded_cols': encoded_cols,  # One-hot encoded columns
        'encoding': {
            'Weld': {
                'type': 'one_hot',
                'original_column': 'Weld',
                'categories': sorted(df['Weld'].unique().tolist())  # All unique Weld values seen during training
            }
        }
    }
    joblib.dump(feature_info, os.path.join(model_path, "feature_info.pkl"))
    
    # Split data once
    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y)
    train_idx = X_train.index
    test_idx = X_test.index
    
    # Define evaluation splits
    splits = {
        'Train': (X_train, y_train),
        'Test': (X_test, y_test),
        'Full': (X, y)
    }
    
    # Collect results for each model and split
    results = []  # collect per-model, per-split rows
    results_table = [] # per model, using structs
    results_total = []  # All raw probabilities for each model
    
    best_models = {}
    
    # Initialize dictionary to store all probabilities and GT
    # Prepare wide-format DataFrame indexed by original DataFrame
    wide_df = pd.DataFrame(index=df.index)
    wide_df['GT'] = y.values
    
    for model_name, model in MODELS.items():
        print(f"\n{'-'*50}\nTraining & evaluating {model_name}...\n")

        # --- prepare per-model accumulators ---
        this_model_results = []     # List[ModelEvaluationResult]
        this_model_probs: dict = {} # Dict[split_name -> np.array]

        # train once, get all split probs and the trained model back
        split_preds, trained_model = train_and_evaluate_model(
            model=model,
            X_train=X_train, y_train=y_train,
            splits=splits,
            model_name=f"meta_model_v2.1_{model_name}",
            model_dir=model_path,
            save_model=True
        )

        # now loop splits exactly as before, but WITHOUT retraining
        for split_name, (proba, y_eval) in split_preds.items():
            print(f"\nEvaluating on {split_name} split...")

            # threshold sweep
            sweep, best_cost, best_acc = threshold_sweep_with_cost(
                proba, y_eval,
                thresholds=np.linspace(0,1,21),
                C_FP=C_FP, C_FN=C_FN
            )
            cost_thr = best_cost['threshold']
            acc_thr  = best_acc['threshold']

            # plot
            plot_threshold_sweep(
                sweep,
                output_path=f"{image_path}/{model_name}_{split_name.lower()}_threshold_sweep.png",
                cost_optimal_thr=cost_thr,
                accuracy_optimal_thr=acc_thr,
                C_FP=C_FP, C_FN=C_FN
            )
            plt.close()

            # compute metrics at both thresholds
            y_cost = (proba >= cost_thr).astype(int)
            y_acc  = (proba >= acc_thr).astype(int)
            mets_cost = compute_metrics(y_eval, y_cost, C_FP=C_FP, C_FN=C_FN)
            mets_acc  = compute_metrics(y_eval, y_acc,  C_FP=C_FP, C_FN=C_FN)

            # append to your old results list
            results.append({
                'model': model_name, 'split': split_name,
                'threshold_type': 'cost', 'threshold': cost_thr,
                **mets_cost
            })
            results.append({
                'model': model_name, 'split': split_name,
                'threshold_type': 'accuracy', 'threshold': acc_thr,
                **mets_acc
            })

            # append to your dataclass table
            cost_result = ModelEvaluationResult(
                model_name    = model_name,
                split         = split_name,
                threshold_type= 'cost',
                threshold     = cost_thr,
                precision     = mets_cost['precision'],
                recall        = mets_cost['recall'],
                accuracy      = mets_cost['accuracy'],
                cost          = mets_cost['cost'],
                tp            = mets_cost['tp'],
                fp            = mets_cost['fp'],
                tn            = mets_cost['tn'],
                fn            = mets_cost['fn'],
            )
            acc_result = ModelEvaluationResult(
                model_name    = model_name,
                split         = split_name,
                threshold_type= 'accuracy',
                threshold     = acc_thr,
                precision     = mets_acc['precision'],
                recall        = mets_acc['recall'],
                accuracy      = mets_acc['accuracy'],
                cost          = mets_acc['cost'],
                tp            = mets_acc['tp'],
                fp            = mets_acc['fp'],
                tn            = mets_acc['tn'],
                fn            = mets_acc['fn'],
            )

            # collect into per-model accumulators
            this_model_results.extend([cost_result, acc_result])
            this_model_probs[split_name] = proba

            # store into wide_df as before
            col = f"{model_name}_{split_name}"
            arr = np.full(len(df), np.nan)
            idx = y_eval.index if hasattr(y_eval, 'index') else range(len(proba))
            arr[idx] = proba
            wide_df[col] = arr

        # --- now that all splits are done, create ONE run per model ---
        run = ModelEvaluationRun(
            model_name   = model_name,
            results      = this_model_results,
            probabilities= this_model_probs
        )
        results_total.append(run)

        # (no need to joblib.dump here — already saved in train_and_evaluate_model)

    # after loop, recreate results_df
    results_df = pd.DataFrame(results)

    # ========== BASE MODEL METRICS SECTION ==========
    # Add base model predictions to wide_df
    for base in ['AD_Decision', 'CL_Decision']:
        wide_df[base] = (df[base] == 'Bad').astype(float)
    # Combined index: fails if either base model is "Bad"
    wide_df['AD_or_CL_Fail'] = ((df['AD_Decision'] == 'Bad') | (df['CL_Decision'] == 'Bad')).astype(float)

    # Compute and append metrics for base models for each split
    base_metrics = []
    for base in ['AD_Decision', 'CL_Decision', 'AD_or_CL_Fail']:
        for split_name, idx in zip(['Train', 'Test', 'Full'], [train_idx, test_idx, df.index]):
            if base == 'AD_or_CL_Fail':
                y_pred = wide_df.loc[idx, 'AD_or_CL_Fail'].values
            else:
                y_pred = wide_df.loc[idx, base].values
            y_true = y.loc[idx].values
            metrics = compute_metrics(y_true, y_pred, C_FP=C_FP, C_FN=C_FN)
            base_metrics.append({
                'model': base,
                'split': split_name,
                'threshold_type': 'cost',
                'threshold': None,
                **metrics
            })
    
    # Concatenate all base metrics at once
    if base_metrics:
        base_metrics_df = pd.DataFrame(base_metrics)
        results_df = pd.concat([results_df, base_metrics_df], ignore_index=True) if not results_df.empty else base_metrics_df

    # ========== Add base models to results_total structure ==========
    for base in ['AD_Decision', 'CL_Decision', 'AD_or_CL_Fail']:
        base_results = []
        base_probs = {}
        for split_name, idx in zip(['Train', 'Test', 'Full'], [train_idx, test_idx, df.index]):
            if base == 'AD_or_CL_Fail':
                y_pred = wide_df.loc[idx, 'AD_or_CL_Fail'].values
            else:
                y_pred = wide_df.loc[idx, base].values
            y_true = y.loc[idx].values
            metrics = compute_metrics(y_true, y_pred, C_FP=C_FP, C_FN=C_FN)
            base_result = ModelEvaluationResult(
                model_name    = base,
                split         = split_name,
                threshold_type= 'base',
                threshold     = None,
                precision     = metrics['precision'],
                recall        = metrics['recall'],
                accuracy      = metrics['accuracy'],
                cost          = metrics['cost'],
                tp            = metrics['tp'],
                fp            = metrics['fp'],
                tn            = metrics['tn'],
                fn            = metrics['fn'],
                is_base_model = True
            )
            base_results.append(base_result)
            base_probs[split_name] = y_pred
        run = ModelEvaluationRun(
            model_name   = base,
            results      = base_results,
            probabilities= base_probs
        )
        results_total.append(run)
    # ========== END BASE MODEL METRICS SECTION ==========

   # Print summary table
    print("\n" + "="*80)
    print("SUMMARY OF MODEL PERFORMANCE")
    print("="*80)

    # Group by model and split, then aggregate metrics
    summary = results_df.groupby(['model', 'split']).agg({
        'accuracy': 'mean',
        'precision': 'mean',
        'recall': 'mean'
    }).reset_index()
    
    # Print formatted summary with confusion matrix and cost by threshold type for meta-models
    for model_name in MODELS.keys():
        print(f"\n{model_name}:")
        model_results = results_df[(results_df['model'] == model_name)]
        for split in ['Train', 'Test', 'Full']:
            for thr_type in ['cost', 'accuracy']:
                row = model_results[(model_results['split'] == split) & (model_results['threshold_type'] == thr_type)]
                if not row.empty:
                    row = row.iloc[0]
                    print(f"  {split} Split ({thr_type}-optimal):")
                    print(f"    Accuracy:  {row['accuracy']:.1f}%  Precision: {row['precision']:.1f}%  Recall: {row['recall']:.1f}%")
                    print(f"    Threshold: {row['threshold']:.3f}")
                    print(f"    Cost:      {row['cost']}")
                    print(f"    Confusion Matrix: [[TN={row['tn']} FP={row['fp']}], [FN={row['fn']} TP={row['tp']}]]")

    # Print summary for base models (preset confidence threshold, not optimized)
    print("\nBASE MODELS PERFORMANCE (preset confidence threshold, not optimized):")
    base_model_names = [name for name in results_df['model'].unique() if name not in MODELS.keys()]
    for model_name in base_model_names:
        print(f"\n{model_name}:")
        model_results = results_df[(results_df['model'] == model_name) & (results_df['threshold_type'] == 'cost')]
        for split in ['Train', 'Test', 'Full']:
            row = model_results[model_results['split'] == split]
            if not row.empty:
                row = row.iloc[0]
                print(f"  {split} Split (preset confidence threshold):")
                print(f"    Accuracy:  {row['accuracy']:.1f}%  Precision: {row['precision']:.1f}%  Recall: {row['recall']:.1f}%")
                print(f"    Cost:      {row['cost']}")
                print(f"    Confusion Matrix: [[TN={row['tn']} FP={row['fp']}], [FN={row['fn']} TP={row['tp']}]]")


    if hasattr(config, 'SAVE_PREDICTIONS') and config.SAVE_PREDICTIONS:
        predictions_dir = getattr(config, 'PREDICTIONS_DIR', os.path.dirname(model_path))
        save_all_model_probabilities_from_structure(results_total, predictions_dir, df.index, y_eval)

    print(results_df['split'].unique())


    # Plot all models at each of the three threshold types using model-specific thresholds
    # 1. Cost-optimized thresholds for test data

    # plot_all_models_at_threshold(
    #     output_path=os.path.join(image_path, 'all_models_test_cost_threshold.png'),
    #     results_df=results_df, 
    #     threshold_type='cost',
    #     split_type='test',
    #     model_thresholds=best_models,
    #     C_FP=C_FP,
    #     C_FN=C_FN
    # )
    # plot_all_models_at_threshold(
    #     output_path=os.path.join(image_path, 'all_models_test_accuracy_threshold.png'),
    #     results_df=results_df, 
    #     threshold_type='accuracy',
    #     split_type='test',
    #     model_thresholds=best_models,
    #     C_FP=C_FP,
    #     C_FN=C_FN
    # )
    # print(f"\nPlotted all models with cost-optimized thresholds (test split)")
    
    # # 2. Cost-optimized thresholds for train data
    # plot_all_models_at_threshold(
    #     output_path=os.path.join(image_path, 'all_models_full_cost_threshold.png'),
    #     results_df=results_df, 
    #     threshold_type='cost',
    #     split_type='all',
    #     model_thresholds=best_models,
    #     C_FP=C_FP,
    #     C_FN=C_FN
    # )
    # plot_all_models_at_threshold(
    #     output_path=os.path.join(image_path, 'all_models_full_accuracy_threshold.png'),
    #     results_df=results_df, 
    #     threshold_type='accuracy',
    #     split_type='all',
    #     model_thresholds=best_models,
    #     C_FP=C_FP,
    #     C_FN=C_FN
    # )

    plot_runs_at_threshold(
        runs=results_total,
        threshold_type='cost',
        split_name='Full',
        C_FP=C_FP,
        C_FN=C_FN,
        output_path=os.path.join(image_path, 'model_comparison_cost_optimized.png')
    )
    plot_runs_at_threshold(
        runs=results_total,
        threshold_type='accuracy',
        split_name='Full',
        C_FP=C_FP,
        C_FN=C_FN,
        output_path=os.path.join(image_path, 'model_comparison_accuracy_optimized.png')
    )

    # print("results_table")
    # print(results_total)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        main(config)
    else:
        main(config)