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
from sklearn.model_selection import StratifiedKFold
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
def prepare_data(df, config):
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
    exclude_cols = config.EXCLUDE_COLS + [config.TARGET]
    X = df.drop(columns=exclude_cols)
    # Map target to 0/1 if needed
    y_raw = df[config.TARGET]
    if y_raw.dtype == object or y_raw.dtype.name == 'category':
        y = y_raw.map({'Good': 0, 'Bad': 1})
    else:
        y = y_raw
    y = y.astype(int)

    # Identify numeric and categorical columns
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()

    # One-hot encode categoricals
    X_encoded = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
    encoded_cols = [col for col in X_encoded.columns if col not in numeric_cols]

    return X_encoded, y, numeric_cols, encoded_cols

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
        loc='upper left',            # anchor "point" is the lower‐center of the legend box
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
        (using the "Full" split when available)

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
        if not probas:
            return None
        if 'Full' in probas:
            return probas['Full']
        # Only consider non-empty arrays
        non_empty = [arr for arr in probas.values() if arr is not None and hasattr(arr, 'shape')]
        if not non_empty:
            return None
        return max(non_empty, key=lambda arr: arr.shape[0])

    # now add exactly one column per model
    for run in results_total:
        model_col = run.model_name
        probas = pick_full_or_longest(run.probabilities)

        # align by length/index
        if probas is None:
            print(f"[WARN] No probabilities found for model {model_col}, skipping.")
            continue
        if len(probas) == len(index):
            wide_df[model_col] = probas
        else:
            # shorter → assume starts at the top
            wide_df[model_col] = pd.Series(probas, index=index[: len(probas)])

    # save to CSV
    out_path = os.path.join(predictions_dir, 'all_model_predictions.csv')
    wide_df.to_csv(out_path, index=False)
    print(f"Saved GT as first column + one prob‐column per model to {out_path}")

def print_performance_summary(
    runs: list[ModelEvaluationRun],
    meta_model_names: set[str],
    splits=('Train', 'Test', 'Full'),
    thr_types=('cost', 'accuracy')
):
    """
    Print a summary of model performance for:
      • meta‐models (cost/accuracy‐optimized)
      • base models (preset threshold)
    
    Args:
      runs             : List of ModelEvaluationRun
      meta_model_names : Names of your meta‐models (e.g. MODELS.keys())
      splits           : Which splits to include
      thr_types        : Which optimized thresholds to print for meta‐models
    """
    # Header
    print("\n" + "="*80)
    print("SUMMARY OF MODEL PERFORMANCE")
    print("="*80)

    # Helper to fetch a result or None
    def _find(model, split, thr):
        for run in runs:
            if run.model_name != model:
                continue
            for r in run.results:
                if r.split == split and r.threshold_type == thr:
                    return r
        return None

    # --- Meta‐model summaries ---
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

    # --- Base‐model summaries ---
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

    # --- K-Fold Averaged Model Summaries ---
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

    SAVE_PLOTS = getattr(config, 'SAVE_PLOTS', True)
    SAVE_PREDICTIONS = getattr(config, 'SAVE_PREDICTIONS', True)
    SAVE_MODEL = getattr(config, 'SAVE_MODEL', True)
    SUMMARY = getattr(config, 'SUMMARY', True)

    # Define model zoo
    MODELS = config.MODELS
    
    # Load data
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Prepare data
    print("Preparing data...")
    X, y, feature_cols, encoded_cols = prepare_data(df, config)
    
    # Save feature columns for later use only if SAVE_MODEL is True
    if SAVE_MODEL:
        # Dynamically build encoding info for all one-hot encoded categoricals
        Save_Feature_Info(model_path, df, feature_cols, encoded_cols)

    # ===== Cross‑Validation vs Single‑Split =====
    if config.USE_KFOLD:
        cv_splits = CV_Split(config, X, y)
    else:
        X_train, y_train, train_idx, test_idx, single_splits = Regular_Split(config, X, y)
# =============================================

    # ──────────────────────────── CORE LOOP ────────────────────────────
    # Collect averaged-fold or single-split results for every model
    results_total = []

    for model_name, model in MODELS.items():
        print(f"\n{'-'*40}\nEvaluating {model_name}\n{'-'*40}")

        # ------------------------------------------------------------------
        # ❶ K-FOLD PATH  ─ gather per-fold results, keep *only* the average
        # ------------------------------------------------------------------
        if config.USE_KFOLD:
            fold_cost_results = []          # one cost-opt ModelEvaluationResult / fold
            fold_acc_results  = []          # one acc-opt  ModelEvaluationResult / fold
            fold_prob_arrays  = []          # raw probability array / fold

            for fold_name, (X_tr, y_tr), (X_te, y_te) in cv_splits:
                print(f"  Fold {fold_name}")

                split_preds, _ = train_and_evaluate_model(
                    model, X_tr, y_tr,
                    splits={fold_name: (X_te, y_te)},
                    model_name=None, model_dir=None, save_model=False
                )
                proba, y_eval = split_preds[fold_name]

                sweep, best_cost, best_acc = threshold_sweep_with_cost(
                    proba, y_eval, C_FP=C_FP, C_FN=C_FN
                )

                # optional per-fold sweep plot
                if SAVE_PLOTS:
                    plot_threshold_sweep(
                        sweep, C_FP, C_FN,
                        cost_optimal_thr=best_cost['threshold'],
                        accuracy_optimal_thr=best_acc['threshold'],
                        output_path=os.path.join(
                            image_path,
                            f"{model_name}_{fold_name.lower()}_threshold_sweep.png")
                    )

                # build result objects for this fold
                def _mk_result(best, thr_type):
                    return ModelEvaluationResult(
                        model_name=model_name,
                        split=fold_name,
                        threshold_type=thr_type,
                        threshold=best['threshold'],
                        precision=best['precision'],
                        recall=best['recall'],
                        accuracy=best['accuracy'],
                        cost=best['cost'],
                        tp=best['tp'], fp=best['fp'],
                        tn=best['tn'], fn=best['fn'],
                        is_base_model=False
                    )

                fold_cost_results.append(_mk_result(best_cost, 'cost'))
                fold_acc_results.append(_mk_result(best_acc,  'accuracy'))
                fold_prob_arrays.append(proba)

            # ----- average helper functions -----
            def _average_results(res_list):
                """Return a single ModelEvaluationResult averaged over folds."""
                fields = ['precision', 'recall', 'accuracy', 'cost',
                        'tp', 'fp', 'tn', 'fn']
                avg = {f: float(np.mean([getattr(r, f) for r in res_list]))
                    for f in fields}
                return ModelEvaluationResult(
                    model_name=f'kfold_avg_{model_name}',
                    split='Full',
                    threshold_type=res_list[0].threshold_type,
                    threshold=float(np.mean([r.threshold for r in res_list])),
                    is_base_model=False,
                    **avg
                )

            def _average_probabilities(prob_arrays):
                if not prob_arrays:
                    return None
                min_len = min(map(len, prob_arrays))
                return np.mean([p[:min_len] for p in prob_arrays], axis=0)

            avg_cost_res = _average_results(fold_cost_results)
            avg_acc_res  = _average_results(fold_acc_results)
            avg_probs    = _average_probabilities(fold_prob_arrays)

            # store *only* the averaged K-fold run
            results_total.append(
                ModelEvaluationRun(
                    model_name=f'kfold_avg_{model_name}',
                    results=[avg_cost_res, avg_acc_res],
                    probabilities={'Full': avg_probs} if avg_probs is not None else {}
                )
            )

            # still train a production model on all data, but **do not**
            # push its metrics into results_total
            train_and_evaluate_model(
                model, X, y,
                splits={'Full': (X, y)},
                model_name=f'kfold_final_{model_name}',
                model_dir=model_path,
                save_model=True
            )

        # ------------------------------------------------------------------
        # ❷ SINGLE-SPLIT PATH  ─ original logic unchanged
        # ------------------------------------------------------------------
        else:
            split_preds, _ = train_and_evaluate_model(
                model, X_train, y_train,
                splits=single_splits,
                model_name=f"meta_model_v2.1_{model_name}",
                model_dir=model_path,
                save_model=SAVE_MODEL
            )

            single_results = []
            single_probs   = {}

            for split_name, (proba, y_eval) in split_preds.items():
                sweep, best_cost, best_acc = threshold_sweep_with_cost(
                    proba, y_eval, C_FP=C_FP, C_FN=C_FN
                )

                if SAVE_PLOTS:
                    plot_threshold_sweep(
                        sweep, C_FP, C_FN,
                        cost_optimal_thr=best_cost['threshold'],
                        accuracy_optimal_thr=best_acc['threshold'],
                        output_path=os.path.join(
                            image_path,
                            f"{model_name}_{split_name.lower()}_threshold_sweep.png")
                    )

                def _mk_result(best, thr_type):
                    return ModelEvaluationResult(
                        model_name=model_name,
                        split=split_name,
                        threshold_type=thr_type,
                        threshold=best['threshold'],
                        precision=best['precision'],
                        recall=best['recall'],
                        accuracy=best['accuracy'],
                        cost=best['cost'],
                        tp=best['tp'], fp=best['fp'],
                        tn=best['tn'], fn=best['fn'],
                        is_base_model=False
                    )

                single_results.extend([
                    _mk_result(best_cost, 'cost'),
                    _mk_result(best_acc,  'accuracy')
                ])
                single_probs[split_name] = proba

            results_total.append(
                ModelEvaluationRun(
                    model_name=model_name,
                    results=single_results,
                    probabilities=single_probs
                )
            )
    # ─────────────────────── END OF CORE LOOP REPLACEMENT ───────────────────────


    # ========== STRUCTURED BASE MODEL METRICS SECTION ==========
    for base in ['AD_Decision', 'CL_Decision', 'AD_or_CL_Fail']:
        base_results = []
        base_probs   = {}

        if config.USE_KFOLD:
            split_names_and_indices = [('Full', df.index)]
        else:
            split_names_and_indices = [
                ('Train', train_idx),
                ('Test', test_idx),
                ('Full', df.index)
            ]

        for split_name, idx in split_names_and_indices:
            # 1) build y_pred for this base model + split
            if base == 'AD_or_CL_Fail':
                # fail if either AD or CL says "Bad"
                y_pred = (
                    (df.loc[idx, 'AD_Decision'] == 'Bad')
                  | (df.loc[idx, 'CL_Decision'] == 'Bad')
                ).astype(int).values
            else:
                y_pred = (df.loc[idx, base] == 'Bad').astype(int).values

            # 2) ground truth
            y_true = y.loc[idx].values

            # 3) compute metrics
            mets = compute_metrics(y_true, y_pred, C_FP=C_FP, C_FN=C_FN)

            # 4) stash probabilities & results
            base_probs[split_name] = y_pred
            base_results.append(
                ModelEvaluationResult(
                    model_name     = base,
                    split          = split_name,
                    threshold_type = 'base',
                    threshold      = None,
                    precision      = mets['precision'],
                    recall         = mets['recall'],
                    accuracy       = mets['accuracy'],
                    cost           = mets['cost'],
                    tp             = mets['tp'],
                    fp             = mets['fp'],
                    tn             = mets['tn'],
                    fn             = mets['fn'],
                    is_base_model  = True
                )
            )

        # 5) add one run per base model
        results_total.append(
            ModelEvaluationRun(
                model_name   = base,
                results      = base_results,
                probabilities= base_probs
            )
        )
    # ========== END STRUCTURED BASE MODEL METRICS SECTION ==========
    
    if SUMMARY:
        print_performance_summary(
        runs=results_total,
        meta_model_names=set(MODELS.keys())
        )

    if SAVE_PREDICTIONS:
        predictions_dir = getattr(config, 'PREDICTIONS_DIR', os.path.dirname(model_path))
        save_all_model_probabilities_from_structure(results_total, predictions_dir, df.index, y_eval)

    if SAVE_PLOTS:
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

def Save_Feature_Info(model_path, df, feature_cols, encoded_cols):
    encoding_info = {}
        # Identify which columns in df were categorical and one-hot encoded
    for col in df.columns:
        if col not in feature_cols and col in df.select_dtypes(include=['object', 'category', 'bool']).columns:
            encoding_info[col] = {
                    'type': 'one_hot',
                    'original_column': col,
                    'categories': sorted(df[col].unique().tolist())
                }
    feature_info = {
            'feature_cols': feature_cols,  # Original numeric features
            'encoded_cols': encoded_cols,  # One-hot encoded columns
            'encoding': encoding_info
        }
    joblib.dump(feature_info, os.path.join(model_path, "feature_info.pkl"))

def Regular_Split(config, X, y):
    print("Running single train/test split…")
    X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.20, random_state=getattr(config,'RANDOM_STATE',42),
            stratify=y
        )
    train_idx = X_train.index
    test_idx  = X_test.index
        # Single-split "splits" dict mirrors your old code
    single_splits = {
            'Train': (X_train, y_train),
            'Test':  (X_test,  y_test),
            'Full':  (X,       y)
        }

    return X_train,y_train,train_idx,test_idx,single_splits

def CV_Split(config, X, y):
    print(f"Running {config.N_SPLITS}‑fold cross‑validation…")
    kf = StratifiedKFold(
            n_splits = config.N_SPLITS,
            shuffle  = True,
            random_state = getattr(config, 'RANDOM_STATE', 42)
        )
        # KFold logic does not use train_idx/test_idx here
    cv_splits = []
    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X, y), start=1):
        print(f"  Fold {fold_idx}/{config.N_SPLITS}")
        X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
        X_te, y_te = X.iloc[test_idx],  y.iloc[test_idx]
        cv_splits.append( (f"Fold{fold_idx}", (X_tr, y_tr), (X_te, y_te)) )
    return cv_splits

if __name__ == "__main__":
    if len(sys.argv) > 1:
        main(config)
    else:
        main(config)