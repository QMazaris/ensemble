#!/usr/bin/env python3
# train_weld_meta_model_v2.1.py
#
# Usage:
#   pip install pandas scikit-learn joblib matplotlib
#   python train_weld_meta_model_v2.1.py "C:/.../ensemble_resultsV2.1.csv"

# Additions that could be made to improve the models:
# 2. use grid search to find the best hyperparameters
# 4. Make the comparison to the other models more fair


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
from sklearn.model_selection import RandomizedSearchCV
from sklearn.base import clone


from helpers import (
    ModelEvaluationResult, ModelEvaluationRun,
    prepare_data, train_and_evaluate_model, compute_metrics, threshold_sweep_with_cost,
    plot_threshold_sweep, plot_runs_at_threshold, save_all_model_probabilities_from_structure,
    print_performance_summary, Save_Feature_Info, Regular_Split, CV_Split,
    _mk_result, _average_results, _average_probabilities, apply_variance_filter, apply_correlation_filter,
    optimize_hyperparams
)

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
    if config.SUMMARY:
        print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Prepare data
    if config.SUMMARY:
        print("Preparing data...")
    X, y, feature_cols, encoded_cols = prepare_data(df, config)

    if config.FilterData:
        # Apply 1.1: Variance Filter
        X = apply_variance_filter(X, threshold=config.VARIANCE_THRESH, SUMMARY=config.SUMMARY)

        # Apply 1.2: Correlation Filter
        X = apply_correlation_filter(X, threshold=config.CORRELATION_THRESH, SUMMARY=config.SUMMARY)

        if config.SUMMARY:
            print("📋 Feature Names:")
            for i, col in enumerate(X.columns, 1):
                print(f"  {i:2d}. {col}")


    
    # Save feature columns for later use only if SAVE_MODEL is True
    if SAVE_MODEL:
        Save_Feature_Info(model_path, df, feature_cols, encoded_cols)

    # 1) Only one split for single-split path
    if not config.USE_KFOLD:
        X_train, y_train, train_idx, test_idx, single_splits = Regular_Split(config, X, y)
        # Do single-split tuning here if requested
        if config.OPTIMIZE_HYPERPARAMS and config.SUMMARY:
            print("🔍 Optimizing hyperparameters for single-split evaluation…")
        if config.OPTIMIZE_HYPERPARAMS:
            for name, mdl in MODELS.items():
                MODELS[name] = optimize_hyperparams(name, mdl, X_train, y_train, config)

    # ──────────────────────────── CORE LOOP ────────────────────────────
    # Collect averaged-fold or single-split results for every model
    results_total = []

    if config.USE_KFOLD:
        outer_cv = StratifiedKFold(
            n_splits     = config.N_SPLITS,
            shuffle      = True,
            random_state = config.RANDOM_STATE
        )

        for model_name, prototype in MODELS.items():
            if config.SUMMARY:
                print(f"\n{'='*40}\nNested‑CV → {model_name}\n{'='*40}")

            fold_cost, fold_acc, fold_probs = [], [], []

            # ── Outer folds ──
            for fold_idx, (tr_idx, te_idx) in enumerate(outer_cv.split(X, y), start=1):
                if config.SUMMARY:
                    print(f"  ➤ Fold {fold_idx}")

                X_tr, y_tr = X.iloc[tr_idx], y.iloc[tr_idx]
                X_te, y_te = X.iloc[te_idx], y.iloc[te_idx]

                # ── Inner tuning ──
                if config.OPTIMIZE_HYPERPARAMS:
                    tuned = optimize_hyperparams(
                        model_name,
                        clone(prototype),
                        X_tr, y_tr,
                        config
                    )
                else:
                    tuned = clone(prototype)  # Use untuned model if optimization is disabled

                # ── Train & predict this fold ──
                split_preds, _ = train_and_evaluate_model(
                    tuned,
                    X_tr, y_tr,
                    splits={f"Fold{fold_idx}": (X_te, y_te)},
                    model_name=None,
                    model_dir=None,
                    save_model=False,
                    SUMMARY=config.SUMMARY
                )
                proba, y_eval = split_preds[f"Fold{fold_idx}"]

                # ── Threshold sweep ──
                sweep, bc, ba = threshold_sweep_with_cost(
                    proba, y_eval, C_FP=C_FP, C_FN=C_FN,
                    SUMMARY=config.SUMMARY, split_name=f"Fold{fold_idx}"
                )

                # ── Collect fold results ──
                fold_cost.append(_mk_result(model_name, f"Fold{fold_idx}", bc, 'cost'))
                fold_acc.append(_mk_result(model_name, f"Fold{fold_idx}", ba, 'accuracy'))
                fold_probs.append(proba)

                # Optional per-fold sweep plot
                if SAVE_PLOTS:
                    plot_threshold_sweep(
                        sweep, C_FP, C_FN,
                        cost_optimal_thr=bc['threshold'],
                        accuracy_optimal_thr=ba['threshold'],
                        output_path=os.path.join(
                            image_path,
                            f"{model_name}_fold{fold_idx}_threshold_sweep.png"),
                        SUMMARY=config.SUMMARY
                    )

            # ── Average & store ──
            avg_c = _average_results(fold_cost, model_name)
            avg_a = _average_results(fold_acc, model_name)
            avg_p = _average_probabilities(fold_probs)
            results_total.append(
                ModelEvaluationRun(
                    model_name=f'kfold_avg_{model_name}',
                    results=[avg_c, avg_a],
                    probabilities={'Full': avg_p} if avg_p is not None else {}
                )
            )

            # ── Final production model on all data ──
            train_and_evaluate_model(
                tuned, X, y,
                splits={'Full': (X, y)},
                model_name=f'kfold_final_{model_name}',
                model_dir=model_path,
                save_model=True,
                SUMMARY=config.SUMMARY
            )
    else:
        # Single-split path: use the already split data and tuned models
        for model_name, model in MODELS.items():
            split_preds, _ = train_and_evaluate_model(
                model, X_train, y_train,
                splits=single_splits,
                model_name=f"meta_model_v2.1_{model_name}",
                model_dir=model_path,
                save_model=SAVE_MODEL,
                SUMMARY=config.SUMMARY
            )

            single_results = []
            single_probs = {}

            for split_name, (proba, y_eval) in split_preds.items():
                sweep, best_cost, best_acc = threshold_sweep_with_cost(
                    proba, y_eval, C_FP=C_FP, C_FN=C_FN, 
                    SUMMARY=config.SUMMARY, split_name=split_name
                )

                if SAVE_PLOTS:
                    plot_threshold_sweep(
                        sweep, C_FP, C_FN,
                        cost_optimal_thr=best_cost['threshold'],
                        accuracy_optimal_thr=best_acc['threshold'],
                        output_path=os.path.join(
                            image_path,
                            f"{model_name}_{split_name.lower()}_threshold_sweep.png"),
                        SUMMARY=config.SUMMARY
                    )

                single_results.extend([
                    _mk_result(model_name, split_name, best_cost, 'cost'),
                    _mk_result(model_name, split_name, best_acc, 'accuracy')
                ])
                single_probs[split_name] = proba

            results_total.append(
                ModelEvaluationRun(
                    model_name=model_name,
                    results=single_results,
                    probabilities=single_probs
                )
            )

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
    
    if config.SUMMARY:
        print_performance_summary(
        runs=results_total,
        meta_model_names=set(MODELS.keys())
        )

    if SAVE_PREDICTIONS:
        predictions_dir = getattr(config, 'PREDICTIONS_DIR', os.path.dirname(model_path))
        save_all_model_probabilities_from_structure(results_total, predictions_dir, df.index, y_eval, SUMMARY = config.SUMMARY)

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

        # Save the final model
        if config.OPTIMIZE_FINAL_MODEL and config.SAVE_MODEL:
            if config.SUMMARY:
                print("🏭 Final hyperparameter tuning on the full dataset…")
            for name, prototype in config.MODELS.items():
                base = clone(prototype)              # fresh instance
                best = optimize_hyperparams(name, base, X, y, config)
                out_path = os.path.join(model_path, f"{name}_production.pkl")
                joblib.dump(best, out_path)
                if config.SUMMARY:
                    print(f"✅ Saved production model: {out_path}")

        
        print("Pipeline complete")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        main(config)
    else:
        main(config)