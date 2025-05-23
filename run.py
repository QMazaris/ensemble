# Additions that could be made to improve the models:
# Refactoring

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
from sklearn.base import clone
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

# Import from helpers package
from helpers import (
    # Data preparation
    prepare_data,
    apply_variance_filter,
    apply_correlation_filter,
    Regular_Split,
    CV_Split,
    Save_Feature_Info,
    
    # Modeling
    optimize_hyperparams,
    train_and_evaluate_model,
    
    # Metrics and evaluation
    ModelEvaluationResult,
    ModelEvaluationRun,
    compute_metrics,
    threshold_sweep_with_cost,
    _mk_result,
    _average_results,
    _average_probabilities,
    
    # Plotting
    plot_threshold_sweep,
    plot_runs_at_threshold,
    
    # Reporting
    print_performance_summary,
    save_all_model_probabilities_from_structure,
    
    # Utils
    create_directories
)

# Explicit import for FinalModelCreateAndAnalyize
from helpers.modeling import FinalModelCreateAndAnalyize

# ---------- Main Function ----------
def main(config):
    """Main function to run the entire pipeline.
    
    Args:
        config: Configuration object containing paths and parameters
    """
    # Create output directories
    create_directories(config.dirs_to_create, summary=config.SUMMARY)

    # Unpack config
    csv_path = config.DATA_PATH
    model_path = config.MODEL_DIR
    image_path = config.PLOT_DIR

    C_FP = config.C_FP
    C_FN = config.C_FN

    # raw‑output lookup
    base_cols = config.BASE_MODEL_OUTPUT_COLUMNS

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

    # Collect averaged-fold or single-split results for every model
    model_zoo_runs = []
    base_model_runs = []

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
            # Add arrays for AD/CL results
            fold_cost_ad, fold_acc_ad, fold_probs_ad = [], [], []
            fold_cost_cl, fold_acc_cl, fold_probs_cl = [], [], []

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

                # ── Threshold sweep for meta model ──
                sweep, bc, ba = threshold_sweep_with_cost(
                    proba, y_eval, C_FP=C_FP, C_FN=C_FN,
                    SUMMARY=config.SUMMARY, split_name=f"Fold{fold_idx}"
                )

                # ── Collect fold results for meta model ──
                fold_cost.append(_mk_result(model_name, f"Fold{fold_idx}", bc, 'cost'))
                fold_acc.append(_mk_result(model_name, f"Fold{fold_idx}", ba, 'accuracy'))
                fold_probs.append(proba)

                # ── Base‑model threshold sweeps ──
                # pull raw scores from df via config
                ad_scores = df.loc[te_idx, base_cols["AD"]].values
                cl_scores = df.loc[te_idx, base_cols["CL"]].values

                sweep_ad, bc_ad, ba_ad = threshold_sweep_with_cost(
                    ad_scores, y_te, C_FP, C_FN,
                    SUMMARY=config.SUMMARY, split_name=f"AD_Fold{fold_idx}"
                )
                sweep_cl, bc_cl, ba_cl = threshold_sweep_with_cost(
                    cl_scores, y_te, C_FP, C_FN,
                    SUMMARY=config.SUMMARY, split_name=f"CL_Fold{fold_idx}"
                )

                # optional: plot them
                if SAVE_PLOTS:
                    plot_threshold_sweep(sweep_ad, C_FP, C_FN,
                                        cost_optimal_thr=bc_ad["threshold"],
                                        accuracy_optimal_thr=ba_ad["threshold"],
                                        output_path=os.path.join(
                                            image_path, f"AD_fold{fold_idx}_threshold_sweep.png"),
                                        SUMMARY=config.SUMMARY)
                    plot_threshold_sweep(sweep_cl, C_FP, C_FN,
                                        cost_optimal_thr=bc_cl["threshold"],
                                        accuracy_optimal_thr=ba_cl["threshold"],
                                        output_path=os.path.join(
                                            image_path, f"CL_fold{fold_idx}_threshold_sweep.png"),
                                        SUMMARY=config.SUMMARY)

                # stash their results/probs in parallel arrays
                fold_cost_ad.append(_mk_result("AD", f"Fold{fold_idx}", bc_ad, 'cost'))
                fold_acc_ad.append(_mk_result("AD", f"Fold{fold_idx}", ba_ad, 'accuracy'))
                fold_probs_ad.append(ad_scores)

                fold_cost_cl.append(_mk_result("CL", f"Fold{fold_idx}", bc_cl, 'cost'))
                fold_acc_cl.append(_mk_result("CL", f"Fold{fold_idx}", ba_cl, 'accuracy'))
                fold_probs_cl.append(cl_scores)

                # Optional per-fold sweep plot for meta model
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

            # ── Average & store meta model results ──
            avg_c = _average_results(fold_cost, model_name)
            avg_a = _average_results(fold_acc, model_name)
            avg_p = _average_probabilities(fold_probs)
            model_zoo_runs.append(
                ModelEvaluationRun(
                    model_name=f'kfold_avg_{model_name}',
                    results=[avg_c, avg_a],
                    probabilities={'Full': avg_p} if avg_p is not None else {}
                )
            )

        # ── Now do the same for AD and CL ──
        # Calculate the kfold averaged AD/CL results.
        # These will be included in base_model_runs if config.USE_KFOLD is True.
        avg_c_ad = _average_results(fold_cost_ad, "AD")
        avg_a_ad = _average_results(fold_acc_ad, "AD")
        avg_p_ad = _average_probabilities(fold_probs_ad)
        kfold_avg_ad_run = ModelEvaluationRun(
            model_name='kfold_avg_AD',
            results=[avg_c_ad, avg_a_ad],
            probabilities={'Full': avg_p_ad} if avg_p_ad is not None else {}
        )

        avg_c_cl = _average_results(fold_cost_cl, "CL")
        avg_a_cl = _average_results(fold_acc_cl, "CL")
        avg_p_cl = _average_probabilities(fold_probs_cl)
        kfold_avg_cl_run = ModelEvaluationRun(
            model_name='kfold_avg_CL',
            results=[avg_c_cl, avg_a_cl],
            probabilities={'Full': avg_p_cl} if avg_p_cl is not None else {}
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

            model_zoo_runs.append(
                ModelEvaluationRun(
                    model_name=model_name,
                    results=single_results,
                    probabilities=single_probs
                )
            )

        # Add base model threshold optimization for single-split mode
        if not config.USE_KFOLD:
            # Process AD and CL scores for single split
            base_model_runs = []
            
            # Get scores for train and test sets
            ad_scores_train = df.loc[train_idx, base_cols["AD"]].values
            cl_scores_train = df.loc[train_idx, base_cols["CL"]].values
            ad_scores_test = df.loc[test_idx, base_cols["AD"]].values
            cl_scores_test = df.loc[test_idx, base_cols["CL"]].values
            
            # Get corresponding y values
            y_train_values = y.loc[train_idx].values
            y_test_values = y.loc[test_idx].values
            y_full_values = y.loc[df.index].values
            
            # Process AD scores
            for split_name, (scores, y_true) in [
                ('Train', (ad_scores_train, y_train_values)),
                ('Test', (ad_scores_test, y_test_values)),
                ('Full', (df.loc[df.index, base_cols["AD"]].values, y_full_values))
            ]:
                sweep_ad, bc_ad, ba_ad = threshold_sweep_with_cost(
                    scores, y_true, C_FP, C_FN,
                    SUMMARY=config.SUMMARY, split_name=f"AD_{split_name}"
                )
                
                if SAVE_PLOTS:
                    plot_threshold_sweep(
                        sweep_ad, C_FP, C_FN,
                        cost_optimal_thr=bc_ad["threshold"],
                        accuracy_optimal_thr=ba_ad["threshold"],
                        output_path=os.path.join(
                            image_path, f"AD_{split_name.lower()}_threshold_sweep.png"),
                        SUMMARY=config.SUMMARY
                    )
                
                base_model_runs.append(
                    ModelEvaluationRun(
                        model_name='AD',
                        results=[
                            _mk_result('AD', split_name, bc_ad, 'cost'),
                            _mk_result('AD', split_name, ba_ad, 'accuracy')
                        ],
                        probabilities={split_name: scores}
                    )
                )
            
            # Process CL scores
            for split_name, (scores, y_true) in [
                ('Train', (cl_scores_train, y_train_values)),
                ('Test', (cl_scores_test, y_test_values)),
                ('Full', (df.loc[df.index, base_cols["CL"]].values, y_full_values))
            ]:
                sweep_cl, bc_cl, ba_cl = threshold_sweep_with_cost(
                    scores, y_true, C_FP, C_FN,
                    SUMMARY=config.SUMMARY, split_name=f"CL_{split_name}"
                )
                
                if SAVE_PLOTS:
                    plot_threshold_sweep(
                        sweep_cl, C_FP, C_FN,
                        cost_optimal_thr=bc_cl["threshold"],
                        accuracy_optimal_thr=ba_cl["threshold"],
                        output_path=os.path.join(
                            image_path, f"CL_{split_name.lower()}_threshold_sweep.png"),
                        SUMMARY=config.SUMMARY
                    )
                
                base_model_runs.append(
                    ModelEvaluationRun(
                        model_name='CL',
                        results=[
                            _mk_result('CL', split_name, bc_cl, 'cost'),
                            _mk_result('CL', split_name, ba_cl, 'accuracy')
                        ],
                        probabilities={split_name: scores}
                    )
                )

    # ========== STRUCTURED BASE MODEL METRICS SECTION ==========
    # This section calculates results for the original base models (AD_Decision, CL_Decision, AD_or_CL_Fail).
    # These runs will always be created here, and added to base_model_runs below.
    structured_base_model_runs = []
    base_models_to_process = ['AD_Decision', 'CL_Decision', 'AD_or_CL_Fail']

    for base in base_models_to_process:
        base_results = []
        base_probs   = {}

        if config.USE_KFOLD:
            # When in k-fold mode, use the full data for structured base models
            split_names_and_indices = [('Full', df.index)]
        else:
            # When in single-split mode, use train, test, and full data
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
            # Need to get the corresponding y values for the current split index
            y_true = y.loc[idx].values

            # 3) compute metrics
            mets = compute_metrics(y_true, y_pred, C_FP=C_FP, C_FN=C_FN)

            # 4) stash probabilities & results
            # For base models based on decisions ('Bad'/'Good'), probability is just the prediction itself (0 or 1)
            base_probs[split_name] = y_pred
            base_results.append(
                ModelEvaluationResult(
                    model_name     = base,
                    split          = split_name,
                    threshold_type = 'base',
                    threshold      = None, # No threshold concept for decision-based base models
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

        # Add the structured base model run to a temporary list
        structured_base_model_runs.append(
            ModelEvaluationRun(
                model_name   = base,
                results      = base_results,
                probabilities= base_probs
            )
        )

    # ========== END STRUCTURED BASE MODEL METRICS SECTION ==========

    # Combine model zoo and base model runs for final reporting and plotting
    # Determine the order of base models based on config.USE_KFOLD
    if config.USE_KFOLD:
        # K-fold mode: kfold_avg_AD, kfold_avg_CL, then structured base models
        base_model_runs.append(kfold_avg_ad_run)
        base_model_runs.append(kfold_avg_cl_run)
        base_model_runs.extend(structured_base_model_runs)
    else:
        # Single-split mode: only structured base models
        base_model_runs.extend(structured_base_model_runs)

    results_total = model_zoo_runs + base_model_runs

    if config.SUMMARY:
        print_performance_summary(
        runs=results_total,
        meta_model_names=set(MODELS.keys())
        )

    if SAVE_PREDICTIONS:
        predictions_dir = getattr(config, 'PREDICTIONS_DIR', os.path.dirname(model_path))
        # Need to determine which set of y_eval to use based on k-fold or single split
        if config.USE_KFOLD:
             # In k-fold, y_eval is from the last fold's test set. This might not be ideal
             # for saving predictions for the 'Full' dataset. We need the true y for the full dataset.
             y_full = y.loc[df.index].values # Get true y for the full dataset
             save_all_model_probabilities_from_structure(results_total, predictions_dir, df.index, y_full, SUMMARY = config.SUMMARY)
        else:
             # In single-split, y_eval from the loop is the test set y. For full dataset, use y.loc[df.index]
             y_full = y.loc[df.index].values # Get true y for the full dataset
             save_all_model_probabilities_from_structure(results_total, predictions_dir, df.index, y_full, SUMMARY = config.SUMMARY)


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

    # ========== FINAL PRODUCTION MODELS SECTION ==========
    # When using k-fold, the final model should be trained on the full data after hyperparameter tuning (if enabled)
    # based on the average performance across folds.
    # When using single-split, the final model is trained on the training split and evaluated on the test split.
    FinalModelCreateAndAnalyize(config, model_path, image_path, C_FP, C_FN, SAVE_PLOTS, X, y)

    print("\nPipeline complete")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        main(config)
    else:
        main(config)