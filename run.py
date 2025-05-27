# Additions that could be made to improve the models:
# Small Quirk: doesnt fill the AD or CL structures properly when k fold is off

# Standard library imports
import os
import sys
from pathlib import Path

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
    plot_class_balance,
    
    # Reporting
    print_performance_summary,
    save_all_model_probabilities_from_structure,
    
    # Utils
    create_directories,
    
    # Streamlit exports
    export_metrics_for_streamlit
)

# Import model export functionality
from helpers.model_export import export_model

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
    
    
    df, X, y = Initalize(config, SAVE_MODEL)

    # Plot class balance if SAVE_PLOTS is enabled
    if SAVE_PLOTS:
        plot_class_balance(
            y,
            output_path=os.path.join(config.PLOT_DIR, 'class_balance.png'),
            SUMMARY=SUMMARY
        )

    # 1) Only one split for single-split path
    X_train, y_train, train_idx, test_idx, single_splits = Split_No_KFold(config, MODELS, X, y)

    # Collect averaged-fold or single-split results for every model
    model_zoo_runs = []
    base_model_runs = []

    if config.USE_KFOLD:
        # K-fold path
        kfold_avg_ad_run, kfold_avg_cl_run = Core_KFold(config, C_FP, C_FN, base_cols, SAVE_PLOTS, MODELS, df, X, y, model_zoo_runs)
        # In K-fold, these are the averaged base model runs
        base_model_runs.append(kfold_avg_ad_run)
        base_model_runs.append(kfold_avg_cl_run)
    else:
        # Single-split path
        if not config.USE_KFOLD:
            X_train, y_train, train_idx, test_idx, single_splits = Split_No_KFold(config, MODELS, X, y)
            base_model_runs = Core_Standard(config, C_FP, C_FN, base_cols, SAVE_PLOTS, SAVE_MODEL, MODELS, df, y, X_train, y_train, train_idx, test_idx, single_splits, model_zoo_runs)

    # ========== STRUCTURED BASE MODEL METRICS SECTION ==========
    # This section calculates results for the original base models (AD_Decision, CL_Decision, AD_or_CL_Fail).
    # These runs will always be created here, and added to base_model_runs below.
    structured_base_model_runs = []
    base_models_to_process = ['AD_Decision', 'CL_Decision', 'AD_or_CL_Fail']

    Legacy_Base(config, C_FP, C_FN, df, y, train_idx, test_idx, structured_base_model_runs, base_models_to_process)


    base_model_runs.extend(structured_base_model_runs)

    # Make the final structure
    results_total = model_zoo_runs + base_model_runs

    if config.SUMMARY:
        print_performance_summary(
        runs=results_total,
        meta_model_names=set(MODELS.keys())
        )

    if SAVE_PREDICTIONS:
        y_full = y.loc[df.index].values # Get true y for the full dataset
        save_all_model_probabilities_from_structure(results_total, config.PREDICTIONS_DIR, df.index, y_full, SUMMARY = config.SUMMARY)
      

    if SAVE_PLOTS:
        # Use Test split for single split mode, Full split for k-fold mode
        comparison_split = 'Full' if config.USE_KFOLD else 'Test'
        
        plot_runs_at_threshold(
            runs=results_total,
            threshold_type='cost',
            split_name=comparison_split,
            C_FP=C_FP,
            C_FN=C_FN,
            output_path=os.path.join(config.PLOT_DIR, 'model_comparison_cost_optimized.png')
        )
        plot_runs_at_threshold(
            runs=results_total,
            threshold_type='accuracy',
            split_name=comparison_split,
            C_FP=C_FP,
            C_FN=C_FN,
            output_path=os.path.join(config.PLOT_DIR, 'model_comparison_accuracy_optimized.png')
        )

    # ========== FINAL PRODUCTION MODELS SECTION ==========
    # When using k-fold, the final model should be trained on the full data after hyperparameter tuning (if enabled)
    # based on the average performance across folds.
    # When using single-split, the final model is trained on the training split and evaluated on the test split.
    FinalModelCreateAndAnalyize(config, config.MODEL_DIR, config.PLOT_DIR, C_FP, C_FN, SAVE_PLOTS, X, y)

    

    print("\nPipeline complete")


    streamlit_output_dir = Path("output") / "streamlit_data"
    export_metrics_for_streamlit(results_total, streamlit_output_dir, meta_model_names=set(MODELS.keys()))

def Legacy_Base(config, C_FP, C_FN, df, y, train_idx, test_idx, structured_base_model_runs, base_models_to_process):
    """Process legacy base models (AD_Decision, CL_Decision, AD_or_CL_Fail) with consistent cost calculation.
    
    When k-fold is off:
        - Calculate costs on train and test splits separately
        - Full split cost is the sum of train and test costs
    When k-fold is on:
        - Calculate cost on full dataset
        - Divide by N_SPLITS to make it comparable to k-fold averaged costs
    """
    for base in base_models_to_process:
        base_results = []
        base_probs = {}
        
        # Get the raw decisions for this base model and convert to binary (1/0)
        if base == 'AD_or_CL_Fail':
            # fail if either AD or CL says "Bad"
            decisions = (
                (df['AD_Decision'] == 'Bad') | 
                (df['CL_Decision'] == 'Bad')
            ).astype(int).values
        else:
            # Convert 'Bad'/'Good' to 1/0
            decisions = (df[base] == 'Bad').astype(int).values
            
        y_true = y.values
        
        if config.USE_KFOLD:
            # For k-fold, calculate on full dataset but divide cost by N_SPLITS
            # to make it comparable to k-fold averaged costs
            tn, fp, fn, tp = confusion_matrix(y_true, decisions).ravel()
            cost = (C_FP * fp + C_FN * fn) / config.N_SPLITS  # Divide by number of folds
            metrics = compute_metrics(y_true, decisions, C_FP, C_FN)
            metrics.update({
                'cost': cost,
                'tp': tp,
                'fp': fp,
                'tn': tn,
                'fn': fn
            })
            
            base_results.append(
                ModelEvaluationResult(
                    model_name=base,
                    split='Full',  # Only use Full split for k-fold
                    threshold_type='base',
                    threshold=None,  # No threshold concept for decision-based base models
                    precision=metrics['precision'],
                    recall=metrics['recall'],
                    accuracy=metrics['accuracy'],
                    cost=cost,
                    tp=tp,
                    fp=fp,
                    tn=tn,
                    fn=fn,
                    is_base_model=True
                )
            )
            base_probs['Full'] = decisions
            
        else:
            # For single split, calculate costs separately for train and test
            for split_name, idx in [('Train', train_idx), ('Test', test_idx)]:
                split_decisions = decisions[idx]
                split_y = y_true[idx]
                
                tn, fp, fn, tp = confusion_matrix(split_y, split_decisions).ravel()
                cost = C_FP * fp + C_FN * fn
                metrics = compute_metrics(split_y, split_decisions, C_FP, C_FN)
                metrics.update({
                    'cost': cost,
                    'tp': tp,
                    'fp': fp,
                    'tn': tn,
                    'fn': fn
                })
                
                base_results.append(
                    ModelEvaluationResult(
                        model_name=base,
                        split=split_name,
                        threshold_type='base',
                        threshold=None,  # No threshold concept for decision-based base models
                        precision=metrics['precision'],
                        recall=metrics['recall'],
                        accuracy=metrics['accuracy'],
                        cost=cost,
                        tp=tp,
                        fp=fp,
                        tn=tn,
                        fn=fn,
                        is_base_model=True
                    )
                )
                base_probs[split_name] = split_decisions
            
            # Calculate Full split as sum of train and test costs
            full_cost = sum(r.cost for r in base_results)
            full_tp = sum(r.tp for r in base_results)
            full_fp = sum(r.fp for r in base_results)
            full_tn = sum(r.tn for r in base_results)
            full_fn = sum(r.fn for r in base_results)
            
            # Calculate metrics for full dataset
            full_metrics = compute_metrics(y_true, decisions, C_FP, C_FN)
            base_results.append(
                ModelEvaluationResult(
                    model_name=base,
                    split='Full',
                    threshold_type='base',
                    threshold=None,
                    precision=full_metrics['precision'],
                    recall=full_metrics['recall'],
                    accuracy=full_metrics['accuracy'],
                    cost=full_cost,
                    tp=full_tp,
                    fp=full_fp,
                    tn=full_tn,
                    fn=full_fn,
                    is_base_model=True
                )
            )
            base_probs['Full'] = decisions

        # Add the structured base model run
        structured_base_model_runs.append(
            ModelEvaluationRun(
                model_name=base,
                results=base_results,
                probabilities=base_probs
            )
        )

def Core_Standard(config, C_FP, C_FN, base_cols, SAVE_PLOTS, SAVE_MODEL, MODELS, df, y, X_train, y_train, train_idx, test_idx, single_splits, model_zoo_runs):
    for model_name, model in MODELS.items():
        split_preds, _ = train_and_evaluate_model(
                model, X_train, y_train,
                splits=single_splits,
                model_name=f"meta_model_v2.1_{model_name}",
                model_dir=config.MODEL_DIR,
                save_model=SAVE_MODEL,
                SUMMARY=config.SUMMARY,
                config=config
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
                            config.PLOT_DIR,
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

    # Process AD and CL scores for single split
    base_model_runs = []
    
    # Get scores for all splits
    ad_scores = {
        'Train': df.loc[train_idx, base_cols["AD"]].values,
        'Test': df.loc[test_idx, base_cols["AD"]].values,
        'Full': df.loc[df.index, base_cols["AD"]].values
    }
    
    cl_scores = {
        'Train': df.loc[train_idx, base_cols["CL"]].values,
        'Test': df.loc[test_idx, base_cols["CL"]].values,
        'Full': df.loc[df.index, base_cols["CL"]].values
    }
    
    # Get corresponding y values
    y_values = {
        'Train': y.loc[train_idx].values,
        'Test': y.loc[test_idx].values,
        'Full': y.loc[df.index].values
    }
    
    # Process AD scores
    ad_results = []
    ad_probs = {}
    
    for split_name in ['Train', 'Test', 'Full']:
        sweep_ad, bc_ad, ba_ad = threshold_sweep_with_cost(
            ad_scores[split_name], y_values[split_name], C_FP, C_FN,
            SUMMARY=config.SUMMARY, split_name=f"AD_{split_name}"
        )
        
        if SAVE_PLOTS:
            plot_threshold_sweep(
                sweep_ad, C_FP, C_FN,
                cost_optimal_thr=bc_ad["threshold"],
                accuracy_optimal_thr=ba_ad["threshold"],
                output_path=os.path.join(
                    config.PLOT_DIR, f"AD_{split_name.lower()}_threshold_sweep.png"),
                SUMMARY=config.SUMMARY
            )
        
        ad_results.extend([
            _mk_result('AD', split_name, bc_ad, 'cost'),
            _mk_result('AD', split_name, ba_ad, 'accuracy')
        ])
        ad_probs[split_name] = ad_scores[split_name]
    
    base_model_runs.append(
        ModelEvaluationRun(
            model_name='AD',
            results=ad_results,
            probabilities=ad_probs
        )
    )
    
    # Process CL scores
    cl_results = []
    cl_probs = {}
    
    for split_name in ['Train', 'Test', 'Full']:
        sweep_cl, bc_cl, ba_cl = threshold_sweep_with_cost(
            cl_scores[split_name], y_values[split_name], C_FP, C_FN,
            SUMMARY=config.SUMMARY, split_name=f"CL_{split_name}"
        )
        
        if SAVE_PLOTS:
            plot_threshold_sweep(
                sweep_cl, C_FP, C_FN,
                cost_optimal_thr=bc_cl["threshold"],
                accuracy_optimal_thr=ba_cl["threshold"],
                output_path=os.path.join(
                    config.PLOT_DIR, f"CL_{split_name.lower()}_threshold_sweep.png"),
                SUMMARY=config.SUMMARY
            )
        
        cl_results.extend([
            _mk_result('CL', split_name, bc_cl, 'cost'),
            _mk_result('CL', split_name, ba_cl, 'accuracy')
        ])
        cl_probs[split_name] = cl_scores[split_name]
    
    base_model_runs.append(
        ModelEvaluationRun(
            model_name='CL',
            results=cl_results,
            probabilities=cl_probs
        )
    )
    
    return base_model_runs

def Core_KFold(config, C_FP, C_FN, base_cols, SAVE_PLOTS, MODELS, df, X, y, model_zoo_runs):
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
                    SUMMARY=config.SUMMARY,
                    config=config
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
                                            config.PLOT_DIR, f"AD_fold{fold_idx}_threshold_sweep.png"),
                                        SUMMARY=config.SUMMARY)
                plot_threshold_sweep(sweep_cl, C_FP, C_FN,
                                        cost_optimal_thr=bc_cl["threshold"],
                                        accuracy_optimal_thr=ba_cl["threshold"],
                                        output_path=os.path.join(
                                            config.PLOT_DIR, f"CL_fold{fold_idx}_threshold_sweep.png"),
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
                            config.PLOT_DIR,
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
    
    return kfold_avg_ad_run,kfold_avg_cl_run

def Split_No_KFold(config, MODELS, X, y):
    X_train, y_train, train_idx, test_idx, single_splits = Regular_Split(config, X, y)
    # Do single-split tuning here if requested
    if config.OPTIMIZE_HYPERPARAMS and config.SUMMARY:
        print("🔍 Optimizing hyperparameters for single-split evaluation…")
    if config.OPTIMIZE_HYPERPARAMS:
        for name, mdl in MODELS.items():
            MODELS[name] = optimize_hyperparams(name, mdl, X_train, y_train, config)
    return X_train,y_train,train_idx,test_idx,single_splits

def Initalize(config, SAVE_MODEL):
    # Load data
    if config.SUMMARY:
        print(f"Loading data from {config.DATA_PATH}...")
    df = pd.read_csv(config.DATA_PATH)
    
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
        Save_Feature_Info(config.MODEL_DIR, df, feature_cols, encoded_cols)
    return df,X,y

if __name__ == "__main__":
    if len(sys.argv) > 1:
        main(config)
    else:
        main(config)