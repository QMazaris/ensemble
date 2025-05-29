# Additions that could be made to improve the models:
# Small Quirk: doesnt fill the AD or CL structures properly when k fold is off

# Standard library imports
import os
import sys
from pathlib import Path
import pickle

# Third-party imports
import joblib
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

from . import config  # Import the config module

# Import from helpers package
from .helpers import (
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
from .helpers.model_export import export_model

# Explicit import for FinalModelCreateAndAnalyize
from .helpers.modeling import FinalModelCreateAndAnalyize

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

    # Add detailed feature information logging and saving
    if config.SUMMARY:
        print("=== TRAINING FEATURE INFO ===")
        print(f"Final X shape: {X.shape}")
        print(f"Number of features: {X.shape[1]}")
        print()
        print("Feature columns in exact order:")
        for i, col in enumerate(X.columns):
            print(f"{i:2d}: {col}")
        print()
        print("First row of X (sample features):")
        print(X.iloc[0].values)
        print()
        print("Feature statistics:")
        print(X.describe())

    # Save the exact feature info for inference
    feature_mapping = {
        'feature_columns': list(X.columns),
        'feature_count': X.shape[1],
        'sample_features': X.iloc[0].to_dict(),
        'feature_dtypes': X.dtypes.to_dict(),
        'preprocessing_steps': {
            'variance_filter': config.FilterData and hasattr(config, 'VARIANCE_THRESH'),
            'correlation_filter': config.FilterData and hasattr(config, 'CORRELATION_THRESH'),
            'variance_threshold': getattr(config, 'VARIANCE_THRESH', None),
            'correlation_threshold': getattr(config, 'CORRELATION_THRESH', None)
        }
    }

    feature_info_path = os.path.join(config.MODEL_DIR, 'exact_training_features.pkl')
    with open(feature_info_path, 'wb') as f:
        pickle.dump(feature_mapping, f)
        
    if config.SUMMARY:
        print(f"Saved exact training features to '{feature_info_path}'")
        print()

    # Generate class balance data if SAVE_PLOTS is enabled
    if SAVE_PLOTS:
        class_balance_data = plot_class_balance(y, SUMMARY=SUMMARY)
        # Store data for potential frontend use

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
        predictions_data = save_all_model_probabilities_from_structure(results_total, config.PREDICTIONS_DIR, df.index, y_full, SUMMARY = config.SUMMARY)
      
    if SAVE_PLOTS:
        # Use Test split for single split mode, Full split for k-fold mode
        comparison_split = 'Full' if config.USE_KFOLD else 'Test'
        
        cost_comparison_data = plot_runs_at_threshold(
            runs=results_total,
            threshold_type='cost',
            split_name=comparison_split,
            C_FP=C_FP,
            C_FN=C_FN
        )
        accuracy_comparison_data = plot_runs_at_threshold(
            runs=results_total,
            threshold_type='accuracy',
            split_name=comparison_split,
            C_FP=C_FP,
            C_FN=C_FN
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
            # For single-split, calculate on train and test separately
            for split_name, split_idx in [('Train', train_idx), ('Test', test_idx)]:
                y_split = y_true[split_idx]
                decisions_split = decisions[split_idx]
                
                tn, fp, fn, tp = confusion_matrix(y_split, decisions_split).ravel()
                cost = C_FP * fp + C_FN * fn
                metrics = compute_metrics(y_split, decisions_split, C_FP, C_FN)
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
                        threshold=None,
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
                base_probs[split_name] = decisions_split
            
            # Calculate Full split as combination of train and test
            tn, fp, fn, tp = confusion_matrix(y_true, decisions).ravel()
            cost = C_FP * fp + C_FN * fn
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
                    split='Full',
                    threshold_type='base',
                    threshold=None,
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
        
        # Create the run for this base model
        structured_base_model_runs.append(
            ModelEvaluationRun(
                model_name=base,
                results=base_results,
                probabilities=base_probs
            )
        )

def Core_Standard(config, C_FP, C_FN, base_cols, SAVE_PLOTS, SAVE_MODEL, MODELS, df, y, X_train, y_train, train_idx, test_idx, single_splits, model_zoo_runs):
    """Core logic for standard (non-k-fold) evaluation."""
    base_model_runs = []
    
    for model_name, prototype in MODELS.items():
        if config.SUMMARY:
            print(f"\n{'='*40}\nTraining → {model_name}\n{'='*40}")

        # Train and evaluate the model
        split_preds, trained_model = train_and_evaluate_model(
            prototype, X_train, y_train,
            splits=single_splits,
            model_name=model_name,
            model_dir=config.MODEL_DIR if SAVE_MODEL else None,
            save_model=SAVE_MODEL,
            SUMMARY=config.SUMMARY,
            config=config
        )

        # Perform threshold sweeps and collect results
        model_results = []
        model_probs = {}
        model_sweeps = {}
        
        for split_name, (proba, y_eval) in split_preds.items():
            sweep, bc, ba = threshold_sweep_with_cost(
                proba, y_eval, C_FP=C_FP, C_FN=C_FN,
                SUMMARY=config.SUMMARY, split_name=split_name
            )
            
            model_results.extend([
                _mk_result(model_name, split_name, bc, 'cost'),
                _mk_result(model_name, split_name, ba, 'accuracy')
            ])
            model_probs[split_name] = proba
            model_sweeps[split_name] = sweep
            
            # Optional per-split sweep plot
            if SAVE_PLOTS:
                sweep_data = plot_threshold_sweep(
                    sweep, C_FP, C_FN,
                    cost_optimal_thr=bc['threshold'],
                    accuracy_optimal_thr=ba['threshold'],
                    SUMMARY=config.SUMMARY
                )

        # Store the run for this model
        model_zoo_runs.append(
            ModelEvaluationRun(
                model_name=model_name,
                results=model_results,
                probabilities=model_probs,
                sweep_data=model_sweeps
            )
        )

    # Process base models (AD, CL) for single-split
    for base_name in ["AD", "CL"]:
        base_results = []
        base_probs = {}
        base_sweeps = {}
        
        for split_name, (X_eval, y_eval) in single_splits.items():
            # Get base model scores for this split
            if split_name == 'Train':
                split_df = df.loc[train_idx]
            elif split_name == 'Test':
                split_df = df.loc[test_idx]
            else:  # Full
                split_df = df
                
            base_scores = split_df[base_cols[base_name]].values
            
            sweep, bc, ba = threshold_sweep_with_cost(
                base_scores, y_eval, C_FP, C_FN,
                SUMMARY=config.SUMMARY, split_name=f"{base_name}_{split_name}"
            )
            
            base_results.extend([
                _mk_result(base_name, split_name, bc, 'cost'),
                _mk_result(base_name, split_name, ba, 'accuracy')
            ])
            base_probs[split_name] = base_scores
            base_sweeps[split_name] = sweep
            
            # Optional per-split sweep plot
            if SAVE_PLOTS:
                sweep_data = plot_threshold_sweep(
                    sweep, C_FP, C_FN,
                    cost_optimal_thr=bc['threshold'],
                    accuracy_optimal_thr=ba['threshold'],
                    SUMMARY=config.SUMMARY
                )

        # Store the run for this base model
        base_model_runs.append(
            ModelEvaluationRun(
                model_name=base_name,
                results=base_results,
                probabilities=base_probs,
                sweep_data=base_sweeps
            )
        )
    
    return base_model_runs

def Core_KFold(config, C_FP, C_FN, base_cols, SAVE_PLOTS, MODELS, df, X, y, model_zoo_runs):
    outer_cv = StratifiedKFold(
            n_splits     = config.N_SPLITS,
            shuffle      = True,
            random_state = config.RANDOM_STATE
        )

    n_samples = len(X)
    # Initialize OOF probability arrays for meta, AD, and CL models
    oof_probs_dict = {model_name: np.full(n_samples, np.nan) for model_name in MODELS}
    oof_probs_ad = np.full(n_samples, np.nan)
    oof_probs_cl = np.full(n_samples, np.nan)

    for model_name, prototype in MODELS.items():
        if config.SUMMARY:
            print(f"\n{'='*40}\nNested‑CV → {model_name}\n{'='*40}")

        fold_cost, fold_acc, fold_probs = [], [], []
        fold_sweeps = []  # Store sweep data for each fold
        # Add arrays for AD/CL results
        fold_cost_ad, fold_acc_ad, fold_probs_ad = [], [], []
        fold_sweeps_ad = []  # Store AD sweep data
        fold_cost_cl, fold_acc_cl, fold_probs_cl = [], [], []
        fold_sweeps_cl = []

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
            oof_probs_dict[model_name][te_idx] = proba  # Store OOF probabilities for meta model

            # ── Threshold sweep for meta model ──
            sweep, bc, ba = threshold_sweep_with_cost(
                    proba, y_eval, C_FP=C_FP, C_FN=C_FN,
                    SUMMARY=config.SUMMARY, split_name=f"Fold{fold_idx}"
                )

            # ── Collect fold results for meta model ──
            fold_cost.append(_mk_result(model_name, f"Fold{fold_idx}", bc, 'cost'))
            fold_acc.append(_mk_result(model_name, f"Fold{fold_idx}", ba, 'accuracy'))
            fold_probs.append(proba)
            fold_sweeps.append(sweep)  # Store sweep data

            # ── Base‑model threshold sweeps ──
            # pull raw scores from df via config
            ad_scores = df.loc[te_idx, base_cols["AD"]].values
            cl_scores = df.loc[te_idx, base_cols["CL"]].values
            oof_probs_ad[te_idx] = ad_scores  # Store OOF for AD
            oof_probs_cl[te_idx] = cl_scores  # Store OOF for CL

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
                ad_sweep_data = plot_threshold_sweep(sweep_ad, C_FP, C_FN,
                                        cost_optimal_thr=bc_ad["threshold"],
                                        accuracy_optimal_thr=ba_ad["threshold"],
                                        SUMMARY=config.SUMMARY)
                cl_sweep_data = plot_threshold_sweep(sweep_cl, C_FP, C_FN,
                                        cost_optimal_thr=bc_cl["threshold"],
                                        accuracy_optimal_thr=ba_cl["threshold"],
                                        SUMMARY=config.SUMMARY)

            # stash their results/probs in parallel arrays
            fold_cost_ad.append(_mk_result("AD", f"Fold{fold_idx}", bc_ad, 'cost'))
            fold_acc_ad.append(_mk_result("AD", f"Fold{fold_idx}", ba_ad, 'accuracy'))
            fold_probs_ad.append(ad_scores)
            fold_sweeps_ad.append(sweep_ad)  # Store AD sweep data

            fold_cost_cl.append(_mk_result("CL", f"Fold{fold_idx}", bc_cl, 'cost'))
            fold_acc_cl.append(_mk_result("CL", f"Fold{fold_idx}", ba_cl, 'accuracy'))
            fold_probs_cl.append(cl_scores)
            fold_sweeps_cl.append(sweep_cl)  # Store CL sweep data

            # Optional per-fold sweep plot for meta model
            if SAVE_PLOTS:
                sweep_data = plot_threshold_sweep(
                        sweep, C_FP, C_FN,
                        cost_optimal_thr=bc['threshold'],
                        accuracy_optimal_thr=ba['threshold'],
                        SUMMARY=config.SUMMARY
                    )

        # ── Average & store meta model results ──
        avg_c = _average_results(fold_cost, model_name)
        avg_a = _average_results(fold_acc, model_name)
        avg_p = _average_probabilities(fold_probs)
        
        # Average sweep data across folds
        avg_sweep = {}
        if fold_sweeps:  # If we have sweep data
            # Get all unique thresholds across all folds
            all_thresholds = sorted(set().union(*[set(s.keys()) for s in fold_sweeps]))
            for thr in all_thresholds:
                # For each threshold, average the metrics across folds
                metrics = {}
                for metric in ['cost', 'accuracy', 'precision', 'recall', 'tp', 'fp', 'tn', 'fn']:
                    values = [s[thr][metric] for s in fold_sweeps if thr in s]
                    if values:  # Only average if we have values for this threshold
                        metrics[metric] = float(np.mean(values))
                if metrics:  # Only add if we have metrics
                    avg_sweep[float(thr)] = metrics

        # Store OOF probabilities for meta model
        model_zoo_runs.append(
                ModelEvaluationRun(
                    model_name=f'kfold_avg_{model_name}',
                    results=[avg_c, avg_a],
                    probabilities={'Full': avg_p, 'oof': oof_probs_dict[model_name].tolist()},
                    sweep_data={'Full': avg_sweep} if avg_sweep else None
                )
            )

        # ── Now do the same for AD and CL ──
        # Calculate the kfold averaged AD/CL results
        avg_c_ad = _average_results(fold_cost_ad, "AD")
        avg_a_ad = _average_results(fold_acc_ad, "AD")
        avg_p_ad = _average_probabilities(fold_probs_ad)
        
        # Average AD sweep data
        avg_sweep_ad = {}
        if fold_sweeps_ad:
            all_thresholds = sorted(set().union(*[set(s.keys()) for s in fold_sweeps_ad]))
            for thr in all_thresholds:
                metrics = {}
                for metric in ['cost', 'accuracy', 'precision', 'recall', 'tp', 'fp', 'tn', 'fn']:
                    values = [s[thr][metric] for s in fold_sweeps_ad if thr in s]
                    if values:
                        metrics[metric] = float(np.mean(values))
                if metrics:
                    avg_sweep_ad[float(thr)] = metrics

        # Store OOF probabilities for AD
        kfold_avg_ad_run = ModelEvaluationRun(
                model_name='kfold_avg_AD',
                results=[avg_c_ad, avg_a_ad],
                probabilities={'Full': avg_p_ad, 'oof': oof_probs_ad.tolist()},
                sweep_data={'Full': avg_sweep_ad} if avg_sweep_ad else None
            )

        avg_c_cl = _average_results(fold_cost_cl, "CL")
        avg_a_cl = _average_results(fold_acc_cl, "CL")
        avg_p_cl = _average_probabilities(fold_probs_cl)
        
        # Average CL sweep data
        avg_sweep_cl = {}
        if fold_sweeps_cl:
            all_thresholds = sorted(set().union(*[set(s.keys()) for s in fold_sweeps_cl]))
            for thr in all_thresholds:
                metrics = {}
                for metric in ['cost', 'accuracy', 'precision', 'recall', 'tp', 'fp', 'tn', 'fn']:
                    values = [s[thr][metric] for s in fold_sweeps_cl if thr in s]
                    if values:
                        metrics[metric] = float(np.mean(values))
                if metrics:
                    avg_sweep_cl[float(thr)] = metrics

        # Store OOF probabilities for CL
        kfold_avg_cl_run = ModelEvaluationRun(
                model_name='kfold_avg_CL',
                results=[avg_c_cl, avg_a_cl],
                probabilities={'Full': avg_p_cl, 'oof': oof_probs_cl.tolist()},
                sweep_data={'Full': avg_sweep_cl} if avg_sweep_cl else None
            )
    
    return kfold_avg_ad_run, kfold_avg_cl_run

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
    df = pd.read_csv(config.DATA_PATH)
    
    # Prepare data
    X, y, numeric_cols, encoded_cols = prepare_data(df, config)
    
    # Apply feature filtering if enabled
    if config.FilterData:
        if config.SUMMARY:
            print(f"Original feature count: {X.shape[1]}")
        
        # Apply variance filter
        X = apply_variance_filter(X, threshold=config.VARIANCE_THRESH, SUMMARY=config.SUMMARY)
        
        # Apply correlation filter
        X = apply_correlation_filter(X, threshold=config.CORRELATION_THRESH, SUMMARY=config.SUMMARY)
        
        if config.SUMMARY:
            print(f"Final feature count after filtering: {X.shape[1]}")
    
    # Save feature information if model saving is enabled
    if SAVE_MODEL:
        Save_Feature_Info(config.MODEL_DIR, df, numeric_cols, encoded_cols)
    
    return df, X, y

if __name__ == "__main__":
    main(config) 