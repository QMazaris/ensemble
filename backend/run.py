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

from . import config_adapter as config  # Import the config adapter

# Import from helpers package
from .helpers import (
    # Data preparation
    prepare_data,
    apply_variance_filter,
    apply_correlation_filter,
    CV_Split,
    Save_Feature_Info,
    get_cv_splitter,
    
    # Modeling
    optimize_hyperparams,
    train_and_evaluate_model,
    process_cv_fold,
    
    # Metrics and evaluation
    ModelEvaluationResult,
    ModelEvaluationRun,
    compute_metrics,
    threshold_sweep_with_cost,
    _mk_result,
    _average_results,
    _average_probabilities,
    _average_sweep_data,
    
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

# Import bitwise logic functionality
from .helpers.stacked_logic import generate_combined_runs

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

    # Collect averaged-fold results for every model (K-fold is always used)
    model_zoo_runs = []
    base_model_runs = []

    # K-fold path (always executed)
    kfold_avg_ad_run, kfold_avg_cl_run = Core_KFold(config, C_FP, C_FN, base_cols, SAVE_PLOTS, MODELS, df, X, y, model_zoo_runs)
    # These are the averaged base model runs
    base_model_runs.append(kfold_avg_ad_run)
    base_model_runs.append(kfold_avg_cl_run)

    # ========== STRUCTURED BASE MODEL METRICS SECTION ==========
    # This section calculates results for the original base models (AD_Decision, CL_Decision, etc.).
    # These runs will always be created here, and added to base_model_runs below.
    structured_base_model_runs = []
    base_models_to_process = config.BASE_MODEL_DECISION_COLUMNS

    # For K-fold, we don't need train/test indices, so pass None
    Legacy_Base(config, C_FP, C_FN, df, y, None, None, structured_base_model_runs, base_models_to_process)

    base_model_runs.extend(structured_base_model_runs)

    # Make the final structure
    results_total = model_zoo_runs + base_model_runs

    # ========== BITWISE LOGIC SECTION ==========
    # Apply bitwise logic rules if enabled and configured
    try:
        from shared.config_manager import config_manager
        
        # Get bitwise logic configuration
        bitwise_config = config_manager.get('models.bitwise_logic', {
            'rules': [],
            'enabled': False
        })
        
        if bitwise_config.get('enabled', False) and bitwise_config.get('rules', []):
            if config.SUMMARY:
                print("\n" + "="*80)
                print("APPLYING BITWISE LOGIC RULES")
                print("="*80)
            
            # Convert bitwise logic rules to the format expected by generate_combined_runs
            combined_logic = {}
            for rule in bitwise_config.get('rules', []):
                combined_logic[rule['name']] = {
                    'columns': rule['columns'],
                    'logic': rule['logic']
                }
            
            if config.SUMMARY:
                print(f"Applying {len(combined_logic)} bitwise logic rules:")
                for rule_name, rule_config in combined_logic.items():
                    print(f"  • {rule_name}: {' '.join(rule_config['columns'])} with {rule_config['logic']} logic")
            
            # Generate combined runs using bitwise logic
            combined_runs = generate_combined_runs(
                runs=results_total,
                combined_logic=combined_logic,
                y_true=y.values,
                C_FP=C_FP,
                C_FN=C_FN,
                N_SPLITS=config.N_SPLITS
            )
            
            # Add combined runs to the total results
            results_total.extend(combined_runs)
            
            if config.SUMMARY:
                print(f"✅ Created {len(combined_runs)} combined models:")
                for run in combined_runs:
                    print(f"  • {run.model_name}")
                print()
        
    except Exception as e:
        if config.SUMMARY:
            print(f"⚠️ Warning: Could not apply bitwise logic rules: {str(e)}")
            print("Continuing without bitwise logic combinations...")
            print()

    if config.SUMMARY:
        print_performance_summary(
        runs=results_total,
        meta_model_names=set(MODELS.keys())
        )

    if SAVE_PREDICTIONS:
        y_full = y.loc[df.index].values # Get true y for the full dataset
        predictions_data = save_all_model_probabilities_from_structure(
            results_total, config.PREDICTIONS_DIR, df.index, y_full, 
            SUMMARY=config.SUMMARY, save_csv_backup=False
        )
      
    if SAVE_PLOTS:
        # Always use Full split for k-fold mode
        comparison_split = 'Full'
        
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

    print("\n🔄 Saving results for frontend...")
    
    streamlit_output_dir = Path("output") / "streamlit_data"

    export_metrics_for_streamlit(results_total, streamlit_output_dir, meta_model_names=set(MODELS.keys()))

def Legacy_Base(config, C_FP, C_FN, df, y, train_idx, test_idx, structured_base_model_runs, base_models_to_process):
    """Process legacy decision-based base models using defined decision columns and configurable good/bad tags."""

    good_tag = config.GOOD_TAG
    bad_tag = config.BAD_TAG
    decision_columns = config.BASE_MODEL_DECISION_COLUMNS  # list of column names

    for column in decision_columns:
        base_results = []
        base_probs = {}

        model_name = column  # use column name as model name

        if column not in df.columns:
            raise ValueError(f"🚨 Column '{column}' not found in dataframe.")

        # Validate the column only contains good/bad tags
        invalid_values = df[column][~df[column].isin([good_tag, bad_tag])]
        if not invalid_values.empty:
            raise ValueError(
                f"🚨 Column '{column}' contains invalid values: {invalid_values.unique().tolist()}.\n"
                f"Expected only: [{good_tag}, {bad_tag}]"
            )

        # Convert to binary: 1 for bad, 0 for good
        decisions = (df[column] == bad_tag).astype(int).values
        y_true = y.values

        # Compute metrics and cost (normalized over N_SPLITS)
        tn, fp, fn, tp = confusion_matrix(y_true, decisions).ravel()
        cost = (C_FP * fp + C_FN * fn) / config.N_SPLITS
        metrics = compute_metrics(y_true, decisions, C_FP, C_FN)
        metrics.update({'cost': cost, 'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn})

        # Store result
        base_results.append(
            ModelEvaluationResult(
                model_name=model_name,
                split='Full',
                threshold_type='base',
                threshold=None,
                precision=metrics['precision'],
                recall=metrics['recall'],
                f1_score=metrics['f1_score'],
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

        structured_base_model_runs.append(
            ModelEvaluationRun(
                model_name=model_name,
                results=base_results,
                probabilities=base_probs
            )
        )


def Core_KFold(config, C_FP, C_FN, base_cols, SAVE_PLOTS, MODELS, df, X, y, model_zoo_runs):
    """Core logic for K-fold cross-validation with improved structure."""
    from .helpers.data import get_cv_splitter
    from .helpers.modeling import process_cv_fold
    from .helpers.metrics import _average_results, _average_probabilities, _average_sweep_data, _mk_result
    
    outer_cv = get_cv_splitter(config)
    n_samples = len(X)
    
    # Initialize OOF probability arrays for meta, AD, and CL models
    oof_probs_dict = {model_name: np.full(n_samples, np.nan) for model_name in MODELS}
    oof_probs_ad = np.full(n_samples, np.nan)
    oof_probs_cl = np.full(n_samples, np.nan)

    for model_name, prototype in MODELS.items():
        if config.SUMMARY:
            print(f"\n{'='*40}\nNested‑CV → {model_name}\n{'='*40}")

        fold_cost, fold_acc, fold_probs, fold_sweeps = [], [], [], []
        fold_cost_ad, fold_acc_ad, fold_probs_ad, fold_sweeps_ad = [], [], [], []
        fold_cost_cl, fold_acc_cl, fold_probs_cl, fold_sweeps_cl = [], [], [], []

        # Process each fold
        for fold_idx, (tr_idx, te_idx) in enumerate(outer_cv.split(X, y), start=1):
            if config.SUMMARY:
                print(f"  ➤ Fold {fold_idx}")

            X_tr, y_tr = X.iloc[tr_idx], y.iloc[tr_idx]
            X_te, y_te = X.iloc[te_idx], y.iloc[te_idx]

            # Process meta model for this fold
            cost_result, acc_result, proba, sweep = process_cv_fold(
                prototype, X_tr, y_tr, X_te, y_te, fold_idx, config, C_FP, C_FN
            )
            
            # Update model name in results
            cost_result.model_name = model_name
            acc_result.model_name = model_name
            
            # Store OOF probabilities for meta model
            oof_probs_dict[model_name][te_idx] = proba
            
            # Collect fold results for meta model
            fold_cost.append(cost_result)
            fold_acc.append(acc_result)
            fold_probs.append(proba)
            fold_sweeps.append(sweep)

            # Process base models (AD/CL) for this fold
            ad_scores = df.loc[te_idx, base_cols["AD"]].values
            cl_scores = df.loc[te_idx, base_cols["CL"]].values
            oof_probs_ad[te_idx] = ad_scores
            oof_probs_cl[te_idx] = cl_scores

            # Threshold sweeps for base models
            sweep_ad, bc_ad, ba_ad = threshold_sweep_with_cost(
                ad_scores, y_te, C_FP, C_FN,
                SUMMARY=config.SUMMARY, split_name=f"AD_Fold{fold_idx}"
            )
            sweep_cl, bc_cl, ba_cl = threshold_sweep_with_cost(
                cl_scores, y_te, C_FP, C_FN,
                SUMMARY=config.SUMMARY, split_name=f"CL_Fold{fold_idx}"
            )

            # Optional plotting for base models
            if SAVE_PLOTS:
                plot_threshold_sweep(sweep_ad, C_FP, C_FN,
                                   cost_optimal_thr=bc_ad["threshold"],
                                   accuracy_optimal_thr=ba_ad["threshold"],
                                   SUMMARY=config.SUMMARY)
                plot_threshold_sweep(sweep_cl, C_FP, C_FN,
                                   cost_optimal_thr=bc_cl["threshold"],
                                   accuracy_optimal_thr=ba_cl["threshold"],
                                   SUMMARY=config.SUMMARY)

            # Collect base model results
            fold_cost_ad.append(_mk_result("AD", f"Fold{fold_idx}", bc_ad, 'cost'))
            fold_acc_ad.append(_mk_result("AD", f"Fold{fold_idx}", ba_ad, 'accuracy'))
            fold_probs_ad.append(ad_scores)
            fold_sweeps_ad.append(sweep_ad)

            fold_cost_cl.append(_mk_result("CL", f"Fold{fold_idx}", bc_cl, 'cost'))
            fold_acc_cl.append(_mk_result("CL", f"Fold{fold_idx}", ba_cl, 'accuracy'))
            fold_probs_cl.append(cl_scores)
            fold_sweeps_cl.append(sweep_cl)

            # Optional per-fold sweep plot for meta model
            if SAVE_PLOTS:
                plot_threshold_sweep(
                    sweep, C_FP, C_FN,
                    cost_optimal_thr=cost_result.threshold,
                    accuracy_optimal_thr=acc_result.threshold,
                    SUMMARY=config.SUMMARY
                )

        # Average results across folds for meta model
        avg_c = _average_results(fold_cost, model_name)
        avg_a = _average_results(fold_acc, model_name)
        avg_p = _average_probabilities(fold_probs)
        avg_sweep = _average_sweep_data(fold_sweeps)

        # Store meta model results
        model_zoo_runs.append(
            ModelEvaluationRun(
                model_name=f'kfold_avg_{model_name}',
                results=[avg_c, avg_a],
                probabilities={'Full': avg_p, 'oof': oof_probs_dict[model_name].tolist()},
                sweep_data={'Full': avg_sweep} if avg_sweep else None
            )
        )

    # Average results for base models (AD/CL)
    avg_c_ad = _average_results(fold_cost_ad, "AD")
    avg_a_ad = _average_results(fold_acc_ad, "AD")
    avg_p_ad = _average_probabilities(fold_probs_ad)
    avg_sweep_ad = _average_sweep_data(fold_sweeps_ad)

    avg_c_cl = _average_results(fold_cost_cl, "CL")
    avg_a_cl = _average_results(fold_acc_cl, "CL")
    avg_p_cl = _average_probabilities(fold_probs_cl)
    avg_sweep_cl = _average_sweep_data(fold_sweeps_cl)

    # Create base model runs
    kfold_avg_ad_run = ModelEvaluationRun(
        model_name='kfold_avg_AD',
        results=[avg_c_ad, avg_a_ad],
        probabilities={'Full': avg_p_ad, 'oof': oof_probs_ad.tolist()},
        sweep_data={'Full': avg_sweep_ad} if avg_sweep_ad else None
    )

    kfold_avg_cl_run = ModelEvaluationRun(
        model_name='kfold_avg_CL',
        results=[avg_c_cl, avg_a_cl],
        probabilities={'Full': avg_p_cl, 'oof': oof_probs_cl.tolist()},
        sweep_data={'Full': avg_sweep_cl} if avg_sweep_cl else None
    )
    
    return kfold_avg_ad_run, kfold_avg_cl_run

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