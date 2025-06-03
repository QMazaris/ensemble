# Standard library imports
import os
import sys
from pathlib import Path
import pickle
import requests

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


def load_config():
    """Load configuration from API endpoint."""
    try:
        response = requests.get("http://localhost:8000/config/load")
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"Error loading config from API: {e}")
        print("Please ensure the API server is running on http://localhost:8000")
        sys.exit(1)


def create_model_instances(config):
    """Create model instances based on configuration."""
    models = {}
    enabled_models = config.get("models", {}).get("enabled", [])
    model_params = config.get("model_params", {})
    
    for model_name in enabled_models:
        if model_name == "XGBoost":
            models[model_name] = xgb.XGBClassifier(**model_params.get("XGBoost", {}))
        elif model_name == "RandomForest":
            models[model_name] = RandomForestClassifier(**model_params.get("RandomForest", {}))
        elif model_name == "LogisticRegression":
            models[model_name] = LogisticRegression(**model_params.get("LogisticRegression", {}))
        elif model_name == "SVM":
            models[model_name] = SVC(**model_params.get("SVM", {}))
        elif model_name == "KNN":
            models[model_name] = KNeighborsClassifier(**model_params.get("KNN", {}))
        elif model_name == "MLP":
            models[model_name] = MLPClassifier(**model_params.get("MLP", {}))
    
    return models


def create_output_directories(config):
    """Create output directories based on configuration."""
    base_dir = config.get("output", {}).get("base_dir", "output")
    subdirs = config.get("output", {}).get("subdirs", {})
    
    dirs_to_create = []
    for subdir_name, subdir_path in subdirs.items():
        full_path = os.path.join(base_dir, subdir_path)
        dirs_to_create.append(full_path)
    
    return dirs_to_create


# ---------- Main Function ----------
def main(config_dict=None):
    """Main function to run the entire pipeline.
    
    Args:
        config_dict: Configuration dictionary. If None, loads from API.
    """
    # Load configuration from API if not provided
    if config_dict is None:
        config = load_config()
    else:
        config = config_dict

    # Create output directories
    dirs_to_create = create_output_directories(config)
    summary = config.get("logging", {}).get("summary", True)
    create_directories(dirs_to_create, summary=summary)

    # Unpack config
    C_FP = config.get("costs", {}).get("false_positive", 1.0)
    C_FN = config.get("costs", {}).get("false_negative", 50.0)

    # raw‑output lookup
    base_cols = config.get("models", {}).get("base_model_columns", ["AnomalyScore", "CL_ConfMax"])

    SAVE_PREDICTIONS = config.get("export", {}).get("save_predictions", True)
    SAVE_MODEL = config.get("export", {}).get("save_models", True)
    SUMMARY = config.get("logging", {}).get("summary", True)

    # Define model zoo
    MODELS = create_model_instances(config)
    
    df, X, y = Initalize(config, SAVE_MODEL)

    # Add detailed feature information logging and saving
    if SUMMARY:
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
    filter_data = config.get("features", {}).get("filter_data", False)
    feature_mapping = {
        'feature_columns': list(X.columns),
        'feature_count': X.shape[1],
        'sample_features': X.iloc[0].to_dict(),
        'feature_dtypes': X.dtypes.to_dict(),
        'preprocessing_steps': {
            'variance_filter': filter_data and config.get("features", {}).get("variance_threshold") is not None,
            'correlation_filter': filter_data and config.get("features", {}).get("correlation_threshold") is not None,
            'variance_threshold': config.get("features", {}).get("variance_threshold"),
            'correlation_threshold': config.get("features", {}).get("correlation_threshold")
        }
    }

    base_dir = config.get("output", {}).get("base_dir", "output")
    model_dir = os.path.join(base_dir, config.get("output", {}).get("subdirs", {}).get("models", "models"))
    feature_info_path = os.path.join(model_dir, 'exact_training_features.pkl')
    with open(feature_info_path, 'wb') as f:
        pickle.dump(feature_mapping, f)
        
    if SUMMARY:
        print(f"Saved exact training features to '{feature_info_path}'")
        print()

    meta, base = Core_KFold(config, C_FP, C_FN, base_cols, MODELS, df, X, y)
    # These are the averaged base model runs from output columns
    results_total = meta + base

    # ========== STRUCTURED BASE MODEL METRICS SECTION ==========
    # This section calculates results for the original base models (AD_Decision, CL_Decision, etc.).
    # These runs will always be created here, and added to base_model_runs below.
    structured_base_model_runs = []
    # Safely handle base_model_decisions - it can be either a list or a dictionary
    base_model_decisions_config = config.get("models", {}).get("base_model_decisions", [])
    
    # Handle both list and dictionary formats
    if isinstance(base_model_decisions_config, list):
        # If it's a list, use it directly as the column names
        base_models_to_process = base_model_decisions_config
    elif isinstance(base_model_decisions_config, dict):
        # If it's a dictionary, get the enabled_columns
        base_models_to_process = base_model_decisions_config.get("enabled_columns", [])
    else:
        # Fallback to empty list if neither format
        base_models_to_process = []

    decision_base_model_runs = Legacy_Base(config, C_FP, C_FN, df, y, base_models_to_process)

    results_total.extend(decision_base_model_runs)

    # ========== BITWISE LOGIC SECTION ==========
    # Apply bitwise logic rules if enabled and configured
    try:
        # Get bitwise logic configuration
        bitwise_config = config.get("models", {}).get("bitwise_logic", {})
        
        if bitwise_config.get('enabled', False) and bitwise_config.get('rules', []):
            if SUMMARY:
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
            
            if SUMMARY:
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
                N_SPLITS=config.get("training", {}).get("n_splits", 5),
                model_thresholds=config.get("model_thresholds", {})
            )
            
            # Add combined runs to the total results
            results_total.extend(combined_runs)
            
            if SUMMARY:
                print(f"✅ Created {len(combined_runs)} combined models:")
                for run in combined_runs:
                    print(f"  • {run.model_name}")
                print()
        
    except Exception as e:
        if SUMMARY:
            print(f"⚠️ Warning: Could not apply bitwise logic rules: {str(e)}")
            print("Continuing without bitwise logic combinations...")
            print()

    if SUMMARY:
        print_performance_summary(
        runs=results_total,
        meta_model_names=set(MODELS.keys())
        )

    if SAVE_PREDICTIONS:
        y_full = y.loc[df.index].values # Get true y for the full dataset
        predictions_dir = os.path.join(base_dir, config.get("output", {}).get("subdirs", {}).get("predictions", "predictions"))
        predictions_data = save_all_model_probabilities_from_structure(
            results_total, predictions_dir, df.index, y_full, 
            SUMMARY=SUMMARY, save_csv_backup=False
        )

    # ========== FINAL PRODUCTION MODELS SECTION ==========
    # When using k-fold, the final model should be trained on the full data after hyperparameter tuning (if enabled)
    # based on the average performance across folds.
    # When using single-split, the final model is trained on the training split and evaluated on the test split.
    plot_dir = os.path.join(base_dir, config.get("output", {}).get("subdirs", {}).get("plots", "plots"))
    FinalModelCreateAndAnalyize(config, model_dir, plot_dir, C_FP, C_FN, X, y)

    print("\nPipeline complete")

    print("\n🔄 Saving results for frontend...")
    
    streamlit_output_dir = Path("output") / "streamlit_data"

    export_metrics_for_streamlit(results_total, streamlit_output_dir, meta_model_names=set(MODELS.keys()))

def Legacy_Base(
    config,
    C_FP,
    C_FN,
    df: pd.DataFrame,
    y: pd.Series,
    base_models_to_process: list[str]
) -> list[ModelEvaluationRun]:
    """
    Process legacy decision-based base models.
    Returns a list of ModelEvaluationRun for each column in base_models_to_process.
    """

    from sklearn.metrics import confusion_matrix
    from .helpers.metrics import compute_metrics
    from .helpers import ModelEvaluationResult, ModelEvaluationRun

    runs: list[ModelEvaluationRun] = []
    # Safely handle base_model_decisions for good/bad tags
    base_model_decisions_config = config.get("models", {}).get("base_model_decisions", {})
    
    # Handle both list and dictionary formats for good/bad tags
    if isinstance(base_model_decisions_config, dict):
        good_tag = base_model_decisions_config.get("good_tag", "Good")
        bad_tag = base_model_decisions_config.get("bad_tag", "Bad")
    else:
        # If it's a list format, use defaults from data section or fallback
        good_tag = config.get("data", {}).get("good_tag", "Good")
        bad_tag = config.get("data", {}).get("bad_tag", "Bad")

    for column in base_models_to_process:
        if column not in df.columns:
            raise ValueError(f"🚨 Column '{column}' not found in dataframe.")

        # Ensure only good/bad values
        invalid = df[column][~df[column].isin([good_tag, bad_tag])]
        if not invalid.empty:
            raise ValueError(
                f"🚨 Column '{column}' contains invalid values {invalid.unique().tolist()}; "
                f"expected only [{good_tag}, {bad_tag}]."
            )

        # 1) Convert decisions to binary
        decisions = (df[column] == bad_tag).astype(int).values
        y_true    = y.values

        # 2) Compute confusion & cost
        tn, fp, fn, tp = confusion_matrix(y_true, decisions).ravel()
        cost = (C_FP * fp + C_FN * fn) / config.get("training", {}).get("n_splits", 5)

        # 3) Compute standard metrics
        metrics = compute_metrics(y_true, decisions, C_FP, C_FN)
        metrics.update({'cost': cost, 'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn})

        # 4) Build the ModelEvaluationResult
        result = ModelEvaluationResult(
            model_name     = column,
            split          = 'Full',
            threshold_type = 'base',
            threshold      = None,
            precision      = metrics['precision'],
            recall         = metrics['recall'],
            f1_score       = metrics['f1_score'],
            accuracy       = metrics['accuracy'],
            cost           = cost,
            tp             = tp,
            fp             = fp,
            tn             = tn,
            fn             = fn,
            is_base_model  = True
        )

        # 5) Wrap in a ModelEvaluationRun, with the "Full"‐split probabilities
        run = ModelEvaluationRun(
            model_name   = column,
            results      = [result],
            probabilities= {'Full': decisions}
        )

        runs.append(run)

    return runs



def Core_KFold(config, C_FP, C_FN, base_cols, MODELS, df, X, y):
    """
    Pure K‑fold cross‑validation:
      - returns (meta_runs, base_runs)
      - does not mutate any input lists
    """
    from .helpers.data    import get_cv_splitter
    from .helpers.modeling import process_cv_fold
    from .helpers.metrics  import (
        _average_results,
        _average_probabilities,
        _average_sweep_data,
        _mk_result
    )

    outer_cv  = get_cv_splitter(config)
    n_samples = len(X)

    # 1) META models
    meta_runs      = []
    oof_probs_meta = {
        mn: np.full(n_samples, np.nan)
        for mn in MODELS
    }

    for model_name, prototype in MODELS.items():
        fold_cost, fold_acc, fold_probs, fold_sweeps = [], [], [], []

        for fold_idx, (tr_idx, te_idx) in enumerate(outer_cv.split(X, y), start=1):
            X_tr, y_tr = X.iloc[tr_idx], y.iloc[tr_idx]
            X_te, y_te = X.iloc[te_idx], y.iloc[te_idx]

            c_res, a_res, proba, sweep = process_cv_fold(
                prototype, X_tr, y_tr, X_te, y_te,
                fold_idx, config, C_FP, C_FN
            )
            # tag with correct name
            c_res.model_name = a_res.model_name = model_name
            oof_probs_meta[model_name][te_idx] = proba

            fold_cost.append(c_res)
            fold_acc.append(a_res)
            fold_probs.append(proba)
            fold_sweeps.append(sweep)



        # average across folds
        avg_c     = _average_results(fold_cost, model_name)
        avg_a     = _average_results(fold_acc,  model_name)
        avg_p     = _average_probabilities(fold_probs)
        avg_sweep = _average_sweep_data(fold_sweeps)

        meta_runs.append(
            ModelEvaluationRun(
                model_name   = f'kfold_avg_{model_name}',
                results      = [avg_c, avg_a],
                probabilities= {'Full': avg_p,
                                'oof' : oof_probs_meta[model_name].tolist()},
                sweep_data   = {'Full': avg_sweep} if avg_sweep else None
            )
        )

    # 2) BASE models (from output columns)
    base_runs       = []
    oof_probs_base  = {
        col: np.full(n_samples, np.nan)
        for col in base_cols
    }
    folds_by_column = {
        col: {'cost':[], 'acc':[], 'probs':[], 'sweeps':[]}
        for col in base_cols
    }

    for fold_idx, (tr_idx, te_idx) in enumerate(outer_cv.split(X, y), start=1):
        y_te = y.iloc[te_idx]

        for column_name in base_cols:
            if column_name not in df:
                if config.get("logging", {}).get("summary", True):
                    print(f"⚠️ missing '{column_name}', skipping.")
                continue

            scores = df.loc[te_idx, column_name].values
            oof_probs_base[column_name][te_idx] = scores

            sweep, cost_opt, acc_opt = threshold_sweep_with_cost(
                scores, y_te, C_FP, C_FN,
                SUMMARY   = config.get("logging", {}).get("summary", True),
                split_name= f"{column_name}_Fold{fold_idx}"
            )

            folds_by_column[column_name]['cost'].append(
                _mk_result(column_name, f"Fold{fold_idx}", cost_opt,'cost')
            )
            folds_by_column[column_name]['acc'].append(
                _mk_result(column_name, f"Fold{fold_idx}", acc_opt,'accuracy')
            )
            folds_by_column[column_name]['probs'].append(scores)
            folds_by_column[column_name]['sweeps'].append(sweep)

           

    # Average each base model
    for column_name, data in folds_by_column.items():
        if not data['cost']:
            continue

        avg_c     = _average_results(data['cost'], column_name)
        avg_a     = _average_results(data['acc'],  column_name)
        avg_p     = _average_probabilities(data['probs'])
        avg_sweep = _average_sweep_data(data['sweeps'])

        base_runs.append(
            ModelEvaluationRun(
                model_name   = f'kfold_avg_{column_name}',
                results      = [avg_c, avg_a],
                probabilities= {'Full': avg_p,
                                'oof' : oof_probs_base[column_name].tolist()},
                sweep_data   = {'Full': avg_sweep} if avg_sweep else None
            )
        )

    return meta_runs, base_runs


def Initalize(config, SAVE_MODEL):
    # Load data
    data_path = config.get("data", {}).get("path", "data/training_data.csv")
    df = pd.read_csv(data_path)
    
    # Prepare data
    X, y, numeric_cols, encoded_cols = prepare_data(df, config)
    
    # Apply feature filtering if enabled
    filter_data = config.get("features", {}).get("filter_data", False)
    if filter_data:
        summary = config.get("logging", {}).get("summary", True)
        if summary:
            print(f"Original feature count: {X.shape[1]}")
        
        # Apply variance filter
        variance_thresh = config.get("features", {}).get("variance_threshold", 0.01)
        X = apply_variance_filter(X, threshold=variance_thresh, SUMMARY=summary)
        
        # Apply correlation filter
        correlation_thresh = config.get("features", {}).get("correlation_threshold", 0.95)
        X = apply_correlation_filter(X, threshold=correlation_thresh, SUMMARY=summary)
        
        if summary:
            print(f"Final feature count after filtering: {X.shape[1]}")
    
    # Save feature information if model saving is enabled
    if SAVE_MODEL:
        base_dir = config.get("output", {}).get("base_dir", "output")
        model_dir = os.path.join(base_dir, config.get("output", {}).get("subdirs", {}).get("models", "models"))
        Save_Feature_Info(model_dir, df, numeric_cols, encoded_cols)
    
    return df, X, y

if __name__ == "__main__":
    main() 