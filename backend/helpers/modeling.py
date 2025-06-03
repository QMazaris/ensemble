# Standard library imports
import os

# Third-party imports
import joblib
import numpy as np
import optuna
from imblearn.over_sampling import SMOTE
from sklearn.base import clone
from sklearn.model_selection import cross_val_score
from sklearn.metrics import log_loss

# Local imports
from .metrics import threshold_sweep_with_cost, _mk_result
from .plotting import plot_threshold_sweep
from .model_export import export_model

def optimize_hyperparams(model_name, model, X_train, y_train, config):
    """Bayesian optimization using Optuna to find optimal hyperparameters.
    
    The hyperparameter space is defined in config hyperparam_spaces with a nested structure:
    - 'shared': Parameters shared across all models
    - model_name: Model-specific parameters
    """
    hyperparam_spaces = config.get("hyperparam_spaces", {})
    if not hyperparam_spaces:
        summary = config.get("logging", {}).get("summary", True)
        if summary:
            print(f"⚠️ No hyperparameter space defined for {model_name}, using default parameters")
        return model

    def objective(trial):
        # Create a fresh model instance
        model_clone = clone(model)
        
        # Build parameters from the config space
        params = {}
        
        # Add shared parameters if they exist
        if 'shared' in hyperparam_spaces:
            for param_name, param_space in hyperparam_spaces['shared'].items():
                if isinstance(param_space, list):
                    # Categorical parameter
                    params[param_name] = trial.suggest_categorical(f"shared_{param_name}", param_space)
                elif isinstance(param_space, tuple):
                    if len(param_space) != 2:
                        raise ValueError(f"Parameter space for {param_name} must be a tuple of (min, max)")
                    if isinstance(param_space[0], int):
                        # Integer parameter
                        params[param_name] = trial.suggest_int(f"shared_{param_name}", param_space[0], param_space[1])
                    else:
                        # Float parameter
                        params[param_name] = trial.suggest_float(f"shared_{param_name}", param_space[0], param_space[1])
                else:
                    raise ValueError(f"Invalid parameter space type for {param_name}. Must be list or tuple.")
        
        # Add model-specific parameters
        if model_name in hyperparam_spaces:
            for param_name, param_space in hyperparam_spaces[model_name].items():
                if isinstance(param_space, list):
                    # Categorical parameter
                    params[param_name] = trial.suggest_categorical(f"{model_name}_{param_name}", param_space)
                elif isinstance(param_space, tuple):
                    if len(param_space) != 2:
                        raise ValueError(f"Parameter space for {param_name} must be a tuple of (min, max)")
                    if isinstance(param_space[0], int):
                        # Integer parameter
                        params[param_name] = trial.suggest_int(f"{model_name}_{param_name}", param_space[0], param_space[1])
                    else:
                        # Float parameter
                        params[param_name] = trial.suggest_float(f"{model_name}_{param_name}", param_space[0], param_space[1])
                else:
                    raise ValueError(f"Invalid parameter space type for {param_name}. Must be list or tuple.")
        
        # Set parameters and evaluate
        try:
            model_clone.set_params(**params)
            scores = cross_val_score(model_clone, X_train, y_train, cv=3, scoring='neg_log_loss', n_jobs=1)
            return scores.mean()
        except Exception as e:
            # Return a poor score if parameter combination is invalid
            return -10.0
    
    # Run optimization
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)  # Reduce optuna log noise
    study = optuna.create_study(direction='maximize')
    
    hyperparams_enabled = config.get("optimization", {}).get("enabled", False)
    if not hyperparams_enabled:
        summary = config.get("logging", {}).get("summary", True)
        if summary:
            print(f"⚠️ Hyperparameter optimization disabled for {model_name}, using default parameters")
        return model
    
    iterations = config.get("optimization", {}).get("iterations", 50)
    summary = config.get("logging", {}).get("summary", True)
    
    study.optimize(
        objective,
        n_trials=iterations,
        show_progress_bar=summary
    )

    # Get the best parameters and create a new model with them
    best_params = study.best_params
    # Remove the prefixes from parameter names
    cleaned_params = {}
    for param_name, value in best_params.items():
        if param_name.startswith('shared_'):
            cleaned_params[param_name[7:]] = value  # Remove 'shared_' prefix
        elif param_name.startswith(f'{model_name}_'):
            cleaned_params[param_name[len(model_name)+1:]] = value  # Remove 'model_name_' prefix
    
    best_model = clone(model)
    best_model.set_params(**cleaned_params)
    
    if summary:
        print(f"✅ {model_name} optimization complete:")
        print(f"   Best parameters: {cleaned_params}")
        print(f"   Best CV score: {-study.best_value:.4f}")
    
    return best_model

def train_and_evaluate_model(model, X_train, y_train, splits: dict, model_name: str = None, model_dir: str = None, save_model: bool = True, SUMMARY = None, config=None, feature_names=None):
    """Train a model and evaluate it on multiple splits."""
    # Apply SMOTE if configured
    use_smote = config.get("training", {}).get("use_smote", True) if config else True
    if use_smote:
        smote_ratio = config.get("training", {}).get("smote_ratio", 0.5) if config else 0.5
        random_state = config.get("data", {}).get("random_state", 42) if config else 42
        smote = SMOTE(sampling_strategy=smote_ratio, random_state=random_state)
        X_bal, y_bal = smote.fit_resample(X_train, y_train)
    else:
        X_bal, y_bal = X_train, y_train

    # Train the model
    model.fit(X_bal, y_bal)

    # Save if requested
    if save_model and model_name and model_dir:
        # Use feature_names if provided, else use X_train columns
        if feature_names is None:
            feature_names = X_train.columns.tolist()
        model_path = export_model(model, model_name, model_dir, config, feature_names=feature_names)
        if SUMMARY:
            print(f"Saved trained model to {model_path}")

    # Generate predictions for each split
    split_preds = {}
    for split_name, (X_eval, y_eval) in splits.items():
        if SUMMARY:
            print(f"Predicting probabilities on {split_name} split...")
        proba = model.predict_proba(X_eval)[:, 1]
        split_preds[split_name] = (proba, y_eval)

    return split_preds, model

def save_final_kfold_model(model, X, y, model_name, model_dir, SUMMARY=None, config=None, feature_names=None):
    """Train and save the final K-Fold model on the full dataset."""
    trained_model, _ = train_and_evaluate_model(
        model, X, y,
        splits={'Full': (X, y)},
        model_name=f'{model_name}_final',
        model_dir=model_dir,
        save_model=True,
        SUMMARY=SUMMARY,
        config=config,
        feature_names=feature_names if feature_names is not None else X.columns.tolist()
    )
    if SUMMARY:
        print(f"Final K-Fold model '{model_name}_final' trained and saved to {model_dir}")
    return trained_model

def FinalModelCreateAndAnalyize(config, model_path, image_path, C_FP, C_FN, X, y):
    """Create and analyze final production models.
    
    Args:
        config: Configuration dictionary containing model settings
        model_path: Directory to save models
        image_path: Directory to save plots
        C_FP: Cost of false positive
        C_FN: Cost of false negative
        X: Feature matrix
        y: Target vector
    """
    optimize_final_model = config.get("optimization", {}).get("optimize_final_model", False)
    if not optimize_final_model:
        summary = config.get("logging", {}).get("summary", True)
        if summary:
            print("\n⚠️ Final model optimization is disabled. Skipping production model training.")
        return

    save_models = config.get("export", {}).get("save_models", True)
    summary = config.get("logging", {}).get("summary", True)
    
    if save_models:
        if summary:
            print("\n" + "="*80)
            print("FINAL PRODUCTION MODELS")
            print("="*80)

        # Create model instances
        from ..run import create_model_instances
        models = create_model_instances(config)

        for name, prototype in models.items():
            if summary:
                print(f"\n🏭 Optimizing final production model: {name}")
            # Get fresh instance and optimize hyperparameters
            base = clone(prototype)
            optimize_hyperparams_enabled = config.get("optimization", {}).get("enabled", False)
            if optimize_hyperparams_enabled:
                best = optimize_hyperparams(name, base, X, y, config)
            else:
                best = base  # Use default parameters if optimization is disabled
            
            # Train the model on full dataset
            best.fit(X, y)
            
            # Perform threshold sweep on full dataset
            if summary:
                print(f"\n📊 Performing threshold sweep for {name} on full dataset...")
            proba = best.predict_proba(X)[:, 1]
            sweep_results, best_cost, best_acc = threshold_sweep_with_cost(
                proba, y, C_FP=C_FP, C_FN=C_FN,
                SUMMARY=summary,
                split_name=f"FINAL PRODUCTION {name}"
            )
            
            
            # Save the model
            out_path = export_model(best, f"{name}_production", model_path, config, feature_names=X.columns.tolist())
            if summary:
                print(f"✅ Saved production model: {out_path}")
                print(f"   • Cost-optimal threshold: {best_cost['threshold']:.3f}")
                print(f"   • Accuracy-optimal threshold: {best_acc['threshold']:.3f}")

def process_cv_fold(model_prototype, X_tr, y_tr, X_te, y_te, fold_idx, config, C_FP, C_FN):
    """Process a single cross-validation fold.
    
    Args:
        model_prototype: Model to train
        X_tr, y_tr: Training data for this fold
        X_te, y_te: Test data for this fold  
        fold_idx: Fold number
        config: Configuration dictionary
        C_FP, C_FN: Cost parameters
        
    Returns:
        tuple: (cost_result, accuracy_result, probabilities, sweep_data)
    """
    from .metrics import threshold_sweep_with_cost, _mk_result
    from sklearn.base import clone
    
    # Hyperparameter optimization for this fold
    optimize_hyperparams_enabled = config.get("optimization", {}).get("enabled", False)
    if optimize_hyperparams_enabled:
        tuned = optimize_hyperparams(
            f"fold_{fold_idx}",
            clone(model_prototype),
            X_tr, y_tr,
            config
        )
    else:
        tuned = clone(model_prototype)
    
    # Train and predict on this fold
    summary = config.get("logging", {}).get("summary", True)
    split_preds, _ = train_and_evaluate_model(
        tuned,
        X_tr, y_tr,
        splits={f"Fold{fold_idx}": (X_te, y_te)},
        model_name=None,
        model_dir=None,
        save_model=False,
        SUMMARY=summary,
        config=config
    )
    
    proba, y_eval = split_preds[f"Fold{fold_idx}"]
    
    # Threshold sweep for this fold
    sweep, bc, ba = threshold_sweep_with_cost(
        proba, y_eval, C_FP=C_FP, C_FN=C_FN,
        SUMMARY=summary, split_name=f"Fold{fold_idx}"
    )
    
    # Create results
    cost_result = _mk_result("temp", f"Fold{fold_idx}", bc, 'cost')
    accuracy_result = _mk_result("temp", f"Fold{fold_idx}", ba, 'accuracy')
    
    return cost_result, accuracy_result, proba, sweep 