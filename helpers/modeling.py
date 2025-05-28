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
from .metrics import threshold_sweep_with_cost
from .plotting import plot_threshold_sweep
from .model_export import export_model

def optimize_hyperparams(model_name, model, X_train, y_train, config):
    """Bayesian optimization using Optuna to find optimal hyperparameters.
    
    The hyperparameter space is defined in config.HYPERPARAM_SPACE with a nested structure:
    - 'shared': Parameters shared across all models
    - model_name: Model-specific parameters
    """
    space = config.HYPERPARAM_SPACE
    if not space or 'shared' not in space:
        if config.SUMMARY:
            print(f"⚠️ No hyperparameter space defined for {model_name}, using default parameters")
        return model

    def objective(trial):
        # Create a fresh model instance
        model_clone = clone(model)
        
        # Build parameters from the config space
        params = {}
        
        # Add shared parameters
        for param_name, param_space in space['shared'].items():
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
        
        # Add model-specific parameters if they exist
        if model_name in space:
            for param_name, param_space in space[model_name].items():
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

        # Set the parameters
        model_clone.set_params(**params)
        
        # Use cross-validation to evaluate the model
        scores = cross_val_score(
            model_clone, X_train, y_train,
            cv=3,
            scoring='neg_log_loss',
            n_jobs=config.N_JOBS
        )
        
        # Return the mean score (Optuna minimizes, so we negate the score)
        return -scores.mean()

    # Create and run the study
    study = optuna.create_study(
        direction='minimize',
        sampler=optuna.samplers.TPESampler(seed=config.RANDOM_STATE)
    )
    
    if config.SUMMARY:
        print(f"\n🔍 Optimizing hyperparameters for {model_name}...")
        print(f"   Shared parameters: {space['shared']}")
        if model_name in space:
            print(f"   Model-specific parameters: {space[model_name]}")
    
    study.optimize(
        objective,
        n_trials=config.HYPERPARAM_ITER,
        show_progress_bar=config.SUMMARY
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
    
    if config.SUMMARY:
        print(f"✅ {model_name} optimization complete:")
        print(f"   Best parameters: {cleaned_params}")
        print(f"   Best CV score: {-study.best_value:.4f}")
    
    return best_model

def train_and_evaluate_model(model, X_train, y_train, splits: dict, model_name: str = None, model_dir: str = None, save_model: bool = True, SUMMARY = None, config=None, feature_names=None):
    """Train a model and evaluate it on multiple splits."""
    # Apply SMOTE if configured
    if getattr(model, 'use_smote', True):  # Default to True if not specified
        smote = SMOTE(random_state=42)
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

def FinalModelCreateAndAnalyize(config, model_path, image_path, C_FP, C_FN, SAVE_PLOTS, X, y):
    """Create and analyze final production models.
    
    Args:
        config: Configuration object containing model settings
        model_path: Directory to save models
        image_path: Directory to save plots
        C_FP: Cost of false positive
        C_FN: Cost of false negative
        SAVE_PLOTS: Whether to save threshold sweep plots
        X: Feature matrix
        y: Target vector
    """
    if not config.OPTIMIZE_FINAL_MODEL:
        if config.SUMMARY:
            print("\n⚠️ Final model optimization is disabled. Skipping production model training.")
        return

    if config.SAVE_MODEL:
        if config.SUMMARY:
            print("\n" + "="*80)
            print("FINAL PRODUCTION MODELS")
            print("="*80)

        for name, prototype in config.MODELS.items():
            if config.SUMMARY:
                print(f"\n🏭 Optimizing final production model: {name}")
            # Get fresh instance and optimize hyperparameters
            base = clone(prototype)
            if config.OPTIMIZE_HYPERPARAMS:
                best = optimize_hyperparams(name, base, X, y, config)
            else:
                best = base  # Use default parameters if optimization is disabled
            
            # Train the model on full dataset
            best.fit(X, y)
            
            # Perform threshold sweep on full dataset
            if config.SUMMARY:
                print(f"\n📊 Performing threshold sweep for {name} on full dataset...")
            proba = best.predict_proba(X)[:, 1]
            sweep_results, best_cost, best_acc = threshold_sweep_with_cost(
                proba, y, C_FP=C_FP, C_FN=C_FN,
                SUMMARY=config.SUMMARY,
                split_name=f"FINAL PRODUCTION {name}"
            )
            
            # Save threshold sweep plot
            if SAVE_PLOTS:
                plot_threshold_sweep(
                    sweep_results, C_FP, C_FN,
                    cost_optimal_thr=best_cost['threshold'],
                    accuracy_optimal_thr=best_acc['threshold'],
                    output_path=os.path.join(
                        image_path,
                        f"{name}_final_production_threshold_sweep.png"
                    ),
                    SUMMARY=config.SUMMARY
                )
            
            # Save the model
            out_path = export_model(best, f"{name}_production", model_path, config, feature_names=X.columns.tolist())
            if config.SUMMARY:
                print(f"✅ Saved production model: {out_path}")
                print(f"   • Cost-optimal threshold: {best_cost['threshold']:.3f}")
                print(f"   • Accuracy-optimal threshold: {best_acc['threshold']:.3f}")
