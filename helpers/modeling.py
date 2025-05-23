import os
import joblib
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import RandomizedSearchCV
from sklearn.base import clone

# Import necessary functions from other helpers
from .metrics import threshold_sweep_with_cost
from .plotting import plot_threshold_sweep

def optimize_hyperparams(model_name, model, X_train, y_train, config):
    """Random-search CV on a single train split."""
    space = config.HYPERPARAM_SPACE.get(model_name, {})
    if not space:
        return model
    search = RandomizedSearchCV(
        estimator          = clone(model),
        param_distributions= space,
        n_iter             = config.HYPERPARAM_ITER,
        cv                 = 3,
        scoring            = 'neg_log_loss',
        n_jobs             = config.N_JOBS,
        random_state       = config.RANDOM_STATE,
        verbose            = 1
    )
    search.fit(X_train, y_train)
    if config.SUMMARY:
        print(f"🔧 {model_name} best params → {search.best_params_}")
    return search.best_estimator_

def train_and_evaluate_model(model, X_train, y_train, splits: dict, model_name: str = None, model_dir: str = None, save_model: bool = True, SUMMARY = None):
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
        os.makedirs(model_dir, exist_ok=True)
        path = os.path.join(model_dir, f"{model_name}.pkl")
        joblib.dump(model, path)
        if SUMMARY:
            print(f"Saved trained model to {path}")

    # Generate predictions for each split
    split_preds = {}
    for split_name, (X_eval, y_eval) in splits.items():
        if SUMMARY:
            print(f"Predicting probabilities on {split_name} split...")
        proba = model.predict_proba(X_eval)[:, 1]
        split_preds[split_name] = (proba, y_eval)

    return split_preds, model

def save_final_kfold_model(model, X, y, model_name, model_dir, SUMMARY=None):
    """Train and save the final K-Fold model on the full dataset."""
    trained_model, _ = train_and_evaluate_model(
        model, X, y,
        splits={'Full': (X, y)},
        model_name=f'{model_name}_final',
        model_dir=model_dir,
        save_model=True,
        SUMMARY=SUMMARY
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
    if config.SAVE_MODEL:
        if config.SUMMARY:
            print("\n" + "="*80)
            print("FINAL PRODUCTION MODELS")
            print("="*80)

        for name, prototype in config.MODELS.items():
            if config.OPTIMIZE_FINAL_MODEL:
                if config.SUMMARY:
                    print(f"\n🏭 Optimizing final production model: {name}")
                # Get fresh instance and optimize hyperparameters
                base = clone(prototype)
                best = optimize_hyperparams(name, base, X, y, config)
                
                # Train the optimized model on full dataset
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
                
                # Save the optimized model
                out_path = os.path.join(model_path, f"{name}_production.pkl")
                joblib.dump(best, out_path)
                if config.SUMMARY:
                    print(f"✅ Saved optimized production model: {out_path}")
                    print(f"   • Cost-optimal threshold: {best_cost['threshold']:.3f}")
                    print(f"   • Accuracy-optimal threshold: {best_acc['threshold']:.3f}")
            else:
                # For non-optimized final models, just train and save
                if config.SUMMARY:
                    print(f"\n🏭 Training final production model: {name}")
                model = clone(prototype)
                model.fit(X, y)
                out_path = os.path.join(model_path, f"{name}_production.pkl")
                joblib.dump(model, out_path)
                if config.SUMMARY:
                    print(f"✅ Saved production model: {out_path}")
