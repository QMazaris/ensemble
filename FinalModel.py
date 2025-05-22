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
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
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

from config import *  # Import all constants and variables
create_directories()  # Ensure output folders exist

# ---------- PATHS (edit these as needed) ----------

# Output plot files
THRESHOLD_SWEEP_PLOT = "MyPlots/ai_threshold_sweep3.0.png"
OUTPUT_IMAGE_PATH = "MyPlots/"

# Constants for deployment
C_FP = 1    # Cost of scrapping a good weld
C_FN = 30   # Cost of shipping a bad weld

# ---------- Data Preparation ----------
def prepare_data(df):
    """Prepare features and target from dataframe.
    
    Args:
        df: Pandas DataFrame with raw data
        
    Returns:
        X: Feature matrix
        y: Target vector
        feature_cols: List of numeric feature names
        weld_columns: List of one-hot encoded weld type columns
    """
    # Start with the original feature list
    feature_cols = [
        "AnomalyScore",
        "RegionArea",
        "MaxVal",
        "RegionCount",
        "BurnThru",
        "Concavity",
        "Good",
        "Porosity",
        "Skip",
        "RegionRow",
        "RegionCol",
        "RegionAreaFrac",
        "CL_ConfMax",
        "CL_ConfMargin",
        "SegThresh",
        "ClassThresh"
    ]

    # One-hot encode the Weld column
    weld_dummies = pd.get_dummies(df['Weld'], prefix='Weld')

    # Combine numeric features with encoded weld types
    X = pd.concat([df[feature_cols], weld_dummies], axis=1)
    
    # Encode target
    y = df["GT_Label"].map({"Good": 0, "Bad": 1})
    
    return X, y, feature_cols, list(weld_dummies.columns)

def train_and_predict(model, X_train, y_train, X_eval):
    """Train the model with SMOTE balancing and make predictions.
    
    Args:
        model: The model to train
        X_train: Training features
        y_train: Training target
        X_eval: Evaluation features

    
    Returns:
        tuple: (probabilities, predictions)
    """
    # Apply SMOTE to balance classes
    print("Applying SMOTE to balance classes...")
    smote = SMOTE(random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
    
    # Train model
    print("Training model...")
    model.fit(X_train_balanced, y_train_balanced)
    
    # Make predictions
    print("Making predictions...")
    proba = model.predict_proba(X_eval)[:, 1]
    return proba

# ---------- Metrics Computation ----------
def compute_metrics(y_true, y_pred, as_dict=True):
    """Compute classification metrics and cost.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        as_dict: If True, return metrics as a dictionary
    
    Returns:
        tuple or dict: (precision, recall, accuracy) or dictionary of metrics
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

def threshold_sweep_with_cost(proba, y_true, thresholds=None, C_FP=None, C_FN=None):
    """Perform threshold sweep analysis with cost calculation.
    
    Args:
        proba: array-like, predicted probabilities
        y_true: array-like, ground truth labels (0 or 1)
        thresholds: list or array of thresholds to evaluate
        C_FP: cost of false positive
        C_FN: cost of false negative
    
    Returns:
        sweep_results: dict mapping threshold -> metrics (including cost)
        best_by_cost: dict with metrics for threshold with lowest cost
        best_by_accuracy: dict with metrics for threshold with highest accuracy (minimum mistakes)
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
        
        metrics = compute_metrics(y_true, y_pred)
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
        ax1.axvline(cost_optimal_thr, color='limegreen', linestyle='--', linewidth=2, label=f'Cost-optimal Threshold ({cost_optimal_thr:.2f})')
    if accuracy_optimal_thr is not None:
        ax1.axvline(accuracy_optimal_thr, color='orange', linestyle=':', linewidth=2, label=f'Accuracy-optimal Threshold ({accuracy_optimal_thr:.2f})')
        # Add a marker at the accuracy-optimal threshold on the accuracy curve
        try:
            acc_idx = thresholds.index(accuracy_optimal_thr)
            acc_val = accuracy[acc_idx]
            ax1.plot(accuracy_optimal_thr, acc_val, marker='o', markersize=10, color='orange', label='Accuracy Peak')
        except ValueError:
            # In case the threshold is not exactly in the list due to float precision
            closest_idx = min(range(len(thresholds)), key=lambda i: abs(thresholds[i] - accuracy_optimal_thr))
            ax1.plot(thresholds[closest_idx], accuracy[closest_idx], marker='o', markersize=10, color='orange', label='Accuracy Peak')

    # Combine legends from both axes
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='best')
    
    # Add title with cost information
    plt.title('Effect of Confidence Threshold on Model Performance and Cost')
    
    # Adjust layout and save if needed
    fig.tight_layout()
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path)
        print(f"Saved threshold sweep plot to {output_path}")
    
    plt.close()

def plot_all_models_at_threshold(output_path, results_df, threshold_type, split_type='test', model_thresholds=None):
    """
    Plot precision, recall, and accuracy for all models using model-specific thresholds.
    
    This function visualizes model performance metrics across different models. It can show:
    - Precision: Percentage of positive predictions that were correct
    - Recall: Percentage of actual positives that were identified
    - Accuracy: Percentage of all predictions that were correct
    - Cost: Calculated cost based on false positives and false negatives
    
    The function can filter results by data split (train, test, or all) and threshold type (cost or accuracy).
    
    Args:
        output_path: Path to save the plot as an image file (use None to not save)
        results_df: DataFrame containing model evaluation metrics
        threshold_type: Type of threshold to use ('cost' or 'accuracy')
        split_type: Which data split to visualize ('train', 'test', or 'all')
        model_thresholds: Dictionary mapping model names to their thresholds (kept for backward compatibility)
        
    Returns:
        None (displays or saves the plot)
        
    Example:
        plot_all_models_at_threshold(
            output_path='model_comparison.png',
            results_df=my_results,
            threshold_type='cost',
            split_type='test'
        )
    """
    # Convert split_type to lowercase for case-insensitive comparison
    split_type = split_type.lower()
    if split_type not in ['train', 'test', 'all']:
        raise ValueError("split_type must be 'train', 'test', or 'all'")
    
    # Filter results_df based on split_type and threshold_type
    if split_type == 'all':
        # Use only the 'Full' split for the full dataset stats
        metrics_df = results_df[(results_df['split'].str.lower() == 'full') & 
                              (results_df['threshold_type'] == threshold_type)]
        base_models_df = results_df[(results_df['split'].str.lower() == 'full') &
                                 (results_df['model'].isin(['AD_Decision', 'CL_Decision', 'AD_or_CL_Fail']))]
    else:
        split_name = split_type.capitalize()
        metrics_df = results_df[(results_df['split'] == split_name) & 
                              (results_df['threshold_type'] == threshold_type)]
        # Get base models for the specified split
        base_models_df = results_df[
            (results_df['split'] == split_name) &
            (results_df['model'].isin(['AD_Decision', 'CL_Decision', 'AD_or_CL_Fail']))
        ]
    
    # Combine both DataFrames
    metrics_df = pd.concat([metrics_df, base_models_df]).drop_duplicates()
    
    # Prepare data for plotting
    models = metrics_df['model'].unique()
    metrics = {
        'Model': [],
        'Precision': [],
        'Recall': [],
        'Accuracy': [],
        'Cost': [],
        'Threshold': []
    }
    
    for model in models:
        model_data = metrics_df[metrics_df['model'] == model].iloc[0]  # Get first row for this model
        metrics['Model'].append(model)
        metrics['Precision'].append(model_data.get('precision', 0))
        metrics['Recall'].append(model_data.get('recall', 0))
        metrics['Accuracy'].append(model_data.get('accuracy', 0))
        metrics['Cost'].append(model_data.get('cost', 0))
        metrics['Threshold'].append(model_data.get('threshold', None))
    
    # Create the plot
    x = np.arange(len(metrics['Model']))
    width = 0.2
    
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Plot metrics
    precision_bars = ax.bar(x - 1.5*width, metrics['Precision'], width, label='Precision', color='#4F81BD')
    recall_bars = ax.bar(x - 0.5*width, metrics['Recall'], width, label='Recall', color='#C0504D')
    accuracy_bars = ax.bar(x + 0.5*width, metrics['Accuracy'], width, label='Accuracy', color='#9BBB59')
    
    # Add cost on secondary y-axis
    ax2 = ax.twinx()
    cost_bars = ax2.bar(x + 1.5*width, metrics['Cost'], width, label='Cost', color='gray', alpha=0.6)
    
    # Add cost ratio annotation to the title
    cost_ratio = C_FN / C_FP
    split_display = split_type.capitalize()
    if split_type == 'all':
        split_display = 'Full Dataset'
    
    # Function to add value labels on top of bars
    def add_value_labels(bars, ax, is_cost=False, decimals=1):
        for bar in bars:
            height = bar.get_height()
            if is_cost:
                # For cost, use integers since it's a count
                label = f'{int(height)}'
                va = 'bottom' if height >= 0 else 'top'
                y = height + (5 if height >= 0 else -5)
            else:
                # For percentages, show 1 decimal place
                label = f'{height:.{decimals}f}%'
                va = 'bottom' if height >= 0 else 'top'
                y = height + 0.5  # Small offset for percentages
                
            ax.text(bar.get_x() + bar.get_width()/2., y,
                    label,
                    ha='center', va=va, rotation=0, fontsize=8)
    
    # Add value labels to all bars
    add_value_labels(precision_bars, ax)
    add_value_labels(recall_bars, ax)
    add_value_labels(accuracy_bars, ax)
    add_value_labels(cost_bars, ax2, is_cost=True)
    
    # Customize the plot
    ax.set_xticks(x)
    ax.set_xticklabels(metrics['Model'], rotation=45, ha='right')
    ax.set_ylabel('Percentage')
    ax2.set_ylabel('Cost', color='gray')
    ax2.tick_params(axis='y', labelcolor='gray')
    
    # Add threshold annotations
    for i, (model, threshold) in enumerate(zip(metrics['Model'], metrics['Threshold'])):
        if threshold is not None and not np.isnan(threshold):
            ax.text(i, -5, f'Thr: {threshold:.2f}', ha='center', va='top', rotation=45, fontsize=8)
    
    # Add legend and title
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    # Set the title with cost ratio information
    plt.title(f'Model Performance Comparison\n({split_display} Data, {threshold_type.capitalize()}-optimized Thresholds, Cost Ratio: {cost_ratio:.1f})')
    plt.subplots_adjust(top=0.9)  # Add more space for the title
    plt.tight_layout()
    
    # Ensure the MyPlots directory exists
    plots_dir = 'MyPlots'
    os.makedirs(plots_dir, exist_ok=True)
    
    # Create default filename if none provided
    if output_path is None:
        output_path = os.path.join(plots_dir, f'model_comparison_{split_type}_{threshold_type}_{pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")}.png')
    
    # Save the plot if output_path is provided
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {os.path.abspath(output_path)}")
    
    # Always show the plot
    # plt.show()
    
    # Don't close the plot to keep it visible

# ---------- Main Function ----------
def main(config):
    """Main function to run the entire pipeline.
    
    Args:
        config: Configuration object containing paths and parameters
    """

    # Unpack config
    csv_path = config.DATA_PATH
    model_path = config.MODEL_OUTPUT_PATH
    
    # Define model zoo
    MODELS = config.MODELS
    
    # Ensure output directories exist
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    # Load data
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Prepare data
    print("Preparing data...")
    X, y, feature_cols, weld_columns = prepare_data(df)
    
    # Save feature columns for later use
    # Update the feature_info dictionary to include encoding details
    feature_info = {
        'feature_cols': feature_cols,  # Original numeric features
        'weld_columns': weld_columns,  # One-hot encoded Weld columns
        'encoding': {
            'Weld': {
                'type': 'one_hot',
                'original_column': 'Weld',
                'categories': sorted(df['Weld'].unique().tolist())  # All unique Weld values seen during training
            }
        }
    }
    joblib.dump(feature_info, os.path.join(os.path.dirname(model_path), "feature_info.pkl"))
    
    # Split data once
    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y)
    train_idx = X_train.index
    test_idx = X_test.index
    
    # Define evaluation splits
    splits = {
        'Train': (X_train, y_train),
        'Test': (X_test, y_test),
        'Full': (X, y)
    }
    
    # Collect results for each model and split
    results = []  # collect per-model, per-split rows
    best_models = {}
    
    # Initialize dictionary to store all probabilities and GT
    # Prepare wide-format DataFrame indexed by original DataFrame
    wide_df = pd.DataFrame(index=df.index)
    wide_df['GT'] = y.values
    
    # Loop through all models
    for model_name, model in MODELS.items():
        print(f"\n{'-'*50}")
        print(f"Training {model_name}...")
        model_instance = model  # Create a fresh instance
        
        # Train on training data
        print(f"Training {model_name}...")
        
        # Loop through all splits for evaluation
        for split_name, (X_eval, y_eval) in splits.items():
            print(f"\nEvaluating on {split_name} split...")
            
            # For training split, we don't need to predict again
            if split_name == 'Train':
                proba = train_and_predict(model_instance, X_train, y_train, X_train)
            else:
                # For test and full, we use the trained model to predict
                proba = train_and_predict(model_instance, X_train, y_train, X_eval)
            
            
            sweep, best_cost, best_acc = threshold_sweep_with_cost(
                proba,
                y_eval,
                thresholds=np.linspace(0,1,21),
                C_FP=C_FP,
                C_FN=C_FN
            )

            cost_optimal_thr = best_cost['threshold']
            accuracy_optimal_thr = best_acc['threshold']
            
            # Plot the threshold sweep with cost and accuracy lines
            plot_threshold_sweep(
                sweep, 
                output_path=f"MyPlots/{model_name}_{split_name.lower()}_threshold_sweep.png",
                cost_optimal_thr=cost_optimal_thr,
                accuracy_optimal_thr=accuracy_optimal_thr,
                C_FP=C_FP,
                C_FN=C_FN
            )
            plt.close()


            # print("Cost‑optimal threshold:    ", cost_optimal_thr)
            # print("Accuracy‑optimal threshold:", accuracy_optimal_thr)

            # Evaluate cost-optimal threshold
            y_pred_cost = (proba >= cost_optimal_thr).astype(int)
            metrics_cost = compute_metrics(y_eval, y_pred_cost)

            # Evaluate accuracy-optimal threshold
            y_pred_acc = (proba >= accuracy_optimal_thr).astype(int)
            metrics_acc = compute_metrics(y_eval, y_pred_acc)

            # print("Cost-optimal metrics:", metrics_cost)
            # print("Accuracy-optimal metrics:", metrics_acc)

            
            results.append({
                'model': model_name,
                'split': split_name,
                'threshold_type': 'cost',
                'threshold': cost_optimal_thr,
                **metrics_cost
            })

            results.append({
                'model': model_name,
                'split': split_name,
                'threshold_type': 'accuracy',
                'threshold': accuracy_optimal_thr,
                **metrics_acc
            })

            # Store probabilities in wide format, aligned to original indices
            col_name = f"{model_name}_{split_name}"
            proba_col = np.full(shape=len(df), fill_value=np.nan)
            # y_eval.index gives the positions in the original DataFrame
            if proba is not None:
                # If X_eval is a DataFrame, get its index; else, try to use y_eval.index
                if hasattr(y_eval, 'index'):
                    proba_col[y_eval.index] = proba
                else:
                    # fallback: assign in order (should not happen)
                    proba_col[:len(proba)] = proba
            wide_df[col_name] = proba_col
            
            # # Print basic metrics
            # print(f"Accuracy: {metrics_cost['accuracy']:.1f}%")
            # print(f"Precision: {metrics_cost['precision']:.1f}%")
            # print(f"Recall: {metrics_cost['recall']:.1f}%")

            


        # Save the trained model for this model_name (modular, dynamic)
        model_output_path = f"{model_path}/meta_model_v2.1_{model_name}.pkl"
        joblib.dump(model_instance, model_output_path)
        print(f"Saved {model_name} model as {model_output_path}")

    # Create a DataFrame from results
    results_df = pd.DataFrame(results)

    # ========== BASE MODEL METRICS SECTION ==========
    # Add base model predictions to wide_df
    for base in ['AD_Decision', 'CL_Decision']:
        wide_df[base] = (df[base] == 'Bad').astype(float)
    # Combined index: fails if either base model is "Bad"
    wide_df['AD_or_CL_Fail'] = ((df['AD_Decision'] == 'Bad') | (df['CL_Decision'] == 'Bad')).astype(float)

    # Compute and append metrics for base models for each split
    base_metrics = []
    for base in ['AD_Decision', 'CL_Decision', 'AD_or_CL_Fail']:
        for split_name, idx in zip(['Train', 'Test', 'Full'], [train_idx, test_idx, df.index]):
            if base == 'AD_or_CL_Fail':
                y_pred = wide_df.loc[idx, 'AD_or_CL_Fail'].values
            else:
                y_pred = wide_df.loc[idx, base].values
            y_true = y.loc[idx].values
            metrics = compute_metrics(y_true, y_pred)
            base_metrics.append({
                'model': base,
                'split': split_name,
                'threshold_type': 'cost',
                'threshold': None,
                **metrics
            })
    
    # Concatenate all base metrics at once
    if base_metrics:
        base_metrics_df = pd.DataFrame(base_metrics)
        results_df = pd.concat([results_df, base_metrics_df], ignore_index=True) if not results_df.empty else base_metrics_df
    # ========== END BASE MODEL METRICS SECTION ==========

   # Print summary table
    print("\n" + "="*80)
    print("SUMMARY OF MODEL PERFORMANCE")
    print("="*80)

    # Group by model and split, then aggregate metrics
    summary = results_df.groupby(['model', 'split']).agg({
        'accuracy': 'mean',
        'precision': 'mean',
        'recall': 'mean'
    }).reset_index()
    
    # Print formatted summary with confusion matrix and cost by threshold type for meta-models
    for model_name in MODELS.keys():
        print(f"\n{model_name}:")
        model_results = results_df[(results_df['model'] == model_name)]
        for split in ['Train', 'Test', 'Full']:
            for thr_type in ['cost', 'accuracy']:
                row = model_results[(model_results['split'] == split) & (model_results['threshold_type'] == thr_type)]
                if not row.empty:
                    row = row.iloc[0]
                    print(f"  {split} Split ({thr_type}-optimal):")
                    print(f"    Accuracy:  {row['accuracy']:.1f}%  Precision: {row['precision']:.1f}%  Recall: {row['recall']:.1f}%")
                    print(f"    Threshold: {row['threshold']:.3f}")
                    print(f"    Cost:      {row['cost']}")
                    print(f"    Confusion Matrix: [[TN={row['tn']} FP={row['fp']}], [FN={row['fn']} TP={row['tp']}]]")

    # Print summary for base models (preset confidence threshold, not optimized)
    print("\nBASE MODELS PERFORMANCE (preset confidence threshold, not optimized):")
    base_model_names = [name for name in results_df['model'].unique() if name not in MODELS.keys()]
    for model_name in base_model_names:
        print(f"\n{model_name}:")
        model_results = results_df[(results_df['model'] == model_name) & (results_df['threshold_type'] == 'cost')]
        for split in ['Train', 'Test', 'Full']:
            row = model_results[model_results['split'] == split]
            if not row.empty:
                row = row.iloc[0]
                print(f"  {split} Split (preset confidence threshold):")
                print(f"    Accuracy:  {row['accuracy']:.1f}%  Precision: {row['precision']:.1f}%  Recall: {row['recall']:.1f}%")
                print(f"    Cost:      {row['cost']}")
                print(f"    Confusion Matrix: [[TN={row['tn']} FP={row['fp']}], [FN={row['fn']} TP={row['tp']}]]")


    # Save all probabilities in wide format to CSV
    predictions_path = os.path.join(os.path.dirname(model_path), 'all_model_predictions.csv')
    wide_df.to_csv(predictions_path, index=False)
    print(f"Saved all model probabilities in wide format to {predictions_path}")

    print(results_df['split'].unique())


    # Plot all models at each of the three threshold types using model-specific thresholds
    # 1. Cost-optimized thresholds for test data
    plot_all_models_at_threshold(
        output_path=os.path.join(OUTPUT_IMAGE_PATH, 'all_models_test_cost_threshold.png'),
        results_df=results_df, 
        threshold_type='cost',
        split_type='test',
        model_thresholds=best_models
    )
    plot_all_models_at_threshold(
        output_path=os.path.join(OUTPUT_IMAGE_PATH, 'all_models_test_accuracy_threshold.png'),
        results_df=results_df, 
        threshold_type='accuracy',
        split_type='test',
        model_thresholds=best_models
    )
    print(f"\nPlotted all models with cost-optimized thresholds (test split)")
    
    # 2. Cost-optimized thresholds for train data
    plot_all_models_at_threshold(
        output_path=os.path.join(OUTPUT_IMAGE_PATH, 'all_models_full_cost_threshold.png'),
        results_df=results_df, 
        threshold_type='cost',
        split_type='all',
        model_thresholds=best_models
    )
    plot_all_models_at_threshold(
        output_path=os.path.join(OUTPUT_IMAGE_PATH, 'all_models_full_accuracy_threshold.png'),
        results_df=results_df, 
        threshold_type='accuracy',
        split_type='all',
        model_thresholds=best_models
    )

if __name__ == "__main__":
    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        main()