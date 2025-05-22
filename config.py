"""
Configuration file containing constants and file paths for the ensemble pipeline.
"""

import os
from pathlib import Path
import numpy as np

# Model imports for config
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

# ===== File System Configuration =====
# Paths
DATA_PATH = 'C:/Users/QuinnMazaris/Desktop/Stacking/Results/ensemble_resultsV3.csv'
OUTPUT_DIR = 'output/'
MODEL_DIR = os.path.join(OUTPUT_DIR, 'models')
PLOT_DIR = os.path.join(OUTPUT_DIR, 'plots')
PREDICTIONS_DIR = os.path.join(OUTPUT_DIR, 'predictions')
LOG_DIR = os.path.join(OUTPUT_DIR, 'logs')

# ===== Data Configuration =====
# Target variable
TARGET = 'GT_Label'  # Update with your target variable name
EXCLUDE_COLS = [    
    "Image",       # File reference, not useful for model
    "AD_Decision", # Base model predictions (used separately)
    "CL_Decision",
    "AD_ClassID",  # Internal IDs
    "CL_ClassID",
    "GT_Label_Num" # Duplicate of GT_Label (encoded)
]

# Data splitting
TEST_SIZE = 0.2  # Proportion of data to use for testing
RANDOM_STATE = 42  # Random seed for reproducibility

# ===== Model Configuration =====
# Define model zoo
MODELS = {
    'XGBoost': xgb.XGBClassifier(
        objective='binary:logistic',
        max_depth=4,                    # Reduced depth to prevent overfitting
        learning_rate=0.1,              # Reduced learning rate for better convergence
        n_estimators=400,               # Increased number of trees
        subsample=0.8,                  # Slightly lower subsample ratio
        colsample_bytree=0.8,           # Slightly lower column sampling
        min_child_weight=5,             # Increased to prevent overfitting
        gamma=0.1,                      # Slightly higher gamma for more conservative learning
        scale_pos_weight=62.5,          # Set to negative/positive ratio (2374/38 ≈ 62.5)
        random_state=42,
        n_jobs=-1,
        eval_metric='logloss'           # Better for imbalanced data
    ),
    'RandomForest': RandomForestClassifier(
        n_estimators=300,               # Increased number of trees
        max_depth=6,                    # Slightly deeper trees
        min_samples_split=10,           # Increased to prevent overfitting
        min_samples_leaf=5,             # Increased to prevent overfitting
        class_weight='balanced_subsample',  # Better handling of class imbalance
        max_features='sqrt',            # Consider square root of features per tree
        bootstrap=True,                 # Enable bootstrap sampling
        oob_score=True,                 # Enable out-of-bag score
        n_jobs=-1,
        random_state=42
    ),
    # 'MLP': MLPClassifier(
    #     hidden_layer_sizes=(64, 32),    # Two hidden layers
    #     activation='relu',
    #     solver='adam',
    #     alpha=1e-4,
    #     batch_size=32,                  # Batch size for mini-batch training
    #     learning_rate_init=0.001,       # Initial learning rate
    #     learning_rate='adaptive',        # Automatically adjust learning rate
    #     early_stopping=True,
    #     validation_fraction=0.1,        # Use 10% of training for validation
    #     beta_1=0.9,                     # Adam optimizer parameters
    #     beta_2=0.999,
    #     epsilon=1e-8,
    #     max_iter=500,                   # Increased max iterations
    #     tol=1e-4,                       # Tolerance for optimization
    #     n_iter_no_change=20,            # Stop if no improvement for 20 epochs
    #     random_state=42
    # ),
    # 'KNN': KNeighborsClassifier(
    #     n_neighbors=10,                 # Increased neighbors for more stable predictions
    #     weights='distance',             # Weight by inverse distance
    #     algorithm='auto',               # Let the algorithm choose the best method
    #     leaf_size=30,                   # Default leaf size
    #     p=2,                           # Euclidean distance
    #     metric='minkowski',            # Standard distance metric
    #     n_jobs=-1                      # Use all available cores
    # ),
    # 'LogisticRegression': LogisticRegression(
    #     class_weight='balanced',
    #     penalty='l2',
    #     C=0.1,                         # More regularization
    #     solver='saga',                  # Better for larger datasets
    #     max_iter=5000,                  # Increased max iterations
    #     tol=1e-4,                       # Tolerance for stopping criteria
    #     random_state=42,
    #     n_jobs=-1,                      # Use all available cores
    #     warm_start=False,               # Don't reuse previous solution
    #     l1_ratio=None                   # Only used when penalty='elasticnet'
    # ),
    # 'SVM': SVC(
    #     kernel='rbf',
    #     C=10.0,                        # Higher C for more complex decision boundary
    #     gamma='scale',                  # Kernel coefficient
    #     class_weight='balanced',
    #     probability=True,              # Enable probability estimates
    #     cache_size=1000,               # Larger cache for better performance
    #     max_iter=5000,                 # Increased max iterations
    #     decision_function_shape='ovr',  # One-vs-rest for binary classification
    #     random_state=42,
    #     tol=1e-4,                      # Tolerance for stopping criteria
    #     shrinking=True                 # Use shrinking heuristic
    # )
}

# Cost weights for evaluation
C_FP = 1  # Cost of false positive
C_FN = 20  # Cost of false negative (typically higher than C_FP)

# ===== Cross-Validation Settings =====
USE_KFOLD = False  # Enable/disable k-fold cross-validation
N_SPLITS = 5  # Number of folds for K-fold cross-validation
STRATIFIED_KFOLD = True  # Use stratified k-fold for classification

# ===== Hyperparameter Optimization =====
OPTIMIZE_HYPERPARAMS = False  # Enable/disable hyperparameter optimization
HYPERPARAM_ITER = 50  # Number of iterations for randomized search
N_JOBS = -1  # Number of jobs to run in parallel (-1 uses all available cores)

# ===== Training Configuration =====
USE_SMOTE = True  # Apply SMOTE for imbalanced data
SMOTE_RATIO = 0.5  # Ratio of minority class to majority class after SMOTE

# ===== Output & Logging =====
SAVE_MODEL = True  # Save trained models
SAVE_PLOTS = True  # Save evaluation plots
SAVE_PREDICTIONS = True  # Save predictions
SUMMARY = True

# ===== Create Output Directories =====
dirs_to_create = [
    OUTPUT_DIR, 
    MODEL_DIR, 
    PLOT_DIR, 
    PREDICTIONS_DIR, 
    LOG_DIR
]

def create_directories():
    """Create all necessary output directories if they don't exist."""
    print("📁 Ensuring output directories exist...")
    for directory in dirs_to_create:
        try:
            os.makedirs(directory, exist_ok=True)
            print(f"✓ Created or found: {directory}")
        except Exception as e:
            print(f"❌ Error creating {directory}: {str(e)}")

