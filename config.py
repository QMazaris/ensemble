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

# which DataFrame columns hold the raw scores for base models
BASE_MODEL_OUTPUT_COLUMNS = {
    "AD": "AnomalyScore",    # raw continuous score from your anomaly detector
    "CL": "CL_ConfMax"       # "good %" column from your classifier
}

# ===== File System Configuration =====
# Paths
DATA_PATH = 'data/training_data.csv'
OUTPUT_DIR = 'output/'
MODEL_DIR = os.path.join(OUTPUT_DIR, 'models')
PLOT_DIR = os.path.join(OUTPUT_DIR, 'plots')
PREDICTIONS_DIR = os.path.join(OUTPUT_DIR, 'predictions')
LOG_DIR = os.path.join(OUTPUT_DIR, 'logs')

# List of directories to create
dirs_to_create = [
    OUTPUT_DIR, 
    MODEL_DIR, 
    PLOT_DIR, 
    PREDICTIONS_DIR, 
    LOG_DIR
]

# ===== Data Configuration =====
# Target variable
TARGET = 'GT_Label'
EXCLUDE_COLS = ['Image', 'AD_Decision', 'CL_Decision', 'AD_ClassID', 'CL_ClassID', 'GT_Label_Num', 'ClassThresh', 'SegThresh']


# Data splitting
TEST_SIZE = 0.2  # Proportion of data to use for testing
RANDOM_STATE = 42  # Random seed for reproducibility

# Very conservative, minor gains
FilterData = False
VARIANCE_THRESH = 0.01
CORRELATION_THRESH = 0.95

# ===== Model Configuration =====
# Define model zoo
# For tree models, more features is almost always better
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
# Cost of false positive
C_FP = 1
#Cost of false negative (typically higher than C_FP)
C_FN = 30

# ===== Cross-Validation Settings =====
USE_KFOLD = False
N_SPLITS = 5

# ===== Hyperparameter Optimization =====
OPTIMIZE_HYPERPARAMS = False
HYPERPARAM_ITER = 50
# Final full-data tuning
OPTIMIZE_FINAL_MODEL = False
N_JOBS = -1  # Number of jobs to run in parallel (-1 uses all available cores)

# Parameter grids for RandomizedSearchCV
HYPERPARAM_SPACE = {
    'XGBoost': {
        'max_depth':        [3,4,5,6],
        'learning_rate':    [0.01,0.05,0.1],
        'subsample':        [0.6,0.8,1.0],
        'colsample_bytree': [0.6,0.8,1.0],
        'n_estimators':     [100,200,400],
        'gamma':            [0,0.1,0.2],
    },
    'RandomForest': {
        'n_estimators':      [100,200,300],
        'max_depth':         [None,5,10],
        'min_samples_split': [2,5,10],
        'min_samples_leaf':  [1,2,5],
    },
    # add other models if you like…
}

# ===== Training Configuration =====
USE_SMOTE = True  # Apply SMOTE for imbalanced data
SMOTE_RATIO = 0.5  # Ratio of minority class to majority class after SMOTE

# ===== Output & Logging =====
SAVE_MODEL = True
SAVE_PLOTS = True
SAVE_PREDICTIONS = True
SUMMARY = True