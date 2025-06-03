import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.feature_selection import VarianceThreshold

def prepare_data(df, config):
    """Prepare data for modeling by encoding categorical variables and handling the target."""
    df = df.copy()
    target = config.get("data", {}).get("target_column", "target")
    
    # Get configurable good/bad tags
    good_tag = config.get("data", {}).get("good_tag", "Good")
    bad_tag = config.get("data", {}).get("bad_tag", "Bad")
    
    # Handle target mapping with configurable tags
    y = df[target].map({good_tag: 0, bad_tag: 1})
    if y.isnull().any():
        raise ValueError(f"Found unmapped values in {target}. Expected values: '{good_tag}' or '{bad_tag}'")
    
    # Safely handle base_model_decisions - it can be either a list or a dictionary
    base_model_decisions_config = config.get("models", {}).get("base_model_decisions", [])
    
    # Handle both list and dictionary formats
    if isinstance(base_model_decisions_config, list):
        # If it's a list, use it directly as the column names
        decision_columns = base_model_decisions_config
    elif isinstance(base_model_decisions_config, dict):
        # If it's a dictionary, get the enabled_columns
        decision_columns = base_model_decisions_config.get("enabled_columns", [])
    else:
        # Fallback to empty list if neither format
        decision_columns = []
    
    decision_columns_in_data = [col for col in decision_columns if col in df.columns]
    
    # Only exclude columns that actually exist in the dataframe
    exclude_cols_config = config.get("data", {}).get("exclude_columns", [])
    exclude_cols_existing = [col for col in exclude_cols_config if col in df.columns]
    
    # Combine regular exclude columns with decision columns and target
    exclude_cols = exclude_cols_existing + [target] + decision_columns_in_data
    
    summary = config.get("logging", {}).get("summary", True)
    if summary and decision_columns_in_data:
        print(f"📊 Automatically excluding decision columns from training: {decision_columns_in_data}")
    
    X = df.drop(columns=exclude_cols)

    if summary:
        features_before_encoding = X.columns.tolist()
        print(f"Features before encoding ({len(features_before_encoding)}):")
        for feat in features_before_encoding:
            print(f"  • {feat}")

    # Ensure y is properly formatted
    y = y.astype(int)
    
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
    X_encoded = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
    encoded_cols = [col for col in X_encoded.columns if col not in numeric_cols]
    
    return X_encoded, y, numeric_cols, encoded_cols

def apply_variance_filter(X, threshold=0.01, SUMMARY=True):
    """Apply variance threshold filter to remove low-variance features."""
    if SUMMARY:
        print(f"\nApplying variance filter (threshold={threshold})...")
        print(f"Features before variance filter: {X.shape[1]}")
    
    selector = VarianceThreshold(threshold=threshold)
    X_filtered = selector.fit_transform(X)
    
    # Get feature names for the filtered dataset
    feature_mask = selector.get_support()
    filtered_feature_names = X.columns[feature_mask]
    X_filtered = pd.DataFrame(X_filtered, columns=filtered_feature_names, index=X.index)
    
    if SUMMARY:
        removed_count = X.shape[1] - X_filtered.shape[1]
        print(f"Features after variance filter: {X_filtered.shape[1]} (removed {removed_count})")
    
    return X_filtered

def apply_correlation_filter(X, threshold=0.95, SUMMARY=True):
    """Remove highly correlated features."""
    if SUMMARY:
        print(f"\nApplying correlation filter (threshold={threshold})...")
        print(f"Features before correlation filter: {X.shape[1]}")
    
    # Calculate correlation matrix
    corr_matrix = X.corr().abs()
    
    # Find pairs of highly correlated features
    upper_tri = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )
    
    # Find features to drop
    to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > threshold)]
    
    # Drop the features
    X_filtered = X.drop(columns=to_drop)
    
    if SUMMARY:
        print(f"Features after correlation filter: {X_filtered.shape[1]} (removed {len(to_drop)})")
        if to_drop and len(to_drop) <= 10:  # Only show if not too many
            print(f"Removed features: {to_drop}")
    
    return X_filtered

def Regular_Split(config, X, y):
    """Create a regular train/test split."""
    test_size = config.get("data", {}).get("test_size", 0.2)
    random_state = config.get("data", {}).get("random_state", 42)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    train_idx = X_train.index
    test_idx  = X_test.index
    single_splits = {
        'Train': (X_train, y_train),
        'Test':  (X_test,  y_test),
        'Full':  (X,       y)
    }
    return X_train, y_train, train_idx, test_idx, single_splits

def get_cv_splitter(config):
    """Get a configured StratifiedKFold splitter for consistent cross-validation."""
    n_splits = config.get("training", {}).get("n_splits", 5)
    random_state = config.get("data", {}).get("random_state", 42)
    
    return StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state
    )

def CV_Split(config, X, y):
    """Create K-fold cross-validation splits."""
    kf = get_cv_splitter(config)
    cv_splits = []
    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X, y), start=1):
        X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
        X_te, y_te = X.iloc[test_idx],  y.iloc[test_idx]
        cv_splits.append((f"Fold{fold_idx}", (X_tr, y_tr), (X_te, y_te)))
    return cv_splits

def Save_Feature_Info(model_dir, df, numeric_cols, encoded_cols):
    """Save feature information for later use."""
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    feature_info = {
        'numeric_cols': numeric_cols,
        'encoded_cols': encoded_cols,
        'all_columns': df.columns.tolist()
    }
    
    feature_info_path = os.path.join(model_dir, 'feature_info.pkl')
    joblib.dump(feature_info, feature_info_path)
    print(f"Feature information saved to {feature_info_path}") 