import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.feature_selection import VarianceThreshold

def prepare_data(df, config):
    """Prepare data for modeling by encoding categorical variables and handling the target."""
    df = df.copy()
    target = config.TARGET
    
    # Get configurable good/bad tags
    good_tag = getattr(config, 'GOOD_TAG', 'Good')
    bad_tag = getattr(config, 'BAD_TAG', 'Bad')
    
    # Handle target mapping with configurable tags
    y = df[target].map({good_tag: 0, bad_tag: 1})
    if y.isnull().any():
        raise ValueError(f"Found unmapped values in {target}. Expected values: '{good_tag}' or '{bad_tag}'")
    
    # Automatically exclude decision columns from training if they exist
    decision_columns = getattr(config, 'BASE_MODEL_DECISION_COLUMNS', [])
    decision_columns_in_data = [col for col in decision_columns if col in df.columns]
    
    # Only exclude columns that actually exist in the dataframe
    exclude_cols_config = getattr(config, 'EXCLUDE_COLS', [])
    exclude_cols_existing = [col for col in exclude_cols_config if col in df.columns]
    
    # Combine regular exclude columns with decision columns and target
    exclude_cols = exclude_cols_existing + [config.TARGET] + decision_columns_in_data
    
    if config.SUMMARY and decision_columns_in_data:
        print(f"📊 Automatically excluding decision columns from training: {decision_columns_in_data}")
    
    X = df.drop(columns=exclude_cols)

    if config.SUMMARY:
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

def apply_variance_filter(X, threshold=0.01, SUMMARY=None):
    """Remove features with variance below threshold."""
    selector = VarianceThreshold(threshold=threshold)
    X_reduced = selector.fit_transform(X)
    kept_cols = X.columns[selector.get_support()]
    if SUMMARY:
        print(f"🔍 Variance Filter: Kept {len(kept_cols)}/{X.shape[1]} features (threshold = {threshold})")
    return pd.DataFrame(X_reduced, columns=kept_cols, index=X.index)

def apply_correlation_filter(X, threshold=0.9, SUMMARY=None):
    """Remove features that are positively correlated above the threshold."""
    corr_matrix = X.corr()  # KEEP the sign
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    # Only drop features with positive correlation above the threshold
    to_drop = [column for column in upper.columns if (upper[column] > threshold).any()]
    X_reduced = X.drop(columns=to_drop)
    if SUMMARY:
        print(f"🔍 Correlation Filter: Dropped {len(to_drop)} features (threshold = {threshold})")
    return X_reduced

def Regular_Split(config, X, y):
    """Create a regular train/test split."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=getattr(config,'RANDOM_STATE',42),
        stratify=y
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
    return StratifiedKFold(
        n_splits=config.N_SPLITS,
        shuffle=True,
        random_state=getattr(config, 'RANDOM_STATE', 42)
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

def Save_Feature_Info(model_path, df, feature_cols, encoded_cols):
    """Save feature encoding information for later use."""
    encoding_info = {}
    for col in df.columns:
        if col not in feature_cols and col in df.select_dtypes(include=['object', 'category', 'bool']).columns:
            encoding_info[col] = {
                'type': 'one_hot',
                'original_column': col,
                'categories': sorted(df[col].unique().tolist())
            }
    feature_info = {
        'feature_cols': feature_cols,  # Original numeric features
        'encoded_cols': encoded_cols,  # One-hot encoded columns
        'encoding': encoding_info
    }
    joblib.dump(feature_info, os.path.join(model_path, "feature_info.pkl")) 