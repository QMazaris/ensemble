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
    y = df[target].map({"Good": 0, "Bad": 1})
    if y.isnull().any():
        raise ValueError(f"Found unmapped values in {target}")
    exclude_cols = config.EXCLUDE_COLS + [config.TARGET]
    X = df.drop(columns=exclude_cols)

    if config.SUMMARY:
        features_before_encoding = X.columns.tolist()
        print(f"Features before encoding ({len(features_before_encoding)}):")
        for feat in features_before_encoding:
            print(f"  • {feat}")

    y_raw = df[config.TARGET]
    if y_raw.dtype == object or y_raw.dtype.name == 'category':
        y = y_raw.map({'Good': 0, 'Bad': 1})
    else:
        y = y_raw
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
    """Remove highly correlated features."""
    corr_matrix = X.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
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

def CV_Split(config, X, y):
    """Create K-fold cross-validation splits."""
    kf = StratifiedKFold(
        n_splits = config.N_SPLITS,
        shuffle  = True,
        random_state = getattr(config, 'RANDOM_STATE', 42)
    )
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
