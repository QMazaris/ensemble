import os
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import confusion_matrix
from dataclasses import dataclass, asdict
from typing import Optional
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import RandomizedSearchCV
from sklearn.base import clone

@dataclass
class ModelEvaluationResult:
    model_name: str
    split: str
    threshold_type: str       # 'cost', 'accuracy', or 'base' for base models
    threshold: float = None   # None for base models
    precision: float = None
    recall: float = None
    accuracy: float = None
    cost: float = None
    tp: int = None
    fp: int = None
    tn: int = None
    fn: int = None
    is_base_model: bool = False  # True for base model results

@dataclass
class ModelEvaluationRun:
    model_name: str
    results: list  # List of ModelEvaluationResult (including base and meta models)
    probabilities: Optional[dict] = None  # Dict of model_name -> np.ndarray (including base models if available)

# ---------- Data Preparation ----------
def prepare_data(df, config):
    # ... function body unchanged ...
    df = df.copy()
    target = config.TARGET
    y = df[target].map({"Good": 0, "Bad": 1})
    if y.isnull().any():
        raise ValueError(f"Found unmapped values in {target}")
    exclude_cols = config.EXCLUDE_COLS + [config.TARGET]
    X = df.drop(columns=exclude_cols)

    if config.SUMMARY:
        # ─── New print ───────────────────────────────────────────────────────────────
        features_before_encoding = X.columns.tolist()
        print(f"Features before encoding ({len(features_before_encoding)}):")
        for feat in features_before_encoding:
            print(f"  • {feat}")
        # ──────────────────────────────────────────────────────────────────────────────

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

def apply_variance_filter(X, threshold=0.01, SUMMARY = None):
    selector = VarianceThreshold(threshold=threshold)
    X_reduced = selector.fit_transform(X)
    kept_cols = X.columns[selector.get_support()]
    if SUMMARY:
        print(f"🔍 Variance Filter: Kept {len(kept_cols)}/{X.shape[1]} features (threshold = {threshold})")
    return pd.DataFrame(X_reduced, columns=kept_cols, index=X.index)

def apply_correlation_filter(X, threshold=0.9, SUMMARY=None):
    corr_matrix = X.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    X_reduced = X.drop(columns=to_drop)
    if SUMMARY:
        print(f"🔍 Correlation Filter: Dropped {len(to_drop)} features (threshold = {threshold})")
    return X_reduced

def optimize_hyperparams(model_name, model, X_train, y_train, config):
    """Random‑search CV on a single train split."""
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
    # ... function body unchanged ...
    smote = SMOTE(random_state=42)
    X_bal, y_bal = smote.fit_resample(X_train, y_train)
    model.fit(X_bal, y_bal)
    if save_model and model_name and model_dir:
        os.makedirs(model_dir, exist_ok=True)
        path = os.path.join(model_dir, f"{model_name}.pkl")
        joblib.dump(model, path)
        if SUMMARY:
            print(f"Saved trained model to {path}")
    split_preds = {}
    for split_name, (X_eval, y_eval) in splits.items():
        if SUMMARY:
            print(f"Predicting probabilities on {split_name} split...")
        proba = model.predict_proba(X_eval)[:, 1]
        split_preds[split_name] = (proba, y_eval)
    return split_preds, model

def compute_metrics(y_true, y_pred, C_FP, C_FN, as_dict=True):
    # ... function body unchanged ...
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    precision = 100 * tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = 100 * tp / (tp + fn) if (tp + fn) > 0 else 0
    accuracy = 100 * (tp + tn) / (tp + tn + fp + fn)
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

def threshold_sweep_with_cost(proba, y_true, C_FP, C_FN, thresholds=None, SUMMARY=None, split_name=None):
    if thresholds is None:
        thresholds = np.linspace(0, 1, 21)
    best_by_cost = {'threshold': None, 'cost': float('inf')}
    best_by_accuracy = {'threshold': None, 'accuracy': -1}
    sweep_results = {}
    for thr in thresholds:
        y_pred = (proba >= thr).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        cost = C_FP * fp + C_FN * fn
        total_mistakes = fp + fn
        metrics = compute_metrics(y_true, y_pred, C_FP, C_FN)
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
        if cost < best_by_cost['cost']:
            best_by_cost = {'threshold': thr, 'cost': cost, **metrics}
        if metrics['accuracy'] > best_by_accuracy['accuracy']:
            best_by_accuracy = {'threshold': thr, 'accuracy': metrics['accuracy'], **metrics}
    if SUMMARY:
        if split_name:
            print(f"\n=== {split_name.upper()} SPLIT ===")
        print(f"Best threshold by cost: {best_by_cost['threshold']} with expected cost {best_by_cost['cost']}")
        print(f"  Confusion Matrix (cost-opt): [[TN={best_by_cost['tn']} FP={best_by_cost['fp']}], [FN={best_by_cost['fn']} TP={best_by_cost['tp']}]]")
        print(f"Best threshold by accuracy: {best_by_accuracy['threshold']} with accuracy {best_by_accuracy['accuracy']:.2f}%")
        print(f"  Confusion Matrix (acc-opt):  [[TN={best_by_accuracy['tn']} FP={best_by_accuracy['fp']}], [FN={best_by_accuracy['fn']} TP={best_by_accuracy['tp']}]]\n")
    return sweep_results, best_by_cost, best_by_accuracy

def plot_threshold_sweep(sweep_results, C_FP, C_FN, output_path=None, cost_optimal_thr=None, accuracy_optimal_thr=None, SUMMARY = None):
    # ... function body unchanged ...
    thresholds = list(sweep_results.keys())
    precision = [sweep_results[t]['precision'] for t in thresholds]
    recall = [sweep_results[t]['recall'] for t in thresholds]
    accuracy = [sweep_results[t]['accuracy'] for t in thresholds]
    cost = [sweep_results[t].get('cost', 0) for t in thresholds]
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(thresholds, precision, 'b-', label='Precision: % of predicted bad welds that were actually bad')
    ax1.plot(thresholds, recall, 'r-', label='Recall: % of actual bad welds caught')
    ax1.plot(thresholds, accuracy, color='black', linewidth=2.5, label='Accuracy: % of all welds correctly classified')
    ax1.set_xlabel('Confidence Threshold (higher = stricter)')
    ax1.set_ylabel('Percentage')
    ax1.set_ylim(0, 105)
    ax1.grid(True)
    ax2 = ax1.twinx()
    cost_line = ax2.plot(thresholds, cost, 'm--', label=f'Cost (FP={C_FP}, FN={C_FN})')
    ax2.set_ylabel('Cost', color='m')
    ax2.tick_params(axis='y', labelcolor='m')
    if cost_optimal_thr is not None:
        ax1.axvline(x=cost_optimal_thr, color='red', linestyle='--', alpha=0.7, label='Cost-Optimal Threshold')
    if accuracy_optimal_thr is not None:
        try:
            acc_val = accuracy[thresholds.index(accuracy_optimal_thr)]
            ax1.axvline(x=accuracy_optimal_thr, color='orange', linestyle='--', alpha=0.7, label='Accuracy-Optimal Threshold')
        except ValueError:
            pass
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    fig.legend(lines1 + lines2, labels1 + labels2, loc='upper center', bbox_to_anchor=(0.5, 0.96), ncol=2, frameon=False)
    plt.suptitle('Effect of Confidence Threshold on Model Performance and Cost', y=0.99)
    plt.subplots_adjust(top=0.85)
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        if SUMMARY:
            print(f"Saved threshold sweep plot to {output_path}")
    plt.close(fig)

def plot_runs_at_threshold(runs, threshold_type, split_name='Test', C_FP=1.0, C_FN=1.0, output_path=None):
    # ... function body unchanged ...
    split_name = split_name.capitalize()
    if threshold_type not in ('cost','accuracy'):
        raise ValueError("threshold_type must be 'cost' or 'accuracy'")
    labels, precisions, recalls, accuracies, costs, thresholds = [], [], [], [], [], []
    for run in runs:
        match = next((r for r in run.results if r.split == split_name and r.threshold_type == threshold_type), None)
        if not match:
            match = next((r for r in run.results if r.split == split_name and r.is_base_model), None)
        if not match:
            continue
        labels.append(run.model_name)
        precisions.append(match.precision or 0.0)
        recalls.append(match.recall or 0.0)
        accuracies.append(match.accuracy or 0.0)
        costs.append(match.cost or 0.0)
        thresholds.append(match.threshold if match.threshold is not None else np.nan)
    if not labels:
        raise RuntimeError(f"No results for split={split_name}, threshold_type={threshold_type}")
    x = np.arange(len(labels))
    w = 0.2
    fig, ax = plt.subplots(figsize=(14, 6))
    p_b = ax.bar(x - 1.5*w, precisions, w, label='Precision')
    r_b = ax.bar(x - 0.5*w, recalls,    w, label='Recall')
    a_b = ax.bar(x + 0.5*w, accuracies, w, label='Accuracy')
    ax2 = ax.twinx()
    c_b = ax2.bar(x + 1.5*w, costs, w, label='Cost', alpha=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha='right')
    ax.set_ylabel('Percentage')
    ax2.set_ylabel('Cost')
    def _annotate(bars, axis, fmt, offset):
        for bar in bars:
            h = bar.get_height()
            axis.text(bar.get_x() + bar.get_width()/2, h + offset, fmt(h), ha='center', va='bottom', fontsize=8)
    _annotate(p_b, ax, lambda h: f"{h:.1f}%", 0.5)
    _annotate(r_b, ax, lambda h: f"{h:.1f}%", 0.5)
    _annotate(a_b, ax, lambda h: f"{h:.1f}%", 0.5)
    max_cost = max(costs) if costs else 0
    cost_offset = max_cost * 0.01
    _annotate(c_b, ax2, lambda h: f"{int(h)}", cost_offset)
    fig.subplots_adjust(bottom=0.25)
    for i, thr in enumerate(thresholds):
        if not np.isnan(thr):
            ax.text(x[i], -0.05, f"Thr: {thr:.2f}", ha='center', va='top', rotation=45, fontsize=8, transform=ax.get_xaxis_transform())
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    combined_handles = h1 + h2
    combined_labels  = l1 + l2
    ax.legend(combined_handles, combined_labels, loc='upper left', ncol=2)
    fig.subplots_adjust(top=0.85)
    ax.relim()
    ax.autoscale_view()
    ylim = ax.get_ylim()
    ax.set_ylim(ylim[0], ylim[1] * 1.15)
    ax2.relim()
    ax2.autoscale_view()
    ylim2 = ax2.get_ylim()
    ax2.set_ylim(ylim2[0], ylim2[1] * 1.15)
    ratio = C_FN / C_FP if C_FP else float('inf')
    ax.set_title(f"Model Performance Comparison\n({split_name} Data, {threshold_type.capitalize()}‑opt Thresholds, Cost Ratio {ratio:.1f})")
    plt.tight_layout()
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    plt.close(fig)

def save_all_model_probabilities_from_structure(results_total, predictions_dir, index, y_true, SUMMARY = None):
    # ... function body unchanged ...
    os.makedirs(predictions_dir, exist_ok=True)
    wide_df = pd.DataFrame(index=index)
    if not results_total:
        raise ValueError("results_total is empty")
    wide_df['GT'] = y_true
    def pick_full_or_longest(probas):
        if not probas:
            return None
        if 'Full' in probas:
            return probas['Full']
        non_empty = [arr for arr in probas.values() if arr is not None and hasattr(arr, 'shape')]
        if not non_empty:
            return None
        return max(non_empty, key=lambda arr: arr.shape[0])
    for run in results_total:
        model_col = run.model_name
        probas = pick_full_or_longest(run.probabilities)
        if probas is None:
            print(f"[WARN] No probabilities found for model {model_col}, skipping.")
            continue
        if len(probas) == len(index):
            wide_df[model_col] = probas
        else:
            wide_df[model_col] = pd.Series(probas, index=index[: len(probas)])
    out_path = os.path.join(predictions_dir, 'all_model_predictions.csv')
    wide_df.to_csv(out_path, index=False)
    if SUMMARY:
        print(f"Saved GT as first column + one prob‐column per model to {out_path}")

def print_performance_summary(runs, meta_model_names, splits=('Train', 'Test', 'Full'), thr_types=('cost', 'accuracy')):
    print("\n" + "="*80)
    print("SUMMARY OF MODEL PERFORMANCE")
    print("="*80)
    def _find(model, split, thr):
        for run in runs:
            if run.model_name != model:
                continue
            for r in run.results:
                if r.split == split and r.threshold_type == thr:
                    return r
        return None
    for model in meta_model_names:
        found_any = False
        for split in splits:
            for thr in thr_types:
                r = _find(model, split, thr)
                if r:
                    found_any = True
                    break
            if found_any:
                break
        if not found_any:
            continue
        print(f"\n{model}:")
        for split in splits:
            for thr in thr_types:
                r = _find(model, split, thr)
                if not r:
                    continue
                print(f"  {split} Split ({thr}-optimal):")
                print(f"    Accuracy:  {r.accuracy:.1f}%  Precision: {r.precision:.1f}%  Recall: {r.recall:.1f}%")
                print(f"    Threshold: {r.threshold:.3f}")
                print(f"    Cost:      {r.cost}")
                print(f"    Confusion Matrix: [[TN={r.tn} FP={r.fp}], [FN={r.fn} TP={r.tp}]]")
    all_models = {run.model_name for run in runs}
    base_models = all_models - set(meta_model_names)
    print("\nBASE MODELS PERFORMANCE (preset confidence threshold):")
    for model in base_models:
        found_any = False
        for split in splits:
            r = _find(model, split, 'base')
            if r:
                found_any = True
                break
        if not found_any:
            continue
        print(f"\n{model}:")
        for split in splits:
            r = _find(model, split, 'base')
            if not r:
                continue
            print(f"  {split} Split:")
            print(f"    Accuracy:  {r.accuracy:.1f}%  Precision: {r.precision:.1f}%  Recall: {r.recall:.1f}%")
            print(f"    Cost:      {r.cost}")
            print(f"    Confusion Matrix: [[TN={r.tn} FP={r.fp}], [FN={r.fn} TP={r.tp}]]")
    # Only print K-FOLD AVERAGED MODEL PERFORMANCE if any kfold_avg_ models exist
    if any(run.model_name.startswith("kfold_avg_") for run in runs):
        print("\nK-FOLD AVERAGED MODEL PERFORMANCE:")
        for run in runs:
            if run.model_name.startswith("kfold_avg_"):
                if not run.results:
                    continue
                print(f"\n{run.model_name}:")
                for r in run.results:
                    print(f"  {r.split} Split ({r.threshold_type}-optimal):")
                    print(f"    Accuracy:  {r.accuracy:.1f}%  Precision: {r.precision:.1f}%  Recall: {r.recall:.1f}%")
                    print(f"    Threshold: {r.threshold:.3f}")
                    print(f"    Cost:      {r.cost}")
                    print(f"    Confusion Matrix: [[TN={r.tn} FP={r.fp}], [FN={r.fn} TP={r.tp}]]")

def Save_Feature_Info(model_path, df, feature_cols, encoded_cols):
    # ... function body unchanged ...
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

def Regular_Split(config, X, y):
    # ... function body unchanged ...
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
    return X_train,y_train,train_idx,test_idx,single_splits

def CV_Split(config, X, y):
    # ... function body unchanged ...
    kf = StratifiedKFold(
            n_splits = config.N_SPLITS,
            shuffle  = True,
            random_state = getattr(config, 'RANDOM_STATE', 42)
        )
    cv_splits = []
    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X, y), start=1):
        X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
        X_te, y_te = X.iloc[test_idx],  y.iloc[test_idx]
        cv_splits.append( (f"Fold{fold_idx}", (X_tr, y_tr), (X_te, y_te)) )
    return cv_splits

def _mk_result(model_name, split_name, best, thr_type):
    return ModelEvaluationResult(
        model_name=model_name,
        split=split_name,
        threshold_type=thr_type,
        threshold=best['threshold'],
        precision=best['precision'],
        recall=best['recall'],
        accuracy=best['accuracy'],
        cost=best['cost'],
        tp=best['tp'], fp=best['fp'],
        tn=best['tn'], fn=best['fn'],
        is_base_model=False
    )

def _average_results(res_list, model_name):
    fields = ['precision', 'recall', 'accuracy', 'cost', 'tp', 'fp', 'tn', 'fn']
    avg = {f: float(np.mean([getattr(r, f) for r in res_list])) for f in fields}
    return ModelEvaluationResult(
        model_name=f'kfold_avg_{model_name}',
        split='Full',
        threshold_type=res_list[0].threshold_type,
        threshold=float(np.mean([r.threshold for r in res_list])),
        is_base_model=False,
        **avg
    )

def _average_probabilities(prob_arrays):
    if not prob_arrays:
        return None
    min_len = min(map(len, prob_arrays))
    return np.mean([p[:min_len] for p in prob_arrays], axis=0)

def save_final_kfold_model(model, X, y, model_name, model_dir, SUMMARY=True):
    """Train and save the final K-Fold model on the full dataset."""
    # Use train_and_evaluate_model to fit and save the model on the full data
    _ = train_and_evaluate_model(
        model, X, y,
        splits={'Full': (X, y)},
        model_name=model_name,
        model_dir=model_dir,
        save_model=True,
        SUMMARY=SUMMARY
    )
    if SUMMARY:
        print(f"Final K-Fold model '{model_name}' trained and saved on the full dataset.")
