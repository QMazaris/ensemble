import os
import pandas as pd
import numpy as np

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