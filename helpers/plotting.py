import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from collections import Counter

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

def plot_class_balance(y, output_path=None, SUMMARY=None):
    """Plot class distribution as a bar chart with polished styling.
    
    Args:
        y: Target vector (0/1 or True/False)
        output_path: Path to save the plot
        SUMMARY: Whether to print status messages
    """
    # Count classes
    counts = Counter(np.array(y))
    labels = ['Good (0)', 'Bad (1)']
    values = [counts.get(0, 0), counts.get(1, 0)]
    total = sum(values)
    percents = [v / total * 100 for v in values]

    # Set up figure
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor('#f5f5f5')
    ax.set_facecolor('#ffffff')

    # Plot bars
    bars = ax.bar(labels, values, 
                  width=0.6, 
                  edgecolor='gray', 
                  linewidth=1.2, 
                  alpha=0.85)

    # Remove top/right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Grid lines
    ax.yaxis.grid(True, linestyle='--', alpha=0.6)
    ax.xaxis.grid(False)

    # Title & axis labels
    ax.set_title('Class Distribution in Dataset', pad=15, fontsize=16, weight='bold')
    ax.set_ylabel('Number of Samples', fontsize=14)
    ax.tick_params(axis='both', which='major', labelsize=12)

    # Annotate counts & percentages
    for bar, pct in zip(bars, percents):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + total*0.01,
                f'{int(h):,}\n({pct:.1f}%)',
                ha='center', va='bottom',
                fontsize=12, fontweight='semibold',
                color='#333333')

    # Tight layout & save
    plt.tight_layout()
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        if SUMMARY:
            print(f"Saved class balance plot to {output_path}")
    plt.close(fig)