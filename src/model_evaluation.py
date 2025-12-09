"""Model Evaluation and Visualization

This module provides functions for evaluating classification models and
generating visualizations including ROC curves, confusion matrices, feature
importance plots, and precision-recall curves.
"""
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Try to import matplotlib, but don't fail if not available
try:
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    print("Warning: matplotlib not available. Plotting functions will be disabled.")


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_pred_proba: Optional[np.ndarray] = None) -> Dict:
    """Compute classification metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_pred_proba: Predicted probabilities (optional, for ROC-AUC)
    
    Returns:
        Dict with metrics
    """
    # Basic metrics
    TP = np.sum((y_true == 1) & (y_pred == 1))
    TN = np.sum((y_true == 0) & (y_pred == 0))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))
    
    accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    metrics = {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'true_positives': int(TP),
        'true_negatives': int(TN),
        'false_positives': int(FP),
        'false_negatives': int(FN)
    }
    
    # ROC-AUC if probabilities provided
    if y_pred_proba is not None:
        try:
            from sklearn.metrics import roc_auc_score
            roc_auc = roc_auc_score(y_true, y_pred_proba)
            metrics['roc_auc'] = float(roc_auc)
        except:
            # Compute manually
            roc_auc = compute_roc_auc_manual(y_true, y_pred_proba)
            metrics['roc_auc'] = float(roc_auc)
    
    return metrics


def compute_roc_auc_manual(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    """Compute ROC-AUC manually using trapezoidal rule.
    
    Args:
        y_true: True binary labels
        y_scores: Predicted scores/probabilities
    
    Returns:
        ROC-AUC score
    """
    # Sort by score
    desc_score_indices = np.argsort(y_scores)[::-1]
    y_true_sorted = y_true[desc_score_indices]
    y_scores_sorted = y_scores[desc_score_indices]
    
    # Compute TPR and FPR at each threshold
    tprs = []
    fprs = []
    
    n_pos = np.sum(y_true == 1)
    n_neg = np.sum(y_true == 0)
    
    if n_pos == 0 or n_neg == 0:
        return 0.5
    
    tp = 0
    fp = 0
    
    for i, label in enumerate(y_true_sorted):
        if label == 1:
            tp += 1
        else:
            fp += 1
        
        tpr = tp / n_pos
        fpr = fp / n_neg
        
        tprs.append(tpr)
        fprs.append(fpr)
    
    # Compute AUC using trapezoidal rule
    auc = 0.0
    for i in range(1, len(fprs)):
        auc += (fprs[i] - fprs[i-1]) * (tprs[i] + tprs[i-1]) / 2
    
    return auc


def plot_roc_curve(y_true: np.ndarray, y_pred_proba: np.ndarray, output_path: str, title: str = "ROC Curve") -> None:
    """Generate and save ROC curve plot.
    
    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        output_path: Path to save plot
        title: Plot title
    """
    if not PLOTTING_AVAILABLE:
        print("Matplotlib not available. Skipping ROC curve plot.")
        return
    
    try:
        from sklearn.metrics import roc_curve, auc
        fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
        roc_auc = auc(fpr, tpr)
    except ImportError:
        # Manual computation
        sorted_indices = np.argsort(y_pred_proba)[::-1]
        y_true_sorted = y_true[sorted_indices]
        
        n_pos = np.sum(y_true == 1)
        n_neg = np.sum(y_true == 0)
        
        tpr_list = []
        fpr_list = []
        tp, fp = 0, 0
        
        for label in y_true_sorted:
            if label == 1:
                tp += 1
            else:
                fp += 1
            tpr_list.append(tp / n_pos if n_pos > 0 else 0)
            fpr_list.append(fp / n_neg if n_neg > 0 else 0)
        
        fpr = np.array(fpr_list)
        tpr = np.array(tpr_list)
        roc_auc = compute_roc_auc_manual(y_true, y_pred_proba)
    
    # Plot
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"ROC curve saved to: {output_path}")


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, output_path: str, title: str = "Confusion Matrix") -> None:
    """Generate and save confusion matrix heatmap.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        output_path: Path to save plot
        title: Plot title
    """
    if not PLOTTING_AVAILABLE:
        print("Matplotlib not available. Skipping confusion matrix plot.")
        return
    
    # Compute confusion matrix
    TP = np.sum((y_true == 1) & (y_pred == 1))
    TN = np.sum((y_true == 0) & (y_pred == 0))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))
    
    cm = np.array([[TN, FP], [FN, TP]])
    
    # Plot
    plt.figure(figsize=(8, 6))
    
    if 'seaborn' in globals():
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Predicted 0', 'Predicted 1'],
                    yticklabels=['Actual 0', 'Actual 1'])
    else:
        plt.imshow(cm, interpolation='nearest', cmap='Blues')
        plt.colorbar()
        
        # Add text annotations
        for i in range(2):
            for j in range(2):
                plt.text(j, i, str(cm[i, j]), ha='center', va='center', fontsize=20)
        
        plt.xticks([0, 1], ['Predicted 0', 'Predicted 1'])
        plt.yticks([0, 1], ['Actual 0', 'Actual 1'])
    
    plt.title(title)
    plt.tight_layout()
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Confusion matrix saved to: {output_path}")


def plot_feature_importance(
    feature_names: List[str],
    coefficients: np.ndarray,
    output_path: str,
    top_n: int = 20,
    title: str = "Feature Importance"
) -> None:
    """Generate and save feature importance bar chart.
    
    Args:
        feature_names: List of feature names
        coefficients: Model coefficients (importance values)
        output_path: Path to save plot
        top_n: Number of top features to show
        title: Plot title
    """
    if not PLOTTING_AVAILABLE:
        print("Matplotlib not available. Skipping feature importance plot.")
        return
    
    # Get absolute importance
    importance = np.abs(coefficients)
    
    # Sort and get top N
    indices = np.argsort(importance)[::-1][:top_n]
    top_features = [feature_names[i] for i in indices]
    top_importance = importance[indices]
    top_coefficients = coefficients[indices]
    
    # Plot
    plt.figure(figsize=(10, 8))
    colors = ['red' if c < 0 else 'green' for c in top_coefficients]
    plt.barh(range(len(top_features)), top_importance, color=colors, alpha=0.7)
    plt.yticks(range(len(top_features)), top_features)
    plt.xlabel('Absolute Coefficient Value')
    plt.title(title)
    plt.gca().invert_yaxis()  # Highest importance at top
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Feature importance saved to: {output_path}")


def plot_precision_recall_curve(y_true: np.ndarray, y_pred_proba: np.ndarray, output_path: str, title: str = "Precision-Recall Curve") -> None:
    """Generate and save precision-recall curve.
    
    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        output_path: Path to save plot
        title: Plot title
    """
    if not PLOTTING_AVAILABLE:
        print("Matplotlib not available. Skipping precision-recall curve.")
        return
    
    # Sort by probability descending
    sorted_indices = np.argsort(y_pred_proba)[::-1]
    y_true_sorted = y_true[sorted_indices]
    
    # Compute precision and recall at each threshold
    precisions = []
    recalls = []
    
    n_pos = np.sum(y_true == 1)
    tp = 0
    fp = 0
    
    for label in y_true_sorted:
        if label == 1:
            tp += 1
        else:
            fp += 1
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / n_pos if n_pos > 0 else 0
        
        precisions.append(precision)
        recalls.append(recall)
    
    # Plot
    plt.figure(figsize=(8, 6))
    plt.plot(recalls, precisions, color='blue', lw=2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(title)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Precision-recall curve saved to: {output_path}")


def evaluate_model(
    model,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    output_dir: str
) -> Dict:
    """Complete model evaluation with metrics and plots.
    
    Args:
        model: Trained model with predict and predict_proba methods
        X: Feature matrix
        y: True labels
        feature_names: List of feature names
        output_dir: Directory to save plots and metrics
    
    Returns:
        Dict with all metrics
    """
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Make predictions
    y_pred = model.predict(X)
    y_pred_proba = model.predict_proba(X)[:, 1]
    
    # Compute metrics
    metrics = compute_metrics(y, y_pred, y_pred_proba)
    
    print(f"\n{'='*70}")
    print("MODEL EVALUATION METRICS")
    print(f"{'='*70}")
    print(f"Accuracy:  {metrics['accuracy']:.3f}")
    print(f"Precision: {metrics['precision']:.3f}")
    print(f"Recall:    {metrics['recall']:.3f}")
    print(f"F1 Score:  {metrics['f1']:.3f}")
    print(f"ROC-AUC:   {metrics['roc_auc']:.3f}")
    print(f"\nConfusion Matrix:")
    print(f"  TN: {metrics['true_negatives']:3d}  FP: {metrics['false_positives']:3d}")
    print(f"  FN: {metrics['false_negatives']:3d}  TP: {metrics['true_positives']:3d}")
    print(f"{'='*70}\n")
    
    # Generate plots
    if PLOTTING_AVAILABLE:
        plot_roc_curve(
            y, y_pred_proba,
            str(output_dir / 'roc_curve.png'),
            title="ROC Curve"
        )
        
        plot_confusion_matrix(
            y, y_pred,
            str(output_dir / 'confusion_matrix.png'),
            title="Confusion Matrix"
        )
        
        if hasattr(model, 'coef_'):
            plot_feature_importance(
                feature_names,
                model.coef_[0],
                str(output_dir / 'feature_importance.png'),
                top_n=20,
                title="Top 20 Most Important Features"
            )
        
        plot_precision_recall_curve(
            y, y_pred_proba,
            str(output_dir / 'precision_recall_curve.png'),
            title="Precision-Recall Curve"
        )
    
    # Save metrics to JSON
    metrics_path = output_dir / 'metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to: {metrics_path}")
    
    return metrics


def generate_evaluation_report(metrics: Dict, output_path: str, asset_name: str) -> None:
    """Generate markdown evaluation report.
    
    Args:
        metrics: Dict with evaluation metrics
        output_path: Path to save markdown report
        asset_name: Name of asset being evaluated
    """
    report = []
    report.append(f"# Model Evaluation Report - {asset_name}")
    report.append(f"\n**Generated**: {Path(output_path).parent.name}")
    
    report.append("\n## Classification Metrics")
    report.append(f"\n| Metric | Value |")
    report.append(f"|--------|-------|")
    report.append(f"| Accuracy | {metrics['accuracy']:.3f} |")
    report.append(f"| Precision | {metrics['precision']:.3f} |")
    report.append(f"| Recall | {metrics['recall']:.3f} |")
    report.append(f"| F1 Score | {metrics['f1']:.3f} |")
    report.append(f"| ROC-AUC | {metrics['roc_auc']:.3f} |")
    
    report.append("\n## Confusion Matrix")
    report.append(f"\n|  | Predicted Negative | Predicted Positive |")
    report.append(f"|--|-------------------|-------------------|")
    report.append(f"| **Actual Negative** | {metrics['true_negatives']} (TN) | {metrics['false_positives']} (FP) |")
    report.append(f"| **Actual Positive** | {metrics['false_negatives']} (FN) | {metrics['true_positives']} (TP) |")
    
    report.append("\n## Visualizations")
    report.append(f"\n### ROC Curve")
    report.append(f"\n![ROC Curve](roc_curve.png)")
    
    report.append(f"\n### Confusion Matrix")
    report.append(f"\n![Confusion Matrix](confusion_matrix.png)")
    
    report.append(f"\n### Feature Importance")
    report.append(f"\n![Feature Importance](feature_importance.png)")
    
    report.append(f"\n### Precision-Recall Curve")
    report.append(f"\n![Precision-Recall Curve](precision_recall_curve.png)")
    
    # Write report
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        f.write('\n'.join(report))
    
    print(f"Evaluation report saved to: {output_path}")


