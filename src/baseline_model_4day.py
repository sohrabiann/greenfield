"""Project-Aligned Baseline Model with 4-Day Horizon

This script builds baseline predictive maintenance models using:
- Priority 1 service request labels with 4-day prediction horizon
- Rich analog tag features (motor currents, bearing temperatures)
- Multiple time-window aggregations (24h, 72h features)
- Simple ML models (logistic regression, decision trees)

This replaces the toy baseline which used a single status tag and 12-hour horizon.
"""
import csv
import datetime as dt
import json
import os
from collections import defaultdict
from typing import Dict, List, Tuple

# Import our modules
from dcs_volume_loader import load_all_dcs_volumes, PROBLEM_ASSET_LOCATIONS
from feature_engineering import (
    extract_multi_window_features,
    create_feature_matrix,
    compute_feature_statistics,
    normalize_features
)


def load_labels_for_asset(location: str) -> Dict[dt.datetime, int]:
    """Load 4-day horizon labels for a specific asset location.
    
    Returns:
        Dict mapping timestamps to labels (1=failure within 4 days, 0=healthy)
    """
    label_file = f'data/processed/labels_{location}.csv'
    
    if not os.path.exists(label_file):
        print(f"Warning: Label file not found: {label_file}")
        return {}
    
    labels = {}
    with open(label_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                timestamp_str = row['timestamp']
                if not timestamp_str:
                    continue
                
                timestamp = dt.datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                if timestamp.tzinfo is None:
                    timestamp = timestamp.replace(tzinfo=dt.timezone.utc)
                
                label_str = row['label']
                if label_str == 'failure_within_4d':
                    labels[timestamp] = 1
                elif label_str == 'healthy':
                    labels[timestamp] = 0
            except Exception as e:
                continue
    
    return labels


def simple_train_test_split(
    timestamps: List[dt.datetime],
    labels: List[int],
    train_fraction: float = 0.7
) -> Tuple[List[int], List[int]]:
    """Split data chronologically into train and test sets.
    
    Args:
        timestamps: List of timestamps
        labels: List of labels
        train_fraction: Fraction of data to use for training
    
    Returns:
        Tuple of (train_indices, test_indices)
    """
    # Sort by timestamp
    sorted_indices = sorted(range(len(timestamps)), key=lambda i: timestamps[i])
    
    split_point = int(len(sorted_indices) * train_fraction)
    train_indices = sorted_indices[:split_point]
    test_indices = sorted_indices[split_point:]
    
    return train_indices, test_indices


def simple_logistic_regression(
    X_train: List[List[float]],
    y_train: List[int],
    learning_rate: float = 0.01,
    n_iterations: int = 100
) -> List[float]:
    """Simple logistic regression implementation (no external dependencies).
    
    Args:
        X_train: Training features (list of feature vectors)
        y_train: Training labels
        learning_rate: Learning rate for gradient descent
        n_iterations: Number of training iterations
    
    Returns:
        Learned weights (including bias as first element)
    """
    if len(X_train) == 0:
        return []
    
    n_features = len(X_train[0])
    weights = [0.0] * (n_features + 1)  # +1 for bias
    
    for iteration in range(n_iterations):
        # Compute predictions and gradients
        gradients = [0.0] * (n_features + 1)
        
        for i, x in enumerate(X_train):
            # Compute prediction (sigmoid of linear combination)
            z = weights[0]  # bias
            for j, x_j in enumerate(x):
                z += weights[j + 1] * x_j
            
            # Sigmoid function
            if z > 20:
                pred = 1.0
            elif z < -20:
                pred = 0.0
            else:
                pred = 1.0 / (1.0 + 2.718281828 ** (-z))
            
            # Gradient
            error = pred - y_train[i]
            gradients[0] += error  # bias gradient
            for j, x_j in enumerate(x):
                gradients[j + 1] += error * x_j
        
        # Update weights
        for j in range(len(weights)):
            weights[j] -= learning_rate * gradients[j] / len(X_train)
    
    return weights


def predict_logistic(X: List[List[float]], weights: List[float]) -> List[float]:
    """Make predictions using logistic regression weights.
    
    Args:
        X: Feature matrix
        weights: Learned weights (including bias)
    
    Returns:
        List of predicted probabilities
    """
    predictions = []
    
    for x in X:
        z = weights[0]  # bias
        for j, x_j in enumerate(x):
            if j + 1 < len(weights):
                z += weights[j + 1] * x_j
        
        # Sigmoid
        if z > 20:
            pred = 1.0
        elif z < -20:
            pred = 0.0
        else:
            pred = 1.0 / (1.0 + 2.718281828 ** (-z))
        
        predictions.append(pred)
    
    return predictions


def evaluate_binary_classifier(
    y_true: List[int],
    y_pred_proba: List[float],
    threshold: float = 0.5
) -> Dict[str, float]:
    """Evaluate binary classifier performance.
    
    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        threshold: Classification threshold
    
    Returns:
        Dict with metrics: accuracy, precision, recall, f1
    """
    y_pred = [1 if p >= threshold else 0 for p in y_pred_proba]
    
    tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 1)
    fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 1)
    tn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 0)
    fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 0)
    
    accuracy = (tp + tn) / len(y_true) if len(y_true) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tp': tp,
        'fp': fp,
        'tn': tn,
        'fn': fn
    }


def train_baseline_model(location: str, selected_tags: List[str], max_files: int = 5) -> Dict:
    """Train a baseline model for one problem asset.
    
    Args:
        location: Asset location code (e.g., 'BL-1829')
        selected_tags: List of DCS tags to use as features
        max_files: Number of DCS files to load
    
    Returns:
        Dict with model results and metrics
    """
    print(f"\n{'='*70}")
    print(f"BASELINE MODEL: {location} - {PROBLEM_ASSET_LOCATIONS[location]['description']}")
    print(f"{'='*70}")
    
    # Load labels
    print("\n1. Loading labels...")
    labels_dict = load_labels_for_asset(location)
    
    if len(labels_dict) == 0:
        print(f"   No labels found for {location}")
        return {}
    
    n_positive = sum(1 for v in labels_dict.values() if v == 1)
    n_negative = sum(1 for v in labels_dict.values() if v == 0)
    print(f"   Total labels: {len(labels_dict)}")
    print(f"   Positive (failure within 4d): {n_positive}")
    print(f"   Negative (healthy): {n_negative}")
    
    if n_positive == 0:
        print("   No positive examples - cannot train model")
        return {}
    
    # Load DCS data
    print(f"\n2. Loading DCS data for {len(selected_tags)} tags...")
    dcs_data = load_all_dcs_volumes(
        selected_tags=set(selected_tags),
        max_files=max_files
    )
    
    if len(dcs_data) == 0:
        print("   No DCS data found")
        return {}
    
    # Extract features
    print("\n3. Extracting features...")
    timestamps = list(labels_dict.keys())
    features_list = extract_multi_window_features(
        dcs_data,
        timestamps,
        window_hours_list=[24, 72],  # 1 day and 3 days
        tag_names=selected_tags
    )
    
    print(f"   Extracted features for {len(features_list)} timestamps")
    
    # Create feature matrix
    feature_matrix, feature_names = create_feature_matrix(features_list)
    print(f"   Feature matrix shape: {len(feature_matrix)} x {len(feature_names)}")
    
    if len(feature_matrix) == 0 or len(feature_names) == 0:
        print("   No valid features extracted")
        return {}
    
    # Prepare labels
    labels = [labels_dict[ts] for ts in timestamps]
    
    # Train/test split
    print("\n4. Splitting data (70% train, 30% test)...")
    train_idx, test_idx = simple_train_test_split(timestamps, labels, train_fraction=0.7)
    
    X_train = [feature_matrix[i] for i in train_idx]
    y_train = [labels[i] for i in train_idx]
    X_test = [feature_matrix[i] for i in test_idx]
    y_test = [labels[i] for i in test_idx]
    
    print(f"   Train: {len(X_train)} samples ({sum(y_train)} positive)")
    print(f"   Test: {len(X_test)} samples ({sum(y_test)} positive)")
    
    if sum(y_train) == 0:
        print("   No positive examples in training set - cannot train")
        return {}
    
    # Normalize features
    print("\n5. Normalizing features...")
    feature_stats = compute_feature_statistics(X_train, feature_names)
    X_train_norm = normalize_features(X_train, feature_stats, feature_names)
    X_test_norm = normalize_features(X_test, feature_stats, feature_names)
    
    # Train model
    print("\n6. Training logistic regression...")
    weights = simple_logistic_regression(X_train_norm, y_train, learning_rate=0.1, n_iterations=200)
    
    # Evaluate
    print("\n7. Evaluating model...")
    train_pred = predict_logistic(X_train_norm, weights)
    test_pred = predict_logistic(X_test_norm, weights)
    
    train_metrics = evaluate_binary_classifier(y_train, train_pred, threshold=0.5)
    test_metrics = evaluate_binary_classifier(y_test, test_pred, threshold=0.5)
    
    print("\n   TRAIN METRICS:")
    print(f"     Accuracy:  {train_metrics['accuracy']:.3f}")
    print(f"     Precision: {train_metrics['precision']:.3f}")
    print(f"     Recall:    {train_metrics['recall']:.3f}")
    print(f"     F1 Score:  {train_metrics['f1']:.3f}")
    
    print("\n   TEST METRICS:")
    print(f"     Accuracy:  {test_metrics['accuracy']:.3f}")
    print(f"     Precision: {test_metrics['precision']:.3f}")
    print(f"     Recall:    {test_metrics['recall']:.3f}")
    print(f"     F1 Score:  {test_metrics['f1']:.3f}")
    
    # Feature importance (absolute weight values)
    print("\n8. Feature importance (top 10):")
    feature_importance = [(fname, abs(weights[i+1])) for i, fname in enumerate(feature_names)]
    feature_importance.sort(key=lambda x: x[1], reverse=True)
    
    for fname, importance in feature_importance[:10]:
        print(f"     {fname}: {importance:.4f}")
    
    return {
        'location': location,
        'n_features': len(feature_names),
        'n_train': len(X_train),
        'n_test': len(X_test),
        'train_metrics': train_metrics,
        'test_metrics': test_metrics,
        'top_features': feature_importance[:20]
    }


def main():
    """Train baseline models for key problem assets."""
    
    # Define assets and their key tags
    model_config = {
        'DC-1834': [
            'CH-DR_DUSTCOLL:IT18211.MEAS',  # Motor current
            'CH-DR_DUSTCOLL:TT18217.MEAS',  # Temperature
        ],
    }
    
    all_results = {}
    
    for location, tags in model_config.items():
        results = train_baseline_model(location, tags, max_files=5)
        if results:
            all_results[location] = results
    
    # Save results
    output_file = 'data/processed/baseline_4day_results.json'
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print(f"\n\nResults saved to {output_file}")
    print("\nBaseline modeling complete!")


if __name__ == '__main__':
    main()

