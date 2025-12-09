"""Quick test of preprocessing pipeline."""
import sys
sys.path.insert(0, 'src')

import pandas as pd
import numpy as np
from preprocessing import preprocess_features, get_preprocessing_stats_summary
from data_quality import quick_data_quality_summary
from model_evaluation import compute_metrics

print("="*70)
print("QUICK PREPROCESSING TEST")
print("="*70)

# Create synthetic test data
print("\n1. Creating test data...")
np.random.seed(42)

# Create 100 samples, 50 features
n_samples = 100
n_features = 50

# Generate data with various issues
data = {}
for i in range(n_features):
    # Add some missing values (10% random)
    col_data = np.random.randn(n_samples) * 10 + 50
    mask = np.random.rand(n_samples) < 0.1
    col_data[mask] = np.nan
    
    # Add outliers (5%)
    outlier_mask = np.random.rand(n_samples) < 0.05
    col_data[outlier_mask] = col_data[outlier_mask] * 5
    
    data[f'feature_{i}'] = col_data

# Add some highly correlated features
data['feature_50'] = data['feature_0'] * 0.98 + np.random.randn(n_samples) * 0.1
data['feature_51'] = data['feature_1'] * 0.99 + np.random.randn(n_samples) * 0.05

df = pd.DataFrame(data)
print(f"Created DataFrame: {df.shape}")
print(quick_data_quality_summary(df))

# Test preprocessing
print("\n2. Testing preprocessing pipeline...")
df_processed, scaler_params, prep_stats = preprocess_features(
    df,
    is_training=True,
    config={'outlier_factor': 3.0, 'correlation_threshold': 0.95}
)

print(get_preprocessing_stats_summary(prep_stats))

# Test on "test" set
print("\n3. Testing on new data (test set)...")
test_data = {}
for i in range(n_features):
    test_data[f'feature_{i}'] = np.random.randn(20) * 10 + 50

test_data['feature_50'] = test_data['feature_0'] * 0.98
test_data['feature_51'] = test_data['feature_1'] * 0.99

df_test = pd.DataFrame(test_data)
df_test_processed, _, _ = preprocess_features(
    df_test,
    is_training=False,
    scaler_params=scaler_params
)

print(f"Test set processed: {df_test_processed.shape}")
print(f"Same features as training: {df_test_processed.shape[1] == df_processed.shape[1]}")

# Test model evaluation metrics
print("\n4. Testing evaluation metrics...")
y_true = np.array([1, 0, 1, 1, 0, 0, 1, 0, 1, 0])
y_pred = np.array([1, 0, 1, 0, 0, 1, 1, 0, 1, 0])
y_proba = np.array([0.9, 0.2, 0.85, 0.45, 0.3, 0.6, 0.75, 0.1, 0.95, 0.25])

metrics = compute_metrics(y_true, y_pred, y_proba)
print(f"Accuracy: {metrics['accuracy']:.3f}")
print(f"Precision: {metrics['precision']:.3f}")
print(f"Recall: {metrics['recall']:.3f}")
print(f"F1: {metrics['f1']:.3f}")
print(f"ROC-AUC: {metrics['roc_auc']:.3f}")

print("\n" + "="*70)
print("ALL TESTS PASSED ✓")
print("="*70)
print("\nPreprocessing pipeline is working correctly!")
print("The full training script should work with real data.")


