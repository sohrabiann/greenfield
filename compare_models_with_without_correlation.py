"""Compare models with and without correlation removal to validate feature dropping."""
import json
import sys
sys.path.insert(0, 'src')

from dcs_volume_loader import load_all_dcs_volumes, PROBLEM_ASSET_LOCATIONS
from feature_engineering import extract_multi_window_features
from preprocessing import preprocess_features
from data_quality import quick_data_quality_summary
from model_evaluation import compute_metrics
import csv
from datetime import datetime, timedelta, timezone
from collections import defaultdict
import random
import pandas as pd
import numpy as np

try:
    from sklearn.linear_model import LogisticRegression
    SKLEARN_AVAILABLE = True
except ImportError:
    print("ERROR: scikit-learn required. Install with: pip install scikit-learn")
    sys.exit(1)

print("="*70)
print("MODEL COMPARISON: WITH vs WITHOUT CORRELATION REMOVAL")
print("="*70)

# Load analysis results
print("\n1. Loading analysis results...")
with open('data/processed/volume_analysis_results.json') as f:
    analysis_results = json.load(f)

assets_with_failures = {}
for asset, data in analysis_results.items():
    failure_data = data.get('failure_aligned', {})
    if failure_data:
        max_windows = max((tag_data.get('n_windows', 0) for tag_data in failure_data.values()), default=0)
        if max_windows > 0:
            assets_with_failures[asset] = max_windows

# Load failures
print("\n2. Loading Priority 1 failure timestamps...")
failures_by_asset = defaultdict(list)
with open('data/processed/problem_asset_failures.csv') as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row['priority'] == '1':
            asset_name = row['asset_name']
            if asset_name in assets_with_failures:
                timestamp_str = row['service_request_timestamp']
                timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                failures_by_asset[asset_name].append({
                    'timestamp': timestamp,
                    'description': row['service_request_text'][:50]
                })

for asset in failures_by_asset:
    failures_by_asset[asset].sort(key=lambda x: x['timestamp'])

print(f"\n3. Loading full DCS dataset...")
dcs_data = load_all_dcs_volumes(max_files=None)
print(f"   Loaded {len(dcs_data):,} data points")

dcs_dict = {dp['timestamp']: dp for dp in dcs_data}
all_timestamps = sorted(dcs_dict.keys())
dcs_start = all_timestamps[0]
dcs_end = all_timestamps[-1]

print("\n4. Analyzing ONE asset in detail (BL-1829)...")
asset_name = 'BL-1829'
tags = ['CH-DRA_PROC_AIR:IT18179.MEAS', 'CH-DRA_PROC_AIR:TT18179A.MEAS', 
        'CH-DRA_PROC_AIR:TT18179B.MEAS', 'CH-DRA_PROC_AIR:TT18179C.MEAS']

asset_failures = failures_by_asset[asset_name]
covered_failures = [f for f in asset_failures if dcs_start <= f['timestamp'] <= dcs_end]

train_failures = covered_failures[:-2]
test_failures = covered_failures[-2:]

print(f"   Train: {len(train_failures)} failures, Test: {len(test_failures)} failures")

# Create samples
positive_samples = []
for failure in train_failures:
    t_2d = failure['timestamp'] - timedelta(days=2)
    closest = min(all_timestamps, key=lambda x: abs(x - t_2d))
    if abs(closest - t_2d) < timedelta(hours=12):
        positive_samples.append(closest)

negative_samples = []
failure_times = [f['timestamp'] for f in covered_failures]
check_time = dcs_start + timedelta(days=7)
while check_time < dcs_end - timedelta(days=7):
    min_dist = min(abs((check_time - ft).days) for ft in failure_times)
    if min_dist > 7:
        closest = min(all_timestamps, key=lambda x: abs(x - check_time))
        if abs(closest - check_time) < timedelta(hours=12):
            negative_samples.append(closest)
    check_time += timedelta(days=3)

random.seed(42)
random.shuffle(negative_samples)
negative_samples = negative_samples[:len(positive_samples) * 2]

print(f"   Samples: {len(positive_samples)} positive, {len(negative_samples)} negative")

# Extract features
print(f"\n5. Extracting features...")
all_samples = positive_samples + negative_samples
labels = [1] * len(positive_samples) + [0] * len(negative_samples)

filtered_dcs = [dp for dp in dcs_data if dp['tag_name'] in tags]

features_raw = extract_multi_window_features(
    filtered_dcs,
    all_samples,
    window_hours_list=[24, 72, 168],
    tag_names=tags
)

features_df = pd.DataFrame(features_raw)
timestamps = features_df['timestamp']
features_df = features_df.drop('timestamp', axis=1)

print(f"   Extracted {features_df.shape[1]} raw features")

# =========================================================================
# MODEL 1: WITH CORRELATION REMOVAL (threshold=0.95)
# =========================================================================
print("\n" + "="*70)
print("MODEL 1: WITH CORRELATION REMOVAL (threshold=0.95)")
print("="*70)

features_processed_1, scaler_params_1, prep_stats_1 = preprocess_features(
    features_df.copy(),
    is_training=True,
    config={'outlier_factor': 3.0, 'correlation_threshold': 0.95}
)

print(f"Features after preprocessing: {features_processed_1.shape[1]}")
print(f"Dropped features: {len(prep_stats_1.get('dropped_features', []))}")

model_1 = LogisticRegression(
    class_weight='balanced',
    max_iter=1000,
    random_state=42
)
model_1.fit(features_processed_1.values, np.array(labels))

y_pred_1 = model_1.predict(features_processed_1.values)
y_proba_1 = model_1.predict_proba(features_processed_1.values)[:, 1]
metrics_1 = compute_metrics(np.array(labels), y_pred_1, y_proba_1)

print(f"\nModel 1 Performance:")
print(f"  Accuracy:  {metrics_1['accuracy']:.3f}")
print(f"  Precision: {metrics_1['precision']:.3f}")
print(f"  Recall:    {metrics_1['recall']:.3f}")
print(f"  ROC-AUC:   {metrics_1['roc_auc']:.3f}")

# Get feature importance
importance_1 = pd.DataFrame({
    'feature': features_processed_1.columns,
    'coefficient': model_1.coef_[0],
    'abs_coefficient': np.abs(model_1.coef_[0])
}).sort_values('abs_coefficient', ascending=False)

print(f"\nTop 10 Most Important Features (Model 1):")
for i, row in importance_1.head(10).iterrows():
    sign = "+" if row['coefficient'] > 0 else "-"
    print(f"  {sign} {row['feature']}: {row['abs_coefficient']:.4f}")

# =========================================================================
# MODEL 2: WITHOUT CORRELATION REMOVAL (threshold=1.0)
# =========================================================================
print("\n" + "="*70)
print("MODEL 2: WITHOUT CORRELATION REMOVAL (keep all features)")
print("="*70)

features_processed_2, scaler_params_2, prep_stats_2 = preprocess_features(
    features_df.copy(),
    is_training=True,
    config={'outlier_factor': 3.0, 'correlation_threshold': 1.0}
)

print(f"Features after preprocessing: {features_processed_2.shape[1]}")
print(f"Dropped features: {len(prep_stats_2.get('dropped_features', []))}")

model_2 = LogisticRegression(
    class_weight='balanced',
    max_iter=1000,
    random_state=42
)
model_2.fit(features_processed_2.values, np.array(labels))

y_pred_2 = model_2.predict(features_processed_2.values)
y_proba_2 = model_2.predict_proba(features_processed_2.values)[:, 1]
metrics_2 = compute_metrics(np.array(labels), y_pred_2, y_proba_2)

print(f"\nModel 2 Performance:")
print(f"  Accuracy:  {metrics_2['accuracy']:.3f}")
print(f"  Precision: {metrics_2['precision']:.3f}")
print(f"  Recall:    {metrics_2['recall']:.3f}")
print(f"  ROC-AUC:   {metrics_2['roc_auc']:.3f}")

# Get feature importance
importance_2 = pd.DataFrame({
    'feature': features_processed_2.columns,
    'coefficient': model_2.coef_[0],
    'abs_coefficient': np.abs(model_2.coef_[0])
}).sort_values('abs_coefficient', ascending=False)

print(f"\nTop 10 Most Important Features (Model 2):")
for i, row in importance_2.head(10).iterrows():
    sign = "+" if row['coefficient'] > 0 else "-"
    print(f"  {sign} {row['feature']}: {row['abs_coefficient']:.4f}")

# =========================================================================
# COMPARISON ANALYSIS
# =========================================================================
print("\n" + "="*70)
print("COMPARISON ANALYSIS")
print("="*70)

# Performance comparison
print("\n1. PERFORMANCE METRICS COMPARISON:")
print(f"\n{'Metric':<15} {'Model 1 (Drop)':<15} {'Model 2 (Keep)':<15} {'Difference':<15}")
print("-" * 60)
for metric in ['accuracy', 'precision', 'recall', 'roc_auc']:
    diff = metrics_2[metric] - metrics_1[metric]
    diff_str = f"{diff:+.4f}"
    print(f"{metric.upper():<15} {metrics_1[metric]:<15.4f} {metrics_2[metric]:<15.4f} {diff_str:<15}")

# Feature analysis
print(f"\n2. FEATURE COUNT:")
print(f"   Model 1 (with removal): {features_processed_1.shape[1]} features")
print(f"   Model 2 (keep all):     {features_processed_2.shape[1]} features")
print(f"   Dropped: {features_processed_2.shape[1] - features_processed_1.shape[1]} features")

# Analyze dropped features
dropped_features = prep_stats_1.get('dropped_features', [])
print(f"\n3. DROPPED FEATURES ({len(dropped_features)} total):")
for feat in dropped_features[:20]:  # Show first 20
    print(f"   - {feat}")
if len(dropped_features) > 20:
    print(f"   ... and {len(dropped_features) - 20} more")

# Check if any dropped features would have been important
print(f"\n4. IMPORTANCE ANALYSIS OF DROPPED FEATURES:")
print(f"   (Checking if dropped features are actually important in Model 2)")

dropped_importance = importance_2[importance_2['feature'].isin(dropped_features)].copy()
dropped_importance = dropped_importance.sort_values('abs_coefficient', ascending=False)

if len(dropped_importance) > 0:
    print(f"\n   Top 10 Most Important DROPPED Features:")
    for i, row in dropped_importance.head(10).iterrows():
        rank_in_model2 = importance_2[importance_2['feature'] == row['feature']].index[0] + 1
        print(f"   - {row['feature']}: {row['abs_coefficient']:.4f} (rank #{rank_in_model2} in Model 2)")
    
    # Check if any are in top 20
    top_20_features = set(importance_2.head(20)['feature'])
    important_dropped = [f for f in dropped_features if f in top_20_features]
    
    if important_dropped:
        print(f"\n   ⚠️  WARNING: {len(important_dropped)} dropped features are in TOP 20 important!")
        for feat in important_dropped:
            rank = importance_2[importance_2['feature'] == feat].index[0] + 1
            coef = importance_2[importance_2['feature'] == feat]['abs_coefficient'].values[0]
            print(f"      - {feat} (rank #{rank}, importance: {coef:.4f})")
    else:
        print(f"\n   ✓ Good news: No dropped features are in TOP 20 important features")

# Check correlation between top features
print(f"\n5. CORRELATION BETWEEN TOP IMPORTANT FEATURES:")
top_10_model2 = importance_2.head(10)['feature'].tolist()
if len(top_10_model2) > 1:
    top_10_df = features_processed_2[top_10_model2]
    corr_matrix = top_10_df.corr().abs()
    
    # Find high correlations
    high_corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            corr_val = corr_matrix.iloc[i, j]
            if corr_val > 0.8:
                high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_val))
    
    if high_corr_pairs:
        print(f"   Found {len(high_corr_pairs)} pairs of top features with r > 0.8:")
        for feat1, feat2, corr in sorted(high_corr_pairs, key=lambda x: -x[2])[:5]:
            print(f"   - {feat1} <-> {feat2}: r={corr:.3f}")
    else:
        print(f"   No high correlations (r > 0.8) among top 10 features")

# =========================================================================
# RECOMMENDATION
# =========================================================================
print("\n" + "="*70)
print("RECOMMENDATION")
print("="*70)

roc_diff = metrics_2['roc_auc'] - metrics_1['roc_auc']
important_dropped = len([f for f in dropped_features if f in set(importance_2.head(20)['feature'])])

if abs(roc_diff) < 0.01 and important_dropped == 0:
    print("\n✓ RECOMMENDATION: USE MODEL 1 (with correlation removal)")
    print("  Reason: Similar performance, fewer features = simpler model")
    print(f"  ROC-AUC difference: {roc_diff:+.4f} (negligible)")
elif roc_diff > 0.02 or important_dropped > 0:
    print("\n✓ RECOMMENDATION: USE MODEL 2 (keep all features)")
    print("  Reason: Better performance and/or important features were dropped")
    print(f"  ROC-AUC difference: {roc_diff:+.4f}")
    if important_dropped > 0:
        print(f"  Important features dropped: {important_dropped}")
else:
    print("\n✓ RECOMMENDATION: USE MODEL 1 (with correlation removal)")
    print("  Reason: Performance difference is small")
    print(f"  ROC-AUC difference: {roc_diff:+.4f}")
    print("  Benefit: Fewer features = faster training, easier interpretation")

print("\n" + "="*70)
print("COMPARISON COMPLETE")
print("="*70)

# Save results
comparison_results = {
    'asset': asset_name,
    'model_1_with_removal': {
        'n_features': int(features_processed_1.shape[1]),
        'n_dropped': len(dropped_features),
        'metrics': metrics_1
    },
    'model_2_keep_all': {
        'n_features': int(features_processed_2.shape[1]),
        'n_dropped': 0,
        'metrics': metrics_2
    },
    'performance_difference': {
        'roc_auc_diff': float(roc_diff),
        'accuracy_diff': float(metrics_2['accuracy'] - metrics_1['accuracy']),
        'precision_diff': float(metrics_2['precision'] - metrics_1['precision']),
        'recall_diff': float(metrics_2['recall'] - metrics_1['recall'])
    },
    'dropped_features': dropped_features[:50],  # Save first 50
    'important_dropped_count': important_dropped
}

with open('data/processed/model_comparison_results.json', 'w') as f:
    json.dump(comparison_results, f, indent=2)

print("\nResults saved to: data/processed/model_comparison_results.json")


