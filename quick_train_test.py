"""Quick training test - one asset, show train vs test performance."""
import json
import sys
sys.path.insert(0, 'src')

from dcs_volume_loader import load_all_dcs_volumes
from feature_engineering import extract_multi_window_features
from preprocessing import preprocess_features
from model_evaluation import compute_metrics
import csv
from datetime import datetime, timedelta
from collections import defaultdict
import random
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

print("="*70)
print("QUICK TRAIN/TEST COMPARISON - BL-1829")
print("="*70)

# Load analysis results
with open('data/processed/volume_analysis_results.json') as f:
    analysis_results = json.load(f)

# Load failures
failures_list = []
with open('data/processed/problem_asset_failures.csv') as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row['priority'] == '1' and row['asset_name'] == 'BL-1829':
            timestamp = datetime.fromisoformat(row['service_request_timestamp'].replace('Z', '+00:00'))
            failures_list.append({'timestamp': timestamp, 'desc': row['service_request_text'][:30]})

failures_list.sort(key=lambda x: x['timestamp'])

print(f"\n1. Loading DCS data...")
dcs_data = load_all_dcs_volumes(max_files=None)
all_timestamps = sorted(set(dp['timestamp'] for dp in dcs_data))
dcs_start, dcs_end = all_timestamps[0], all_timestamps[-1]

print(f"   {len(dcs_data):,} data points loaded")

# Get failures in range
covered = [f for f in failures_list if dcs_start <= f['timestamp'] <= dcs_end]
train_failures = covered[:-2]
test_failures = covered[-2:]

print(f"\n2. Dataset split:")
print(f"   Train: {len(train_failures)} failures")
print(f"   Test: {len(test_failures)} failures (held-out)")

# Create samples
tags = ['CH-DRA_PROC_AIR:IT18179.MEAS', 'CH-DRA_PROC_AIR:TT18179A.MEAS', 
        'CH-DRA_PROC_AIR:TT18179B.MEAS', 'CH-DRA_PROC_AIR:TT18179C.MEAS']

# Training positive samples
train_pos = []
for f in train_failures:
    t_2d = f['timestamp'] - timedelta(days=2)
    closest = min(all_timestamps, key=lambda x: abs(x - t_2d))
    if abs(closest - t_2d) < timedelta(hours=12):
        train_pos.append(closest)

# Training negative samples
failure_times = [f['timestamp'] for f in covered]
train_neg = []
check_time = dcs_start + timedelta(days=7)
while check_time < dcs_end - timedelta(days=7) and len(train_neg) < len(train_pos) * 2:
    min_dist = min(abs((check_time - ft).days) for ft in failure_times)
    if min_dist > 7:
        closest = min(all_timestamps, key=lambda x: abs(x - check_time))
        if abs(closest - t) < timedelta(hours=12):
            train_neg.append(closest)
    check_time += timedelta(days=3)

# Test positive samples
test_pos = []
for f in test_failures:
    t_2d = f['timestamp'] - timedelta(days=2)
    closest = min(all_timestamps, key=lambda x: abs(x - t_2d))
    if abs(closest - t_2d) < timedelta(hours=12):
        test_pos.append(closest)

print(f"\n3. Sample counts:")
print(f"   Train: {len(train_pos)} positive, {len(train_neg)} negative")
print(f"   Test: {len(test_pos)} positive (failures to detect)")

# Extract features
print(f"\n4. Extracting features...")
filtered_dcs = [dp for dp in dcs_data if dp['tag_name'] in tags]

train_features = extract_multi_window_features(filtered_dcs, train_pos + train_neg, 
                                                window_hours_list=[24, 72, 168], tag_names=tags)
test_features = extract_multi_window_features(filtered_dcs, test_pos,
                                               window_hours_list=[24, 72, 168], tag_names=tags)

train_df = pd.DataFrame(train_features).drop('timestamp', axis=1)
test_df = pd.DataFrame(test_features).drop('timestamp', axis=1)

print(f"   Train features: {train_df.shape}")
print(f"   Test features: {test_df.shape}")

# Preprocess
print(f"\n5. Preprocessing...")
train_processed, scaler_params, _ = preprocess_features(
    train_df, is_training=True, 
    config={'outlier_factor': 3.0, 'correlation_threshold': 1.0}
)

test_processed, _, _ = preprocess_features(
    test_df, is_training=False, scaler_params=scaler_params
)

print(f"   Train processed: {train_processed.shape}")
print(f"   Test processed: {test_processed.shape}")
print(f"   ✓ Feature counts match: {train_processed.shape[1] == test_processed.shape[1]}")

# Train model
print(f"\n6. Training with STRONG regularization (C=0.01)...")
y_train = np.array([1]*len(train_pos) + [0]*len(train_neg))

model = LogisticRegression(penalty='l2', C=0.01, class_weight='balanced', max_iter=1000, random_state=42)
model.fit(train_processed.values, y_train)

# Training performance
y_train_pred = model.predict(train_processed.values)
y_train_proba = model.predict_proba(train_processed.values)[:, 1]
train_metrics = compute_metrics(y_train, y_train_pred, y_train_proba)

print(f"\n7. RESULTS:")
print(f"\n   TRAINING SET:")
print(f"     Accuracy:  {train_metrics['accuracy']:.3f}")
print(f"     Precision: {train_metrics['precision']:.3f}")
print(f"     Recall:    {train_metrics['recall']:.3f}")
print(f"     ROC-AUC:   {train_metrics['roc_auc']:.3f}")

# Test performance
y_test = np.array([1]*len(test_pos))
y_test_pred = model.predict(test_processed.values)
y_test_proba = model.predict_proba(test_processed.values)[:, 1]
test_metrics = compute_metrics(y_test, y_test_pred, y_test_proba)

print(f"\n   TEST SET (HELD-OUT FAILURES) ← THE REAL METRIC:")
print(f"     Accuracy:  {test_metrics['accuracy']:.3f}")
print(f"     Precision: {test_metrics['precision']:.3f}")
print(f"     Recall:    {test_metrics['recall']:.3f} ← Did we detect them?")
print(f"     ROC-AUC:   {test_metrics['roc_auc']:.3f}")

for i, (f, pred, prob) in enumerate(zip(test_failures, y_test_pred, y_test_proba)):
    status = "✓ DETECTED" if pred == 1 else "✗ MISSED"
    print(f"     Failure {i+1} ({f['timestamp'].date()}): {status} (prob={prob:.3f})")

print(f"\n" + "="*70)
if train_metrics['accuracy'] == 1.0 and test_metrics['recall'] < 0.9:
    print("⚠️  OVERFITTING DETECTED!")
    print("Train: Perfect (1.0) | Test: Poor → Model memorized, didn't learn")
elif test_metrics['recall'] >= 0.5:
    print("✓ Model is working! Test recall >= 50%")
else:
    print("✗ Model failed - couldn't detect held-out failures")
print("="*70)


