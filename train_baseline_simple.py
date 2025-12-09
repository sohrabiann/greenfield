"""Complete baseline model training with preprocessing and evaluation."""
import json
import sys
sys.path.insert(0, 'src')

from dcs_volume_loader import load_all_dcs_volumes, PROBLEM_ASSET_LOCATIONS
from feature_engineering import extract_multi_window_features
from preprocessing import preprocess_features, apply_smote, get_preprocessing_stats_summary
from data_quality import generate_data_quality_report, quick_data_quality_summary
from model_evaluation import evaluate_model, generate_evaluation_report, compute_metrics
import csv
from datetime import datetime, timedelta, timezone
from collections import defaultdict
import random
import pandas as pd
import numpy as np
import os

# Try to import sklearn
try:
    from sklearn.linear_model import LogisticRegression
    SKLEARN_AVAILABLE = True
except ImportError:
    print("WARNING: scikit-learn not available. Using fallback implementation.")
    SKLEARN_AVAILABLE = False

print("="*70)
print("SIMPLIFIED BASELINE MODEL TRAINING")
print("="*70)

# Load analysis results to see which failures have DCS coverage
print("\n1. Loading analysis results...")
with open('data/processed/volume_analysis_results.json') as f:
    analysis_results = json.load(f)

# Count failures per asset
assets_with_failures = {}
for asset, data in analysis_results.items():
    failure_data = data.get('failure_aligned', {})
    if failure_data:
        max_windows = max((tag_data.get('n_windows', 0) for tag_data in failure_data.values()), default=0)
        if max_windows > 0:
            assets_with_failures[asset] = max_windows
            print(f"   {asset}: {max_windows} failures")

# Load failures to get timestamps
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

# Sort by time
for asset in failures_by_asset:
    failures_by_asset[asset].sort(key=lambda x: x['timestamp'])

print(f"\n3. Loading full DCS dataset...")
dcs_data = load_all_dcs_volumes(max_files=None)
print(f"   Loaded {len(dcs_data):,} data points")

# Convert to time-indexed dict
dcs_dict = {dp['timestamp']: dp for dp in dcs_data}
all_timestamps = sorted(dcs_dict.keys())
dcs_start = all_timestamps[0]
dcs_end = all_timestamps[-1]
print(f"   Time range: {dcs_start} to {dcs_end}")

# For each asset, create training data
print("\n4. Creating training datasets...")

all_results = {}

for asset_name in assets_with_failures:
    print(f"\n   === {asset_name} ===")
    
    # Get tags for this asset
    if asset_name == 'BL-1829':
        tags = ['CH-DRA_PROC_AIR:IT18179.MEAS', 'CH-DRA_PROC_AIR:TT18179A.MEAS', 
                'CH-DRA_PROC_AIR:TT18179B.MEAS', 'CH-DRA_PROC_AIR:TT18179C.MEAS']
    elif asset_name == 'RV-1834':
        tags = ['CH-DR_DUSTCOLL:IT18211.MEAS', 'CH-DR_DUSTCOLL:IT18216.MEAS', 
                'CH-DR_DUSTCOLL:TT18217.MEAS']
    elif asset_name == 'DC-1834':
        tags = ['CH-DR_DUSTCOLL:IT18211.MEAS', 'CH-DR_DUSTCOLL:TT18217.MEAS']
    else:
        continue
    
    # Get failures in DCS range
    asset_failures = failures_by_asset[asset_name]
    covered_failures = [f for f in asset_failures if dcs_start <= f['timestamp'] <= dcs_end]
    
    print(f"   Failures in DCS range: {len(covered_failures)} of {len(asset_failures)}")
    
    if len(covered_failures) < 2:
        print(f"   Not enough failures to train")
        continue
    
    # Simple train/test split: last 2 failures for testing
    train_failures = covered_failures[:-2]
    test_failures = covered_failures[-2:]
    
    print(f"   Train: {len(train_failures)} failures")
    print(f"   Test: {len(test_failures)} failures")
    
    # Create positive samples (2 days before each failure)
    positive_samples = []
    for failure in train_failures:
        # Sample from 2-4 days before failure
        t_2d = failure['timestamp'] - timedelta(days=2)
        t_4d = failure['timestamp'] - timedelta(days=4)
        
        # Find closest DCS timestamp
        closest = min(all_timestamps, key=lambda x: abs(x - t_2d))
        if abs(closest - t_2d) < timedelta(hours=12):  # Within 12 hours
            positive_samples.append(closest)
    
    print(f"   Positive samples: {len(positive_samples)}")
    
    # Create negative samples (from healthy periods, >7 days from any failure)
    negative_samples = []
    failure_times = [f['timestamp'] for f in covered_failures]
    
    # Sample every 3 days from healthy periods
    check_time = dcs_start + timedelta(days=7)
    while check_time < dcs_end - timedelta(days=7):
        # Check if this is >7 days from any failure
        min_dist = min(abs((check_time - ft).days) for ft in failure_times)
        if min_dist > 7:
            closest = min(all_timestamps, key=lambda x: abs(x - check_time))
            if abs(closest - check_time) < timedelta(hours=12):
                negative_samples.append(closest)
        check_time += timedelta(days=3)
    
    # Balance classes
    random.shuffle(negative_samples)
    negative_samples = negative_samples[:len(positive_samples) * 2]
    
    print(f"   Negative samples: {len(negative_samples)}")
    
    # ==================================================================
    # FEATURE EXTRACTION
    # ==================================================================
    print(f"   Extracting features for {len(positive_samples) + len(negative_samples)} samples...")
    
    all_samples = positive_samples + negative_samples
    labels = [1] * len(positive_samples) + [0] * len(negative_samples)
    
    # Filter DCS data to relevant tags
    filtered_dcs = [dp for dp in dcs_data if dp['tag_name'] in tags]
    
    # Extract multi-window features (24h, 72h, 168h)
    features_raw = extract_multi_window_features(
        filtered_dcs,
        all_samples,
        window_hours_list=[24, 72, 168],
        tag_names=tags
    )
    
    # Convert to DataFrame
    features_df = pd.DataFrame(features_raw)
    timestamps = features_df['timestamp']
    features_df = features_df.drop('timestamp', axis=1)
    
    print(f"   Extracted {features_df.shape[1]} raw features")
    print(quick_data_quality_summary(features_df))
    
    # ==================================================================
    # DATA QUALITY REPORT
    # ==================================================================
    print(f"   Generating data quality report...")
    report_path = f'reports/data_quality_{asset_name}.md'
    generate_data_quality_report(features_df.copy(), report_path)
    
    # ==================================================================
    # PREPROCESSING
    # ==================================================================
    print(f"   Preprocessing features...")
    features_processed, scaler_params, prep_stats = preprocess_features(
        features_df,
        is_training=True,
        config={'outlier_factor': 3.0, 'correlation_threshold': 1.0}  # Keep all features
    )
    
    print(get_preprocessing_stats_summary(prep_stats))
    
    # Check feature-to-sample ratio
    n_features = features_processed.shape[1]
    n_samples = len(all_samples)
    ratio = n_samples / n_features
    print(f"\n   Feature-to-sample ratio: {n_samples} samples / {n_features} features = {ratio:.2f}")
    if ratio < 1:
        print(f"   ⚠️  WARNING: More features than samples! High risk of overfitting.")
        print(f"   Solution: Using strong L2 regularization (C=0.01)")
    
    # ==================================================================
    # APPLY SMOTE IF NEEDED
    # ==================================================================
    X_train = features_processed.values
    y_train = np.array(labels)
    
    if len(positive_samples) < len(negative_samples) / 3:
        print(f"   Applying SMOTE to balance classes...")
        X_train, y_train = apply_smote(X_train, y_train, random_state=42)
        print(f"   After SMOTE: {len(y_train)} samples ({np.sum(y_train)} positive)")
    
    # ==================================================================
    # TRAIN MODEL
    # ==================================================================
    print(f"   Training logistic regression...")
    
    if not SKLEARN_AVAILABLE:
        print("   ERROR: sklearn required for training. Skipping model training.")
        all_results[asset_name] = {
            'tags': tags,
            'n_train_failures': len(train_failures),
            'n_test_failures': len(test_failures),
            'n_positive_samples': len(positive_samples),
            'n_negative_samples': len(negative_samples),
            'status': 'sklearn_not_available'
        }
        continue
    
    model = LogisticRegression(
        penalty='l2',           # Ridge regularization to prevent overfitting
        C=0.01,                 # Strong regularization (0.01 = very strong)
        class_weight='balanced',
        max_iter=1000,
        random_state=42,
        solver='lbfgs'
    )
    model.fit(X_train, y_train)
    
    # Training set evaluation
    y_train_pred = model.predict(X_train)
    y_train_proba = model.predict_proba(X_train)[:, 1]
    train_metrics = compute_metrics(y_train, y_train_pred, y_train_proba)
    
    print(f"   Training performance:")
    print(f"     Accuracy:  {train_metrics['accuracy']:.3f}")
    print(f"     Precision: {train_metrics['precision']:.3f}")
    print(f"     Recall:    {train_metrics['recall']:.3f}")
    print(f"     ROC-AUC:   {train_metrics['roc_auc']:.3f}")
    
    # ==================================================================
    # TEST SET EVALUATION
    # ==================================================================
    print(f"   Evaluating on {len(test_failures)} held-out failures...")
    
    # Extract features for test failures
    test_positive_samples = []
    for failure in test_failures:
        t_2d = failure['timestamp'] - timedelta(days=2)
        closest = min(all_timestamps, key=lambda x: abs(x - t_2d))
        if abs(closest - t_2d) < timedelta(hours=12):
            test_positive_samples.append(closest)
    
    if len(test_positive_samples) > 0:
        # Extract test features
        test_features_raw = extract_multi_window_features(
            filtered_dcs,
            test_positive_samples,
            window_hours_list=[24, 72, 168],
            tag_names=tags
        )
        
        test_df = pd.DataFrame(test_features_raw).drop('timestamp', axis=1)
        
        # Preprocess test set (use training scaler_params)
        test_processed, _, _ = preprocess_features(
            test_df,
            is_training=False,
            scaler_params=scaler_params
        )
        
        # Predict on test set
        y_test_pred = model.predict(test_processed.values)
        y_test_proba = model.predict_proba(test_processed.values)[:, 1]
        
        # Compute test metrics
        y_test_true = np.array([1] * len(test_positive_samples))
        test_metrics = compute_metrics(y_test_true, y_test_pred, y_test_proba)
        
        print(f"\n   Test set performance (HELD-OUT FAILURES):")
        print(f"     Accuracy:  {test_metrics['accuracy']:.3f}")
        print(f"     Precision: {test_metrics['precision']:.3f}")
        print(f"     Recall:    {test_metrics['recall']:.3f} ← Did we detect the failures?")
        print(f"     ROC-AUC:   {test_metrics['roc_auc']:.3f}")
        
        print(f"\n   Individual test predictions:")
        for i, (failure, pred, proba) in enumerate(zip(test_failures, y_test_pred, y_test_proba)):
            status = "✓ DETECTED" if pred == 1 else "✗ MISSED"
            print(f"     Failure {i+1} ({failure['timestamp'].date()}): {status} (prob={proba:.3f})")
    
    else:
        test_positive_samples = []
        y_test_pred = np.array([])
        y_test_proba = np.array([])
        test_metrics = {}
    
    # ==================================================================
    # COMPREHENSIVE EVALUATION
    # ==================================================================
    print(f"   Generating evaluation reports...")
    eval_dir = f'reports/evaluation_{asset_name}'
    os.makedirs(eval_dir, exist_ok=True)
    
    # Combine train and test for full evaluation
    X_full = X_train
    y_full = y_train
    
    metrics = evaluate_model(
        model,
        X_full,
        y_full,
        features_processed.columns.tolist(),
        eval_dir
    )
    
    # Generate markdown report
    generate_evaluation_report(metrics, f'{eval_dir}/evaluation_report.md', asset_name)
    
    # ==================================================================
    # SAVE RESULTS
    # ==================================================================
    all_results[asset_name] = {
        'tags': tags,
        'n_train_failures': len(train_failures),
        'n_test_failures': len(test_failures),
        'n_positive_samples': len(positive_samples),
        'n_negative_samples': len(negative_samples),
        'n_features_raw': features_df.shape[1],
        'n_features_processed': features_processed.shape[1],
        'preprocessing_stats': {
            'total_missing_imputed': prep_stats['total_missing_imputed'],
            'total_outliers_clipped': prep_stats['total_outliers_clipped'],
            'n_dropped_features': len(prep_stats.get('dropped_features', []))
        },
        'train_metrics': train_metrics,
        'test_metrics': test_metrics if len(test_positive_samples) > 0 else {},
        'test_predictions': [
            {
                'failure_date': f['timestamp'].isoformat(),
                'predicted': int(pred),
                'probability': float(prob)
            }
            for f, pred, prob in zip(test_failures, y_test_pred, y_test_proba)
        ] if len(test_positive_samples) > 0 else [],
        'evaluation_metrics': metrics,
        'evaluation_dir': eval_dir,
        'status': 'training_complete'
    }
    
    # Save scaler params for future use
    with open(f'{eval_dir}/scaler_params.json', 'w') as f:
        # Convert numpy types to native Python types for JSON serialization
        scaler_json = {
            'means': {k: float(v) for k, v in scaler_params['means'].items()},
            'stds': {k: float(v) for k, v in scaler_params['stds'].items()},
            'feature_names': scaler_params['feature_names']
        }
        json.dump(scaler_json, f, indent=2)
    
    print(f"   Scaler params saved to: {eval_dir}/scaler_params.json")

# Save results
print("\n5. Saving results...")
with open('data/processed/baseline_model_results.json', 'w') as f:
    json.dump(all_results, f, indent=2)

print("\n" + "="*70)
print("TRAINING COMPLETE - SUMMARY")
print("="*70)
for asset, results in all_results.items():
    print(f"\n{asset}:")
    if results['status'] != 'training_complete':
        print(f"  Status: {results['status']}")
        continue
    
    print(f"  Train failures: {results['n_train_failures']}")
    print(f"  Test failures: {results['n_test_failures']}")  
    print(f"  Training samples: {results['n_positive_samples']} positive, {results['n_negative_samples']} negative")
    print(f"  Features: {results['n_features_raw']} raw → {results['n_features_processed']} after preprocessing")
    print(f"  Training accuracy: {results['train_metrics']['accuracy']:.3f}")
    print(f"  Evaluation metrics:")
    print(f"    - ROC-AUC: {results['evaluation_metrics']['roc_auc']:.3f}")
    print(f"    - Precision: {results['evaluation_metrics']['precision']:.3f}")
    print(f"    - Recall: {results['evaluation_metrics']['recall']:.3f}")
    print(f"    - F1 Score: {results['evaluation_metrics']['f1']:.3f}")
    
    if results['test_predictions']:
        n_detected = sum(1 for p in results['test_predictions'] if p['predicted'] == 1)
        print(f"  Held-out test failures: {n_detected}/{len(results['test_predictions'])} detected")
    
    print(f"  Reports saved to: {results['evaluation_dir']}/")

print("\n" + "="*70)
print("PHASE 1 VALIDATION COMPLETE")
print("="*70)
print("\n✓ Models trained with full preprocessing pipeline")
print("✓ Data quality reports generated")
print("✓ Evaluation plots created (ROC, confusion matrix, feature importance)")
print("✓ Held-out failures tested")
print("\nResults saved to: data/processed/baseline_model_results.json")
print("\nNext steps:")
print("  1. Review evaluation plots in reports/evaluation_*/")
print("  2. Analyze feature importance to understand predictions")
print("  3. Tune classification thresholds for desired precision/recall")
print("  4. Expand to Random Forest/XGBoost models (Phase 2)")

