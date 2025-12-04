"""Feature Engineering Pipeline

This module provides functions to extract time-windowed features from DCS data
for predictive maintenance modeling. It supports multiple window sizes and
aggregation functions to capture both short-term anomalies and long-term trends.
"""
import datetime as dt
from collections import defaultdict
from typing import Dict, List, Optional, Tuple


def extract_window_features(
    dcs_data: List[Dict],
    timestamps: List[dt.datetime],
    window_hours: int = 24,
    tag_names: Optional[List[str]] = None
) -> List[Dict[str, float]]:
    """Extract aggregated features from DCS data for given timestamps.
    
    For each timestamp, looks back `window_hours` and computes statistics
    (mean, std, min, max, last_value, slope) for each tag.
    
    Args:
        dcs_data: List of DCS data points with keys: timestamp, tag_name, value
        timestamps: List of timestamps to extract features for
        window_hours: Hours to look back from each timestamp
        tag_names: Optional list of tag names to include (None = all)
    
    Returns:
        List of feature dicts, one per timestamp
    """
    # Group DCS data by tag
    tag_data = defaultdict(list)
    for point in dcs_data:
        if tag_names is None or point['tag_name'] in tag_names:
            tag_data[point['tag_name']].append(point)
    
    # Sort each tag's data by timestamp
    for tag in tag_data:
        tag_data[tag].sort(key=lambda x: x['timestamp'])
    
    # Extract features for each timestamp
    features_list = []
    
    for ts in timestamps:
        window_start = ts - dt.timedelta(hours=window_hours)
        features = {'timestamp': ts}
        
        for tag_name, points in tag_data.items():
            # Filter to window
            window_points = [
                p for p in points
                if window_start <= p['timestamp'] < ts
            ]
            
            if len(window_points) == 0:
                # No data in window
                features[f'{tag_name}_count'] = 0
                features[f'{tag_name}_mean'] = None
                features[f'{tag_name}_std'] = None
                features[f'{tag_name}_min'] = None
                features[f'{tag_name}_max'] = None
                features[f'{tag_name}_last'] = None
                features[f'{tag_name}_slope'] = None
                continue
            
            values = [p['value'] for p in window_points]
            
            # Basic statistics
            mean_val = sum(values) / len(values)
            min_val = min(values)
            max_val = max(values)
            last_val = values[-1]
            
            # Standard deviation
            if len(values) > 1:
                variance = sum((x - mean_val) ** 2 for x in values) / (len(values) - 1)
                std_val = variance ** 0.5
            else:
                std_val = 0.0
            
            # Slope (linear trend)
            if len(values) > 1:
                # Simple linear regression
                n = len(values)
                x_vals = list(range(n))
                x_mean = sum(x_vals) / n
                y_mean = mean_val
                
                numerator = sum((x_vals[i] - x_mean) * (values[i] - y_mean) for i in range(n))
                denominator = sum((x - x_mean) ** 2 for x in x_vals)
                
                if denominator > 0:
                    slope_val = numerator / denominator
                else:
                    slope_val = 0.0
            else:
                slope_val = 0.0
            
            # Store features
            features[f'{tag_name}_count'] = len(values)
            features[f'{tag_name}_mean'] = mean_val
            features[f'{tag_name}_std'] = std_val
            features[f'{tag_name}_min'] = min_val
            features[f'{tag_name}_max'] = max_val
            features[f'{tag_name}_last'] = last_val
            features[f'{tag_name}_slope'] = slope_val
        
        features_list.append(features)
    
    return features_list


def extract_multi_window_features(
    dcs_data: List[Dict],
    timestamps: List[dt.datetime],
    window_hours_list: List[int] = [24, 72, 168],
    tag_names: Optional[List[str]] = None
) -> List[Dict[str, float]]:
    """Extract features at multiple time scales.
    
    Args:
        dcs_data: List of DCS data points
        timestamps: List of timestamps to extract features for
        window_hours_list: List of window sizes in hours
        tag_names: Optional list of tag names to include
    
    Returns:
        List of feature dicts with multi-scale features
    """
    all_features = []
    
    for ts in timestamps:
        combined_features = {'timestamp': ts}
        
        for window_hours in window_hours_list:
            window_features = extract_window_features(
                dcs_data, [ts], window_hours, tag_names
            )[0]
            
            # Add window prefix to feature names
            for key, value in window_features.items():
                if key != 'timestamp':
                    combined_features[f'w{window_hours}h_{key}'] = value
        
        all_features.append(combined_features)
    
    return all_features


def create_feature_matrix(
    features_list: List[Dict[str, float]],
    feature_names: Optional[List[str]] = None
) -> Tuple[List[List[float]], List[str]]:
    """Convert feature dicts to a matrix format.
    
    Args:
        features_list: List of feature dicts
        feature_names: Optional list of feature names to include (None = all numeric)
    
    Returns:
        Tuple of (feature_matrix, feature_names)
    """
    if len(features_list) == 0:
        return [], []
    
    # Determine feature names
    if feature_names is None:
        # Extract all numeric feature names (exclude timestamp)
        feature_names = sorted([
            k for k in features_list[0].keys()
            if k != 'timestamp' and features_list[0][k] is not None
        ])
    
    # Build matrix
    matrix = []
    for features in features_list:
        row = []
        for fname in feature_names:
            value = features.get(fname)
            if value is None:
                row.append(0.0)  # Impute missing as 0
            else:
                row.append(float(value))
        matrix.append(row)
    
    return matrix, feature_names


def compute_feature_statistics(
    feature_matrix: List[List[float]],
    feature_names: List[str]
) -> Dict[str, Dict[str, float]]:
    """Compute statistics for each feature across samples.
    
    Args:
        feature_matrix: List of feature vectors
        feature_names: List of feature names
    
    Returns:
        Dict mapping feature names to their statistics
    """
    if len(feature_matrix) == 0:
        return {}
    
    n_features = len(feature_names)
    stats = {}
    
    for i, fname in enumerate(feature_names):
        values = [row[i] for row in feature_matrix if len(row) > i]
        
        if len(values) == 0:
            continue
        
        mean_val = sum(values) / len(values)
        min_val = min(values)
        max_val = max(values)
        
        if len(values) > 1:
            variance = sum((x - mean_val) ** 2 for x in values) / (len(values) - 1)
            std_val = variance ** 0.5
        else:
            std_val = 0.0
        
        stats[fname] = {
            'mean': mean_val,
            'std': std_val,
            'min': min_val,
            'max': max_val,
            'count': len(values)
        }
    
    return stats


def normalize_features(
    feature_matrix: List[List[float]],
    feature_stats: Dict[str, Dict[str, float]],
    feature_names: List[str]
) -> List[List[float]]:
    """Normalize features using z-score normalization.
    
    Args:
        feature_matrix: List of feature vectors
        feature_stats: Feature statistics (from compute_feature_statistics)
        feature_names: List of feature names
    
    Returns:
        Normalized feature matrix
    """
    normalized = []
    
    for row in feature_matrix:
        norm_row = []
        for i, fname in enumerate(feature_names):
            if i >= len(row):
                norm_row.append(0.0)
                continue
            
            stats = feature_stats.get(fname, {})
            mean_val = stats.get('mean', 0.0)
            std_val = stats.get('std', 1.0)
            
            if std_val > 0:
                norm_value = (row[i] - mean_val) / std_val
            else:
                norm_value = 0.0
            
            norm_row.append(norm_value)
        
        normalized.append(norm_row)
    
    return normalized


if __name__ == '__main__':
    # Demo: Feature extraction
    print("Feature Engineering Pipeline Demo")
    print("=" * 70)
    
    # Create sample DCS data
    import datetime as dt
    base_time = dt.datetime(2023, 1, 1, tzinfo=dt.timezone.utc)
    
    sample_data = []
    for i in range(100):
        sample_data.append({
            'timestamp': base_time + dt.timedelta(minutes=i*10),
            'tag_name': 'TAG_A',
            'value': 50.0 + i * 0.1 + (i % 10) * 0.5
        })
        sample_data.append({
            'timestamp': base_time + dt.timedelta(minutes=i*10),
            'tag_name': 'TAG_B',
            'value': 100.0 - i * 0.05
        })
    
    # Extract features for a single timestamp
    target_time = base_time + dt.timedelta(hours=10)
    features = extract_window_features(sample_data, [target_time], window_hours=6)
    
    print(f"\nFeatures extracted for {target_time}:")
    print(f"  Window: last 6 hours")
    for key, value in features[0].items():
        if key != 'timestamp' and value is not None:
            print(f"  {key}: {value:.2f}")
    
    # Multi-window features
    print(f"\n\nMulti-window features:")
    multi_features = extract_multi_window_features(
        sample_data, [target_time], window_hours_list=[3, 6, 12]
    )
    print(f"  Total features: {len([k for k in multi_features[0].keys() if k != 'timestamp'])}")
    
    print("\nFeature engineering pipeline ready!")

