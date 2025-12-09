"""Preprocessing Pipeline for Predictive Maintenance Models

This module provides comprehensive preprocessing functions including:
- Missing value imputation
- Outlier detection and clipping
- Feature correlation removal
- Feature standardization (z-score normalization)
- SMOTE oversampling (optional)
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple


def handle_missing_values(df: pd.DataFrame, method: str = 'forward_median') -> Tuple[pd.DataFrame, Dict]:
    """Handle missing values with forward fill and median imputation.
    
    Also creates binary indicator columns for originally missing values.
    
    Args:
        df: DataFrame with features (no timestamp column)
        method: Imputation method ('forward_median', 'median', 'zero')
    
    Returns:
        Tuple of (imputed_df, stats_dict)
    """
    df = df.copy()
    stats = {
        'total_missing': df.isna().sum().sum(),
        'missing_by_feature': df.isna().sum().to_dict(),
        'missing_percentage': (df.isna().sum() / len(df) * 100).to_dict()
    }
    
    # Create missing indicators for features with >0% missing
    for col in df.columns:
        if df[col].isna().any():
            df[f'{col}_missing'] = df[col].isna().astype(int)
    
    # Impute missing values
    if method == 'forward_median':
        # Forward fill first
        df = df.ffill()
        # Backward fill for any remaining (start of series)
        df = df.bfill()
        # Median fill for any still remaining
        df = df.fillna(df.median())
    elif method == 'median':
        df = df.fillna(df.median())
    elif method == 'zero':
        df = df.fillna(0)
    
    # Fill any remaining NaN with 0 (in case median is also NaN)
    df = df.fillna(0)
    
    stats['imputation_method'] = method
    
    return df, stats


def clip_outliers(df: pd.DataFrame, method: str = 'iqr', factor: float = 3.0) -> Tuple[pd.DataFrame, Dict]:
    """Clip extreme outliers using IQR method.
    
    Args:
        df: DataFrame with features
        method: 'iqr' or 'zscore'
        factor: IQR multiplier (default 3.0) or z-score threshold
    
    Returns:
        Tuple of (clipped_df, stats_dict)
    """
    df = df.copy()
    stats = {
        'method': method,
        'factor': factor,
        'outliers_clipped': {}
    }
    
    for col in df.columns:
        if '_missing' in col:
            continue  # Skip missing indicator columns
        
        if method == 'iqr':
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - factor * IQR
            upper_bound = Q3 + factor * IQR
        elif method == 'zscore':
            mean = df[col].mean()
            std = df[col].std()
            lower_bound = mean - factor * std
            upper_bound = mean + factor * std
        
        # Count outliers before clipping
        n_outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
        
        # Clip
        df[col] = df[col].clip(lower_bound, upper_bound)
        
        if n_outliers > 0:
            stats['outliers_clipped'][col] = {
                'count': int(n_outliers),
                'percentage': float(n_outliers / len(df) * 100),
                'lower_bound': float(lower_bound),
                'upper_bound': float(upper_bound)
            }
    
    return df, stats


def remove_correlated_features(
    df: pd.DataFrame,
    threshold: float = 0.95,
    exclude_missing: bool = True
) -> Tuple[pd.DataFrame, List[str]]:
    """Remove highly correlated features.
    
    Args:
        df: DataFrame with features
        threshold: Correlation threshold (default 0.95)
        exclude_missing: If True, don't drop missing indicator columns
    
    Returns:
        Tuple of (cleaned_df, dropped_feature_names)
    """
    df = df.copy()
    
    # Separate missing indicators if needed
    if exclude_missing:
        missing_cols = [col for col in df.columns if '_missing' in col]
        other_cols = [col for col in df.columns if '_missing' not in col]
        df_to_check = df[other_cols]
    else:
        missing_cols = []
        df_to_check = df
    
    # Compute correlation matrix
    corr_matrix = df_to_check.corr().abs()
    
    # Get upper triangle
    upper = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )
    
    # Find features to drop
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    
    # Drop from main dataframe
    df = df.drop(columns=to_drop)
    
    return df, to_drop


def standardize_features(
    df: pd.DataFrame,
    scaler_params: Optional[Dict] = None,
    exclude_missing: bool = True
) -> Tuple[pd.DataFrame, Dict]:
    """Standardize features using z-score normalization.
    
    Args:
        df: DataFrame with features
        scaler_params: If provided (test set), use these means/stds
        exclude_missing: If True, don't standardize missing indicators
    
    Returns:
        Tuple of (standardized_df, scaler_params)
    """
    df = df.copy()
    
    # Separate missing indicators if needed
    if exclude_missing:
        missing_cols = [col for col in df.columns if '_missing' in col]
        other_cols = [col for col in df.columns if '_missing' not in col]
        df_missing = df[missing_cols] if missing_cols else pd.DataFrame()
        df_to_scale = df[other_cols]
    else:
        df_missing = pd.DataFrame()
        df_to_scale = df
    
    # Compute or use scaler parameters
    if scaler_params is None:
        # Training set - compute statistics
        means = df_to_scale.mean()
        stds = df_to_scale.std()
        
        scaler_params = {
            'means': means.to_dict(),
            'stds': stds.to_dict(),
            'feature_names': df_to_scale.columns.tolist()
        }
    else:
        # Test set - use provided statistics
        means = pd.Series(scaler_params['means'])
        stds = pd.Series(scaler_params['stds'])
    
    # Standardize
    df_standardized = (df_to_scale - means) / (stds + 1e-8)  # Add epsilon to avoid division by zero
    
    # Recombine with missing indicators
    if not df_missing.empty:
        df_standardized = pd.concat([df_standardized, df_missing], axis=1)
    
    return df_standardized, scaler_params


def preprocess_features(
    features_df: pd.DataFrame,
    is_training: bool = True,
    scaler_params: Optional[Dict] = None,
    config: Optional[Dict] = None
) -> Tuple[pd.DataFrame, Dict, Dict]:
    """Complete preprocessing pipeline.
    
    Steps:
    1. Handle missing values (forward fill -> median)
    2. Clip outliers (IQR with factor=3)
    3. Remove correlated features (r > 0.95, training only)
    4. Standardize (z-score)
    
    Args:
        features_df: DataFrame with features (no timestamp)
        is_training: If True, fit parameters; if False, use scaler_params
        scaler_params: Parameters from training set (for test set)
        config: Optional dict with settings:
            - outlier_factor: IQR factor (default 3.0)
            - correlation_threshold: Threshold for dropping features (default 0.95)
            - imputation_method: 'forward_median', 'median', or 'zero'
    
    Returns:
        Tuple of (processed_df, scaler_params, preprocessing_stats)
    """
    if config is None:
        config = {}
    
    outlier_factor = config.get('outlier_factor', 3.0)
    corr_threshold = config.get('correlation_threshold', 0.95)
    imputation_method = config.get('imputation_method', 'forward_median')
    
    preprocessing_stats = {
        'is_training': is_training,
        'original_shape': features_df.shape,
        'steps': []
    }
    
    # Step 1: Handle missing values
    df, missing_stats = handle_missing_values(features_df, method=imputation_method)
    preprocessing_stats['steps'].append({
        'step': 'missing_value_imputation',
        'stats': missing_stats
    })
    preprocessing_stats['total_missing_imputed'] = missing_stats['total_missing']
    
    # Step 2: Clip outliers
    df, outlier_stats = clip_outliers(df, method='iqr', factor=outlier_factor)
    preprocessing_stats['steps'].append({
        'step': 'outlier_clipping',
        'stats': outlier_stats
    })
    preprocessing_stats['total_outliers_clipped'] = sum(
        v['count'] for v in outlier_stats['outliers_clipped'].values()
    )
    
    # Step 3: Remove correlated features (training only)
    if is_training:
        df, dropped_features = remove_correlated_features(df, threshold=corr_threshold)
        preprocessing_stats['steps'].append({
            'step': 'correlation_removal',
            'dropped_features': dropped_features,
            'n_dropped': len(dropped_features)
        })
        preprocessing_stats['dropped_features'] = dropped_features
    else:
        # Test set - use same features as training
        if scaler_params and 'feature_names' in scaler_params:
            train_features = scaler_params['feature_names']
            
            # Add missing columns with zeros
            for col in train_features:
                if col not in df.columns:
                    df[col] = 0.0
            
            # Reorder to match training features exactly
            df = df[train_features]
        preprocessing_stats['dropped_features'] = []
    
    # Step 4: Standardize
    df, scaler_params = standardize_features(df, scaler_params=scaler_params)
    preprocessing_stats['steps'].append({
        'step': 'standardization',
        'method': 'z-score'
    })
    
    # Save final column list for test set alignment
    if is_training:
        scaler_params['all_columns'] = df.columns.tolist()
    else:
        # Ensure test set has exactly the same columns as training
        if scaler_params and 'all_columns' in scaler_params:
            train_columns = scaler_params['all_columns']
            
            # Add missing columns with zeros
            for col in train_columns:
                if col not in df.columns:
                    df[col] = 0.0
            
            # Reorder to match training exactly
            df = df[train_columns]
    
    preprocessing_stats['final_shape'] = df.shape
    preprocessing_stats['n_features'] = df.shape[1]
    
    return df, scaler_params, preprocessing_stats


def apply_smote(X: np.ndarray, y: np.ndarray, random_state: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """Apply SMOTE oversampling to balance classes.
    
    Args:
        X: Feature matrix
        y: Labels
        random_state: Random seed
    
    Returns:
        Tuple of (X_resampled, y_resampled)
    """
    try:
        from imblearn.over_sampling import SMOTE
        
        smote = SMOTE(random_state=random_state)
        X_resampled, y_resampled = smote.fit_resample(X, y)
        
        return X_resampled, y_resampled
    
    except ImportError:
        print("Warning: imbalanced-learn not installed. Skipping SMOTE.")
        print("Install with: pip install imbalanced-learn")
        return X, y


def get_preprocessing_stats_summary(preprocessing_stats: Dict) -> str:
    """Generate human-readable summary of preprocessing statistics.
    
    Args:
        preprocessing_stats: Stats dict from preprocess_features
    
    Returns:
        Formatted string summary
    """
    summary = []
    summary.append("=" * 70)
    summary.append("PREPROCESSING SUMMARY")
    summary.append("=" * 70)
    
    summary.append(f"\nOriginal shape: {preprocessing_stats['original_shape']}")
    summary.append(f"Final shape: {preprocessing_stats['final_shape']}")
    
    summary.append(f"\nMissing values imputed: {preprocessing_stats['total_missing_imputed']}")
    summary.append(f"Outliers clipped: {preprocessing_stats['total_outliers_clipped']}")
    
    if preprocessing_stats.get('dropped_features'):
        summary.append(f"Correlated features dropped: {len(preprocessing_stats['dropped_features'])}")
        if len(preprocessing_stats['dropped_features']) <= 5:
            for feat in preprocessing_stats['dropped_features']:
                summary.append(f"  - {feat}")
    
    summary.append(f"\nFinal feature count: {preprocessing_stats['n_features']}")
    summary.append("=" * 70)
    
    return "\n".join(summary)

