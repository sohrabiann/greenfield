"""Data Quality Analysis and Reporting

This module provides functions to analyze and report on data quality issues
including missing values, outliers, feature correlations, and distributions.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from pathlib import Path


def analyze_missing_values(df: pd.DataFrame) -> Dict:
    """Analyze missing values in the dataset.
    
    Args:
        df: DataFrame to analyze
    
    Returns:
        Dict with missing value statistics
    """
    total_cells = df.shape[0] * df.shape[1]
    total_missing = df.isna().sum().sum()
    
    missing_by_feature = df.isna().sum()
    missing_pct_by_feature = (missing_by_feature / len(df) * 100)
    
    # Features with missing values
    features_with_missing = missing_by_feature[missing_by_feature > 0].sort_values(ascending=False)
    
    return {
        'total_cells': total_cells,
        'total_missing': int(total_missing),
        'missing_percentage': float(total_missing / total_cells * 100),
        'features_with_missing': {
            k: {'count': int(v), 'percentage': float(missing_pct_by_feature[k])}
            for k, v in features_with_missing.items()
        },
        'n_features_with_missing': len(features_with_missing)
    }


def detect_outliers(df: pd.DataFrame, method: str = 'iqr', factor: float = 3.0) -> Dict:
    """Detect outliers in each feature.
    
    Args:
        df: DataFrame to analyze
        method: 'iqr' or 'zscore'
        factor: IQR multiplier or z-score threshold
    
    Returns:
        Dict with outlier statistics
    """
    outliers_by_feature = {}
    
    for col in df.columns:
        if df[col].dtype in [np.float64, np.int64]:
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
            
            outliers = ((df[col] < lower_bound) | (df[col] > upper_bound))
            n_outliers = outliers.sum()
            
            if n_outliers > 0:
                outlier_values = df[col][outliers].values
                outliers_by_feature[col] = {
                    'count': int(n_outliers),
                    'percentage': float(n_outliers / len(df) * 100),
                    'lower_bound': float(lower_bound),
                    'upper_bound': float(upper_bound),
                    'min_outlier': float(outlier_values.min()),
                    'max_outlier': float(outlier_values.max())
                }
    
    total_outliers = sum(v['count'] for v in outliers_by_feature.values())
    
    return {
        'method': method,
        'factor': factor,
        'total_outliers': total_outliers,
        'features_with_outliers': outliers_by_feature,
        'n_features_with_outliers': len(outliers_by_feature)
    }


def compute_correlation_matrix(df: pd.DataFrame, threshold: float = 0.95) -> Dict:
    """Compute correlation matrix and identify highly correlated pairs.
    
    Args:
        df: DataFrame to analyze
        threshold: Correlation threshold for identifying high correlation
    
    Returns:
        Dict with correlation statistics
    """
    # Compute correlation matrix
    corr_matrix = df.corr().abs()
    
    # Get upper triangle
    upper = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )
    
    # Find highly correlated pairs
    high_corr_pairs = []
    for column in upper.columns:
        for index in upper.index:
            corr_value = upper.loc[index, column]
            if pd.notna(corr_value) and corr_value > threshold:
                high_corr_pairs.append({
                    'feature1': index,
                    'feature2': column,
                    'correlation': float(corr_value)
                })
    
    # Sort by correlation value
    high_corr_pairs.sort(key=lambda x: -x['correlation'])
    
    return {
        'threshold': threshold,
        'n_high_corr_pairs': len(high_corr_pairs),
        'high_corr_pairs': high_corr_pairs[:20],  # Top 20
        'correlation_matrix_shape': corr_matrix.shape
    }


def generate_feature_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """Generate summary statistics for all features.
    
    Args:
        df: DataFrame to analyze
    
    Returns:
        DataFrame with statistics
    """
    stats = df.describe().T
    
    # Add additional statistics
    stats['missing_count'] = df.isna().sum()
    stats['missing_pct'] = (df.isna().sum() / len(df) * 100)
    stats['n_unique'] = df.nunique()
    stats['dtype'] = df.dtypes
    
    return stats


def generate_data_quality_report(features_df: pd.DataFrame, output_path: str) -> None:
    """Generate comprehensive data quality report.
    
    Creates a markdown report with:
    - Missing value summary
    - Outlier detection results
    - Feature correlation analysis
    - Summary statistics table
    
    Args:
        features_df: DataFrame to analyze
        output_path: Path to save markdown report
    """
    # Ensure output directory exists
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Run analyses
    missing_analysis = analyze_missing_values(features_df)
    outlier_analysis = detect_outliers(features_df, method='iqr', factor=3.0)
    correlation_analysis = compute_correlation_matrix(features_df, threshold=0.95)
    feature_stats = generate_feature_statistics(features_df)
    
    # Generate markdown report
    report = []
    report.append("# Data Quality Report")
    report.append(f"\n**Generated**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"\n**Dataset Shape**: {features_df.shape[0]} rows × {features_df.shape[1]} features")
    
    # Missing Values Section
    report.append("\n## Missing Values Analysis")
    report.append(f"\n- **Total cells**: {missing_analysis['total_cells']:,}")
    report.append(f"- **Missing cells**: {missing_analysis['total_missing']:,} ({missing_analysis['missing_percentage']:.2f}%)")
    report.append(f"- **Features with missing values**: {missing_analysis['n_features_with_missing']}")
    
    if missing_analysis['features_with_missing']:
        report.append("\n### Top Features with Missing Values")
        report.append("\n| Feature | Missing Count | Missing % |")
        report.append("|---------|---------------|-----------|")
        
        for feat, stats in list(missing_analysis['features_with_missing'].items())[:10]:
            report.append(f"| {feat} | {stats['count']} | {stats['percentage']:.2f}% |")
    else:
        report.append("\n**No missing values detected!** ✓")
    
    # Outliers Section
    report.append("\n## Outlier Detection")
    report.append(f"\n- **Method**: {outlier_analysis['method']} (factor={outlier_analysis['factor']})")
    report.append(f"- **Total outliers detected**: {outlier_analysis['total_outliers']:,}")
    report.append(f"- **Features with outliers**: {outlier_analysis['n_features_with_outliers']}")
    
    if outlier_analysis['features_with_outliers']:
        report.append("\n### Top Features with Outliers")
        report.append("\n| Feature | Count | % | Min Outlier | Max Outlier |")
        report.append("|---------|-------|---|-------------|-------------|")
        
        sorted_outliers = sorted(
            outlier_analysis['features_with_outliers'].items(),
            key=lambda x: -x[1]['count']
        )[:10]
        
        for feat, stats in sorted_outliers:
            report.append(
                f"| {feat} | {stats['count']} | {stats['percentage']:.1f}% | "
                f"{stats['min_outlier']:.2f} | {stats['max_outlier']:.2f} |"
            )
    
    # Correlation Section
    report.append("\n## Feature Correlation Analysis")
    report.append(f"\n- **Correlation threshold**: {correlation_analysis['threshold']}")
    report.append(f"- **High correlation pairs found**: {correlation_analysis['n_high_corr_pairs']}")
    
    if correlation_analysis['high_corr_pairs']:
        report.append("\n### Highly Correlated Feature Pairs (Top 10)")
        report.append("\n| Feature 1 | Feature 2 | Correlation |")
        report.append("|-----------|-----------|-------------|")
        
        for pair in correlation_analysis['high_corr_pairs'][:10]:
            report.append(
                f"| {pair['feature1']} | {pair['feature2']} | {pair['correlation']:.3f} |"
            )
    
    # Summary Statistics Section
    report.append("\n## Summary Statistics")
    report.append(f"\n**Top 10 features by standard deviation (most variable):**")
    report.append("\n| Feature | Mean | Std | Min | Max | Missing % |")
    report.append("|---------|------|-----|-----|-----|-----------|")
    
    top_variable = feature_stats.nlargest(10, 'std')
    for feat_name, row in top_variable.iterrows():
        report.append(
            f"| {feat_name} | {row['mean']:.2f} | {row['std']:.2f} | "
            f"{row['min']:.2f} | {row['max']:.2f} | {row['missing_pct']:.1f}% |"
        )
    
    # Data Types Section
    report.append("\n## Data Types")
    dtype_counts = feature_stats['dtype'].value_counts()
    for dtype, count in dtype_counts.items():
        report.append(f"- **{dtype}**: {count} features")
    
    # Recommendations Section
    report.append("\n## Recommendations")
    
    recommendations = []
    
    if missing_analysis['missing_percentage'] > 1:
        recommendations.append("- **Missing values**: Significant missing data detected. Use forward-fill + median imputation.")
    
    if outlier_analysis['total_outliers'] > 0:
        recommendations.append(f"- **Outliers**: {outlier_analysis['total_outliers']:,} outliers detected. Consider IQR clipping.")
    
    if correlation_analysis['n_high_corr_pairs'] > 0:
        recommendations.append(f"- **Correlation**: {correlation_analysis['n_high_corr_pairs']} highly correlated pairs. Consider removing redundant features.")
    
    if not recommendations:
        recommendations.append("- **Good quality**: No major data quality issues detected!")
    
    for rec in recommendations:
        report.append(rec)
    
    # Write report
    with open(output_path, 'w') as f:
        f.write('\n'.join(report))
    
    print(f"Data quality report saved to: {output_path}")


def quick_data_quality_summary(features_df: pd.DataFrame) -> str:
    """Generate quick console summary of data quality.
    
    Args:
        features_df: DataFrame to analyze
    
    Returns:
        Formatted string summary
    """
    missing_analysis = analyze_missing_values(features_df)
    outlier_analysis = detect_outliers(features_df)
    
    summary = []
    summary.append("=" * 70)
    summary.append("DATA QUALITY QUICK SUMMARY")
    summary.append("=" * 70)
    summary.append(f"Shape: {features_df.shape[0]} rows × {features_df.shape[1]} features")
    summary.append(f"Missing values: {missing_analysis['total_missing']:,} ({missing_analysis['missing_percentage']:.2f}%)")
    summary.append(f"Outliers (IQR, factor=3): {outlier_analysis['total_outliers']:,}")
    summary.append(f"Features with missing: {missing_analysis['n_features_with_missing']}")
    summary.append(f"Features with outliers: {outlier_analysis['n_features_with_outliers']}")
    summary.append("=" * 70)
    
    return "\n".join(summary)


