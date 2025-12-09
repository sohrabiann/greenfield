"""Phase 1 Volume Analysis - Failure-Aligned & Temporal Analysis

This script performs analysis on the rich DCS volume data for Dryer A problem assets.
It replaces the toy analysis from phase1_analysis.py with production-quality analysis
using real analog tags (currents, temperatures, pressures).
"""
import csv
import datetime as dt
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Import our volume loader
from dcs_volume_loader import (
    load_all_dcs_volumes,
    summarize_available_tags,
    PROBLEM_ASSET_LOCATIONS
)


def load_priority1_failures() -> Dict[str, List[dt.datetime]]:
    """Load Priority 1 service request timestamps by location.
    
    Returns:
        Dict mapping location codes to lists of failure timestamps
    """
    failures_by_location = defaultdict(list)
    
    with open('data/processed/problem_asset_failures.csv', 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['priority'] == '1':
                timestamp_str = row['service_request_timestamp']
                try:
                    timestamp = dt.datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                    # Map asset_name to location
                    asset_name = row['asset_name']
                    for location in PROBLEM_ASSET_LOCATIONS.keys():
                        if location in asset_name:
                            failures_by_location[location].append(timestamp)
                            break
                except Exception as e:
                    print(f"Warning: Could not parse timestamp {timestamp_str}: {e}")
    
    # Sort timestamps
    for location in failures_by_location:
        failures_by_location[location].sort()
    
    return dict(failures_by_location)


def extract_failure_windows(
    dcs_data: List[Dict],
    failure_times: List[dt.datetime],
    window_before_hours: int = 168,  # 7 days
    window_after_hours: int = 24
) -> List[Dict]:
    """Extract DCS data windows around each failure event.
    
    Args:
        dcs_data: List of DCS data points
        failure_times: List of failure timestamps
        window_before_hours: Hours before failure to include
        window_after_hours: Hours after failure to include
    
    Returns:
        List of dicts with keys: failure_time, tag_name, data_points
    """
    windows = []
    
    for failure_time in failure_times:
        window_start = failure_time - dt.timedelta(hours=window_before_hours)
        window_end = failure_time + dt.timedelta(hours=window_after_hours)
        
        # Group data by tag
        tag_data = defaultdict(list)
        for point in dcs_data:
            if window_start <= point['timestamp'] <= window_end:
                tag_data[point['tag_name']].append({
                    'timestamp': point['timestamp'],
                    'value': point['value'],
                    'hours_before_failure': (failure_time - point['timestamp']).total_seconds() / 3600
                })
        
        for tag_name, points in tag_data.items():
            if len(points) > 0:
                windows.append({
                    'failure_time': failure_time,
                    'tag_name': tag_name,
                    'data_points': sorted(points, key=lambda x: x['timestamp'])
                })
    
    return windows


def compute_window_statistics(
    data_points: List[Dict],
    window_hours: float
) -> Dict[str, float]:
    """Compute statistics for a time window.
    
    Args:
        data_points: List of data points with 'hours_before_failure' and 'value'
        window_hours: Window size in hours before failure (e.g., 72 for last 3 days)
    
    Returns:
        Dict with mean, std, min, max, count
    """
    # Filter to window
    window_points = [p for p in data_points if 0 <= p['hours_before_failure'] <= window_hours]
    
    if len(window_points) == 0:
        return {'mean': None, 'std': None, 'min': None, 'max': None, 'count': 0}
    
    values = [p['value'] for p in window_points]
    mean_val = sum(values) / len(values)
    
    if len(values) > 1:
        variance = sum((x - mean_val) ** 2 for x in values) / (len(values) - 1)
        std_val = variance ** 0.5
    else:
        std_val = 0.0
    
    return {
        'mean': mean_val,
        'std': std_val,
        'min': min(values),
        'max': max(values),
        'count': len(values)
    }


def analyze_failure_aligned_patterns(
    location: str,
    selected_tags: List[str],
    max_files: int = None
) -> Dict:
    """Analyze pre-failure patterns for a specific location.
    
    Args:
        location: Asset location code (e.g., 'BL-1829')
        selected_tags: List of tag names to analyze
        max_files: Number of DCS files to load
    
    Returns:
        Analysis results dict
    """
    print(f"\n{'='*70}")
    print(f"FAILURE-ALIGNED ANALYSIS: {location}")
    print(f"{'='*70}")
    
    # Load failures
    failures_by_location = load_priority1_failures()
    failure_times = failures_by_location.get(location, [])
    
    if len(failure_times) == 0:
        print(f"No Priority 1 failures found for {location}")
        return {}
    
    print(f"Found {len(failure_times)} Priority 1 failures")
    print(f"Date range: {min(failure_times)} to {max(failure_times)}")
    
    # Load DCS data for selected tags
    print(f"\nLoading DCS data for {len(selected_tags)} tags...")
    dcs_data = load_all_dcs_volumes(
        selected_tags=set(selected_tags),
        max_files=max_files
    )
    
    if len(dcs_data) == 0:
        print("No DCS data found")
        return {}
    
    # Extract failure windows
    print(f"\nExtracting failure windows...")
    windows = extract_failure_windows(dcs_data, failure_times)
    
    # Analyze each tag
    results = {}
    for tag in selected_tags:
        tag_windows = [w for w in windows if w['tag_name'] == tag]
        
        if len(tag_windows) == 0:
            continue
        
        print(f"\n  Tag: {tag}")
        print(f"    Windows: {len(tag_windows)}")
        
        # Compute statistics for different time windows before failure
        window_stats = {}
        for window_hours in [24, 72, 168]:  # 1 day, 3 days, 7 days
            stats_list = []
            for window in tag_windows:
                stats = compute_window_statistics(window['data_points'], window_hours)
                if stats['count'] > 0:
                    stats_list.append(stats)
            
            if len(stats_list) > 0:
                # Aggregate across failures
                avg_mean = sum(s['mean'] for s in stats_list) / len(stats_list)
                avg_std = sum(s['std'] for s in stats_list) / len(stats_list)
                window_stats[f'{window_hours}h'] = {
                    'avg_mean': avg_mean,
                    'avg_std': avg_std,
                    'n_failures': len(stats_list)
                }
                print(f"    Last {window_hours}h: mean={avg_mean:.2f}, std={avg_std:.2f}")
        
        results[tag] = {
            'n_windows': len(tag_windows),
            'window_stats': window_stats
        }
    
    return results


def compute_autocorrelation(values: List[float], max_lag: int = 12) -> List[float]:
    """Compute autocorrelation function.
    
    Args:
        values: Time series values
        max_lag: Maximum lag to compute
    
    Returns:
        List of autocorrelation coefficients
    """
    n = len(values)
    if n < max_lag + 1:
        return []
    
    mean_val = sum(values) / n
    c0 = sum((x - mean_val) ** 2 for x in values) / n
    
    if c0 == 0:
        return [1.0] * (max_lag + 1)
    
    acf = [1.0]  # lag 0
    for lag in range(1, max_lag + 1):
        c_lag = sum((values[i] - mean_val) * (values[i - lag] - mean_val) 
                    for i in range(lag, n)) / n
        acf.append(c_lag / c0)
    
    return acf


def analyze_temporal_structure(
    location: str,
    selected_tags: List[str],
    max_files: int = None
) -> Dict:
    """Analyze temporal structure and autocorrelation.
    
    Args:
        location: Asset location code
        selected_tags: List of tag names to analyze
        max_files: Number of DCS files to load
    
    Returns:
        Analysis results dict
    """
    print(f"\n{'='*70}")
    print(f"TEMPORAL STRUCTURE ANALYSIS: {location}")
    print(f"{'='*70}")
    
    # Load DCS data
    print(f"Loading DCS data for {len(selected_tags)} tags...")
    dcs_data = load_all_dcs_volumes(
        selected_tags=set(selected_tags),
        max_files=max_files
    )
    
    if len(dcs_data) == 0:
        print("No DCS data found")
        return {}
    
    # Group by tag
    tag_data = defaultdict(list)
    for point in dcs_data:
        tag_data[point['tag_name']].append(point)
    
    results = {}
    for tag in selected_tags:
        if tag not in tag_data:
            continue
        
        points = sorted(tag_data[tag], key=lambda x: x['timestamp'])
        values = [p['value'] for p in points]
        
        if len(values) < 20:
            continue
        
        print(f"\n  Tag: {tag}")
        print(f"    Points: {len(values)}")
        
        # Compute ACF
        acf = compute_autocorrelation(values, max_lag=12)
        
        if len(acf) > 0:
            print(f"    ACF[1]: {acf[1]:.3f}")
            print(f"    ACF[6]: {acf[6]:.3f}" if len(acf) > 6 else "")
            print(f"    ACF[12]: {acf[12]:.3f}" if len(acf) > 12 else "")
        
        # Basic stats
        mean_val = sum(values) / len(values)
        variance = sum((x - mean_val) ** 2 for x in values) / len(values)
        std_val = variance ** 0.5
        
        results[tag] = {
            'n_points': len(values),
            'mean': mean_val,
            'std': std_val,
            'min': min(values),
            'max': max(values),
            'acf': acf
        }
    
    return results


def main():
    """Run failure-aligned and temporal analysis for key problem assets."""
    
    # Create output directory
    os.makedirs('reports/figures/02_failure_aligned_analysis', exist_ok=True)
    os.makedirs('reports/figures/03_temporal_structure_and_windows', exist_ok=True)
    
    # Define key tags to analyze for each location
    analysis_config = {
        'BL-1829': [
            'CH-DRA_PROC_AIR:IT18179.MEAS',  # Motor current
            'CH-DRA_PROC_AIR:TT18179A.MEAS',  # Bearing temp A
            'CH-DRA_PROC_AIR:TT18179B.MEAS',  # Bearing temp B
            'CH-DRA_PROC_AIR:TT18179C.MEAS',  # Bearing temp C
        ],
        'RV-1834': [
            'CH-DR_DUSTCOLL:IT18211.MEAS',  # Motor current
            'CH-DR_DUSTCOLL:IT18216.MEAS',  # Motor current
            'CH-DR_DUSTCOLL:TT18217.MEAS',  # Temperature
        ],
        'DC-1834': [
            'CH-DR_DUSTCOLL:IT18211.MEAS',  # Motor current
            'CH-DR_DUSTCOLL:TT18217.MEAS',  # Temperature
        ],
    }
    
    all_results = {}
    
    for location, tags in analysis_config.items():
        print(f"\n\n{'#'*70}")
        print(f"# ANALYZING: {location} - {PROBLEM_ASSET_LOCATIONS[location]['description']}")
        print(f"{'#'*70}")
        
        # Failure-aligned analysis
        failure_results = analyze_failure_aligned_patterns(location, tags, max_files=None)
        
        # Temporal structure analysis
        temporal_results = analyze_temporal_structure(location, tags, max_files=None)
        
        all_results[location] = {
            'failure_aligned': failure_results,
            'temporal_structure': temporal_results
        }
    
    # Save results
    output_file = 'data/processed/volume_analysis_results.json'
    with open(output_file, 'w') as f:
        # Convert datetime objects to strings for JSON serialization
        json.dump(all_results, f, indent=2, default=str)
    
    print(f"\n\nResults saved to {output_file}")
    print("\nAnalysis complete!")


if __name__ == '__main__':
    main()

