"""DCS Volume Data Loader

This module loads and processes the rich DCS data from the volume extracts
(26 files with ~400 tags each). It handles wide-format CSV files where each 
row is a timestamp and each column is a DCS tag reading.

Key functions:
- load_all_dcs_volumes(): Load all 26 DCS extract files
- extract_problem_asset_tags(): Filter to tags for Dryer A problem assets
- get_analog_tags(): Prioritize analog tags (IT, TT, PT, FT)
"""
import csv
import datetime as dt
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

# Base path to DCS volume data
DCS_VOLUMES_PATH = Path('greenfield_data_context/volumes/dcsrawdataextracts')

# State mappings for binary tags
STATE_MAP = {
    'Open/Running': 1.0,
    'Closed/Stopped': 0.0,
    'On': 1.0,
    'Off': 0.0,
}

# Known problem asset locations and their DCS system prefixes
PROBLEM_ASSET_LOCATIONS = {
    'BL-1829': {
        'description': 'Main Dryer A Blower',
        'tag_patterns': ['CH-DRA_PROC_AIR:', 'BL1829'],
        'key_numbers': ['18179']  # IT18179, TT18179A-E
    },
    'RV-1834': {
        'description': 'Rotary Valve (Dust Collector)',
        'tag_patterns': ['CH-DR_DUSTCOLL:', 'RV1834'],
        'key_numbers': ['18211', '18216', '18222']  # IT tags
    },
    'DC-1834': {
        'description': 'Dust Collector',
        'tag_patterns': ['CH-DR_DUSTCOLL:', 'DC1834'],
        'key_numbers': []
    },
    'E-1834': {
        'description': 'Product Cooler',
        'tag_patterns': ['CH-DR_DUSTCOLL:', 'E1834'],
        'key_numbers': []
    },
    'P-1837': {
        'description': 'Utility Water Pump',
        'tag_patterns': ['CH-DR_UTIL_WTR:', 'P1837'],
        'key_numbers': []
    },
    'CV-1828': {
        'description': 'Conveyor',
        'tag_patterns': ['CH-DRYER_A:', 'CV1828'],
        'key_numbers': []
    },
}


def parse_timestamp(val: str) -> Optional[dt.datetime]:
    """Parse timestamp string to datetime (timezone-aware UTC)."""
    try:
        # Try ISO format with timezone
        return dt.datetime.fromisoformat(val.replace('Z', '+00:00'))
    except Exception:
        # Try without timezone, then make it UTC
        try:
            naive_dt = dt.datetime.strptime(val, '%Y-%m-%d %H:%M')
            return naive_dt.replace(tzinfo=dt.timezone.utc)
        except Exception:
            return None


def parse_value(val: str, data_type: Optional[str] = None) -> Optional[float]:
    """Parse a value string to float, handling special cases."""
    if val is None or val == '':
        return None
    
    val = val.strip()
    
    # Handle special values
    if val in ('No Data', 'Bad Input', 'Over Range', 'Configure', 'Calc Failed', 'I/O Timeout'):
        return None
    
    # Try state mapping first
    if val in STATE_MAP:
        return STATE_MAP[val]
    
    # Try numeric parsing
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def get_dcs_volume_files() -> List[Path]:
    """Get all DCS extract sample files."""
    if not DCS_VOLUMES_PATH.exists():
        return []
    
    # Get all sample files (not the filelist CSVs)
    files = sorted([
        f for f in DCS_VOLUMES_PATH.glob('dcsrawdataextracts_sample*.csv')
        if not f.name.endswith('filelist.csv')
    ])
    return files


def load_dcs_file_headers(file_path: Path) -> Tuple[List[str], Dict[str, str]]:
    """Load headers and data types from a DCS file.
    
    Returns:
        Tuple of (tag_names, tag_data_types)
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        
        # Row 1: Headers (tag names)
        headers = next(reader)
        
        # Row 2: Data types
        data_types = next(reader)
        
    # Build data type mapping
    tag_data_types = {tag: dtype for tag, dtype in zip(headers, data_types) if tag}
    
    return headers, tag_data_types


def identify_problem_asset_tags(headers: List[str]) -> Dict[str, List[str]]:
    """Identify tags related to each problem asset location.
    
    Returns:
        Dict mapping location codes to lists of tag names
    """
    location_tags = defaultdict(list)
    
    for tag in headers:
        if not tag or tag == 'Time':
            continue
        
        tag_upper = tag.upper()
        
        # Check each problem asset location
        for location, config in PROBLEM_ASSET_LOCATIONS.items():
            matched = False
            
            # Check tag patterns (system prefixes and direct asset codes)
            for pattern in config['tag_patterns']:
                if pattern.upper() in tag_upper:
                    matched = True
                    break
            
            # Also check key tag numbers
            if not matched:
                for num in config['key_numbers']:
                    if num in tag:
                        matched = True
                        break
            
            if matched:
                location_tags[location].append(tag)
    
    return dict(location_tags)


def get_analog_tags(tags: List[str]) -> Dict[str, List[str]]:
    """Categorize tags by type (IT=current, TT=temp, PT=pressure, FT=flow).
    
    Returns:
        Dict with keys 'current', 'temperature', 'pressure', 'flow', 'other'
    """
    categorized = {
        'current': [],
        'temperature': [],
        'pressure': [],
        'flow': [],
        'status': [],
        'other': []
    }
    
    for tag in tags:
        tag_upper = tag.upper()
        
        # Current tags (IT prefix)
        if re.search(r':IT\d+', tag_upper):
            categorized['current'].append(tag)
        # Temperature tags (TT prefix)
        elif re.search(r':TT\d+', tag_upper):
            categorized['temperature'].append(tag)
        # Pressure tags (PT prefix)
        elif re.search(r':PT\d+', tag_upper):
            categorized['pressure'].append(tag)
        # Flow tags (FT or FIC prefix)
        elif re.search(r':(FT|FIC)\d+', tag_upper):
            categorized['flow'].append(tag)
        # Status tags (STAIND suffix)
        elif '.STAIND' in tag_upper:
            categorized['status'].append(tag)
        else:
            categorized['other'].append(tag)
    
    return categorized


def load_dcs_volume_data(
    file_path: Path,
    selected_tags: Optional[Set[str]] = None,
    start_time: Optional[dt.datetime] = None,
    end_time: Optional[dt.datetime] = None
) -> List[Dict[str, any]]:
    """Load data from a single DCS volume file.
    
    Args:
        file_path: Path to CSV file
        selected_tags: Set of tag names to include (None = all)
        start_time: Optional start timestamp filter
        end_time: Optional end timestamp filter
    
    Returns:
        List of dicts with keys: timestamp, tag_name, value, data_type
    """
    headers, tag_data_types = load_dcs_file_headers(file_path)
    
    # Find Time column index
    try:
        time_idx = headers.index('Time')
    except ValueError:
        time_idx = 0  # Assume first column if not found
    
    # Determine which columns to extract
    if selected_tags:
        extract_indices = [
            (i, tag) for i, tag in enumerate(headers)
            if tag and tag != 'Time' and tag in selected_tags
        ]
    else:
        extract_indices = [
            (i, tag) for i, tag in enumerate(headers)
            if tag and tag != 'Time'
        ]
    
    # Load data
    data_points = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        
        # Skip headers and data types
        next(reader)
        next(reader)
        
        # Read data rows
        for row_num, row in enumerate(reader, start=3):
            if len(row) <= time_idx:
                continue
            
            # Parse timestamp
            timestamp = parse_timestamp(row[time_idx])
            if timestamp is None:
                continue
            
            # Ensure timezone-aware (make UTC if naive)
            if timestamp.tzinfo is None:
                timestamp = timestamp.replace(tzinfo=dt.timezone.utc)
            
            # Apply time filters
            if start_time and timestamp < start_time:
                continue
            if end_time and timestamp > end_time:
                continue
            
            # Extract tag values
            for col_idx, tag_name in extract_indices:
                if col_idx >= len(row):
                    continue
                
                raw_value = row[col_idx]
                data_type = tag_data_types.get(tag_name, 'Numeric')
                value = parse_value(raw_value, data_type)
                
                if value is not None:
                    data_points.append({
                        'timestamp': timestamp,
                        'tag_name': tag_name,
                        'value': value,
                        'data_type': data_type,
                        'raw_value': raw_value
                    })
    
    return data_points


def load_all_dcs_volumes(
    selected_tags: Optional[Set[str]] = None,
    start_time: Optional[dt.datetime] = None,
    end_time: Optional[dt.datetime] = None,
    max_files: Optional[int] = None
) -> List[Dict[str, any]]:
    """Load data from all DCS volume files.
    
    Args:
        selected_tags: Set of tag names to include (None = all)
        start_time: Optional start timestamp filter
        end_time: Optional end timestamp filter
        max_files: Optional limit on number of files to load
    
    Returns:
        List of data point dicts sorted by timestamp
    """
    files = get_dcs_volume_files()
    
    if max_files:
        files = files[:max_files]
    
    print(f"Loading {len(files)} DCS volume files...")
    
    all_data = []
    for i, file_path in enumerate(files, 1):
        print(f"  [{i}/{len(files)}] Loading {file_path.name}...")
        file_data = load_dcs_volume_data(file_path, selected_tags, start_time, end_time)
        all_data.extend(file_data)
        print(f"    -> {len(file_data):,} data points")
    
    # Sort by timestamp
    all_data.sort(key=lambda x: x['timestamp'])
    
    print(f"\nTotal loaded: {len(all_data):,} data points")
    if all_data:
        print(f"Time range: {all_data[0]['timestamp']} to {all_data[-1]['timestamp']}")
    
    return all_data


def summarize_available_tags(max_files: int = 3) -> Dict[str, any]:
    """Summarize available tags across DCS volume files.
    
    Args:
        max_files: Number of files to scan
    
    Returns:
        Dict with tag inventory by location and type
    """
    files = get_dcs_volume_files()[:max_files]
    
    if not files:
        return {}
    
    # Load headers from first file (assume all files have same structure)
    headers, tag_data_types = load_dcs_file_headers(files[0])
    
    # Identify problem asset tags
    location_tags = identify_problem_asset_tags(headers)
    
    # Categorize by type
    summary = {}
    for location, tags in location_tags.items():
        analog_tags = get_analog_tags(tags)
        config = PROBLEM_ASSET_LOCATIONS.get(location, {})
        summary[location] = {
            'description': config.get('description', 'Unknown'),
            'total_tags': len(tags),
            'current_tags': analog_tags['current'],
            'temperature_tags': analog_tags['temperature'],
            'pressure_tags': analog_tags['pressure'],
            'flow_tags': analog_tags['flow'],
            'status_tags': analog_tags['status'],
            'other_tags': analog_tags['other']
        }
    
    return summary


if __name__ == '__main__':
    # Demo: Summarize available tags
    print("=" * 70)
    print("DCS VOLUME DATA SUMMARY")
    print("=" * 70)
    
    summary = summarize_available_tags()
    
    for location, info in sorted(summary.items()):
        print(f"\n{location} - {info['description']}")
        print(f"  Total tags: {info['total_tags']}")
        print(f"  Current tags (IT): {len(info['current_tags'])}")
        for tag in info['current_tags'][:3]:
            print(f"    - {tag}")
        print(f"  Temperature tags (TT): {len(info['temperature_tags'])}")
        for tag in info['temperature_tags'][:3]:
            print(f"    - {tag}")
        print(f"  Pressure tags (PT): {len(info['pressure_tags'])}")
        print(f"  Flow tags (FT/FIC): {len(info['flow_tags'])}")
        print(f"  Status tags (STAIND): {len(info['status_tags'])}")

