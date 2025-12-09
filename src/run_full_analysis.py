"""Run full analysis with all 26 DCS sample files - with verbose logging."""
import json
import sys
from datetime import datetime
from pathlib import Path

# Add logging
LOG_FILE = Path('analysis_run.log')

def log(msg):
    """Write log message to both console and file."""
    timestamp = datetime.now().strftime('%H:%M:%S')
    full_msg = f"[{timestamp}] {msg}"
    print(full_msg, flush=True)
    with open(LOG_FILE, 'a') as f:
        f.write(full_msg + '\n')

# Clear log file
LOG_FILE.write_text('')

log("=" * 70)
log("FULL ANALYSIS RUN - ALL 26 DCS SAMPLES")
log("=" * 70)

# Import modules
log("\n1. Loading modules...")
try:
    from dcs_volume_loader import load_all_dcs_volumes, get_dcs_volume_files
    from phase1_volume_analysis import analyze_failure_aligned_patterns, analyze_temporal_structure
    import pandas as pd
    log("   ✓ Modules loaded successfully")
except Exception as e:
    log(f"   ✗ Error loading modules: {e}")
    sys.exit(1)

# Check available files
log("\n2. Checking DCS sample files...")
try:
    files = get_dcs_volume_files()
    log(f"   Found {len(files)} DCS sample files")
    for i, f in enumerate(files[:5], 1):
        log(f"   - {f.name}")
    if len(files) > 5:
        log(f"   ... and {len(files) - 5} more")
except Exception as e:
    log(f"   ✗ Error checking files: {e}")
    sys.exit(1)

# Load failures
log("\n3. Loading Priority 1 failures...")
try:
    failures = pd.read_csv('data/processed/problem_asset_failures.csv')
    p1_failures = failures[failures['priority'] == 1]
    log(f"   Found {len(p1_failures)} Priority 1 failures")
    log(f"   Date range: {p1_failures['service_request_timestamp'].min()} to {p1_failures['service_request_timestamp'].max()}")
    
    # Count by asset
    by_asset = p1_failures.groupby('asset_name').size().to_dict()
    for asset, count in sorted(by_asset.items(), key=lambda x: -x[1])[:5]:
        log(f"   - {asset}: {count} failures")
except Exception as e:
    log(f"   ✗ Error loading failures: {e}")
    sys.exit(1)

# Load DCS data
log("\n4. Loading ALL DCS sample files (this may take a few minutes)...")
try:
    dcs_data = load_all_dcs_volumes(max_files=None)
    log(f"   ✓ Loaded {len(dcs_data)} data points")
    
    if dcs_data:
        timestamps = sorted(dcs_data.keys())
        log(f"   DCS data range: {timestamps[0]} to {timestamps[-1]}")
        
        # Count tags
        all_tags = set()
        for ts_data in dcs_data.values():
            all_tags.update(ts_data.keys())
        log(f"   Total unique tags: {len(all_tags)}")
    else:
        log("   ✗ No DCS data loaded!")
        sys.exit(1)
except Exception as e:
    log(f"   ✗ Error loading DCS data: {e}")
    import traceback
    log(traceback.format_exc())
    sys.exit(1)

# Analyze each asset
log("\n5. Running failure-aligned analysis...")

assets_to_analyze = [
    ('BL-1829', ['CH-DRA_PROC_AIR:IT18179.MEAS', 'CH-DRA_PROC_AIR:TT18179A.MEAS']),
    ('RV-1834', ['CH-DR_DUSTCOLL:IT18211.MEAS', 'CH-DR_DUSTCOLL:TT18217.MEAS']),
    ('DC-1834', ['CH-DR_DUSTCOLL:IT18211.MEAS', 'CH-DR_DUSTCOLL:TT18217.MEAS']),
]

results = {}
for location, tags in assets_to_analyze:
    log(f"\n   Analyzing {location}...")
    try:
        # Get failures for this asset
        asset_failures = p1_failures[p1_failures['asset_name'] == location]
        log(f"   - {len(asset_failures)} Priority 1 failures for {location}")
        
        # Run analysis
        failure_results = analyze_failure_aligned_patterns(location, tags, max_files=None)
        log(f"   - Failure-aligned analysis complete")
        
        temporal_results = analyze_temporal_structure(location, tags, max_files=None)
        log(f"   - Temporal structure analysis complete")
        
        results[location] = {
            'failure_aligned': failure_results,
            'temporal_structure': temporal_results
        }
        
        # Report on failure windows found
        for tag, tag_results in failure_results.items():
            n_windows = tag_results.get('n_windows', 0)
            log(f"   - {tag}: {n_windows} failure windows with DCS data")
            
    except Exception as e:
        log(f"   ✗ Error analyzing {location}: {e}")
        import traceback
        log(traceback.format_exc())

# Save results
log("\n6. Saving results...")
try:
    output_file = Path('data/processed/volume_analysis_results.json')
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    log(f"   ✓ Results saved to {output_file}")
except Exception as e:
    log(f"   ✗ Error saving results: {e}")

log("\n" + "=" * 70)
log("ANALYSIS COMPLETE!")
log("=" * 70)
log(f"\nLog saved to: {LOG_FILE.absolute()}")



