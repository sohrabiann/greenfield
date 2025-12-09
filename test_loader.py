#!/usr/bin/env python
"""Quick test of DCS loader."""
import sys
sys.path.insert(0, 'src')

print("Starting test...", flush=True)

try:
    from dcs_volume_loader import get_dcs_volume_files, load_all_dcs_volumes
    print("Imported modules successfully", flush=True)
    
    files = get_dcs_volume_files()
    print(f"\nFound {len(files)} DCS sample files:", flush=True)
    for f in files[:10]:
        print(f"  - {f.name}", flush=True)
    
    print(f"\nLoading ALL {len(files)} files (this may take 2-3 minutes)...", flush=True)
    dcs_data = load_all_dcs_volumes(max_files=None)
    
    print(f"\nLoaded {len(dcs_data)} timestamped data points", flush=True)
    
    if dcs_data:
        timestamps = sorted(dcs_data.keys())
        print(f"Date range: {timestamps[0]} to {timestamps[-1]}", flush=True)
        
        # Count unique tags
        all_tags = set()
        for ts_data in dcs_data.values():
            all_tags.update(ts_data.keys())
        print(f"Total unique tags across all files: {len(all_tags)}", flush=True)
        
        # Write summary to file
        with open('loader_test_results.txt', 'w') as f:
            f.write(f"Files found: {len(files)}\n")
            f.write(f"Data points loaded: {len(dcs_data)}\n")
            f.write(f"Date range: {timestamps[0]} to {timestamps[-1]}\n")
            f.write(f"Unique tags: {len(all_tags)}\n")
        print("\n✓ Results written to loader_test_results.txt", flush=True)
    
    print("\nTest complete!", flush=True)
    
except Exception as e:
    print(f"\nERROR: {e}", flush=True)
    import traceback
    traceback.print_exc()
    with open('loader_test_error.txt', 'w') as f:
        f.write(f"ERROR: {e}\n")
        f.write(traceback.format_exc())



