"""Count DCS sample files and failures - no complex imports."""
from pathlib import Path
import csv

print("="*60)
print("SIMPLE ANALYSIS - Counting files and failures")
print("="*60)

# Count DCS sample files
dcs_path = Path('greenfield_data_context/volumes/dcsrawdataextracts')
sample_files = list(dcs_path.glob('*sample*.csv'))
print(f"\n1. DCS Sample Files: {len(sample_files)} files found")

# Sample first few rows from sample1 to get date range
if sample_files:
    sample1 = sorted(sample_files)[0]
    print(f"\n2. Checking {sample1.name}...")
    with open(sample1) as f:
        reader = csv.reader(f)
        header = next(reader)
        types = next(reader)
        first_data = next(reader)
        print(f"   First timestamp: {first_data[0]}")
        print(f"   Number of tags (columns): {len(header)}")

# Count Priority 1 failures
failure_file = Path('data/processed/problem_asset_failures.csv')
print(f"\n3. Checking failures in {failure_file.name}...")
with open(failure_file) as f:
    reader = csv.DictReader(f)
    failures = list(reader)
    p1_failures = [f for f in failures if f.get('priority') == '1']
    
    print(f"   Total failures: {len(failures)}")
    print(f"   Priority 1 failures: {len(p1_failures)}")
    
    # Count by asset
    from collections import Counter
    asset_counts = Counter(f.get('asset_name') for f in p1_failures)
    print(f"\n   Top assets with Priority 1 failures:")
    for asset, count in asset_counts.most_common(5):
        print(f"   - {asset}: {count}")
    
    # Get date range
    dates = [f.get('service_request_timestamp', '') for f in p1_failures if f.get('service_request_timestamp')]
    if dates:
        print(f"\n   Date range: {min(dates)[:10]} to {max(dates)[:10]}")

print("\n" + "="*60)
print("COUNT COMPLETE")
print("="*60)

# Write to file
with open('count_results.txt', 'w') as f:
    f.write(f"DCS sample files: {len(sample_files)}\n")
    f.write(f"Priority 1 failures: {len(p1_failures)}\n")
    f.write(f"Date range: {min(dates)[:10]} to {max(dates)[:10]}\n")

print("\nResults saved to count_results.txt")



