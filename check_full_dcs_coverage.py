"""Quick check of AllDCSData.csv date coverage."""
import csv
from datetime import datetime
from pathlib import Path

print("="*70)
print("ANALYZING: AllDCSData.csv")
print("="*70)

file_path = Path('greenfield_data_context/volumes/dcsrawdataextracts/AllDCSData.csv')

print(f"\nFile size: {file_path.stat().st_size / (1024**2):.2f} MB")

print("\nReading first and last rows to determine date range...")
with open(file_path, 'r') as f:
    reader = csv.reader(f)
    
    # Header and data types
    header = next(reader)
    data_types = next(reader)
    
    print(f"Number of columns (tags): {len(header)}")
    
    # First data row
    first_row = next(reader)
    first_date = first_row[0]
    
    # Count rows and get last
    row_count = 1
    last_row = first_row
    
    # Sample every 10,000 rows for speed
    for i, row in enumerate(reader):
        last_row = row
        row_count += 1
        if row_count % 10000 == 0:
            print(f"  Scanned {row_count:,} rows...", end='\r')
    
    last_date = last_row[0]
    
    print(f"\n  Total rows: {row_count:,}")

print(f"\nDate range:")
print(f"  Start: {first_date}")
print(f"  End:   {last_date}")

# Parse dates to get duration
try:
    start_dt = datetime.strptime(first_date.split()[0], '%Y-%m-%d')
    end_dt = datetime.strptime(last_date.split()[0], '%Y-%m-%d')
    duration_days = (end_dt - start_dt).days
    print(f"  Duration: {duration_days} days (~{duration_days/365:.1f} years)")
except Exception as e:
    print(f"  Could not parse dates: {e}")

# Now check which Priority 1 failures fall in this range
print("\n" + "="*70)
print("CHECKING PRIORITY 1 FAILURE COVERAGE")
print("="*70)

with open('data/processed/problem_asset_failures.csv') as f:
    reader = csv.DictReader(f)
    failures = [row for row in reader if row['priority'] == '1']

print(f"\nTotal Priority 1 failures: {len(failures)}")

# Parse DCS date range
dcs_start = datetime.strptime(first_date.split()[0], '%Y-%m-%d')
dcs_end = datetime.strptime(last_date.split()[0], '%Y-%m-%d')

# Count failures in range
covered_failures = []
for f in failures:
    try:
        fail_date_str = f['service_request_timestamp'].split('T')[0]
        fail_date = datetime.strptime(fail_date_str, '%Y-%m-%d')
        
        # Check if failure is within DCS range (with some buffer)
        if dcs_start <= fail_date <= dcs_end:
            covered_failures.append(f)
    except Exception:
        continue

print(f"Failures within DCS date range: {len(covered_failures)} ({len(covered_failures)/len(failures)*100:.1f}%)")

# Count by asset
from collections import Counter
asset_counts = Counter(f['asset_name'] for f in covered_failures)

print(f"\nCovered failures by asset:")
for asset, count in asset_counts.most_common():
    print(f"  {asset}: {count}")

# Save results
with open('full_dcs_coverage_results.txt', 'w') as f:
    f.write(f"AllDCSData.csv Coverage Report\n")
    f.write(f"="*70 + "\n\n")
    f.write(f"File size: {file_path.stat().st_size / (1024**2):.2f} MB\n")
    f.write(f"Total rows: {row_count:,}\n")
    f.write(f"Columns: {len(header)}\n")
    f.write(f"Date range: {first_date} to {last_date}\n")
    f.write(f"Duration: {duration_days} days\n\n")
    f.write(f"Priority 1 failures: {len(failures)}\n")
    f.write(f"Covered failures: {len(covered_failures)} ({len(covered_failures)/len(failures)*100:.1f}%)\n\n")
    f.write(f"By asset:\n")
    for asset, count in asset_counts.most_common():
        f.write(f"  {asset}: {count}\n")

print(f"\n✓ Results saved to: full_dcs_coverage_results.txt")
print("\n" + "="*70)


