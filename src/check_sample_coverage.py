"""Quick script to check date coverage of DCS sample files."""
import csv
from pathlib import Path
from datetime import datetime

DCS_PATH = Path('greenfield_data_context/volumes/dcsrawdataextracts')

print('DCS Sample File Coverage:\n')
print(f"{'File':<35} {'Start Date':<20} {'End Date':<20} {'Rows':>10}")
print('=' * 90)

total_rows = 0
all_dates = []

for sample_file in sorted(DCS_PATH.glob('*sample*.csv')):
    try:
        with open(sample_file) as f:
            reader = csv.reader(f)
            header = next(reader)  # Skip header
            types = next(reader)   # Skip data types
            
            first_row = next(reader, None)
            if not first_row:
                print(f'{sample_file.name:<35} EMPTY FILE')
                continue
            
            row_count = 1
            last_row = first_row
            for row in reader:
                last_row = row
                row_count += 1
            
            total_rows += row_count
            all_dates.append((first_row[0], last_row[0]))
            print(f'{sample_file.name:<35} {first_row[0]:<20} {last_row[0]:<20} {row_count:>10,}')
            
    except Exception as e:
        print(f'{sample_file.name:<35} ERROR: {str(e)[:45]}')

print('=' * 90)
print(f'Total rows across all samples: {total_rows:,}')

if all_dates:
    earliest = min(d[0] for d in all_dates)
    latest = max(d[1] for d in all_dates)
    print(f'Overall coverage: {earliest} to {latest}')



