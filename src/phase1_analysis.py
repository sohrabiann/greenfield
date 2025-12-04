import csv
import datetime as dt
import json
import re
from collections import Counter, defaultdict
from typing import Dict, List, Tuple

BASE_PATH = 'greenfield_data_context/samples/'

DCS_FILE = BASE_PATH + 'dcsdata_silver_sample.csv.csv'
SHUTDOWN_FILE = BASE_PATH + 'dryerashutdownsgold_sample.csv.csv'
ASSET_FILE = BASE_PATH + 'assetlist_silver_sample.csv.csv'
INSTRUMENTATION_FILE = BASE_PATH + 'instrumentation_descriptions_silver_sample.csv.csv'
INSTR_STATS_FILE = BASE_PATH + 'instrumentation_statistics_gold_sample.csv.csv'
WORKORDER_FILE = BASE_PATH + 'workordersdetaledviewgold_sample.csv.csv'

STATE_MAP = {
    'Open/Running': 1.0,
    'Closed/Stopped': 0.0,
}

ALLOWED_PLANT_AREAS = {
    'DRYER A',
    'DRYER D',
    'DDG',
    'TGF',
}


def parse_float(val: str):
    try:
        if val is None:
            return None
        val = val.strip()
        if val == '':
            return None
        return float(val)
    except Exception:
        return None


def parse_timestamp(val: str):
    try:
        return dt.datetime.fromisoformat(val.replace('Z', '+00:00'))
    except Exception:
        return None


def load_csv(path: str) -> List[Dict[str, str]]:
    with open(path, newline='') as f:
        return list(csv.DictReader(f))


def clean_plant_area(value: str) -> str:
    raw = (value or '').strip()
    if not raw:
        return 'Unknown'
    # Remove obvious CSV bleed-through by trimming to the first comma-separated token
    if ',' in raw:
        raw = raw.split(',', 1)[0]
    cleaned = raw.strip('"').strip().upper()
    if cleaned in ALLOWED_PLANT_AREAS:
        return cleaned
    # Handle patterns like " SIZE 454" or long malformed strings
    if len(cleaned) > 20 or re.search(r'\\n|\\"', cleaned):
        return 'ParsingError'
    return 'Unknown'


def summarize_assets():
    rows = load_csv(ASSET_FILE)
    for r in rows:
        r['Plant_Area_Clean'] = clean_plant_area(r.get('Plant_Area'))
    problem_assets = [
        r for r in rows if r.get('Plant_Area_Clean') != 'ParsingError' and (
            r.get('IsProblematic', '').lower() == 'yes' or r.get('IsProblematic') == '1'
        )
    ]
    area_counts_raw = Counter(r.get('Plant_Area', '') for r in rows)
    area_counts_clean = Counter(r.get('Plant_Area_Clean', '') for r in rows)
    return {
        'problem_asset_count': len(problem_assets),
        'problem_assets': problem_assets,
        'area_counts_raw': area_counts_raw,
        'area_counts_clean': area_counts_clean,
    }


def summarize_dcs():
    rows = load_csv(DCS_FILE)
    tag_stats: Dict[str, Dict[str, object]] = {}
    for row in rows:
        tag = row.get('Tag_Full')
        ts = parse_timestamp(row.get('Timestamp', ''))
        raw_numeric = parse_float(row.get('Value_Numeric'))
        mapped_state = STATE_MAP.get((row.get('Value_String') or '').strip())
        numeric_value = raw_numeric if raw_numeric is not None else mapped_state
        is_missing = (row.get('Reason_No_Data') or '').strip() != '' or (
            numeric_value is None and (row.get('Value_String') or '').strip() == ''
        )
        tag_entry = tag_stats.setdefault(tag, {
            'total': 0,
            'missing': 0,
            'reason_counts': Counter(),
            'datatype_counts': Counter(),
            'numeric_values': [],
            'timestamps': [],
            'state_counts': Counter(),
        })
        tag_entry['total'] += 1
        if is_missing:
            tag_entry['missing'] += 1
            reason = (row.get('Reason_No_Data') or 'Unspecified').strip() or 'Unspecified'
            tag_entry['reason_counts'][reason] += 1
        dtype = (row.get('DataType') or 'Unknown').strip()
        tag_entry['datatype_counts'][dtype] += 1
        tag_entry['state_counts'][(row.get('Value_String') or '').strip()] += 1
        if numeric_value is not None:
            tag_entry['numeric_values'].append(numeric_value)
            if ts:
                tag_entry['timestamps'].append((ts, numeric_value))
    # compute aggregate summaries
    summary = []
    for tag, info in tag_stats.items():
        numeric_values = info['numeric_values']
        basic_stats = None
        if numeric_values:
            basic_stats = {
                'min': min(numeric_values),
                'max': max(numeric_values),
                'mean': sum(numeric_values) / len(numeric_values),
            }
        summary.append({
            'tag': tag,
            'total': info['total'],
            'missing_pct': info['missing'] / info['total'] * 100 if info['total'] else 0.0,
            'reason_breakdown': info['reason_counts'],
            'datatype_breakdown': info['datatype_counts'],
            'state_breakdown': info['state_counts'],
            'numeric_stats': basic_stats,
            'numeric_series': info['timestamps'],
        })
    summary.sort(key=lambda x: x['missing_pct'], reverse=True)
    return summary


def load_shutdowns():
    rows = load_csv(SHUTDOWN_FILE)
    events = []
    for row in rows:
        start = parse_timestamp(row.get('Shutdown_Start', ''))
        rec = parse_timestamp(row.get('Recovery_Timestamp', ''))
        if start:
            events.append({
                'start': start,
                'recovery': rec,
                'length_hours': parse_float(row.get('Shutdown_Length_Hours')),
            })
    events.sort(key=lambda x: x['start'])
    return events


def failure_aligned(tag_summary, shutdowns):
    window = dt.timedelta(days=7)
    post_window = dt.timedelta(days=1)
    pre_focus = dt.timedelta(days=3)
    results = []
    for entry in tag_summary:
        series: List[Tuple[dt.datetime, float]] = entry['numeric_series']
        if not series:
            continue
        series.sort(key=lambda x: x[0])
        for event in shutdowns:
            start = event['start']
            window_values = [v for t, v in series if start - window <= t <= start + post_window]
            failure_values = [v for t, v in series if start - pre_focus <= t < start]
            healthy_values = [v for t, v in series if t < start - window]
            if not window_values:
                continue
            def mean_or_none(vals):
                return sum(vals) / len(vals) if vals else None
            results.append({
                'tag': entry['tag'],
                'event_time': start.isoformat(),
                'window_count': len(window_values),
                'failure_mean': mean_or_none(failure_values),
                'healthy_mean': mean_or_none(healthy_values),
                'delta': None if not failure_values or not healthy_values else mean_or_none(failure_values) - mean_or_none(healthy_values),
            })
    return results


def autocorrelation(values: List[float], lag: int) -> float:
    if lag >= len(values):
        return 0.0
    mean = sum(values) / len(values)
    num = sum((values[i] - mean) * (values[i - lag] - mean) for i in range(lag, len(values)))
    den = sum((v - mean) ** 2 for v in values)
    return num / den if den else 0.0


def temporal_structure(tag_summary):
    candidate = None
    for entry in tag_summary:
        if entry['numeric_series']:
            if candidate is None or len(entry['numeric_series']) > len(candidate['numeric_series']):
                candidate = entry
    if candidate is None:
        return None
    series = [v for _, v in sorted(candidate['numeric_series'], key=lambda x: x[0])]
    lags = list(range(1, min(13, len(series))))
    acf = {lag: autocorrelation(series, lag) for lag in lags}
    return {'tag': candidate['tag'], 'lags': acf}


def reliability_metrics(shutdowns, workorders):
    mtbf_hours = None
    if len(shutdowns) >= 2:
        diffs = []
        for prev, curr in zip(shutdowns, shutdowns[1:]):
            diffs.append((curr['start'] - prev['start']).total_seconds() / 3600.0)
        mtbf_hours = sum(diffs) / len(diffs)
    mttr_hours = None
    lengths = [s['length_hours'] for s in shutdowns if s.get('length_hours') is not None]
    if lengths:
        mttr_hours = sum(lengths) / len(lengths)

    priority_counts = Counter()
    asset_counts = Counter()
    for row in workorders:
        pr = row.get('Priority')
        priority_counts[pr] += 1
        asset = row.get('Asset') or row.get('Assets') or 'Unknown'
        asset_counts[asset] += 1
    return {
        'mtbf_hours': mtbf_hours,
        'mttr_hours': mttr_hours,
        'priority_counts': priority_counts,
        'asset_counts': asset_counts,
    }


def build_tag_asset_links(problem_assets: List[Dict[str, str]]):
    instrument_rows = load_csv(INSTRUMENTATION_FILE)

    def normalize(text: str) -> str:
        return re.sub(r'[^A-Z0-9]', '', (text or '').upper())

    instr_by_loc = defaultdict(list)
    for row in instrument_rows:
        instr_by_loc[normalize(row.get('INstrumentation_Location', ''))].append(row)

    links = []
    seen = set()
    for asset in problem_assets:
        loc_key = normalize(asset.get('Location', ''))
        for row in instr_by_loc.get(loc_key, []):
            key = (asset.get('Asset'), row.get('Tag_Full'))
            if key in seen:
                continue
            seen.add(key)
            links.append({
                'asset_id': asset.get('Asset'),
                'asset_name': asset.get('Description'),
                'location': asset.get('Location'),
                'plant_area': asset.get('Plant_Area_Clean'),
                'tag_full': row.get('Tag_Full'),
                'tag_role': row.get('Instrumentation_Data_Type_Meaning'),
                'comment': row.get('Instrumentation_Type_'),
            })
    return links


def baseline_model(tag_summary, shutdowns):
    label_window = dt.timedelta(hours=12)
    datasets = []
    for entry in tag_summary:
        for ts, value in entry['numeric_series']:
            label = 0
            for event in shutdowns:
                if event['start'] - label_window <= ts <= event['start']:
                    label = 1
                    break
            datasets.append({'tag': entry['tag'], 'timestamp': ts, 'value': value, 'label': label})
    if not datasets:
        return None
    failures = [d['value'] for d in datasets if d['label'] == 1]
    healthy = [d['value'] for d in datasets if d['label'] == 0]
    if not failures or not healthy:
        return None
    thr = (sum(failures) / len(failures) + sum(healthy) / len(healthy)) / 2
    tp = fp = tn = fn = 0
    for d in datasets:
        pred = 1 if d['value'] >= thr else 0
        if pred == 1 and d['label'] == 1:
            tp += 1
        elif pred == 1:
            fp += 1
        elif d['label'] == 1:
            fn += 1
        else:
            tn += 1
    precision = tp / (tp + fp) if (tp + fp) else None
    recall = tp / (tp + fn) if (tp + fn) else None
    return {
        'threshold': thr,
        'precision': precision,
        'recall': recall,
        'counts': {'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn},
    }


def main():
    assets = summarize_assets()
    dcs_summary = summarize_dcs()
    shutdowns = load_shutdowns()
    workorders = load_csv(WORKORDER_FILE)
    tag_links = build_tag_asset_links(assets['problem_assets'])
    aligned = failure_aligned(dcs_summary, shutdowns)
    temporal = temporal_structure(dcs_summary)
    reliability = reliability_metrics(shutdowns, workorders)
    baseline = baseline_model(dcs_summary, shutdowns)

    output = {
        'assets': {
            'problem_asset_count': assets['problem_asset_count'],
            'area_counts_raw': assets['area_counts_raw'].most_common(),
            'area_counts_clean': assets['area_counts_clean'].most_common(),
            'sample_problem_assets': assets['problem_assets'][:10],
        },
        'tag_summary': [
            {
                'tag': e['tag'],
                'total': e['total'],
                'missing_pct': e['missing_pct'],
                'top_reason': e['reason_breakdown'].most_common(3),
                'top_datatype': e['datatype_breakdown'].most_common(3),
                'state_breakdown': e['state_breakdown'].most_common(3),
                'numeric_stats': e['numeric_stats'],
            } for e in dcs_summary
        ][:50],
        'failure_aligned': aligned,
        'temporal_structure': temporal,
        'reliability': {
            'mtbf_hours': reliability['mtbf_hours'],
            'mttr_hours': reliability['mttr_hours'],
            'priority_counts': reliability['priority_counts'].most_common(5),
            'asset_counts': reliability['asset_counts'].most_common(5),
        },
        'baseline_model': baseline,
        'tag_asset_links': tag_links,
    }

    with open('data/processed/phase1_summary.json', 'w') as f:
        json.dump(output, f, indent=2, default=str)

    if tag_links:
        fieldnames = ['asset_id', 'asset_name', 'location', 'plant_area', 'tag_full', 'tag_role', 'comment']
        with open('docs/dryer_problem_asset_tags.csv', 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(tag_links)


if __name__ == '__main__':
    main()
