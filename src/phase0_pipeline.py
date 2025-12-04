"""Phase 0 processing pipeline for Dryer A problem assets.

This module avoids third-party dependencies to remain runnable in
restricted environments. It provides helpers to parse Excel files
using only the standard library, consolidate service request data,
produce failure labels, and write human-readable asset summaries.
"""
from __future__ import annotations

import csv
import re
import zipfile
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence
import xml.etree.ElementTree as ET


EXCEL_NS = {"a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}


@dataclass
class ServiceRequest:
    service_request_id: str
    service_request_timestamp: datetime
    priority: str
    service_request_text: str
    asset_id: str
    asset_name: str
    failure_description: str


@dataclass
class LabelledTimestamp:
    timestamp: datetime
    label: str


# -----------------
# Excel utilities
# -----------------

def _column_index(cell_ref: str) -> int:
    """Convert an Excel cell reference (e.g., "A1") to a zero-based column index."""
    match = re.match(r"([A-Z]+)", cell_ref)
    if not match:
        raise ValueError(f"Invalid cell reference: {cell_ref}")
    letters = match.group(1)
    idx = 0
    for ch in letters:
        idx = idx * 26 + (ord(ch) - 64)
    return idx - 1


def _load_shared_strings(zf: zipfile.ZipFile) -> List[str]:
    try:
        with zf.open("xl/sharedStrings.xml") as f:
            tree = ET.parse(f)
    except KeyError:
        return []

    shared: List[str] = []
    for si in tree.findall(".//a:si", EXCEL_NS):
        parts = [t.text or "" for t in si.findall(".//a:t", EXCEL_NS)]
        shared.append("".join(parts))
    return shared


def parse_xlsx_to_records(path: Path) -> List[Dict[str, str]]:
    """Parse the first worksheet of an XLSX file into a list of dictionaries.

    This simplified parser supports the cell types used in the provided
    spreadsheets (shared strings, inline strings, and numbers).
    """

    with zipfile.ZipFile(path) as zf:
        shared_strings = _load_shared_strings(zf)
        with zf.open("xl/worksheets/sheet1.xml") as f:
            sheet_tree = ET.parse(f)

    rows: List[List[str]] = []
    for row in sheet_tree.findall(".//a:sheetData/a:row", EXCEL_NS):
        values: Dict[int, str] = {}
        for cell in row.findall("a:c", EXCEL_NS):
            ref = cell.get("r")
            if ref is None:
                continue
            idx = _column_index(ref)
            cell_type = cell.get("t")
            text_value: Optional[str]

            v_elem = cell.find("a:v", EXCEL_NS)
            if v_elem is not None:
                text_value = v_elem.text
            else:
                inline = cell.find("a:is", EXCEL_NS)
                if inline is not None:
                    inline_text = [t.text or "" for t in inline.findall("a:t", EXCEL_NS)]
                    text_value = "".join(inline_text)
                else:
                    text_value = ""

            if text_value is None:
                value = ""
            elif cell_type == "s":
                value = shared_strings[int(text_value)]
            else:
                value = text_value
            values[idx] = value

        if not values:
            continue
        max_idx = max(values.keys())
        row_values = [""] * (max_idx + 1)
        for i, v in values.items():
            row_values[i] = v
        rows.append(row_values)

    if not rows:
        return []

    header = rows[0]
    records: List[Dict[str, str]] = []
    for row in rows[1:]:
        if all((not cell or str(cell).strip() == "") for cell in row):
            continue
        record: Dict[str, str] = {}
        for idx, column in enumerate(header):
            if not column:
                continue
            if idx < len(row):
                record[column] = row[idx]
            else:
                record[column] = ""
        records.append(record)
    return records


# -----------------
# Data preparation
# -----------------

def excel_serial_to_datetime(value: str) -> datetime:
    """Convert an Excel serial date string to a timezone-aware datetime."""
    try:
        serial = float(value)
    except (TypeError, ValueError):
        # Already an ISO string
        return datetime.fromisoformat(value.replace("Z", "+00:00"))

    base = datetime(1899, 12, 30, tzinfo=timezone.utc)
    return base + timedelta(days=serial)


def load_failure_descriptions(path: Path) -> Dict[str, str]:
    records = parse_xlsx_to_records(path)
    return {row.get("Location", ""): row.get("Description of failures", "") for row in records if row.get("Location")}


def load_service_requests(path: Path) -> List[ServiceRequest]:
    raw_records = parse_xlsx_to_records(path)
    requests: List[ServiceRequest] = []
    for row in raw_records:
        try:
            timestamp = excel_serial_to_datetime(row.get("Reported Date", ""))
        except Exception:
            # Skip rows with invalid timestamps
            continue
        requests.append(
            ServiceRequest(
                service_request_id=row.get("Service Request", ""),
                service_request_timestamp=timestamp,
                priority=row.get("Priority", ""),
                service_request_text=row.get("Summary", ""),
                asset_id=row.get("Asset", ""),
                asset_name=row.get("Location", ""),
                failure_description="",
            )
        )
    return requests


def consolidate_failures(
    failure_desc: Dict[str, str], service_requests: Sequence[ServiceRequest]
) -> List[ServiceRequest]:
    problem_locations = set(failure_desc.keys())
    consolidated: List[ServiceRequest] = []
    for request in service_requests:
        if request.asset_name not in problem_locations:
            continue
        consolidated.append(
            ServiceRequest(
                service_request_id=request.service_request_id,
                service_request_timestamp=request.service_request_timestamp,
                priority=request.priority,
                service_request_text=request.service_request_text,
                asset_id=request.asset_id,
                asset_name=request.asset_name,
                failure_description=failure_desc.get(request.asset_name, ""),
            )
        )
    consolidated.sort(key=lambda r: r.service_request_timestamp)
    return consolidated


# -----------------
# Label generation
# -----------------

def _parse_timestamp(ts: str) -> Optional[datetime]:
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except ValueError:
        return None


def _next_failure_within_horizon(timestamp: datetime, failures: Sequence[datetime], horizon_days: int) -> bool:
    horizon = timedelta(days=horizon_days)
    for failure_time in failures:
        if failure_time < timestamp:
            continue
        if failure_time - timestamp <= horizon:
            return True
        break
    return False


def _asset_key(asset_id: str, asset_name: str) -> str:
    return asset_id or asset_name


def _safe_slug(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", value).strip("_")
    return cleaned or "asset"


def generate_labels(
    consolidated: Sequence[ServiceRequest],
    instrumentation_path: Path,
    dcs_path: Path,
    output_dir: Path,
    horizon_days: int = 4,
) -> Dict[str, List[LabelledTimestamp]]:
    """Generate labelled timestamps for each asset based on Priority 1 events."""

    output_dir.mkdir(parents=True, exist_ok=True)

    priority_one_failures: Dict[str, List[datetime]] = {}
    key_to_name: Dict[str, str] = {}
    for request in consolidated:
        if str(request.priority).strip() != "1":
            continue
        key = _asset_key(request.asset_id, request.asset_name)
        if not key:
            continue
        priority_one_failures.setdefault(key, []).append(request.service_request_timestamp)
        key_to_name.setdefault(key, request.asset_name)

    for failure_times in priority_one_failures.values():
        failure_times.sort()

    name_to_keys: Dict[str, List[str]] = {}
    for key, name in key_to_name.items():
        name_to_keys.setdefault(name, []).append(key)

    # Map tags to assets for filtering DCS data
    tag_to_keys: Dict[str, List[str]] = {}
    with instrumentation_path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            tag = row.get("Tag_Full", "")
            asset = row.get("Asset", "")
            location = row.get("Location", "")
            if not tag:
                continue

            candidate_keys: List[str] = []
            if asset and asset in priority_one_failures:
                candidate_keys.append(asset)
            if location and location in name_to_keys:
                candidate_keys.extend(name_to_keys[location])

            if not candidate_keys:
                continue

            tag_to_keys.setdefault(tag, [])
            for key in candidate_keys:
                if key not in tag_to_keys[tag]:
                    tag_to_keys[tag].append(key)

    asset_timestamps: Dict[str, List[datetime]] = {key: [] for key in priority_one_failures}
    with dcs_path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            tag = row.get("Tag_Full", "")
            if tag not in tag_to_keys:
                continue
            timestamp = _parse_timestamp(row.get("Timestamp", ""))
            if timestamp is None:
                continue
            for key in tag_to_keys[tag]:
                asset_timestamps[key].append(timestamp)

    labels: Dict[str, List[LabelledTimestamp]] = {}
    for key, timestamps in asset_timestamps.items():
        timestamps.sort()
        failures = priority_one_failures.get(key, [])
        asset_labels: List[LabelledTimestamp] = []
        for ts in timestamps:
            label = "failure_within_4d" if _next_failure_within_horizon(ts, failures, horizon_days) else "healthy"
            asset_labels.append(LabelledTimestamp(timestamp=ts, label=label))
        labels[key] = asset_labels

        asset_name = key_to_name.get(key, key)
        filename = _safe_slug(key if key.strip() else asset_name)
        output_path = output_dir / f"labels_{filename}.csv"
        with output_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "label"])
            if asset_labels:
                for item in asset_labels:
                    writer.writerow([item.timestamp.isoformat(), item.label])
            else:
                writer.writerow(["", "no_dcs_data_for_problem_asset"])

    return labels


# -----------------
# Asset summary
# -----------------

def build_asset_summary(
    failure_desc: Dict[str, str], consolidated: Sequence[ServiceRequest], instrumentation_path: Path
) -> List[Dict[str, str]]:
    location_to_assets: Dict[str, set] = {}
    for request in consolidated:
        location_to_assets.setdefault(request.asset_name, set()).add(request.asset_id)

    location_to_tags: Dict[str, List[str]] = {loc: [] for loc in failure_desc}
    with instrumentation_path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            location = row.get("Location", "")
            tag = row.get("Tag_Full", "")
            if location in location_to_tags and tag:
                location_to_tags[location].append(tag)

    summary_rows: List[Dict[str, str]] = []
    for location, description in failure_desc.items():
        assets = sorted(location_to_assets.get(location, []))
        tags = location_to_tags.get(location, [])
        unique_tags = []
        for tag in tags:
            if tag not in unique_tags:
                unique_tags.append(tag)
        example_tags = ", ".join(unique_tags[:5])
        summary_rows.append(
            {
                "asset_name": location,
                "asset_ids": ", ".join(assets) if assets else "",
                "failure_mode": description,
                "key_tags": example_tags,
            }
        )
    return summary_rows


# -----------------
# Writing helpers
# -----------------

def write_consolidated_csv(records: Sequence[ServiceRequest], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "asset_id",
                "asset_name",
                "service_request_id",
                "service_request_timestamp",
                "priority",
                "service_request_text",
                "failure_description",
            ]
        )
        for record in records:
            writer.writerow(
                [
                    record.asset_id,
                    record.asset_name,
                    record.service_request_id,
                    record.service_request_timestamp.isoformat(),
                    record.priority,
                    record.service_request_text,
                    record.failure_description,
                ]
            )


def write_asset_summary_markdown(summary_rows: Sequence[Dict[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Dryer Area problem assets", "", "| Asset Name | Asset IDs | Failure Mode | Key DCS Tags |", "| --- | --- | --- | --- |"]
    for row in summary_rows:
        lines.append(
            f"| {row['asset_name']} | {row['asset_ids']} | {row['failure_mode']} | {row['key_tags']} |"
        )
    path.write_text("\n".join(lines))


def write_label_definition(path: Path) -> None:
    text = """# Label Definition

Failure labels are anchored on Priority 1 service requests for Dryer Area problem assets.

- **Failure anchor:** the reported timestamp of each Priority 1 service request.
- **Horizon:** 4 days after each failure anchor.
- **Failure window:** any DCS timestamp that occurs within 4 days **before** a Priority 1 request is marked `failure_within_4d`.
- **Healthy window:** timestamps with no Priority 1 request in the next 4 days are marked `healthy`.

Healthy periods deliberately exclude future-failure timestamps to avoid leakage.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


# -----------------
# Orchestration
# -----------------

def run_phase0(
    samples_dir: Path,
    instrumentation_path: Path,
    dcs_path: Path,
    processed_dir: Path,
    docs_dir: Path,
) -> None:
    failure_desc = load_failure_descriptions(samples_dir / "failure_desc_problem_Equipment_List.xlsx")
    service_requests = load_service_requests(samples_dir / "service_requests_for_problem_eq.xlsx")
    consolidated = consolidate_failures(failure_desc, service_requests)
    write_consolidated_csv(consolidated, processed_dir / "problem_asset_failures.csv")

    generate_labels(
        consolidated,
        instrumentation_path=instrumentation_path,
        dcs_path=dcs_path,
        output_dir=processed_dir,
    )

    summary_rows = build_asset_summary(
        failure_desc, consolidated, instrumentation_path=instrumentation_path
    )
    write_asset_summary_markdown(summary_rows, docs_dir / "dryer_problem_assets_summary.md")
    write_label_definition(docs_dir / "target_definition.md")


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent.parent
    run_phase0(
        samples_dir=base_dir / "greenfield_data_context" / "samples",
        instrumentation_path=base_dir / "greenfield_data_context" / "samples" / "problemassetdetails_sample.csv.csv",
        dcs_path=base_dir / "greenfield_data_context" / "samples" / "dcsdata_silver_sample.csv.csv",
        processed_dir=base_dir / "data" / "processed",
        docs_dir=base_dir / "docs",
    )
