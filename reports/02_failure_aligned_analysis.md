# Task 2 – Failure-Aligned Analysis

## What was done
- Parsed Dryer A shutdown events and aligned the blower-state DCS series around each event using +/-7 days of history and 1 day post-event windows.
- Focused on the last 72 hours before each shutdown to contrast pre-failure behaviour against the earlier healthy baseline.

## Key findings
- Two shutdown events overlapped with recorded DCS data; each window contained hundreds of points (924 and 1,153 respectively).
- The blower state remained **100% running** during the 72-hour pre-shutdown windows and showed no deviation from the longer historical average, producing a delta of **0.0** for the second event.
- No additional tags were available to triangulate pre-failure patterns, so the analysis currently indicates **no observable precursor** in the available tag.

## Implications for the project
- The shutdowns captured in the sample data do not show an operational slowdown or cycling pattern on the available tag. Either the failure modes are unrelated to the blower run state or additional tags (currents, temperatures, pressures) are required to detect degradation.
- When broader tag coverage is accessible, repeat the alignment to look for signatures in load/temperature/pressure signals that precede the shutdowns.
