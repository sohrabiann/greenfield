# Task 4 – Reliability and PM Analytics

## What was done
- Calculated mean time between shutdowns (MTBF) and mean time to recovery (MTTR) from the Dryer A shutdown log.
- Tallied work order volumes by priority and by asset grouping using the detailed work order extract.

## Key findings
- The sample shows an **MTBF of ~110 hours** between shutdown starts and an **MTTR of ~84 hours** based on the recorded recovery durations, indicating outages are long relative to their spacing.
- Work orders are dominated by lower priorities: **Priority 4 accounts for 1,723 records**, while only **13 Priority 1** items appear in the sample.
- Asset references are often grouped; the largest bundle (`2541, 2542, ...`) carries 473 work orders, but 714 records lack an explicit asset identifier.

## Implications for the project
- The long MTTR suggests significant recovery time when the dryer goes down; predictive maintenance that prevents even a few shutdowns could meaningfully improve uptime.
- Priority distributions imply that labeling purely from Priority 1 events may be sparse; consider incorporating Priority 2 data or shutdown-derived labels for model training.
- Improving asset linkage in the work order system would enable more precise reliability scoring per problem asset.
