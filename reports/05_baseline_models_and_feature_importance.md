# Task 5 – Baseline Models and Feature Importance

## What was done
- Built a lightweight baseline using the blower run-state as a single feature and labeling any point within 12 hours before a shutdown as a failure window.
- Derived a global decision threshold at the midpoint between the mean failure value and the mean healthy value (equivalent to checking for a `Closed/Stopped` state).
- Evaluated precision and recall on the full sample without external ML libraries.

## Key findings
- The derived threshold (≈0.97) effectively flags nearly all pre-shutdown windows, yielding a **recall of 0.99** but low **precision of 0.08** because the blower is usually running.
- High false positives stem from the inherent imbalance (most observations are healthy yet in the running state), highlighting the need for richer features.

## Implications for the project
- Even a simple state-based rule can catch nearly every shutdown but generates too many alerts to be practical. Additional tags (currents, temperatures, pressures) and class balancing will be essential for actionable precision.
- Future baselines should incorporate short history windows (30–60 minutes per Task 3) and multiple signals to allow feature importance ranking once more data is available.
