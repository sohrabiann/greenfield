# Task 3 – Temporal Structure and Window Exploration

## What was done
- Selected the densest numeric DCS series (blower run-state mapped to 0/1) and computed autocorrelation coefficients for 10-minute lags up to 2 hours.
- Evaluated whether shorter or longer history windows capture meaningful variation in the state signal.

## Key findings
- Autocorrelation stays very high (≥0.81) even out to 120 minutes, reflecting the fact that the blower remains in a stable state for long stretches.
- Because of the strong persistence, short windows (≤1 hour) convey nearly the same information as longer windows; most lag values differ by less than 0.1.
- No clear daily or weekly cycles are present in the sample, reinforcing that run/stop events are rare and not tied to a regular schedule in this excerpt.

## Implications for the project
- For state-based tags like this blower indicator, **compact windows (30–60 minutes)** should be sufficient for feature extraction, reducing computation without losing signal.
- Richer tags (currents, temperatures) should be re-evaluated once available; those may exhibit cycle-driven structure requiring longer context (e.g., 24–72 hours).
