# Phase 1 Findings & Updated Plan – Greenfield Dryer A Predictive Maintenance

This document:

- Summarizes key findings from **Phase 0 & Phase 1** (sample-based analysis).
- Highlights **limitations** of the current sample pipeline.
- Defines **updated tasks** for Codex to:
  - Fix parsing issues.
  - Expand tag coverage.
  - Align baseline modeling with the **4-day horizon** spec.
  - Move from toy demo to realistic per-asset baselines.

It complements `GREENFIELD_CONTEXT_FOR_CODEX.md` and `CODEX_TASK_PLAN.md`.

---

## 1. Summary of Phase 0 & Phase 1 Findings (Samples)

### 1.1 Assets & problem-asset detection

From the sample asset list (`assetlist_silver_sample.csv`):

- Detected **14 problem assets** with `IsProblematic = Yes`, mainly in **DRYER A**.
- Counted **234 assets** in the DRYER A plant area (in the sample).
- Sample of problem assets includes:
  - Main Dryer A fan and motor at **BL-1829**.
  - Dust collector **DC-1834** and associated flameless vents.
  - Product cooler **E-1834**.
  - Ejector **EJ-1837**.
  - Thermal mass flowmeters **FT-18110 / FT-18111**.
  - Syrup/utility pumps **P-1837**, etc.

**Issue discovered:**

- Some rows have **corrupted / concatenated** fields in the “area” column, leading to spurious entries like:

  - `" SIZE 454"`, `" 24\"\" DIA\"\""`, and very long strings with embedded CSV content.

> ➜ Conclusion:  
> Problem-asset detection works, but **CSV parsing is not fully clean**, and `Plant_Area` / related fields are not always trustworthy.

---

### 1.2 DCS tag summary (current state)

From `dcsdata_silver_sample.csv.csv`:

- Only one example DCS tag was analyzed in detail:
  - `CH-DRA_COMB_AIR:BL1825.STAIND`
    - 2,000 points.
    - **0% missing**.
    - Categorical `Running/Stopped` mapped to numeric 1.0/0.0.
    - State distribution:
      - `Open/Running`: 95.3%
      - `Closed/Stopped`: 4.7%

> ➜ Conclusion:  
> Data quality for this tag is excellent, but it is a **binary state indicator** with **strong class imbalance** and limited information on its own.

---

### 1.3 Failure-aligned analysis (shutdown-aligned)

Using `dryerashutdownsgold_sample.csv.csv`:

- Two shutdown events overlapped with the DCS sample.
- For each, DCS windows were extracted (T−7 days to T+1 day); focus on last 72 hours before shutdown.
- For `CH-DRA_COMB_AIR:BL1825.STAIND`:
  - Mean value in the 72h pre-shutdown window: **1.0 (always running)**.
  - Healthy baseline mean (when available): also 1.0.
  - **Delta = 0.0** → no measurable change.

> ➜ Conclusion:  
> There is **no observable pre-failure signature** in this state-only tag. It stays “running” until shutdown.  
> Predictive signals must come from **other tags** (currents, temperatures, pressures, flows), not just the on/off status.

---

### 1.4 Temporal structure

For the same tag:

- Autocorrelation (ACF) was computed for lags up to 12 steps and stayed very high (≥0.81).
- Reflects that the blower state is highly persistent (long stretches of 1, occasional 0).

> ➜ Conclusion:  
> For this kind of state tag, **short windows (30–60 minutes)** already capture most of the information.  
> This does **not yet generalize** to analog tags, which may require longer windows (e.g., 24–72 hours or more) and will need separate analysis.

---

### 1.5 Reliability & PM metrics (sample-based)

From sample shutdowns + a workorder extract:

- **Mean Time Between Failures (MTBF)**: ~110 hours between shutdown starts (in the sample).
- **Mean Time To Repair (MTTR)**: ~84 hours recovery duration.
- Work order priorities:
  - Priority 4: 1,723 records.
  - Priority 2: 255 records.
  - Priority 1: 13 records.
- Asset references are often grouped (e.g., `"2541, 2542, ..."`); many records have `Unknown` asset.

> ➜ Conclusion:  
> Even in a small sample, the **MTTR is high**, so preventing a few shutdowns can materially increase uptime.  
> Priority 1 events are **rare**, reinforcing the need for careful label definition using Priority 1 service requests as **true failures**.  
> **Priority 2 events will not be treated as failures themselves, but as potential precursors or “can-lead-to-failure” signals** that may be used:
> - As input features (e.g., recent P2 activity as a risk indicator).
> - For exploratory analysis of early warning patterns.
> - To understand how issues escalate from P2 to P1 over time.
---

### 1.6 Baseline model (toy prototype)

A simple baseline was implemented:

- Single feature: the numeric blower state.
- Label: any point within **12 hours before a shutdown** was treated as “failure window”.
- Threshold: mid-point between mean “failure” value and mean “healthy” value (effectively checking for “Running” vs “Stopped”).

Results:

- Threshold ≈ 0.97.
- Precision ≈ 0.08.
- Recall ≈ 0.99.
- False positives are very high (due to the blower being almost always running).

> ➜ Conclusion:  
> A trivial state-based rule can catch nearly all shutdowns but is **not practically usable** (too many alerts).  
> This underscores the need for:
> - **Richer feature sets** (multiple tags, aggregated stats, trends).
> - Alignment with the **4-day horizon** defined in the project spec.

---

## 2. Limitations vs. Project Spec

Comparing the Phase 1 sample pipeline to the intended design:

1. **Label horizon mismatch**
   - Spec: model should predict **Priority 1 failure in next 4 days**.
   - Prototype: used a **12-hour window before shutdown**, not tied to service requests.

2. **Tag coverage**
   - Spec: per-asset models using **relevant DCS tags** (currents, temps, pressures, flows) for each problem asset.
   - Prototype: only one binary state tag (`CH-DRA_COMB_AIR:BL1825.STAIND`) was actually analyzed.

3. **Parsing / data integrity**
   - Spec: trustable plant areas and asset groupings.
   - Prototype: some `Plant_Area` values are clearly mis-parsed strings, which can pollute counts and group-level analyses.

> ➜ These issues are expected in a **first pass on sample files**, but must be addressed before moving to serious per-asset baselines and Phase 1 validation.

---

## 3. Updated Tasks for Codex (Post–Phase 1)

This section updates and extends the previous `CODEX_TASK_PLAN.md`.  
Codex should treat these as **next steps** after the existing Phase 0 & Phase 1 tasks.

### 3.1 Task A – Fix CSV parsing & validate asset/area fields

**Goal:** Ensure asset and area fields are clean and trustworthy for DRYER A analyses.

**Steps:**

1. Inspect `assetlist_silver_sample.csv` parsing:
   - Identify rows where `Plant_Area` or `Description` contain:
     - Comma-separated strings spanning multiple logical fields.
     - Embedded quotes and newline fragments.
   - Confirm the expected column schema from a header sample (manually if needed).

2. Update the CSV loading logic to:
   - Use correct delimiter and quote handling.
   - Trim and sanitize `Plant_Area` values.
   - Optionally define an **allowed list** of plant areas:
     - e.g., `["DRYER A", "DRYER D", "DDG", "TGF", ...]`
   - Treat non-matching values as either:
     - `Unknown`, or
     - assign them to a “ParsingError” category for later cleanup.

3. Recompute:
   - `area_counts`.
   - `problem_asset_count`.
   - Sample problem asset list.
   - Confirm DRYER A problem assets list remains unchanged (or document the differences).

**Deliverable:**

- Updated parsing code (e.g., in `src/phase1_analysis.py`).
- A short section in `reports/01_tag_and_asset_level_eda.md` documenting:
  - What changed in parsing.
  - Before vs after `area_counts`.
  - Confirmation that DRYER A assets/problem assets are correctly recognized.

---

### 3.2 Task B – Expand tag coverage for DRYER A problem assets

**Goal:** Move beyond a single binary tag and identify a **set of relevant DCS tags per problem asset**.

**Steps:**

1. Use `instrumentation_descriptions_silver_sample.csv` (or equivalent) and:
   - Map tags to:
     - Plant area (DRYER A).
     - Equipment locations (e.g., `BL-1829`, `E-1834`, `DC-1834`, etc.).
   - Link tags to problem assets based on:
     - Matching locations.
     - Matching asset identifiers where available.

2. For each DRYER A problem asset:
   - Build a list of associated DCS tags, focusing on:
     - Motor currents.
     - Temperatures (bearing, gas, exhaust).
     - Pressures.
     - Flows.
     - Key status/command signals beyond simple “Running/Stopped”.

3. Create a reference table, e.g. `docs/dryer_problem_asset_tags.md` or `.csv`:
   - Columns:
     - `asset_id`, `asset_name`, `location`, `plant_area`
     - `tag_full`
     - `tag_role` (e.g., “bearing temp”, “motor current”, “exhaust temperature”)
     - `comment` (brief description if available)

**Deliverable:**

- Tag–asset linkage table.
- Updated EDA in `reports/01_tag_and_asset_level_eda.md` to include a small set of these additional tags.

---
Codex has now implemented Plant_Area_Clean and area_counts_clean to isolate parsing errors, and created docs/dryer_problem_asset_tags.csv as a first-pass mapping from Dryer A problem assets to their status indication tags (*.STAIND).


### 3.3 Task C – Re-run failure-aligned & temporal analysis on richer tags

**Goal:** Re-do Tasks 2 and 3 (failure-aligned and temporal structure) on **multiple, more meaningful tags**.

**Steps:**

1. For each DRYER A problem asset:
   - Select 3–5 key analog tags (e.g., current, temperature, pressure, flow).
   - Re-run the **failure-aligned analysis** around Priority 1 SRs (once labels are driven by service requests) or shutdowns as a proxy, if necessary.

2. For each selected tag:
   - Plot:
     - Time-aligned trajectories (T−7 days to T+1 day).
     - Median + quantiles across failures.
   - Compare distributions in:
     - Failure windows (last X hours before the failure).
     - Healthy windows far from any failures.

3. Re-run **temporal structure** analysis:
   - Compute ACF/PACF for analog tags.
   - Check for daily/weekly patterns and typical correlation lengths.

**Deliverables:**

- Updated `reports/02_failure_aligned_analysis.md` and `reports/03_temporal_structure_and_windows.md`:
  - Highlight clear pre-failure patterns (if any) across the new tags.
  - Comment on:
    - Whether your **4-day horizon** appears realistic.
    - Whether shorter windows (24–72 hours) capture most of the signal.

---

### 3.4 Task D – Align baseline models with the 4-day horizon spec

**Goal:** Upgrade the toy baseline into a baseline that respects the **4-day horizon** and uses **aggregated features** over realistic windows.

**Steps:**

1. Use labels based on **Priority 1 service requests** (as defined in `GREENFIELD_CONTEXT_FOR_CODEX.md`):
   - For each problem asset, label timepoints with:
     - `1` if a Priority 1 SR occurs within the next **4 days**.
     - `0` otherwise.
   - **Priority 2 service requests are not failures, but Codex may include “recent P2 activity” as a feature (e.g., count of P2s in the last X days) to capture early-warning conditions that can lead up to a Priority 1 failure.**


2. Define feature windows:
   - Start with a shorter dynamic window, such as **last 24 or 72 hours** before each labeled timestamp.
   - For each DCS tag associated with the asset:
     - Compute aggregated features over that window:
       - mean, max, min, std, last value, slope, maybe quantiles.

3. Train baseline models for **1–2 representative problem assets**:
   - Logistic regression with L1.
   - Random forest or XGBoost on tabular features.

4. Evaluate with a **time-based split**:
   - Train on early history.
   - Validate on later history that includes held-out failures.

**Deliverable:**

- A new section in `reports/05_baseline_models_and_feature_importance.md`:
  - Clearly differentiating:
    - The earlier **toy baseline** (single state tag, 12h horizon).
    - The updated **4-day horizon baseline** with richer features.
  - Performance metrics (ROC-AUC, precision/recall).
  - Updated feature importance for the richer tag set.

---

### 3.5 Task E – Maintain clear documentation of “toy” vs “real” baselines

**Goal:** Avoid confusion between quick sample demos and production-aligned designs.

**Steps:**

1. In `reports/05_baseline_models_and_feature_importance.md`, add a structure like:

   - **Section 1 – Prototype baseline (sample, single state tag, 12h horizon)**
   - **Section 2 – Project-aligned baseline (Priority 1 SR labels, 4-day horizon, multi-tag features)**

2. Briefly explain:
   - Why the prototype baseline was built (to validate the pipeline, not as a final model).
   - Why the updated baseline better reflects the real project requirements.

**Deliverable:**

- Clear narrative separation in the report so that:
  - Supervisors/Greenfield can see the evolution.
  - Future work always references the **project-aligned baseline**, not the toy one.

---

## 4. How Codex Should Use This File

When using Codex Plan:

1. Open `CODEX_TASK_PLAN.md` (existing) **and** this file (`PHASE1_FINDINGS_AND_NEXT_STEPS.md`).
2. Treat the tasks here (A–E) as **extensions / corrections** to Phase 1.
3. A reasonable sequence is:
   - Task A (parsing cleanup).
   - Task B (tag expansion).
   - Task C (re-run analysis).
   - Task D (4-day horizon baseline).
   - Task E (documentation cleanup).

Example Plan prompt:

> “Open `GREENFIELD_CONTEXT_FOR_CODEX.md`, `CODEX_TASK_PLAN.md`, and `PHASE1_FINDINGS_AND_NEXT_STEPS.md`.  
> Apply the updated tasks A–E from `PHASE1_FINDINGS_AND_NEXT_STEPS.md`, starting with Task A (fix CSV parsing) and Task B (expand tag coverage for Dryer A problem assets). For each analysis-like task, update the existing reports or create new ones as described.”

This ensures Codex moves from the initial sample/demo analysis toward the **real Phase 1 baseline** aligned with your 4-day, per-problem-asset predictive maintenance spec.
