# Codex Task Plan – Greenfield Dryer A Predictive Maintenance

This file defines the order of tasks Codex should follow.

It assumes:
- The repo already contains:
  - Data exports (DCS, shutdowns, work orders, etc.).
  - The Excel files:
    - `failure_desc_problem_Equipment_List.xlsx`
    - `service_requests_for_problem_eq.xlsx`
  - The context file:
    - `GREENFIELD_CONTEXT_FOR_CODEX.md`
- Labels are based on **Priority 1 service requests** (failures) for **Dryer A problem assets**.

For each task below:
- Codex may create notebooks/scripts as needed (e.g., under `notebooks/` or `src/`).
- If the task is an “Analysis Task” (Phase 1), Codex should also create a **Markdown report** under `reports/` with the task ID in the filename, explaining:
  - What was done.
  - Key findings.
  - What it means for the project.

---

## Phase 0 – Label & Asset Groundwork (do these BEFORE deeper analysis)

These tasks establish the core labels and asset summaries.  
They should be done **before** the “Analysis TODO for Codex (Phase 1)” tasks.

### Task 0.1 – Parse failure description and service request files

**Goal:** build a consolidated table of failures and their context for Dryer Area problem assets.

Steps:
- Read:
  - `failure_desc_problem_Equipment_List.xlsx`
  - `service_requests_for_problem_eq.xlsx`
- Produce a consolidated table (CSV or Parquet) with at least:
  - `asset_id` / `asset_name`
  - `service_request_id`
  - `service_request_timestamp`
  - `priority`
  - `service_request_text`
  - `failure_description` (from failure_desc sheet, if joinable)
- Filter this table to **Dryer Area problem assets** only.
- Save it somewhere like `data/processed/problem_asset_failures.parquet` (or similar).

---

### Task 0.2 – Implement failure labels from Priority 1 service requests

**Goal:** create labelled timelines for each problem asset.

Steps:
- Using the consolidated table from Task 0.1:
  - For each problem asset, identify all **Priority 1** service requests (these are failures).
- Implement a labeling function (Python or PySpark) that, given a timestamped DCS series for an asset:
  - Marks timestamps as **failure windows** if a Priority 1 request occurs within the next **4 days**.
  - Marks timestamps as **healthy** if no Priority 1 request occurs within the next **4 days**.
- Save resulting label tables per asset (e.g., `data/processed/labels_<asset_id>.parquet`).

Note:
- Use **service request time** as the failure time anchor.
- Horizon: 4 days into the future.
- Healthy windows = no Priority 1 event in the next 4 days.

---

### Task 0.3 – Summarize Dryer Area problem assets and failure modes

**Goal:** create a human-readable reference for problem assets.

Steps:
- From `failure_desc_problem_Equipment_List.xlsx` and service request text:
  - Build a small table: one row per **Dryer Area problem asset**, with:
    - `asset_id`, `asset_name`
    - A short **canonical failure mode description** (from failure_desc sheet)
    - A note/list of **key DCS tags** relevant to that failure (if specified).
- Save this summary as:
  - `docs/dryer_problem_assets_summary.md` **or**
  - `data/processed/dryer_problem_assets_summary.csv`

This will later feed into `plant_overview.md` and `target_definition.md`.

---

### Task 0.4 – Draft label logic section for `target_definition.md`

**Goal:** document the actual label logic based on implemented code.

Steps:
- After Task 0.2 is implemented and sanity-checked:
  - Write a short, human-readable description of:
    - How failures are detected (Priority 1 service requests on problem assets).
    - How the **4-day horizon** and **healthy windows** are constructed.
- Insert this text into a draft of `target_definition.md` under a “Label Definition” section.

This is a documentation task; it ensures the code and docs match.

---

## Phase 1 – Analysis TODO for Codex (requires Phase 0 to be mostly done)

These are deeper analysis tasks.  
Tasks 1–5 assume Phase 0 is done, especially labels and asset summaries.  
Each analysis task must also output a **Markdown report** in `reports/`.

### Task 1 – Tag- and asset-level EDA (`01_tag_and_asset_level_eda`)

**Goal:** understand data quality and basic behaviour of key DCS tags for Dryer A problem assets.

Steps:
- Identify all DCS tags associated with Dryer A problem assets (use instrumentation metadata + asset summary from Task 0.3).
- For each tag:
  - Compute:
    - % missing / `Reason_No_Data` breakdown.
    - Basic stats (min, max, mean, std, quantiles).
  - Plot:
    - Histograms / densities.
    - Example time-series slices.
- Compute correlations:
  - Between tags on the **same asset**.
  - Between tags across subsystems (fan current vs exhaust temp vs feed rate, etc.).
- (Optional) Cluster normal operation (e.g., k-means on 1–4h aggregations) to see regimes.

**Report:**
- Create `reports/01_tag_and_asset_level_eda.md` summarizing:
  - Data quality.
  - Key tag relationships.
  - Any operating regimes.
  - Recommendations (tags to drop, cleaning rules, etc.).

---

### Task 2 – Failure-aligned analysis around Priority 1 events (`02_failure_aligned_analysis`)

**Goal:** see how tags behave before/after failures.

Steps:
- For each Dryer A problem asset:
  - Align DCS data around each Priority 1 SR time (T).
  - Extract windows from `T - 7 days` to `T + 1 day`.
- For each tag:
  - Plot individual trajectories and overlaid median/quantiles.
  - Compare distributions in:
    - Failure windows (e.g., last 24–72h before T).
    - Healthy windows (from labels in Task 0.2).
- Run simple statistical contrasts (mean/median differences, effect sizes).
- Optionally, fit quick baseline models on aggregated features to rank tag importance.

**Report:**
- Create `reports/02_failure_aligned_analysis.md` with:
  - Plots and descriptions of pre-failure patterns.
  - Tags that show clear early warning.
  - Comment on whether **4-day horizon + 1-week history** looks reasonable.

---

### Task 3 – Temporal structure & window-size exploration (`03_temporal_structure_and_windows`)

**Goal:** understand how much history matters and whether daily/weekly patterns exist.

Steps:
- For key tags (fan current, bearing temps, exhaust temp, feed rate, etc.):
  - Compute and plot ACF/PACF.
  - Look for daily/weekly cycles and memory length.
- Run simple anomaly/change-point detection on normal periods:
  - Check if anomaly scores increase near failures.
- For one asset:
  - Compare simple models using:
    - 24h window.
    - 72h window.
    - 7-day window.
  - Evaluate which window length works best.

**Report:**
- Create `reports/03_temporal_structure_and_windows.md` with:
  - Evidence of time dependencies.
  - Daily/weekly behaviour.
  - Window-length experiment results.
  - Recommended default history window (e.g., last 24h/72h/7d).

---

### Task 4 – Reliability and PM analytics (`04_reliability_and_pm_analysis`)

**Goal:** provide classical reliability/PM insights for problem assets.

Steps:
- From service requests / work orders:
  - Build Pareto charts of:
    - Failures per asset.
    - Downtime per asset (if duration available).
    - Cost per asset (if cost available).
- Compute per problem asset:
  - MTBF (Mean Time Between Failures).
  - MTTR (Mean Time To Repair), if possible.
- If enough failures:
  - Fit simple lifetime distributions (e.g., Weibull).
- If PM data exists:
  - Compare failure rates before vs after PM events.

**Report:**
- Create `reports/04_reliability_and_pm_analysis.md` summarizing:
  - Top offender assets.
  - MTBF/MTTR.
  - Any wear-out vs random behaviour.
  - PM effectiveness insights.

---

### Task 5 – Baseline models & feature importance (`05_baseline_models_and_feature_importance`)

**Goal:** build quick baseline models to see which tags/features matter.

Steps:
- For 1–2 representative problem assets:
  - Use labels from Task 0.2 and aggregated features (e.g., last 24h stats per tag).
- Train:
  - Logistic regression with L1.
  - Random forest / XGBoost.
- Evaluate with time-based splits:
  - ROC-AUC, precision/recall, etc.
- Analyze feature importance / SHAP.

**Report:**
- Create `reports/05_baseline_models_and_feature_importance.md` with:
  - Model performance.
  - Ranked feature importance.
  - Discussion of:
    - Which tags/stats are most predictive.
    - Whether simple tabular models might already be strong.
    - How this informs deeper time-series models.

---

## Phase 2 – Modeling Prototype (after Phase 0 & key analysis tasks)

These tasks connect labels + analysis to actual modeling for Phase 1.

### Task 2.1 – DCS feature window extraction for one asset (prototype)

**Goal:** prototype the feature extraction pipeline for an LSTM/GRU/TCN-style model.

Steps:
- Choose a representative problem asset (e.g., main fan).
- Using labels from Task 0.2 and raw DCS data:
  - Implement a function that extracts feature windows (e.g., last 24h/72h/7d) before each labelled timestamp.
- Save:
  - Feature arrays.
  - Corresponding labels.
- Summarize:
  - Number of failure vs healthy windows.
  - Basic stats on the features.

(Reporting can be rolled into `05_baseline_models_and_feature_importance.md` or a new report if needed.)

