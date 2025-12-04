# Greenfield Global – Dryer A Preventative Maintenance Project  
_Context & Instructions for ChatGPT Codex_

## 1. High-Level Project Summary

This project is an MME4499 capstone with **Greenfield Global – Chatham ethanol plant**.

**Main objective**:  
Optimize **preventative maintenance (PM) procedures** using **AI + IoT**, starting with a **pilot on Dryer A and its problem assets** in the Dryer Area.

This is **not just** a model-building project. It has two layers:

1. **Maintenance strategy layer**  
   - Understand where PM is effective, where it’s overdone, and where it’s failing.  
   - Use historical failures + work orders + PM tasks to rationalize and standardize PM strategies.

2. **AI / predictive maintenance layer**  
   - Use historical sensor/time-series data + work orders + shutdowns to train models that:
     - Predict high-risk failure events on Dryer A problem assets.
     - Provide actionable lead time for planners/operators.
   - Integrate outputs into Greenfield’s existing tools (Maximo, dashboards, etc.) over time.

**Pilot scope for modeling**:  
- **Dryer Area only**, focusing on **Dryer A and its problem assets**.  
- The plant overview can mention the overall facility, but all modeling is Dryer A-centric for now.


## 2. Key Artifacts & Data Sources

Codex will have access to a GitHub repo and (eventually) Databricks resources. The following artifacts already exist in some form:

### 2.1 Documents / Presentations

- **Dryer A Training Deck**  
  - Detailed description of Dryer A process, major equipment, and control loops.
  - Includes:
    - Burner, combustion chamber, heat exchanger, drum, cyclones, cooler.
    - Major control loops, e.g.:
      - PIC-18133 – Thermal oxidizer pressure controller.
      - WIC-18153 – Dosing bin weight controller.
      - FIC-18116 – Burner (natural gas flow) demand.
      - TIC-18117A/B – Thermal oxidizer temperature control.
      - TIC-18147A – Dryer exhaust temperature control.
      - TIC-18147B + FIC-18161 + PIC-18164 – Quench water / atomizing steam.
      - IIC-18179 – Main fan amp controller.
      - PIC-18215 – Combustion fan inlet pressure controller.
      - SIC-18114 – Combustion fan speed controller.
    - Shutdown scenarios:
      - Feed permissive loss → controlled shutdown.
      - Normal shutdown (sequence).
      - Fast shutdown (safety interlock).
      - Power loss / load shed.
      - Cleanout sequence (critical for explosion risk).
    - Startup sequences, troubleshooting / “what if” scenarios.

- **Concept Design Presentation**  
  - Describes:
    - Current-state problems in maintenance (over/under-maintenance, repeat failures).
    - Three main work packages:
      1. **WP1 – Current State Analysis** (Maximo WOs, PMs, failures, data sources).
      2. **WP2 – AI Model Development** (data prep, feature eng., model selection).
      3. **WP3 – IoT Expansion** (gap analysis, new sensors if needed).
    - Focus on Dryer Area and problem equipment.
    - Emphasis on **unified data model** linking:
      - Assets ↔ Tags ↔ Shutdowns ↔ Work Orders.

- **Report #1 – Greenfield Global – PM Procedure Optimization**  
  - Problem framing:
    - PM procedures are not data-driven enough; there are repeat failures and some over-maintenance.
  - Approach:
    - Use historical Maximo work orders and DCS data.
    - Use time-series models (Prophet/LSTM/etc.) for failure prediction.
    - Use RCM/FMEA and standards (e.g., ISO 17359/20816) conceptually.
  - High-level comparison to existing tools (Fiix, Merlion, etc.) and literature.

- **AI-System DFMEA (PDF)**  
  - FMEA of the **AI pipeline**, not plant equipment.
  - Covers:
    - Data integration pipeline failures (time misalignment, missing data).
    - Feature engineering issues (scaling, encoding).
    - Model training pitfalls (overfitting, insufficient failure samples).
    - Deployment and streaming issues.
    - UI and user-interpretation risks.
    - IoT sensor drift, calibration, etc.

- **Project Timing (Gantt) PDFs**  
  - Show stages:
    - WP1 – Current State Report.
    - WP2 – AI Model.
    - WP3 – IoT Expansion.
    - Reports and design milestones.

### 2.2 Data (in/through Databricks)

The user previously provided a ZIP with Databricks samples and schemas (not all of this may be visible to Codex, but the structure matters):

- **Key curated tables (Silver/Gold samples)**:
  - `assetlist_silver` – assets with metadata (location, problem flags, cost fields, etc.).
  - `problemassetdetails` – more detailed info on assets flagged as problematic.
  - `dcsdata_silver` – cleaned time-series:
    - `Timestamp`, `Tag_Full`, `Value_Numeric`, `Value_String`, `Reason_No_Data`, etc.
  - `instrumentation_descriptions_silver` – tag dictionary:
    - `Area`, `Tag_Full`, `Instrumentation_Location`, `Instrumentation_Type`, `Role`, `Description`, etc.
  - `instrumentation_statistics_gold` – long-term per-tag stats and percentiles.
  - `instr_stats_(numeric/onoff/manual/running)` – detailed per-tag behavior stats.
  - `dryerashutdownsgold` / `dryerdshutdownsgold` – shutdown events and per-tag pre/post stats.
  - `shutdown_stats_(numeric/onoff/manual/running)` – aggregated features around shutdowns per tag & event.
  - `workorders_silver` – Maximo work orders (type, asset, description, dates, etc.).
  - `workordersdetailview_gold` – richer WO view.
  - `shutdownworkordersmapped` – mapping of shutdown events to work orders.

- **Problem Equipment / Current State Excel files**  
  - `problemassetlist` (Excel):  
    - Contains the **true list of problem assets** for the Dryer Area and other areas.
    - This is currently the **most accurate source** of *which assets give issues* and high-level failure descriptions.
  - `Current State Analysis` (Excel):  
    - Likely contains PM vs failure stats, criticality, and other baseline metrics for assets.

These Excel files are crucial for **real failure modes** and must be parsed by Codex from the repo.


## 3. Main Goal for Codex

We want Codex to help us build **three markdown files** which together provide all the context needed for future modeling, analysis, and collaboration:

1. **`plant_overview.md`**  
2. **`target_definition.md`**  
3. **`modeling_goals.md`**

These files should be **self-contained and repo-friendly**, so that:

- Anyone (including future LLM tools) can understand the Dryer Area system, the modeling targets, and the success criteria without re-reading all PPTs and reports.
- They can be used as the “source of truth” for how we frame predictive maintenance on Dryer A.

Below is what each file is intended to contain.


## 4. Intended Content of the Three Markdown Files

### 4.1 `plant_overview.md`

**Scope**:  
- Briefly introduce the entire Chatham plant.
- Then focus in detail on the **Dryer Area** and **Dryer A + its problem assets**.

**Content** (high-level):

- **Plant context** (short)  
  - Where Dryer A fits in the overall ethanol / DDGS process.

- **Dryer A process overview** (detailed)  
  - Narrative flow: Decanters → Syrup → Heat exchanger → Drum → Cyclones → Cooler → Storage.
  - Describe roles of:
    - Burner, combustion chamber, heat exchanger, drum, cyclones, cooler, main fan, combustion fan, recycle screws, dosing bin, syrup, etc.

- **Major equipment & problem assets**  
  - List all **problem assets in the Dryer Area** (from `problemassetlist`), e.g.:
    - Main fan (BL-1829).
    - Drum trunnions.
    - Cyclones.
    - Cooler drag conveyor.
    - Others flagged by Greenfield.
  - For each: high-level function and how it tends to fail (from `problemassetlist`).

- **Key control loops**  
  - Detailed descriptions of the important loops with:
    - Loop ID and technical description.
    - **Plain-language explanation in brackets** (e.g., “this loop keeps the dryer exhaust temperature stable by adjusting feed rate / quench water”).
  - Include loops such as:
    - PIC-18133, WIC-18153, FIC-18116, TIC-18117A/B, TIC-18147A/B, FIC-18161, IIC-18179, PIC-18215, SIC-18114, etc.
  - Highlight which loops are most relevant to the problem assets / failure modes.

- **Shutdown & startup sequences**  
  - Summarize:
    - Normal shutdown, fast shutdown, feed permissive loss, power loss, cleanout sequence.
    - Startup steps (burner light-off, feed sequence, establishing recycle, etc.).
  - Explain why these sequences matter for:
    - Failure labeling.
    - Safety.
    - ML target selection (e.g., predicting fast shutdowns vs controlled shutdowns).

- **Data sources and unified data model**  
  - Explain at a conceptual level how:
    - DCS tags, shutdown records, and Maximo work orders are linked.
  - Point to the Databricks tables and Excel files used.

### 4.2 `target_definition.md`

**Purpose**:  
Define exactly **what the models will predict**, how labels are constructed, and how data from DCS + Maximo + problem asset list is used.

**Content** (high-level):

- **Problem assets and failure modes**  
  - Enumerated from `problemassetlist` (for Dryer A only).
  - For each problem asset:
    - Asset ID / name.
    - Failure mode (wording from `problemassetlist`).
    - Short plain-language description.

- **How failures are detected in reality**  
  - For each failure mode:
    - How operators / maintenance currently know it happened:
      - Fast/unplanned Dryer A shutdowns.
      - Specific alarms (high temperature, high current, etc.).
      - Corrective/emergency Maximo work orders.
      - Offline inspections.

- **Canonical label rule**  
  - Clear, single rule for what counts as a **positive label** in the data, e.g.:
    - Rule A: A failure = Dryer A shutdown + linked CM/EM WO on a problem asset within ±24–48 hours.
    - Or Rule B/C if chosen differently.
  - Instructions for how this label can be computed from:
    - `shutdownworkordersmapped`.
    - `workorders_silver` / `workordersdetailview_gold`.
    - `dryerashutdownsgold` tables.

- **Negative (healthy) class definition**  
  - How “non-failure” periods are defined:
    - e.g., periods where no shutdown/WO occurs for that asset within the horizon.
    - Clarify whether warnings/alarms without WOs/trips are allowed in the “healthy” class.

- **Prediction targets**  
  - Decision on model framing, for example:
    - Binary classifier per failure mode:
      - **Target**: “Failure mode X occurs within the next T hours.”  
    - Possibly: overall “unplanned Dryer A shutdown risk” classifier.
  - Time horizon T:
    - e.g., 24 hours for most failure modes (or mode-specific horizons if defined).

- **Data windows & features (conceptual)**  
  - Reference the use of:
    - Raw DCS signals (from `dcsdata_silver`).
    - Pre-computed stats (from `instrumentation_statistics_gold`, `shutdown_stats_*`, `instr_stats_*`, etc.).
  - Describe the time windows around each labeled event that will be used to construct features (e.g., 1h/3h windows before failure).

- **Future refinements**  
  - Mention that as more failure history is processed, labels may be refined (e.g., better keyword matching in WO descriptions, additional modes).


### 4.3 `modeling_goals.md`

**Purpose**:  
Capture the **business and technical goals** of the modeling, KPIs, and constraints.

**Content** (high-level):

- **Business goals (ranked)**  
  - e.g., (final wording to be set after questions are answered)
    - Reduce unplanned downtime for Dryer A.
    - Reduce repeat failures on Dryer A problem assets.
    - Optimize PM frequencies (reduce over-maintenance, address under-maintained assets).
    - Improve safety (lower fire/explosion risk).
    - Build a scalable template for other assets and sites.

- **Primary users and use cases**  
  - Rank: maintenance planners, reliability engineers, operators, management.
  - Describe how each will use prediction outputs:
    - Planners: schedule PMs based on risk windows.
    - Reliability: refine strategies, root cause investigations.
    - Operators: early warnings for interventions.
    - Management: KPI dashboards.

- **Metrics and success criteria**  
  - Model performance:
    - e.g., per-mode precision/recall, F1, ROC-AUC, calibration.
    - Lead time quality (e.g., % of failures predicted at least X hours in advance).
  - Operational:
    - Reduction in unplanned downtime.
    - Reduction in repeat failures on problem assets.
    - Acceptable false alarm rates.

- **Alert philosophy**  
  - How conservative vs aggressive predictions should be:
    - e.g., “1–2 false alarms per month per critical asset is acceptable if it means catching most real events.”

- **Prediction cadence**  
  - How often predictions are updated:
    - Real-time (every few minutes) but summarized.
    - Once per shift.
    - Daily scores for planners.

- **Output format**  
  - How predictions are presented:
    - Probability 0–100%.
    - Categorical (Low/Medium/High).
    - Both (score + category).

- **IoT expansion stance**  
  - Clarify:
    - Phase 1: models rely on **existing instrumentation**.
    - Later phases: new sensors considered if performance is limited by data gaps.
  - This ensures everyone understands sensor expansion is conditional, not assumed.

- **Model families & standards**  
  - Briefly mention:
    - Potential time-series models (LSTM/GRU/TCN/XGBoost, etc.) as options.
    - Reference to condition-monitoring standards (ISO 17359, ISO 20816) and RCM/FMEA methodology, at least at a conceptual level.


## 5. Key Open Questions Codex Should Help Clarify or Structure

Some questions must be answered by the team (not by Codex), but Codex can help **organize** or **derive partial answers** from the repo.

### 5.1 Problem Assets & Failure Modes

1. From `problemassetlist` (Excel), extract all **Dryer Area problem assets**, including:
   - Asset ID / tag.
   - Description.
   - How Greenfield describes “this asset gives us problems.”

2. For each asset, synthesize a **failure mode description**:
   - e.g.,:
     - “Frequent imbalance and bearing issues”  
     - “Plugging / buildup causing trips”
   - These become the canonical failure modes in `target_definition.md`.

3. Cross-check with:
   - Current State Analysis file.
   - Work order descriptions (if accessible) to refine wording.

### 5.2 Failure Labels (How to Define a Failure Event)

4. Decide, with the team, which label rule to use:

   - **Rule A – Shutdown + WO**  
     A failure = a Dryer A shutdown with a **CM/EM-type WO** on a problem asset within ±24–48h.

   - **Rule B – WO only**  
     A failure = any CM/EM/CO work order on a problem asset, regardless of shutdown status.

   - **Rule C – Shutdown only**  
     A failure = any unplanned / fast Dryer A shutdown, regardless of WO linkage.

5. Once the rule is chosen, Codex should:
   - Draft the **SQL / PySpark pseudo-logic** to compute these labels from Databricks tables.
   - Summarize that logic in `target_definition.md`.

### 5.3 Prediction Configuration

6. Decide with the team:
   - **Prediction type**:
     - Binary per failure mode.
     - Single binary for any unplanned Dryer A shutdown.
     - Hybrid approach.
   - **Horizon T** (e.g., 24h, 72h, or mode-specific).

7. Define:
   - The **negative (healthy) class** rule:
     - Strict (no WOs/shutdowns within T).
     - Or lenient (allow alarms but no trips/WOs).

Codex should not invent these rules but can help document and structure them once the team decides.


### 5.4 Business & UX Details

8. With the supervisor/client, confirm:

   - Top 3 business outcomes (e.g., downtime, repeat failures, PM optimization).
   - Primary user group (e.g., planners vs reliability vs operators).
   - Acceptable false alarm rate (roughly).
   - Target prediction cadence (real-time vs shift vs daily).
   - Preferred output format (probability vs categories vs both).

These answers refine `modeling_goals.md` but do not change the core architecture.


## 6. How Codex Should Use This Context

When working in this repo, Codex should:

1. **Read this context file first**  
   - Understand scope: Dryer Area, Dryer A, problem assets, three markdown deliverables.

2. **Parse key input files**  
   - `problemassetlist` Excel → derive problem assets & failure modes.  
   - `Current State Analysis` Excel → understand PM vs failure patterns.  
   - Relevant Databricks samples/schemas → confirm table structure and naming.

3. **Draft or refine**:
   - `plant_overview.md` structure and content using:
     - Dryer A training deck.
     - Problem asset list.
   - `target_definition.md` structure using:
     - problemassetlist, shutdown/WO mapping tables.
   - `modeling_goals.md` using:
     - Report #1, concept deck, and clarified business questions.

4. **Iterate with the human team**  
   - Present drafts.
   - Highlight fields where human decisions are required (label rules, horizons, KPIs).
   - Update the markdown files based on feedback.

---

**In short:**  
This file tells Codex:

- What the project is.  
- What artifacts exist.  
- What the three key markdown files should contain.  
- What still needs to be decided.  

The end goal is for `plant_overview.md`, `target_definition.md`, and `modeling_goals.md` to become the **canonical documentation** for the Dryer A PM pilot that both humans and LLMs can rely on.

