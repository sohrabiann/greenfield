### Model configuration

Prediction type:
- We use a **binary classifier per problem asset**:
  - Target: “Will this problem asset experience a **Priority 1 failure** in the next 4 days?”
- Long-term stretch goal:
  - Investigate whether we can also **predict the specific failure description/type** (from `failure_desc_problem_equipment_list`) rather than only “fail / not fail”.
  - Phase 1 focuses on binary failure prediction; multi-class failure type prediction is optional and can be explored later.

Use of failure description and sensors:
- `failure_desc_problem_equipment_list` describes:
  - The **failure type** in text (e.g., what actually went wrong).
  - Which **DCS sensors** trip and typical **signal patterns** at failure.
- These descriptions must be used to:
  - Inform **feature selection** (which tags to include for each asset).
  - Optionally define a **multi-class label** (failure type) in later phases.

Feature windows and data:
- We rely on **raw DCS time-series data**, not just precomputed stats, because the long-term goal is to:
  - Connect these models directly to **live DCS streams** and score in (near) real time.
- Feature history:
  - Initial plan: use up to **the last week** of DCS history for feature construction.
  - We will also likely test shorter windows such as **24 hours** to see what history horizon yields the best predictive performance.
- We are using **one model per problem asset**, trained on:
  - All **relevant tags for that problem asset**.
  - Potentially some shared Dryer Area context tags (e.g., main fan, exhaust temperature, etc.) if relevant.

Model families:
- Candidate model types include:
  - Sequence models: **LSTM**, **GRU**, **TCN**.
  - Tree-based models on engineered time-window features: **XGBoost** or similar.

### Use of failure descriptions and text fields

We have two key text-based sources describing failures:

- `failure_desc_problem_Equipment_List.xlsx` – engineering descriptions of how each problem asset tends to fail and which sensors/alarms trip.
- `service_requests_for_problem_eq.xlsx` – free-text descriptions from Priority 1 service requests created when something goes wrong.

We will NOT feed the "future" failure description into the model as an input (that would leak information from the label into the features).

Instead, we will use these text fields in two ways:

1. As **derived labels / categories for analysis**
   - Map similar textual descriptions into grouped failure type codes, for example:
     - `"bearing issue"`, `"bearing overheat"`, `"bearing lubrication failure"` → `BEARING_FAILURE`
     - `"fan imbalance"`, `"product build-up on fan"`, `"clean fan wheel"` → `IMBALANCE_BUILDUP`
     - `"cyclone plugging"`, `"dust build-up in cyclone"` → `CLOGGING`
   - These codes can be used for:
     - Exploratory analysis of which failure types are most common for each asset.
     - Clustering and visualization of failure patterns.
     - (Future work) multi-class failure type prediction once enough examples exist.

2. As **context for feature selection and thresholds**
   - Use the text descriptions to identify:
     - Which **DCS tags** are relevant for each failure type (e.g., bearing temperatures, fan current, exhaust temperature, vibration if available).
     - Typical **signal patterns** at failure (e.g., rising current, rising temperature followed by trip).
   - This information guides:
     - Which tags to include in the per-asset models.
     - Which time-window aggregations or thresholds may be meaningful.

Summary:
- Text fields (failure descriptions, service request text) are used to **create structured labels and guide feature selection**, but **not** as direct inputs that would leak future information into the model.
