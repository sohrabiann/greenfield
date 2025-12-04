### Failure criteria and label definitions

Data sources for failures:
- `failure_desc_problem_equipment_list`:
  - Contains a description of several past failures for each problem asset.
  - Also describes which DCS sensors/alarms are involved and what the DCS reads when these failures occur.
- `service_requests_for_problem_eq`:
  - Contains service requests for the same problem assets.
  - **This is our primary timeline of when things went wrong**, because service requests are created immediately when a problem is observed (earlier than work orders).

Canonical failure rule:
- A **failure event** is defined as any **service request on a problem asset with `Priority = 1`**.
- In Phase 1, we treat “Priority 1 service request” as equivalent to:
  - “Unplanned failure” and typically “unplanned shutdown” for that problem asset.

Per-asset failure modes:
- Each problem asset in `failure_desc_problem_equipment_list` has:
  - A short **failure description** (e.g., what went wrong, what tripped, etc.).
  - An association with the relevant DCS tags and what they looked like at failure time.
- These per-asset failure descriptions are considered the **canonical text descriptions of the failure modes**.
- We will train **separate models per problem asset** (one model per asset), each predicting failures according to the service request criteria above.

Prediction horizon and history window:
- For all Dryer A problem asset failure modes:
  - The model predicts the **probability that a failure (Priority 1 service request) will occur in the next 4 days**.
  - Features are computed from **preceding history windows** of DCS data.
    - Initial design: use up to **1 week of historical DCS data** as a candidate feature window.
    - We may also experiment with **shorter windows (e.g., last 24 hours)** during analysis to see what works best.

Healthy (negative) windows:
- A **healthy** window is any time period where **no failure event (Priority 1 service request)** for that problem asset occurs within the subsequent 4 days.
- These healthy windows can still contain normal alarms and operator actions, as long as they do *not* lead to a Priority 1 service request within the horizon.
