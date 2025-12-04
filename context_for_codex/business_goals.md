### Business goals and users

Top 3 business outcomes (ranked):
1. **Reduce unplanned downtime on Dryer A.**
2. **Reduce repeat failures on problem assets in the Dryer Area.**
3. **Shift maintenance at Greenfield from corrective to preventative (and eventually predictive).**

Primary users (ranked):
1. **Maintenance planners / schedulers** – use risk predictions to schedule preventative work before failure.
2. **Control room operators** – see upcoming risk for Dryer A problem assets and cross-check with existing alarms and process knowledge.
3. **Management** – consume high-level KPIs and dashboards showing downtime reduction, failure reduction, and PM strategy improvements.

Alert philosophy:
- The client prefers **catching more true failures**, even at the cost of some false alarms:
  - Approximately **1–2 false alerts per month per critical asset** is considered acceptable.
- Control room engineers already have conventional alarms and can cross-check model alerts with their own diagnostics.

Prediction cadence:
- Models will generate updated **risk scores every hour**, but:
  - **Daily summaries** will be the primary input for planners and management.

Output format:
- Each model outputs a **0–100% failure probability** for the asset over the next 4 days.
- (Optionally, risk bands like Low/Medium/High can be derived later, but the primary output is the numerical probability.)

Sensors and IoT:
- Phase 1 must use **existing sensors only**.
- Additional IoT instrumentation can be considered in later phases (WP3) if model performance is limited by data gaps.

Standards and methodologies:
- The approach is aligned with **RCM/FMEA principles** and **condition monitoring guidelines such as ISO 17359 / ISO 20816**.
- Candidate model families include **LSTM/GRU/TCN** and **tree-based methods (e.g., XGBoost)** for tabular time-window features.
