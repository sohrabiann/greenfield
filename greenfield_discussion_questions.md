# Greenfield Discussion - Questions & Preferences

**Date**: December 5, 2025  
**Purpose**: Clarify business priorities, validation criteria, and Phase 2 direction  
**Project**: Dryer A Predictive Maintenance Pilot (Phase 1 → Phase 2 Transition)

---

## 📋 **Meeting Objectives**

1. Validate Phase 1 progress and data coverage
2. Confirm business priorities for Phase 2
3. Clarify alert philosophy and user workflows
4. Discuss deployment timeline and integration requirements
5. Identify any data gaps or additional asset priorities

---

## 🎯 **Section 1: Business Priorities & Success Metrics**

### 1.1 Top Business Outcomes (Please Rank 1-5)

We want to confirm which outcomes matter most to Greenfield:

- [ ] **Reduce unplanned downtime on Dryer A**
- [ ] **Reduce repeat failures on problem assets** (DC-1834, BL-1829, RV-1834)
- [ ] **Shift from corrective to preventative maintenance**
- [ ] **Improve safety** (reduce fire/explosion/injury risk)
- [ ] **Build a scalable template** for other assets and sites

**Question**: Which 2-3 outcomes are the highest priority for the next 6 months?

---

### 1.2 Acceptable Performance Thresholds

For Phase 1 validation and Phase 2 deployment:

**Precision (when model alerts, how often is it right?)**
- [ ] 20-30% is acceptable (catch most failures, tolerate false alarms)
- [ ] 40-50% preferred (balance)
- [ ] 60%+ required (minimize false alarms)

**Recall (what % of real failures should we catch?)**
- [ ] 60%+ is acceptable
- [ ] 70-80% preferred
- [ ] 90%+ required

**Lead Time (how far in advance do you need warnings?)**
- [ ] 24 hours minimum
- [ ] 48-72 hours preferred
- [ ] 4+ days ideal

**Question**: What would make this pilot a "win" in your view? What performance would justify expanding beyond these 3 assets?

---

## 🚨 **Section 2: Alert Philosophy & False Alarm Tolerance**

### 2.1 False Alarm Tolerance

**Currently**: We're designing models that favor catching failures over minimizing false alarms.

**Question**: How many false alerts per month per asset are acceptable?

- [ ] **1-2 per month per asset** (aggressive, catch everything)
- [ ] **1 per month per asset** (moderate)
- [ ] **1 every 2 months per asset** (conservative)
- [ ] **Other**: _____________

**Context**: With 3 assets (DC-1834, BL-1829, RV-1834), this means:
- Aggressive: 3-6 false alerts/month total
- Moderate: ~3 false alerts/month total
- Conservative: ~1-2 false alerts/month total

---

### 2.2 Alert Presentation

**Question**: How should predictions be presented?

- [ ] **Probability score (0-100%)** only
- [ ] **Risk categories (Low/Medium/High)** only
- [ ] **Both** (score + category)

**Question**: Should alerts be differentiated by severity/confidence?

- [ ] Yes - show "High Confidence" vs "Medium Confidence" alerts
- [ ] No - treat all alerts equally
- [ ] Not sure yet

---

### 2.3 Prediction Cadence

**Question**: How often should risk scores be updated?

- [ ] **Real-time** (every 10 minutes, as new DCS data arrives)
- [ ] **Hourly** (updated every hour)
- [ ] **Per shift** (3 updates/day)
- [ ] **Daily** (once per day summary)

**Question**: Who reviews the predictions?

- Control room operators (real-time monitoring)
- Maintenance planners (daily planning)
- Reliability engineers (weekly reviews)
- Management (monthly dashboards)

---

## 👥 **Section 3: Primary Users & Workflows**

### 3.1 User Groups (Please Rank by Priority)

Who will use these predictions most?

- [ ] **Maintenance planners/schedulers** (plan PM windows, order parts)
- [ ] **Control room operators** (cross-check with alarms, take immediate action)
- [ ] **Reliability engineers** (root cause analysis, improve PM strategies)
- [ ] **Management** (KPI dashboards, downtime reporting)

**Question**: For each user group, what actions would they take when a high-risk alert appears?

1. **Maintenance planners**: _____________________________________________
2. **Control room operators**: ___________________________________________
3. **Reliability engineers**: ____________________________________________
4. **Management**: _______________________________________________________

---

### 3.2 Integration with Existing Systems

**Question**: How should predictions be delivered?

- [ ] **Email/text alerts** to key personnel
- [ ] **Dashboard** (web-based or local display)
- [ ] **Integrated into Maximo** (auto-generate PM work orders?)
- [ ] **Control room display** (alongside existing DCS alarms)
- [ ] **Excel/CSV reports** (daily/weekly exports)
- [ ] **Other**: _____________

**Question**: Do you want alerts to automatically create work orders in Maximo?

- [ ] Yes - auto-create PM work orders when risk is high
- [ ] No - humans review and decide whether to create WOs
- [ ] Maybe in Phase 3, not Phase 2

---

## 📊 **Section 4: Data Coverage & Gaps**

### 4.1 Current Coverage (Jan 2023 - Oct 2025)

We have **20 Priority 1 failures** with complete DCS coverage:

| Asset | Failures | Training | Testing |
|-------|----------|----------|---------|
| DC-1834 (Dust Collector) | 9 | 7 | 2 |
| BL-1829 (Main Blower) | 6 | 4 | 2 |
| RV-1834 (Rotary Valve) | 5 | 3 | 2 |

**Missing**: 44 failures from 2021-2022 (before DCS data starts)

**Question**: Is the 2023-2025 coverage sufficient for Phase 2, or should we request 2021-2022 DCS data?

- [ ] **Current coverage is sufficient** (20 failures enough to demonstrate value)
- [ ] **Request 2021-2022 data** (would add ~24 more failures, improve models)
- [ ] **Not sure** (need to see Phase 1 results first)

---

### 4.2 Priority 2 Service Requests

Our analysis shows there are **Priority 2 service requests** (non-failure events) that might be early warning signals.

**Question**: Should we incorporate Priority 2 events as:

- [ ] **Positive labels** (treat as "early stage failures")
- [ ] **Warning signals** (intermediate risk level between healthy and failure)
- [ ] **Ignore for now** (focus on Priority 1 only)
- [ ] **Use for feature engineering** (count P2 events in recent history as a feature)

---

### 4.3 Other Assets to Prioritize

We're currently modeling 3 assets. We've identified 5+ other problem assets:

- **CV-1828** (10 Priority 1 failures, but 0 complete windows)
- **EJ-1837** (3 failures)
- **E-1834** (2 failures)
- Others with incomplete coverage

**Question**: After validating DC-1834, BL-1829, RV-1834, which assets should we expand to next?

1. _______________
2. _______________
3. _______________

**Question**: Are there any **new** problem assets (not in our current list) that have become priorities recently?

---

## 🗓️ **Section 5: Deployment Timeline & Resources**

### 5.1 Phase 2 Timeline

**Question**: What's your preferred timeline for Phase 2 deployment?

- [ ] **Immediate** (deploy baseline models ASAP, iterate in production)
- [ ] **2-4 weeks** (refine models, add Random Forest/XGBoost)
- [ ] **1-2 months** (wait for LSTM/deep learning models)
- [ ] **3+ months** (comprehensive validation before deployment)

---

### 5.2 Pilot vs Production

**Question**: Should Phase 2 be a "silent pilot" or "production"?

- [ ] **Silent pilot** (models run in background, alerts go to small team for validation)
- [ ] **Production** (alerts go directly to operators/planners)
- [ ] **Hybrid** (operators see alerts but marked as "pilot/experimental")

---

### 5.3 Resources & Support

**Question**: Who at Greenfield will be the primary contact for Phase 2?

- **Maintenance lead**: _____________
- **Reliability engineer**: _____________
- **Data/IT contact**: _____________

**Question**: Will we have access to:

- [ ] Updated DCS data extracts (monthly/quarterly)?
- [ ] Work order data from Maximo (to validate predictions)?
- [ ] Operator feedback on false alarms?
- [ ] Control room access for dashboard installation?

---

## 🔧 **Section 6: Technical Preferences**

### 6.1 Model Interpretability vs Accuracy

**Question**: How important is understanding *why* the model predicts a failure?

- [ ] **Very important** - we need to see which sensors drove the alert (use simpler models)
- [ ] **Moderately important** - helpful but not required (can use complex models)
- [ ] **Not critical** - accuracy matters more than interpretability

---

### 6.2 Per-Asset vs Unified Models

**Currently**: We're building separate models for each asset (DC-1834, BL-1829, RV-1834).

**Question**: Should we:

- [ ] **Keep separate models** (each asset has unique failure modes)
- [ ] **Build a unified model** (one model for all Dryer A problem assets)
- [ ] **Both** (compare approaches)

---

### 6.3 Failure Definition

**Currently**: A "failure" = Priority 1 failure event requiring corrective maintenance and/or causing unplanned shutdown.

**Question**: Is this definition correct and complete?

- [ ] Yes
- [ ] No - should also include: _____________________________________________
- [ ] Need to discuss further

---

## 💡 **Section 7: Open Discussion**

### 7.1 Concerns or Constraints

**Question**: Are there any concerns or constraints we should be aware of?

- Budget limitations for Phase 2?
- IT security/network restrictions for deployment?
- Maintenance schedule conflicts (e.g., planned turnarounds)?
- Organizational change management (getting buy-in from operators)?

---

### 7.2 Success Stories or Benchmarks

**Question**: Are you aware of similar predictive maintenance projects (internally or at other plants)?

- What worked well?
- What didn't work?
- Any lessons learned we should apply?

---

### 7.3 Long-Term Vision

**Question**: If this pilot succeeds, what's the long-term vision?

- [ ] Expand to **all Dryer A assets**
- [ ] Expand to **Dryer D**
- [ ] Expand to **other plant areas** (fermentation, distillation, evaporation)
- [ ] Expand to **other Greenfield facilities**
- [ ] Integrate with **corporate-wide asset management strategy**

---

## 📝 **Section 8: Phase 1 Validation - Feedback**

### 8.1 Data Summary (for your review)

We've analyzed:
- **60.7 million DCS data points** (595 MB dataset)
- **20 Priority 1 failures** with complete sensor coverage
- **430 DCS tags** (temperatures, currents, pressures, flows)
- **3 problem assets** (DC-1834, BL-1829, RV-1834)

**Question**: Does this data coverage match your expectations?

- [ ] Yes - this is what we expected
- [ ] No - we thought there would be more/less data
- [ ] Surprised by: _____________

---

### 8.2 Key Findings (for your review)

Our analysis found:
- **Temperature sensors** (bearing temps) show gradual changes before failures
- **Current sensors** (motor currents) show load variations before failures
- **4-day lead time** appears feasible for most failure modes
- **Multi-window features** (24h, 72h, 168h) capture both short and long-term trends

**Question**: Do these findings align with your operational experience?

- What failure precursors do operators currently look for?
- Are there other sensors/tags we should incorporate?
- Are there failure modes that happen too quickly for 4-day prediction?

---

### 8.3 Missing Context

**Question**: Is there any context or domain knowledge we're missing?

- Seasonal patterns (winter vs summer operations)?
- Production schedule impacts (high vs low throughput)?
- Maintenance history insights (recently rebuilt equipment)?
- Known sensor issues (drift, calibration problems)?

---

## ✅ **Section 9: Next Steps & Action Items**

After this discussion, we will:

1. [ ] **Update model configurations** based on your preferences
2. [ ] **Finalize Phase 1 validation report** with your feedback
3. [ ] **Create Phase 2 project plan** with timeline and deliverables
4. [ ] **Schedule follow-up** to review initial model results

**Question**: What's your preferred cadence for updates during Phase 2?

- [ ] Weekly check-ins
- [ ] Bi-weekly check-ins
- [ ] Monthly check-ins
- [ ] As-needed / milestone-based

---

## 📧 **Contact & Follow-Up**

**Prepared by**: MME4499 Capstone Team  
**Date**: December 5, 2025  
**Document**: greenfield_discussion_questions.md

**Please review these questions before our meeting. We'll focus on your highest priorities and can skip sections that aren't relevant.**

---

## 🎯 **Quick Priority Checklist** (Fill out before meeting)

**Top 3 must-discuss topics:**

1. _______________________________________________________
2. _______________________________________________________
3. _______________________________________________________

**Top 3 concerns or questions Greenfield has for us:**

1. _______________________________________________________
2. _______________________________________________________
3. _______________________________________________________

**Ideal outcome from this meeting:**

_________________________________________________________________
_________________________________________________________________
_________________________________________________________________

---

*End of Discussion Guide*

