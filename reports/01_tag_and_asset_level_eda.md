# Task 1 – Tag and Asset Level EDA

## What was done
- Loaded the sample Dryer A datasets from `greenfield_data_context/samples/` using a custom lightweight parser in `src/phase1_analysis.py`.
- Normalized binary DCS states (`Open/Running`, `Closed/Stopped`) into numeric surrogates to allow basic statistics without external packages.
- Summarized problematic assets from the asset list and aggregated DCS tag completeness, data type coverage, and state distributions.

## Key findings
- The asset list shows **14 problem assets** with 234 total assets in the **DRYER A** area; other areas (DDG, Dryer D) are much smaller in the sample.
- The primary DCS tag available (`CH-DRA_COMB_AIR:BL1825.STAIND`) has **0% missing records** across 2,000 readings; the only `DataType` is `Running/Stopped` and the `Reason_No_Data` field is never populated.
- State balance is skewed toward `Open/Running` (**95.3%**) vs `Closed/Stopped` (4.7%), but both states are present, which is sufficient for binary analytics.

## Implications for the project
- Data completeness is strong for the available tag, so downstream feature engineering can focus on state dynamics rather than cleaning.
- The strong imbalance suggests that any classifier will need attention to class weighting or threshold tuning.
- Additional tags should be incorporated as they become available to widen coverage beyond the single blower state indicator.
