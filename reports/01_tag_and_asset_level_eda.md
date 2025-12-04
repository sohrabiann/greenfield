# Task 1 – Tag and Asset Level EDA

## What was done
- Loaded the sample Dryer A datasets from `greenfield_data_context/samples/` using a custom lightweight parser in `src/phase1_analysis.py`.
- Fixed CSV parsing for the asset list by trimming malformed `Plant_Area` strings and mapping unexpected values to `Unknown`/`ParsingError` so spurious entries (e.g., long comma-delimited fragments) no longer inflate area totals.
- Normalized binary DCS states (`Open/Running`, `Closed/Stopped`) into numeric surrogates to allow basic statistics without external packages.
- Summarized problematic assets from the asset list and aggregated DCS tag completeness, data type coverage, and state distributions.
- Built a first pass **tag-to-problem-asset** lookup using instrumentation metadata (saved to `docs/dryer_problem_asset_tags.csv`).

## Key findings
- The asset list still shows **14 problem assets in DRYER A**, and cleaning the area field reduced noisy categories: raw counts included malformed strings and singletons, while the cleaned view consolidates to `DRYER A` (234), `DRYER D` (20), `DDG` (28), `TGF` (11), `Unknown` (15), and `ParsingError` (5).
- The primary DCS tag available (`CH-DRA_COMB_AIR:BL1825.STAIND`) has **0% missing records** across 2,000 readings; the only `DataType` is `Running/Stopped` and the `Reason_No_Data` field is never populated.
- State balance is skewed toward `Open/Running` (**95.3%**) vs `Closed/Stopped` (4.7%), but both states are present, which is sufficient for binary analytics.
- Initial tag-asset linkage surfaces status tags for the main Dryer A blower (BL-1829), product cooler (E-1834), utility pump (P-1837), and dust collector rotary valves (RV-1834); richer analog tags will be added as we extend the mapping beyond status indicators.

## Implications for the project
- The parsing fix keeps DRYER A counts trustworthy while isolating bad rows for cleanup; downstream grouping by plant area can safely rely on the `Plant_Area_Clean` field.
- Data completeness is strong for the available tag, so downstream feature engineering can focus on state dynamics rather than cleaning.
- The strong imbalance suggests that any classifier will need attention to class weighting or threshold tuning.
- Additional tags should be incorporated as they become available to widen coverage beyond the single blower state indicator; the new tag-asset table provides a starting scaffold for multi-tag selection per problem asset.
