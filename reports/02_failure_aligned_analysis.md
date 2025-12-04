# Task 2 – Failure-Aligned Analysis

## What was done

### Phase 1 (Sample Data - Single Status Tag)
- Parsed Dryer A shutdown events and aligned the blower-state DCS series around each event using +/-7 days of history and 1 day post-event windows.
- Focused on the last 72 hours before each shutdown to contrast pre-failure behaviour against the earlier healthy baseline.
- **Result**: No observable precursor in the single available status tag (blower remained 100% running until shutdown).

### Phase 1 Extended (Volume Data - Rich Analog Tags)
- Loaded rich DCS volume data (26 files with ~400 tags each) including motor currents, bearing temperatures, and process measurements.
- Analyzed Priority 1 service request events for three key problem assets:
  - **BL-1829** (Main Dryer A Blower): 12 Priority 1 failures
  - **RV-1834** (Rotary Valve): 9 Priority 1 failures
  - **DC-1834** (Dust Collector): 18 Priority 1 failures
- Extracted failure windows (T-7 days to T+1 day) and computed statistics for last 24h, 72h, and 168h before each failure.
- Analyzed key analog tags:
  - Motor currents (IT tags): IT18179, IT18211, IT18216
  - Bearing temperatures (TT tags): TT18179A-E, TT18217

## Key findings

### Data Coverage Limitations
- DCS volume data spans **Jan 2023 - Feb 2023** (sample files)
- Priority 1 failures span **2021-2025**
- **Limited temporal overlap** between available DCS data and failure events
- Only **1 failure window** (DC-1834) had sufficient DCS data for analysis

### DC-1834 (Dust Collector) - One Analyzable Failure
For the single failure event with data (2023-01-06):

**Motor Current (IT18211)**:
- Last 24h before failure: mean=0.64 amps, std=0.32
- Last 72h before failure: mean=0.74 amps, std=0.20
- Last 168h before failure: mean=0.77 amps, std=0.13
- **Pattern**: Slight decrease in current in final 24h (possible reduced load or intermittent operation)

**Temperature (TT18217)**:
- Last 24h before failure: mean=29.01°F, std=4.15
- Last 72h before failure: mean=26.02°F, std=3.44
- Last 168h before failure: mean=25.91°F, std=5.75
- **Pattern**: Slight temperature increase in final 24h with increased variability

### Temporal Characteristics of Analog Tags

**Motor Current Tags** (IT18179, IT18211, IT18216):
- ACF[1]: 0.76-0.96 (moderate to high persistence)
- ACF[12]: 0.25-0.82 (decays over ~2 hours)
- **Implication**: Capture meaningful short-term dynamics; 24-72h windows appropriate

**Temperature Tags** (TT18179A-C, TT18217):
- ACF[1]: 0.996-1.000 (very high persistence)
- ACF[12]: 0.92-0.97 (slow decay)
- **Implication**: Very stable; changes are gradual; longer windows (72h+) may be needed to detect trends

## Implications for the project

### Data Availability
- **Critical finding**: The sample DCS data does not overlap significantly with the failure timeline
- To build production models, need to:
  1. Load DCS data from the full `AllDCSData.csv` (624 MB) covering the 2021-2025 period
  2. Or focus on recent failures (2023+) where DCS coverage is better

### Pre-Failure Signatures
- The single analyzable failure shows **subtle changes** in the 24h before failure:
  - Decreased motor current (possible intermittent operation or reduced load)
  - Increased temperature variability
- These patterns are **not dramatic** but could be detectable with proper feature engineering

### Feature Window Recommendations
Based on temporal structure analysis:
- **Motor currents**: 24-72 hour windows capture most dynamics
- **Temperatures**: 72-168 hour windows to detect gradual trends
- **Combined approach**: Use multiple window sizes (24h, 72h, 168h) as separate feature sets

### Next Steps
1. **Load full DCS data** to get better failure-event coverage
2. **Aggregate features** over multiple time windows (mean, std, min, max, slope)
3. **Compare failure vs healthy distributions** once more data is available
4. **Test simple anomaly detection** on normal periods to see if pre-failure periods show elevated anomaly scores

### Comparison to Original Sample Analysis
- Original analysis: Single binary status tag, no pre-failure signature
- Extended analysis: Rich analog tags show subtle patterns, but limited by data coverage
- **Conclusion**: The infrastructure is now in place to analyze the full dataset and build meaningful baselines
