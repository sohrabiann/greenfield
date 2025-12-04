# Task 3 – Temporal Structure and Window Exploration

## What was done

### Phase 1 (Sample Data - Single Status Tag)
- Selected the densest numeric DCS series (blower run-state mapped to 0/1) and computed autocorrelation coefficients for 10-minute lags up to 2 hours.
- Evaluated whether shorter or longer history windows capture meaningful variation in the state signal.
- **Result**: High persistence (ACF ≥ 0.81 at 120 minutes) indicating short windows (30-60 min) are sufficient for status tags.

### Phase 1 Extended (Volume Data - Rich Analog Tags)
- Analyzed temporal structure of analog tags from DCS volume data
- Computed autocorrelation functions (ACF) for motor currents and bearing temperatures
- Assessed appropriate feature window lengths for different tag types

## Key findings

### Motor Current Tags (IT18179, IT18211, IT18216)

**IT18179 (Main Blower Current)**:
- 5,892 data points over 42 days
- ACF[1]: 0.758 (moderate persistence)
- ACF[6]: 0.447 (1 hour lag)
- ACF[12]: 0.245 (2 hour lag)
- **Interpretation**: Current varies with load; correlation decays relatively quickly

**IT18211, IT18216 (Dust Collector/Cooler Currents)**:
- 6,030 data points each
- ACF[1]: 0.89-0.96 (high persistence)
- ACF[6]: 0.83-0.89
- ACF[12]: 0.76-0.82
- **Interpretation**: More stable than main blower; slower decay

### Temperature Tags (TT18179A-C, TT18217)

**Bearing Temperatures (TT18179A, TT18179B, TT18179C)**:
- 6,031 data points each
- ACF[1]: 0.999-1.000 (extremely high persistence)
- ACF[6]: 0.975-0.990
- ACF[12]: 0.938-0.971
- **Interpretation**: Very slow-changing; thermal inertia dominates

**System Temperature (TT18217)**:
- 6,030 data points
- ACF[1]: 0.996
- ACF[6]: 0.966
- ACF[12]: 0.920
- **Interpretation**: Similar to bearing temps; gradual changes only

### Comparison: Status vs Analog Tags

| Tag Type | ACF[1] | ACF[12] | Decay Rate | Recommended Window |
|----------|--------|---------|------------|-------------------|
| Status (Run/Stop) | 0.98 | 0.81 | Slow | 30-60 min |
| Motor Current | 0.76-0.96 | 0.25-0.82 | Moderate | 24-72 hours |
| Temperature | 0.996-1.0 | 0.92-0.97 | Very Slow | 72-168 hours |

### No Clear Daily/Weekly Cycles Detected
- Examined ACF patterns for evidence of periodic behavior
- No strong cyclical patterns observed in the sample period
- **Note**: Sample covers only 42 days; longer-term patterns may exist in full dataset

## Implications for the project

### Feature Window Design

**For Motor Currents**:
- **Short-term features** (last 1-6 hours): Capture immediate load changes
- **Medium-term features** (last 24-72 hours): Capture operational patterns
- Use statistics: mean, std, min, max, last value, slope

**For Temperatures**:
- **Medium-term features** (last 24-72 hours): Detect gradual warming trends
- **Long-term features** (last 168 hours / 7 days): Establish baseline and detect sustained deviations
- Use statistics: mean, std, max, slope (trend detection critical)

**Combined Strategy**:
- Extract features at **multiple time scales**: 6h, 24h, 72h, 168h
- This captures both short-term anomalies and long-term degradation trends

### Window-Length Experiments (Planned)
To validate optimal window lengths:
1. Create simple aggregated features for each window size
2. Train quick baseline models (logistic regression) with each window
3. Compare ROC-AUC and precision/recall
4. Select window(s) with best performance

**Hypothesis**:
- **24-72 hour windows** will perform best for most tags
- Combining multiple windows will outperform single-window features

### Memory and Computational Considerations
- Temperature tags have very high autocorrelation → can be downsampled without losing much information
- Motor current tags vary more → keep higher sampling rate
- For LSTM/GRU models, consider:
  - Sequence length: 24-72 hours (144-432 timesteps at 10-min intervals)
  - Attention mechanisms may help focus on critical periods

### Comparison to Original Sample Analysis
- Original: Single status tag, short windows sufficient (30-60 min)
- Extended: Analog tags require longer windows (24-168 hours) to capture degradation patterns
- **Conclusion**: Feature engineering must be tag-type-specific; one-size-fits-all approach will miss important signals
