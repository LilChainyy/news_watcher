# Market Intelligence Engine - Correlation Analysis

## 🎯 What This Does

This is your **quantitative analysis component** for the smart news system. It discovers hidden relationships between news events and stock price movements.

**Example Discovery:** When CoreWeave has a GPU shortage, Nebius stock drops 3.2% even though Nebius isn't mentioned in the news. Your system learns this automatically from historical data.

## 🧠 Your Learning Focus (Quant Skills)

This project gives you hands-on experience with:

1. **Event Study Methodology** - The gold standard for analyzing how events impact prices
2. **Statistical Hypothesis Testing** - Separating real correlations from noise
3. **Risk Metrics** - VaR, CVaR, Sharpe ratios, beta calculation
4. **Time Series Analysis** - Understanding how markets react over time
5. **Correlation Networks** - Detecting indirect market relationships

These are core skills for:
- Market Risk Analyst roles
- Quantitative Research positions
- Systematic Trading strategies

## 📁 File Structure

```
.
├── correlation_engine.py       # YOUR CORE WORK - Event impact analysis
├── statistical_analysis.py     # Statistical validation & risk metrics
├── data_loader.py             # Data pipeline & feature engineering
├── main.py                    # Orchestration script
├── README.md                  # This file
└── data/                      # Your historical data (you provide)
    ├── historical_events.csv
    └── historical_prices.csv
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install pandas numpy scipy scikit-learn statsmodels --break-system-packages
```

### 2. Prepare Your Data

You need two CSV files:

**historical_events.csv:**
```csv
date,ticker,event_type,headline,sentiment,entities,source
2024-10-15,CRWV,infrastructure_failure,"CoreWeave GPU shortage",−0.7,"[CoreWeave, GPU]",NewsAPI
2024-10-16,HOOD,earnings,"Robinhood beats earnings",0.8,"[Robinhood]",AlphaVantage
```

**historical_prices.csv:**
```csv
date,ticker,open,high,low,close,volume
2024-10-15,CRWV,45.20,46.10,44.50,45.00,2500000
2024-10-15,NEBIUS,32.50,33.00,31.80,32.00,1800000
```

### 3. Run Analysis

```bash
# Quick test with sample data
python main.py
# Choose option 2

# Full analysis with your data
python main.py
# Choose option 1
```

### 4. Review Results

Check `correlation_report.json` for:
- Indirect correlations discovered
- Risk profiles per stock
- Statistical significance tests

## 🎓 What You'll Implement

### Phase 1: Core Correlation Engine (High Priority)

In `correlation_engine.py`, fill in these TODOs:

1. **`calculate_price_impact()`**
   - Calculate returns in different time windows (1h, 4h, 24h)
   - Measure volume changes around events
   - Track volatility spikes

2. **`build_correlation_network()`**
   - Build event-impact matrix
   - Calculate cross-ticker correlations
   - Identify event clusters

3. **`detect_indirect_correlations()`**
   - Statistical testing for hidden relationships
   - Filter out noise using p-values
   - Build confidence scores

### Phase 2: Statistical Validation (Medium Priority)

In `statistical_analysis.py`, understand and use:

1. **Hypothesis Testing**
   - T-tests for event impact significance
   - Correlation significance tests
   - Multiple comparison corrections (Bonferroni, FDR)

2. **Risk Metrics**
   - Value at Risk (VaR)
   - Conditional VaR (CVaR)
   - Sharpe & Sortino ratios
   - Maximum drawdown

3. **Event Study Analysis**
   - Abnormal return calculation
   - Market model estimation
   - Statistical significance of CARs

### Phase 3: Feature Engineering (Lower Priority)

In `data_loader.py`, enhance features:

1. Technical indicators (RSI, Bollinger Bands)
2. Volume patterns
3. Volatility regimes
4. Time-based features

## 💡 Example Usage

```python
from correlation_engine import EventImpactAnalyzer
from data_loader import DataPipeline

# Load data
pipeline = DataPipeline()
events_df, prices_df = pipeline.run_pipeline()

# Initialize analyzer
analyzer = EventImpactAnalyzer(lookback_days=180)
analyzer.load_historical_data(events_df, prices_df)

# Discover correlations
indirect_corr = analyzer.detect_indirect_correlations()

# Check what we found
for relationship, stats in indirect_corr.items():
    print(f"{relationship}: {stats['avg_impact']:.2%} impact")
    print(f"Confidence: {stats['confidence']:.1%}")

# Test new event
new_event = {
    'ticker': 'CRWV',
    'event_type': 'infrastructure_failure',
    'entities': ['CoreWeave', 'GPU']
}
predictions = analyzer.predict_impact(new_event)
```

## 🎯 Key Concepts You'll Master

### 1. Event Study Methodology

```
Event Timeline:
│
├─── Estimation Window (120 days) ──┤ Event ├─── Event Window (3 days) ──┤
│                                    │   │   │                           │
│ Learn normal behavior              │   ↓   │  Measure abnormal impact  │
```

**What you calculate:**
- Normal returns (based on historical relationship with market)
- Abnormal returns (actual - expected)
- Statistical significance (is the impact real or luck?)

### 2. Correlation vs Causation

**Direct Correlation:** CoreWeave news → CoreWeave stock moves
**Indirect Correlation:** CoreWeave news → Nebius stock moves (hidden relationship)

Your system discovers these automatically by:
1. Testing every event against every stock
2. Statistical validation (p-values < 0.05)
3. Requiring minimum sample size
4. Calculating confidence scores

### 3. Risk Metrics Explained

**VaR (Value at Risk):**
"95% of the time, losses won't exceed X%"

**CVaR (Conditional VaR):**
"When things go bad (worst 5%), average loss is X%"

**Sharpe Ratio:**
"Return per unit of risk" (higher = better)

**Beta:**
"How much stock moves relative to market" (1.0 = same as market)

## 🔍 Debugging Tips

### Issue: No correlations found

**Causes:**
- Not enough historical data (need 6+ months)
- Events too sparse (need 20+ events per type)
- Price data doesn't cover event dates

**Fix:**
- Check date alignment: `validator.check_date_alignment(events_df, prices_df)`
- Verify event count: `events_df.groupby('event_type').size()`

### Issue: All correlations significant

**Problem:** Likely bug in p-value calculation

**Fix:**
- Check sample sizes: `stats['sample_size']` should be > 5
- Verify statistical test implementation
- Use `bonferroni_correction()` to adjust for multiple tests

### Issue: Predictions seem random

**Causes:**
- Not enough training data
- Events too diverse (need event clustering)
- Missing important features

**Fix:**
- Increase lookback period
- Cluster similar events together
- Add more event context (entities, sentiment)

## 📊 What Good Results Look Like

### Strong Correlation Example:
```
CRWV infrastructure_failure → NEBIUS
  Avg Impact: -3.2%
  P-value: 0.003
  Sample Size: 15 events
  Confidence: 99.7%
```

This means:
- Every time CoreWeave has infrastructure issues, Nebius drops ~3%
- 99.7% confident this isn't random
- Seen 15 times in historical data
- **Actionable:** When CoreWeave news breaks, check Nebius position

### Weak Correlation Example:
```
AMD product_launch → RKLB
  Avg Impact: +0.5%
  P-value: 0.23
  Sample Size: 8 events
  Confidence: 77%
```

This means:
- Small positive correlation but not statistically significant
- P-value too high (> 0.05)
- Don't trade on this

## 🎓 Learning Path

### Week 1-2: Setup & Understanding
- Read through all files
- Understand the data flow
- Run quick test mode
- Ask questions about concepts

### Week 3-4: Core Implementation
- Implement `calculate_price_impact()`
- Build correlation matrix
- Add statistical tests
- Validate with sample data

### Week 5-6: Advanced Features
- Implement indirect correlation detection
- Add event clustering
- Build risk profiles
- Test with real data

### Week 7-8: Refinement
- Optimize for speed
- Add more features
- Improve accuracy
- Document findings

## 🤝 Integration with Dashboard

Once this works, your software engineer will:

1. **Pull predictions** from `analyzer.predict_impact()`
2. **Display on dashboard** with confidence scores
3. **Trigger alerts** for high-confidence predictions
4. **Update model** weekly with new data

Your job: Make sure predictions are statistically sound.

Their job: Make it look good and accessible.

## 📚 Resources for Learning

**Event Studies:**
- "Event Studies in Economics and Finance" by MacKinlay
- Classic methodology used in finance research

**Statistical Testing:**
- Scipy documentation for hypothesis tests
- "Statistical Inference" by Casella & Berger

**Risk Metrics:**
- "Value at Risk" by Jorion
- Industry standard risk measurement

**Python Skills:**
- Pandas time series operations
- NumPy vectorization
- Scipy statistical functions

## ⚠️ Important Notes

### Data Quality Matters
- Garbage in = garbage out
- Need clean, aligned event and price data
- Validate data before running analysis

### Statistical Rigor
- Don't trade on p-value > 0.05
- Use multiple comparison corrections
- Require minimum sample sizes
- Be skeptical of "too good" results

### Overfitting Risk
- Don't test too many hypotheses
- Use out-of-sample validation
- Update model regularly with new data
- Monitor prediction accuracy

## 🎯 Success Metrics

You'll know it's working when:

1. ✅ You find 5-10 strong indirect correlations (p < 0.05)
2. ✅ Predictions have 60%+ accuracy on new events
3. ✅ You can explain WHY correlations exist (not just that they do)
4. ✅ Risk metrics align with your trading experience
5. ✅ You can articulate this in interviews for quant roles

## 🚀 Next Steps

1. **Get data:** Historical news + prices for your watchlist
2. **Start coding:** Fill in TODOs in correlation_engine.py
3. **Test thoroughly:** Use statistical_analysis.py for validation
4. **Document findings:** What correlations did you discover?
5. **Iterate:** Improve based on what you learn

## 💬 Questions?

Think through:
- What does this correlation mean economically?
- Why would these stocks move together?
- Is the sample size large enough?
- Could this be spurious correlation?

This mindset is what separates good quants from bad ones.

---

**Remember:** This isn't just code practice. This is building the same analytical framework that market risk teams at banks use for event analysis. Take your time to understand the concepts, not just implement the code.
