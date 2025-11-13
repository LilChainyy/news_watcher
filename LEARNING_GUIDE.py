# STEP-BY-STEP LEARNING GUIDE
# Complete Implementation Walkthrough with Code Explanations

"""
This guide walks you through every piece of code in the correlation engine.
Read this BEFORE diving into the implementation to understand what each part does.
"""

# ============================================================================
# PART 1: UNDERSTANDING THE CORE CONCEPT
# ============================================================================

"""
WHAT WE'RE BUILDING:
A system that discovers hidden market relationships by analyzing historical data.

EXAMPLE DISCOVERY:
"When CoreWeave (CRWV) has infrastructure problems, Nebius drops 3.2% 
even though Nebius isn't mentioned in the news"

WHY THIS MATTERS:
- You can front-run the market (act before others notice the pattern)
- Hedge positions proactively
- Find pairs trading opportunities
- Understand systemic risk

HOW WE DO IT:
1. Collect historical events + price data
2. Measure price changes after each event
3. Look for patterns across different stocks
4. Validate patterns statistically (separate signal from noise)
5. Use patterns to predict future events
"""

# ============================================================================
# PART 2: DATA STRUCTURES
# ============================================================================

"""
TWO KEY DATAFRAMES:

1. EVENTS_DF:
   date         ticker  event_type              sentiment
   2024-10-15   CRWV    infrastructure_failure  -0.7
   2024-10-16   HOOD    earnings                 0.8
   
   This tells us WHAT happened and WHEN

2. PRICES_DF:
   date         ticker  open    high    low     close   volume
   2024-10-15   CRWV    45.20   46.10   44.50   45.00   2500000
   2024-10-15   NEBIUS  32.50   33.00   31.80   32.00   1800000
   
   This tells us HOW prices moved
"""

# ============================================================================
# PART 3: STEP-BY-STEP CODE WALKTHROUGH
# ============================================================================

# --- STEP 1: CALCULATE PRICE IMPACT ---

def calculate_price_impact_explained():
    """
    GOAL: Measure how much a stock moved after an event
    
    INPUTS:
    - event_date: When the news broke
    - ticker: Which stock to analyze
    - window_hours: How long to measure (default 24h)
    
    OUTPUTS:
    - return_1h: Price change in first hour
    - return_4h: Price change in 4 hours
    - return_24h: Price change in full day
    - volume_change: How much trading volume increased
    - volatility_change: Did the stock become more volatile?
    
    WHY MULTIPLE TIME WINDOWS?
    - Immediate reaction (1h): HFT algorithms
    - Early reaction (4h): Day traders
    - Full reaction (24h): Market fully digests news
    
    CODE BREAKDOWN:
    """
    
    # 1. Get price BEFORE event
    pre_event_price = 45.00  # Example: CRWV at $45
    
    # 2. Get price AFTER event (24 hours later)
    post_event_price = 41.50  # Dropped to $41.50
    
    # 3. Calculate return (percentage change)
    return_24h = (post_event_price - pre_event_price) / pre_event_price
    # = (41.50 - 45.00) / 45.00 = -0.078 = -7.8%
    
    # 4. Compare volume
    normal_volume = 2_000_000  # Average daily volume
    event_day_volume = 5_500_000  # Volume on event day
    volume_change = (event_day_volume - normal_volume) / normal_volume
    # = 175% increase (people were freaking out!)
    
    return {
        'return_24h': -0.078,  # -7.8% drop
        'volume_change': 1.75   # 175% volume surge
    }

# --- STEP 2: BUILD CORRELATION NETWORK ---

def build_correlation_network_explained():
    """
    GOAL: Find which stocks move together when events happen
    
    THE KEY INSIGHT:
    We calculate impact on ALL stocks, not just the one in the news!
    
    EXAMPLE:
    Event: CoreWeave GPU shortage (10/15/2024)
    
    Calculate impact on:
    - CRWV (direct): -7.8%
    - NEBIUS (indirect): -3.2%  ← THIS IS THE DISCOVERY!
    - AMD (indirect): -1.1%
    - RKLB (unrelated): +0.5%
    
    WHY NEBIUS MOVED:
    - Both companies in GPU cloud infrastructure
    - Investors worry about sector-wide issues
    - But not obvious from the news headline
    
    CODE PROCESS:
    """
    
    # 1. For each event, build impact vector
    event_impact_vector = {
        'CRWV': -0.078,   # Direct impact
        'NEBIUS': -0.032,  # Indirect (hidden correlation!)
        'AMD': -0.011,
        'RKLB': 0.005
    }
    
    # 2. Repeat for ALL events
    all_event_impacts = [
        # Event 1 impacts
        {'CRWV': -0.078, 'NEBIUS': -0.032, 'AMD': -0.011, 'RKLB': 0.005},
        # Event 2 impacts
        {'CRWV': -0.042, 'NEBIUS': -0.028, 'AMD': -0.005, 'RKLB': -0.002},
        # ... many more events
    ]
    
    # 3. Calculate correlation matrix
    # "How often do CRWV and NEBIUS move together?"
    correlation_matrix = calculate_correlations(all_event_impacts)
    
    # Result might look like:
    #          CRWV    NEBIUS   AMD    RKLB
    # CRWV     1.00    0.87    0.45   -0.02
    # NEBIUS   0.87    1.00    0.52    0.01
    # AMD      0.45    0.52    1.00    0.10
    # RKLB    -0.02    0.01    0.10    1.00
    
    # 0.87 correlation = VERY STRONG relationship!

# --- STEP 3: DETECT INDIRECT CORRELATIONS ---

def detect_indirect_correlations_explained():
    """
    GOAL: Statistically validate which correlations are REAL vs RANDOM
    
    THE PROBLEM:
    If you test enough things, some will look correlated just by luck.
    Like flipping a coin 100 times - you'll get some long streaks by chance.
    
    THE SOLUTION:
    Statistical hypothesis testing
    
    NULL HYPOTHESIS (H0):
    "There is no relationship between CRWV news and NEBIUS price"
    
    ALTERNATIVE HYPOTHESIS (H1):
    "CRWV news DOES affect NEBIUS price"
    
    TEST PROCESS:
    """
    
    # 1. Collect all returns for NEBIUS when CRWV has news
    nebius_returns_during_crwv_events = [
        -0.032,  # Event 1
        -0.028,  # Event 2
        -0.041,  # Event 3
        -0.025,  # Event 4
        # ... 15 total events
    ]
    
    # 2. Calculate average impact
    avg_impact = mean(nebius_returns_during_crwv_events)
    # = -3.2% average
    
    # 3. Statistical test: Is this significantly different from 0?
    from scipy.stats import ttest_1samp
    
    t_statistic, p_value = ttest_1samp(
        nebius_returns_during_crwv_events,
        0  # Testing against "no effect"
    )
    
    # Results:
    # t_statistic = -3.45
    # p_value = 0.003
    
    # INTERPRETATION:
    # p_value = 0.003 means:
    # "Only 0.3% chance this is random"
    # or "99.7% confident this is real"
    
    # Since p < 0.05 (our threshold), WE ACCEPT THE CORRELATION!
    
    if p_value < 0.05:
        print("✅ Correlation is REAL!")
        return {
            'relationship': 'CRWV → NEBIUS',
            'avg_impact': -0.032,
            'confidence': 0.997,
            'is_tradable': True
        }

# --- STEP 4: CALCULATE RISK METRICS ---

def calculate_risk_metrics_explained():
    """
    GOAL: Understand the risk/reward of trading on these correlations
    
    KEY METRICS:
    """
    
    # Example: NEBIUS returns during CRWV infrastructure events
    returns = [-0.032, -0.028, -0.041, -0.025, -0.035, -0.019, -0.038, ...]
    
    # 1. VaR (Value at Risk) - 95% confidence
    # "What's the maximum loss I'll see 95% of the time?"
    var_95 = np.percentile(returns, 5)  # 5th percentile
    # = -4.8%
    # Interpretation: "95% of the time, NEBIUS won't drop more than 4.8%"
    
    # 2. CVaR (Conditional VaR) - Expected Shortfall
    # "When things go bad (worst 5%), what's the average loss?"
    worst_5_percent = returns[returns <= var_95]
    cvar_95 = np.mean(worst_5_percent)
    # = -5.2%
    # Interpretation: "In bad scenarios, expect 5.2% loss"
    
    # 3. Sharpe Ratio - Risk-adjusted return
    mean_return = np.mean(returns)  # = -3.2%
    std_dev = np.std(returns)       # = 0.8%
    sharpe = mean_return / std_dev
    # = -3.2 / 0.8 = -4.0
    # Interpretation: "Consistent negative returns" (good for shorting!)
    
    # 4. Max Drawdown - Worst single event
    max_loss = np.min(returns)  # = -5.8%
    # Interpretation: "Worst case we've seen is 5.8% drop"
    
    return {
        'mean_return': -3.2,
        'var_95': -4.8,
        'cvar_95': -5.2,
        'sharpe_ratio': -4.0,
        'max_drawdown': -5.8,
        'trading_strategy': 'Short NEBIUS when CRWV has infrastructure issues'
    }

# ============================================================================
# PART 4: PUTTING IT ALL TOGETHER - REAL TRADING WORKFLOW
# ============================================================================

def real_world_trading_example():
    """
    SCENARIO: It's 10am on a Tuesday. Breaking news:
    "CoreWeave reports GPU cluster outage affecting 30% of capacity"
    
    YOUR SYSTEM IN ACTION:
    """
    
    # STEP 1: News hits your system
    new_event = {
        'ticker': 'CRWV',
        'event_type': 'infrastructure_failure',
        'sentiment': -0.8,
        'timestamp': '2024-11-15 10:00:00'
    }
    
    # STEP 2: System looks up historical patterns
    predictions = analyzer.predict_impact(new_event)
    
    # Results:
    predictions = {
        'CRWV': {
            'predicted_return': -0.068,  # -6.8%
            'confidence': 0.92,
            'type': 'direct'
        },
        'NEBIUS': {
            'predicted_return': -0.032,  # -3.2%
            'confidence': 0.89,
            'type': 'indirect'  # ← HIDDEN CORRELATION!
        },
        'AMD': {
            'predicted_return': -0.011,
            'confidence': 0.67,
            'type': 'indirect'
        }
    }
    
    # STEP 3: Trading decision tree
    for ticker, pred in predictions.items():
        if pred['confidence'] > 0.85 and abs(pred['predicted_return']) > 0.02:
            # High confidence + meaningful magnitude
            
            if pred['predicted_return'] < 0:
                action = f"Consider shorting {ticker} or buying puts"
                reasoning = f"Expected {pred['predicted_return']:.1%} drop with {pred['confidence']:.0%} confidence"
            else:
                action = f"Consider going long {ticker} or selling puts"
                reasoning = f"Expected {pred['predicted_return']:.1%} rise with {pred['confidence']:.0%} confidence"
            
            print(f"🚨 TRADING SIGNAL: {ticker}")
            print(f"   Action: {action}")
            print(f"   Reasoning: {reasoning}")
    
    # OUTPUT:
    """
    🚨 TRADING SIGNAL: CRWV
       Action: Consider shorting CRWV or buying puts
       Reasoning: Expected -6.8% drop with 92% confidence
       
    🚨 TRADING SIGNAL: NEBIUS
       Action: Consider shorting NEBIUS or buying puts
       Reasoning: Expected -3.2% drop with 89% confidence
    """
    
    # THE EDGE:
    # Most traders only see CRWV news → Short CRWV
    # YOU see: CRWV news → Short CRWV AND NEBIUS
    # NEBIUS still trading near highs because nobody made the connection!

# ============================================================================
# PART 5: COMMON MISTAKES AND HOW TO AVOID THEM
# ============================================================================

def common_mistakes():
    """
    MISTAKE 1: Trading on weak correlations
    """
    # ❌ BAD
    if p_value < 0.15:  # Too lenient!
        trade_on_this()
    
    # ✅ GOOD
    if p_value < 0.05 and sample_size > 10 and abs(avg_impact) > 0.02:
        trade_on_this()
    
    """
    MISTAKE 2: Ignoring sample size
    """
    # ❌ BAD: Found correlation with only 3 events
    correlations = analyze_with_3_events()  # Not enough data!
    
    # ✅ GOOD: Require minimum 10-20 events
    if sample_size >= 10:
        correlations = analyze_correlations()
    
    """
    MISTAKE 3: Not testing out-of-sample
    """
    # ❌ BAD: Use all data for training
    correlations = find_correlations(all_data)
    accuracy = test_predictions(all_data)  # Overfitting!
    
    # ✅ GOOD: Split data
    train_data = data[:80%]
    test_data = data[80%:]
    correlations = find_correlations(train_data)
    accuracy = test_predictions(test_data)  # Real accuracy
    
    """
    MISTAKE 4: Confusing correlation with causation
    """
    # Just because two stocks move together doesn't mean one causes the other!
    
    # ❌ BAD: "Ice cream sales correlate with drownings → ice cream causes drowning"
    # ✅ GOOD: "Both caused by summer weather (confounding variable)"
    
    # ALWAYS ask: "WHY would these be related?"
    # CRWV → NEBIUS makes sense (both GPU infrastructure)
    # CRWV → RKLB doesn't make sense (rockets unrelated to GPUs)

# ============================================================================
# PART 6: YOUR LEARNING CHECKLIST
# ============================================================================

"""
WEEK 1-2: CONCEPTUAL UNDERSTANDING
□ Understand what event studies measure
□ Know the difference between correlation and causation
□ Grasp why statistical testing matters
□ Understand p-values and confidence levels

WEEK 3-4: CODE COMPREHENSION
□ Trace through calculate_price_impact() line by line
□ Understand how correlation matrix is built
□ Know how t-tests validate correlations
□ Grasp the predict_impact() logic

WEEK 5-6: HANDS-ON IMPLEMENTATION
□ Run tutorial.py and understand each step
□ Modify code to add new features
□ Test with your own data
□ Calculate risk metrics manually to verify

WEEK 7-8: ADVANCED APPLICATION
□ Find at least 3 significant correlations in your data
□ Explain WHY each correlation exists economically
□ Calculate all risk metrics (VaR, CVaR, Sharpe)
□ Build a simple trading strategy based on findings
□ Track prediction accuracy
"""

# ============================================================================
# PART 7: MATHEMATICAL CONCEPTS SIMPLIFIED
# ============================================================================

def math_concepts_explained():
    """
    CONCEPT 1: P-VALUE
    """
    # Imagine flipping a coin 10 times and getting 8 heads
    # Is the coin unfair, or just luck?
    
    # P-value answers: "What's the probability of seeing this result 
    # if the coin is actually fair?"
    
    # p = 0.5 (50%) → Could easily be luck, coin is probably fair
    # p = 0.05 (5%) → Very unlikely to be luck, coin might be unfair
    # p = 0.001 (0.1%) → Almost certainly unfair!
    
    # Same for correlations:
    # p = 0.003 → Only 0.3% chance this correlation is random
    
    """
    CONCEPT 2: CORRELATION vs CAUSATION
    """
    # CORRELATION: Two things move together
    # CAUSATION: One thing CAUSES the other to move
    
    # Example:
    # - Ice cream sales and drowning deaths are correlated
    # - But ice cream doesn't cause drowning!
    # - Both caused by summer weather (third factor)
    
    # In trading:
    # - CRWV and NEBIUS correlated? Yes
    # - Does CRWV news CAUSE NEBIUS to drop? Probably yes (same sector)
    # - Look for economic logic to support causation
    
    """
    CONCEPT 3: STANDARD DEVIATION
    """
    # Measures "how spread out" data is
    
    returns = [-2%, -1%, 0%, +1%, +2%]  # Low std dev (predictable)
    returns = [-10%, -5%, 0%, +5%, +10%]  # High std dev (volatile)
    
    # In trading:
    # Low std dev = Consistent (good for strategies)
    # High std dev = Unpredictable (risky)
    
    """
    CONCEPT 4: SHARPE RATIO
    """
    # Risk-adjusted return
    # Formula: (Average Return) / (Standard Deviation)
    
    # Strategy A: 10% return, 5% volatility → Sharpe = 2.0 (GREAT!)
    # Strategy B: 10% return, 20% volatility → Sharpe = 0.5 (MEH)
    
    # Same return, but A is much better because it's more consistent

# ============================================================================
# PART 8: DEBUGGING GUIDE
# ============================================================================

"""
PROBLEM: "I found 100 correlations!"

DIAGNOSIS: You're probably getting false positives

FIX:
1. Check p-values - all should be < 0.05
2. Check sample sizes - need at least 10 events each
3. Apply Bonferroni correction for multiple testing
4. Verify correlations make economic sense

---

PROBLEM: "No correlations found"

DIAGNOSIS: Insufficient data or wrong time windows

FIX:
1. Check if you have 6+ months of data
2. Verify events and prices are aligned (same dates)
3. Try different time windows (1h, 4h, 24h)
4. Ensure event types have 20+ samples each

---

PROBLEM: "Predictions are random"

DIAGNOSIS: Overfitting or insufficient validation

FIX:
1. Split data into train/test sets
2. Increase sample sizes
3. Check if correlations still exist in recent data
4. Add more features (sentiment, volume, etc.)

---

PROBLEM: "Can't explain why correlation exists"

DIAGNOSIS: Might be spurious correlation

FIX:
1. Research the companies - what connects them?
2. Check if there's a third factor (sector trends)
3. If no economic logic exists, DON'T trade on it
4. Correlations without causation break down over time
"""

# ============================================================================
# FINAL THOUGHTS
# ============================================================================

"""
REMEMBER:

1. This is REAL quantitative analysis
   - Same methodology banks use for market risk
   - Same stats that academic researchers publish
   - Same approach systematic traders employ

2. Focus on UNDERSTANDING, not just running code
   - Why does this correlation exist?
   - Is the sample size large enough?
   - Could this be spurious?
   - What's my confidence level?

3. Start small, think critically
   - Test one correlation thoroughly
   - Calculate metrics manually to verify
   - Paper trade before risking real money
   - Track accuracy religiously

4. This is portfolio-quality work
   - Graduate schools look for rigorous analysis
   - Employers want statistical thinking
   - Demonstrates both technical and domain skills
   - Shows you can work on ambiguous problems

GOOD LUCK! 🚀

You're not just learning to code.
You're learning to think like a quantitative analyst.
That skill is worth more than any single trading strategy.
"""

if __name__ == "__main__":
    print(__doc__)
    print("\n" + "="*70)
    print("Read through each section carefully.")
    print("Then run tutorial.py to see the code in action.")
    print("="*70)
