"""
COMPLETE WORKING EXAMPLE
Run this file to see the entire system in action with sample data

This demonstrates the full workflow from data loading to predictions.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys

# Import our modules
from correlation_engine import EventImpactAnalyzer
from statistical_analysis import CorrelationValidator, RiskMetricsCalculator
from data_loader import DataPipeline


def generate_realistic_sample_data():
    """
    Generate realistic sample data for demonstration
    This mimics real market behavior with correlations
    """
    print("📊 Generating sample data...")
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # ========== GENERATE PRICE DATA ==========
    
    # 6 months of daily data
    dates = pd.date_range(start='2024-05-01', end='2024-11-01', freq='D')
    tickers = ['CRWV', 'NEBIUS', 'AMD', 'HOOD', 'SPY']  # SPY = market benchmark
    
    prices_data = []
    
    # Base prices and correlations
    base_prices = {
        'CRWV': 45.0,
        'NEBIUS': 32.0,
        'AMD': 150.0,
        'HOOD': 18.0,
        'SPY': 450.0
    }
    
    # Create correlated random walks
    # CRWV and NEBIUS will be highly correlated (same sector)
    market_returns = np.random.normal(0.0005, 0.015, len(dates))
    
    for ticker in tickers:
        price = base_prices[ticker]
        
        # Different beta to market
        if ticker == 'CRWV':
            beta = 1.5  # More volatile than market
            correlation_with_market = 0.7
        elif ticker == 'NEBIUS':
            beta = 1.4  # Similar to CRWV (same sector)
            correlation_with_market = 0.68
        elif ticker == 'AMD':
            beta = 1.2
            correlation_with_market = 0.65
        elif ticker == 'HOOD':
            beta = 1.3
            correlation_with_market = 0.6
        else:  # SPY
            beta = 1.0
            correlation_with_market = 1.0
        
        for i, date in enumerate(dates):
            # Mix of market return + idiosyncratic return
            market_component = market_returns[i] * beta
            idiosyncratic = np.random.normal(0, 0.01 * (1 - correlation_with_market))
            daily_return = market_component + idiosyncratic
            
            price = price * (1 + daily_return)
            
            # OHLCV data
            daily_vol = np.random.uniform(0.015, 0.025)
            open_price = price * (1 + np.random.uniform(-daily_vol, daily_vol))
            high_price = max(price, open_price) * (1 + np.random.uniform(0, daily_vol))
            low_price = min(price, open_price) * (1 - np.random.uniform(0, daily_vol))
            
            volume = int(np.random.uniform(1_000_000, 5_000_000))
            
            prices_data.append({
                'date': date,
                'ticker': ticker,
                'open': round(open_price, 2),
                'high': round(high_price, 2),
                'low': round(low_price, 2),
                'close': round(price, 2),
                'volume': volume
            })
    
    prices_df = pd.DataFrame(prices_data)
    
    # ========== GENERATE EVENT DATA ==========
    
    # Create events that cause realistic price movements
    events_data = []
    
    # CRWV infrastructure events (will impact NEBIUS too)
    crwv_event_dates = pd.date_range(start='2024-05-15', end='2024-10-15', freq='20D')
    
    for event_date in crwv_event_dates:
        events_data.append({
            'date': event_date,
            'ticker': 'CRWV',
            'event_type': 'infrastructure_failure',
            'headline': 'CoreWeave reports GPU cluster issues',
            'sentiment': np.random.uniform(-0.9, -0.6),
            'entities': str(['CoreWeave', 'GPU', 'infrastructure']),
            'source': 'NewsAPI'
        })
        
        # Inject correlated price drops for CRWV and NEBIUS
        # Find the next trading day
        next_days = prices_df[
            (prices_df['date'] > event_date) & 
            (prices_df['date'] <= event_date + timedelta(days=3))
        ]['date'].unique()
        
        if len(next_days) > 0:
            impact_date = next_days[0]
            
            # CRWV drops 5-8%
            crwv_drop = np.random.uniform(0.05, 0.08)
            prices_df.loc[
                (prices_df['ticker'] == 'CRWV') & (prices_df['date'] == impact_date),
                'close'
            ] *= (1 - crwv_drop)
            
            # NEBIUS drops 2-4% (indirect effect)
            nebius_drop = np.random.uniform(0.02, 0.04)
            prices_df.loc[
                (prices_df['ticker'] == 'NEBIUS') & (prices_df['date'] == impact_date),
                'close'
            ] *= (1 - nebius_drop)
            
            # Volume surges
            prices_df.loc[
                (prices_df['ticker'].isin(['CRWV', 'NEBIUS'])) & 
                (prices_df['date'] == impact_date),
                'volume'
            ] *= np.random.uniform(2.0, 3.5)
    
    # HOOD earnings events (mostly positive)
    hood_event_dates = pd.date_range(start='2024-06-01', end='2024-10-01', freq='45D')
    
    for event_date in hood_event_dates:
        events_data.append({
            'date': event_date,
            'ticker': 'HOOD',
            'event_type': 'earnings',
            'headline': 'Robinhood reports quarterly earnings',
            'sentiment': np.random.uniform(0.5, 0.9),
            'entities': str(['Robinhood', 'earnings']),
            'source': 'AlphaVantage'
        })
        
        next_days = prices_df[
            (prices_df['date'] > event_date) & 
            (prices_df['date'] <= event_date + timedelta(days=3))
        ]['date'].unique()
        
        if len(next_days) > 0:
            impact_date = next_days[0]
            
            # HOOD usually beats, +3-6%
            hood_gain = np.random.uniform(0.03, 0.06)
            prices_df.loc[
                (prices_df['ticker'] == 'HOOD') & (prices_df['date'] == impact_date),
                'close'
            ] *= (1 + hood_gain)
    
    # AMD product launches
    amd_event_dates = pd.date_range(start='2024-06-15', end='2024-09-15', freq='30D')
    
    for event_date in amd_event_dates:
        events_data.append({
            'date': event_date,
            'ticker': 'AMD',
            'event_type': 'product_launch',
            'headline': 'AMD announces new chip architecture',
            'sentiment': np.random.uniform(0.6, 0.8),
            'entities': str(['AMD', 'chip', 'technology']),
            'source': 'TechCrunch'
        })
        
        next_days = prices_df[
            (prices_df['date'] > event_date) & 
            (prices_df['date'] <= event_date + timedelta(days=3))
        ]['date'].unique()
        
        if len(next_days) > 0:
            impact_date = next_days[0]
            
            # AMD gains 2-4%
            amd_gain = np.random.uniform(0.02, 0.04)
            prices_df.loc[
                (prices_df['ticker'] == 'AMD') & (prices_df['date'] == impact_date),
                'close'
            ] *= (1 + amd_gain)
    
    events_df = pd.DataFrame(events_data)
    
    print(f"✓ Generated {len(prices_df)} price records")
    print(f"✓ Generated {len(events_df)} events")
    print(f"✓ Date range: {prices_df['date'].min()} to {prices_df['date'].max()}")
    
    return events_df, prices_df


def run_complete_analysis():
    """
    Run the complete correlation analysis workflow
    """
    
    print("\n" + "="*70)
    print("COMPLETE WORKING EXAMPLE - CORRELATION ANALYSIS")
    print("="*70 + "\n")
    
    # ========== STEP 1: GENERATE DATA ==========
    
    events_df, prices_df = generate_realistic_sample_data()
    
    input("\n▶ Press Enter to continue to analysis...\n")
    
    # ========== STEP 2: INITIALIZE ANALYZER ==========
    
    print("🔧 Initializing correlation engine...")
    analyzer = EventImpactAnalyzer(lookback_days=180)
    analyzer.load_historical_data(events_df, prices_df)
    print("✓ Loaded data into analyzer\n")
    
    # ========== STEP 3: BUILD CORRELATION NETWORK ==========
    
    print("=" * 70)
    print("STEP 1: BUILDING CORRELATION NETWORK")
    print("=" * 70 + "\n")
    
    correlation_matrix = analyzer.build_correlation_network()
    
    if len(correlation_matrix) > 0:
        print("\n📊 Correlation Matrix (how tickers move together):")
        print(correlation_matrix.round(2))
        print("\nNote: Values close to 1.0 = strong positive correlation")
        print("      Values close to -1.0 = strong negative correlation")
    
    input("\n▶ Press Enter to detect indirect correlations...\n")
    
    # ========== STEP 4: DETECT INDIRECT CORRELATIONS ==========
    
    print("=" * 70)
    print("STEP 2: DETECTING INDIRECT CORRELATIONS")
    print("=" * 70 + "\n")
    
    indirect_correlations = analyzer.detect_indirect_correlations(significance_level=0.05)
    
    if len(indirect_correlations) > 0:
        print(f"\n🎯 DISCOVERED {len(indirect_correlations)} SIGNIFICANT RELATIONSHIPS:\n")
        
        # Sort by impact magnitude
        sorted_corr = sorted(
            indirect_correlations.items(),
            key=lambda x: abs(x[1]['avg_impact']),
            reverse=True
        )
        
        for i, (relationship, stats) in enumerate(sorted_corr, 1):
            print(f"{i}. {relationship}")
            print(f"   Average Impact: {stats['avg_impact']:+.2%}")
            print(f"   Standard Dev: {stats['std_dev']:.2%}")
            print(f"   P-value: {stats['p_value']:.4f}")
            print(f"   T-statistic: {stats['t_statistic']:.2f}")
            print(f"   Sample Size: {stats['sample_size']} events")
            print(f"   Confidence: {stats['confidence']:.1%}")
            print(f"   Effect Size (Cohen's d): {stats['effect_size']:.2f}")
            
            # Interpretation
            if stats['p_value'] < 0.01:
                sig_level = "VERY STRONG"
            elif stats['p_value'] < 0.05:
                sig_level = "STRONG"
            else:
                sig_level = "MODERATE"
            
            print(f"   ✓ {sig_level} evidence of relationship")
            
            # Trading implication
            event_ticker, affected_ticker = relationship.split(" → ")
            if stats['avg_impact'] < 0:
                print(f"   💡 When {event_ticker} has bad news, {affected_ticker} typically drops")
            else:
                print(f"   💡 When {event_ticker} has good news, {affected_ticker} typically rises")
            
            print()
    else:
        print("❌ No significant correlations found")
        print("   This could mean:")
        print("   - Not enough data (need 6+ months)")
        print("   - Events too sparse (need 20+ per type)")
        print("   - Correlations are weak in this dataset")
    
    input("\n▶ Press Enter to calculate risk metrics...\n")
    
    # ========== STEP 5: RISK ANALYSIS ==========
    
    print("=" * 70)
    print("STEP 3: RISK ANALYSIS")
    print("=" * 70 + "\n")
    
    # Analyze CRWV infrastructure events
    risk = analyzer.calculate_risk_metrics('CRWV', 'infrastructure_failure')
    
    print("📊 Risk Profile: CRWV during infrastructure_failure events\n")
    print(f"Return Statistics:")
    print(f"  Mean Return: {risk['mean_return']:+.2%}")
    print(f"  Volatility (Std Dev): {risk['volatility']:.2%}")
    print(f"  Best Case: {risk['max_gain']:+.2%}")
    print(f"  Worst Case: {risk['max_drawdown']:+.2%}")
    
    print(f"\nRisk Metrics:")
    print(f"  VaR (95%): {risk['var_95']:.2%}")
    print(f"    → 95% of the time, loss won't exceed this")
    print(f"  CVaR (95%): {risk['cvar_95']:.2%}")
    print(f"    → Average loss in worst 5% of cases")
    
    print(f"\nRisk-Adjusted Performance:")
    print(f"  Sharpe Ratio: {risk['sharpe_ratio']:.2f}")
    
    if risk['sharpe_ratio'] > 2:
        interpretation = "EXCELLENT"
    elif risk['sharpe_ratio'] > 1:
        interpretation = "GOOD"
    elif risk['sharpe_ratio'] > 0:
        interpretation = "ACCEPTABLE"
    else:
        interpretation = "POOR (negative returns)"
    
    print(f"    → {interpretation}")
    
    input("\n▶ Press Enter to test prediction on new event...\n")
    
    # ========== STEP 6: PREDICT NEW EVENT ==========
    
    print("=" * 70)
    print("STEP 4: PREDICTING NEW EVENT IMPACT")
    print("=" * 70 + "\n")
    
    # Simulate breaking news
    new_event = {
        'ticker': 'CRWV',
        'event_type': 'infrastructure_failure',
        'description': 'CoreWeave reports major GPU cluster outage',
        'sentiment': -0.85
    }
    
    print("🚨 BREAKING NEWS:")
    print(f"   {new_event['description']}")
    print(f"   Sentiment: {new_event['sentiment']:.2f} (very negative)")
    print(f"   Event Type: {new_event['event_type']}")
    
    print("\n🤖 Running prediction model...\n")
    
    predictions = analyzer.predict_impact(new_event)
    
    print("=" * 70)
    print("PREDICTED MARKET IMPACTS")
    print("=" * 70 + "\n")
    
    # Sort by magnitude
    sorted_predictions = sorted(
        predictions.items(),
        key=lambda x: abs(x[1]['predicted_return']),
        reverse=True
    )
    
    for ticker, pred in sorted_predictions:
        impact_type = "🎯 DIRECT" if pred['type'] == 'direct' else "🔗 INDIRECT"
        
        print(f"{impact_type}: {ticker}")
        print(f"  Predicted Return: {pred['predicted_return']:+.2%}")
        print(f"  Confidence: {pred['confidence']:.1%}")
        
        if 'risk_range' in pred:
            print(f"  Risk Range: {pred['risk_range'][0]:.2%} to {pred['risk_range'][1]:.2%}")
        
        # Trading recommendation
        if abs(pred['predicted_return']) > 0.02 and pred['confidence'] > 0.70:
            print(f"  ")
            if pred['predicted_return'] > 0:
                print(f"  💰 SIGNAL: Consider LONG position or sell puts")
            else:
                print(f"  ⚠️  SIGNAL: Consider SHORT position or buy puts")
        else:
            print(f"  ")
            print(f"  ⏸️  SIGNAL: WAIT (low confidence or small impact)")
        
        print()
    
    # ========== STEP 7: STATISTICAL VALIDATION ==========
    
    input("\n▶ Press Enter to see statistical validation...\n")
    
    print("=" * 70)
    print("STEP 5: STATISTICAL VALIDATION")
    print("=" * 70 + "\n")
    
    validator = CorrelationValidator(significance_level=0.05)
    
    if len(indirect_correlations) > 0:
        # Test the strongest correlation
        strongest = sorted_corr[0]
        relationship, stats = strongest
        
        print(f"Testing strongest correlation: {relationship}\n")
        
        # Get the returns
        event_ticker = relationship.split(" → ")[0]
        affected_ticker = relationship.split(" → ")[1]
        
        # Get returns data
        relevant_impacts = analyzer.detailed_impacts[
            (analyzer.detailed_impacts['event_ticker'] == event_ticker) &
            (analyzer.detailed_impacts['affected_ticker'] == affected_ticker)
        ]['return_24h'].values
        
        relevant_impacts = relevant_impacts[relevant_impacts != 0]
        
        if len(relevant_impacts) > 0:
            # Calculate risk metrics
            print("Risk Metrics for this correlation:")
            print(f"  VaR (95%): {RiskMetricsCalculator.calculate_var(relevant_impacts):.2%}")
            print(f"  CVaR (95%): {RiskMetricsCalculator.calculate_cvar(relevant_impacts):.2%}")
            print(f"  Sharpe Ratio: {RiskMetricsCalculator.calculate_sharpe_ratio(relevant_impacts):.2f}")
            
            print(f"\nMultiple Testing Adjustment:")
            # Get all p-values
            all_p_values = [s['p_value'] for _, s in indirect_correlations.items()]
            
            # Bonferroni correction
            bonferroni = validator.bonferroni_correction(all_p_values)
            
            # FDR correction
            fdr = validator.false_discovery_rate(all_p_values)
            
            print(f"  Original p-values: {len(all_p_values)}")
            print(f"  Significant after Bonferroni: {sum(bonferroni)}")
            print(f"  Significant after FDR: {sum(fdr)}")
            
            if stats['p_value'] < 0.05 / len(all_p_values):  # Bonferroni adjusted
                print(f"\n  ✅ This correlation survives Bonferroni correction")
                print(f"     Safe to trade on this relationship")
            else:
                print(f"\n  ⚠️  This correlation doesn't survive Bonferroni correction")
                print(f"     Use with caution, might be false positive")
    
    # ========== FINAL SUMMARY ==========
    
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE - SUMMARY")
    print("=" * 70 + "\n")
    
    print(f"📊 Data Analyzed:")
    print(f"   • {len(events_df)} events")
    print(f"   • {len(prices_df)} price records")
    print(f"   • {len(prices_df['ticker'].unique())} tickers")
    print(f"   • {(prices_df['date'].max() - prices_df['date'].min()).days} days\n")
    
    print(f"🔍 Discoveries:")
    print(f"   • {len(indirect_correlations)} significant correlations found")
    print(f"   • Confidence levels: 95%+\n")
    
    print(f"💼 Trading Applications:")
    print(f"   • Use predictions for position sizing")
    print(f"   • Hedge correlated positions")
    print(f"   • Find pairs trading opportunities")
    print(f"   • Understand systemic risk\n")
    
    print(f"📚 Next Steps:")
    print(f"   1. Run this with YOUR actual data")
    print(f"   2. Validate correlations make economic sense")
    print(f"   3. Paper trade your strategies")
    print(f"   4. Track prediction accuracy over time\n")
    
    print("="*70)
    print("This example demonstrates the complete workflow.")
    print("Now modify the code and experiment with your own data!")
    print("="*70 + "\n")


if __name__ == "__main__":
    try:
        run_complete_analysis()
    except KeyboardInterrupt:
        print("\n\nAnalysis interrupted by user.")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nDebug info:")
        import traceback
        traceback.print_exc()
        print("\nIf you see import errors, make sure all files are in the same directory.")
