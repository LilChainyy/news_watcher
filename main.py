"""
Main Execution Script
Orchestrates the full correlation analysis workflow
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json

# Import your modules
from data_loader import DataPipeline, DataLoader
from correlation_engine import EventImpactAnalyzer, SectorContagionAnalyzer, TimeSeriesAnalyzer
from statistical_analysis import CorrelationValidator, RiskMetricsCalculator, EventStudyAnalysis


class MarketIntelligenceEngine:
    """
    Main orchestrator that brings all components together
    """
    
    def __init__(self, watchlist: list):
        """
        Args:
            watchlist: List of ticker symbols to monitor
        """
        self.watchlist = watchlist
        self.data_pipeline = DataPipeline()
        self.event_analyzer = EventImpactAnalyzer()
        self.sector_analyzer = SectorContagionAnalyzer()
        self.validator = CorrelationValidator()
        
        # Storage for results
        self.correlation_matrix = None
        self.indirect_correlations = None
        self.risk_profiles = {}
        
    def initialize(self):
        """Load and prepare all historical data"""
        print("🚀 Initializing Market Intelligence Engine")
        print(f"📋 Watchlist: {', '.join(self.watchlist)}")
        print("-" * 60)
        
        # Run data pipeline
        self.events_df, self.prices_df = self.data_pipeline.run_pipeline()
        
        # Load into analyzers
        self.event_analyzer.load_historical_data(self.events_df, self.prices_df)
        
        print("\n✅ Initialization complete\n")
        
    def discover_correlations(self):
        """Run the correlation discovery process"""
        print("🔍 Phase 1: Discovering Market Correlations")
        print("-" * 60)
        
        # Build correlation network
        print("\n1️⃣  Building correlation matrix...")
        self.correlation_matrix = self.event_analyzer.build_correlation_network()
        print(f"   ✓ Analyzed {len(self.correlation_matrix)} relationships")
        
        # Detect indirect correlations (THE MONEY PART)
        print("\n2️⃣  Detecting indirect correlations...")
        self.indirect_correlations = self.event_analyzer.detect_indirect_correlations()
        
        if len(self.indirect_correlations) > 0:
            print(f"   ✓ Found {len(self.indirect_correlations)} significant indirect relationships")
            print("\n   Top Discoveries:")
            
            # Sort by impact magnitude
            sorted_corr = sorted(
                self.indirect_correlations.items(),
                key=lambda x: abs(x[1]['avg_impact']),
                reverse=True
            )
            
            for i, (relationship, stats) in enumerate(sorted_corr[:5], 1):
                print(f"   {i}. {relationship}")
                print(f"      → Avg Impact: {stats['avg_impact']:.2%}")
                print(f"      → Confidence: {stats['confidence']:.1%}")
                print(f"      → Sample Size: {stats['sample_size']} events")
        else:
            print("   ⚠️  No significant indirect correlations found")
            print("      (Need more historical data)")
        
        print("\n✅ Correlation discovery complete\n")
        
    def calculate_risk_profiles(self):
        """Calculate risk metrics for each ticker-event combination"""
        print("📊 Phase 2: Calculating Risk Profiles")
        print("-" * 60)
        
        event_types = self.events_df['event_type'].unique()
        
        for ticker in self.watchlist:
            print(f"\n📌 {ticker}:")
            self.risk_profiles[ticker] = {}
            
            for event_type in event_types:
                risk = self.event_analyzer.calculate_risk_metrics(ticker, event_type)
                
                if risk['mean_return'] != 0:  # Has data
                    self.risk_profiles[ticker][event_type] = risk
                    
                    print(f"   {event_type}:")
                    print(f"      Mean Return: {risk['mean_return']:+.2%}")
                    print(f"      Volatility: {risk['volatility']:.2%}")
                    print(f"      VaR(95%): {risk['var_95']:.2%}")
                    print(f"      Sharpe: {risk['sharpe_ratio']:.2f}")
        
        print("\n✅ Risk profiling complete\n")
        
    def test_new_event(self, event: dict):
        """
        Test what would happen with a new event
        This is your real-time prediction capability
        """
        print("🎯 Phase 3: Testing New Event Impact")
        print("-" * 60)
        print(f"\nEvent: {event['event_type']} for {event['ticker']}")
        print(f"Details: {event.get('description', 'N/A')}")
        
        # Get predictions
        predictions = self.event_analyzer.predict_impact(event)
        
        print("\n📈 Predicted Price Impacts:\n")
        
        # Sort by predicted magnitude
        sorted_predictions = sorted(
            predictions.items(),
            key=lambda x: abs(x[1]['predicted_return']),
            reverse=True
        )
        
        for ticker, pred in sorted_predictions:
            impact_type = "🎯 DIRECT" if pred['type'] == 'direct' else "🔗 INDIRECT"
            
            print(f"{impact_type} {ticker}:")
            print(f"   Predicted Return: {pred['predicted_return']:+.2%}")
            print(f"   Confidence: {pred['confidence']:.1%}")
            
            if 'risk_range' in pred:
                print(f"   Risk Range: {pred['risk_range'][0]:.2%} to {pred['risk_range'][1]:.2%}")
            
            # Trading suggestion
            if abs(pred['predicted_return']) > 0.02 and pred['confidence'] > 0.7:
                if pred['predicted_return'] > 0:
                    print(f"   💡 Trade Idea: Consider selling puts or going long")
                else:
                    print(f"   💡 Trade Idea: Avoid new positions, consider hedging")
            
            print()
        
        print("✅ Event analysis complete\n")
        
    def generate_report(self, output_file: str = "correlation_report.json"):
        """Generate comprehensive report of findings"""
        print("📝 Generating Correlation Report")
        print("-" * 60)
        
        report = {
            'generated_at': datetime.now().isoformat(),
            'watchlist': self.watchlist,
            'analysis_summary': {
                'total_events_analyzed': len(self.events_df),
                'date_range': {
                    'start': str(self.events_df['date'].min()),
                    'end': str(self.events_df['date'].max())
                },
                'indirect_correlations_found': len(self.indirect_correlations)
            },
            'indirect_correlations': self.indirect_correlations,
            'risk_profiles': self.risk_profiles
        }
        
        # Save to JSON
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"✅ Report saved to {output_file}\n")
        
        return report
    
    def run_full_analysis(self):
        """Execute complete analysis workflow"""
        print("\n" + "="*60)
        print("MARKET INTELLIGENCE ENGINE - FULL ANALYSIS")
        print("="*60 + "\n")
        
        # Phase 1: Initialize
        self.initialize()
        
        # Phase 2: Discover correlations
        self.discover_correlations()
        
        # Phase 3: Risk profiling
        self.calculate_risk_profiles()
        
        # Phase 4: Generate report
        report = self.generate_report()
        
        print("="*60)
        print("ANALYSIS COMPLETE")
        print("="*60)
        
        return report


def example_workflow():
    """
    Example of how to use the system
    Replace this with your actual watchlist and data
    """
    
    # Define your watchlist
    my_watchlist = ['HOOD', 'AMD', 'NEBIUS', 'CRWV', 'RKLB']
    
    # Initialize engine
    engine = MarketIntelligenceEngine(watchlist=my_watchlist)
    
    # Run full analysis
    report = engine.run_full_analysis()
    
    # Test a new event
    print("\n" + "="*60)
    print("TESTING NEW EVENT SCENARIO")
    print("="*60 + "\n")
    
    new_event = {
        'ticker': 'CRWV',
        'event_type': 'infrastructure_failure',
        'description': 'CoreWeave experiences GPU shortage',
        'sentiment': -0.7
    }
    
    engine.test_new_event(new_event)
    
    # Show key findings
    print("\n" + "="*60)
    print("KEY FINDINGS SUMMARY")
    print("="*60 + "\n")
    
    if engine.indirect_correlations:
        print("🎯 Hidden Market Relationships Discovered:")
        for relationship, stats in list(engine.indirect_correlations.items())[:3]:
            print(f"\n   {relationship}")
            print(f"   When this event happens, expect {stats['avg_impact']:+.2%} move")
            print(f"   Confidence: {stats['confidence']:.1%}")
    
    print("\n💡 Next Steps:")
    print("   1. Review correlation_report.json for detailed findings")
    print("   2. Integrate into your dashboard (feed to software engineer)")
    print("   3. Use predictions for trade planning")
    print("   4. Update model weekly with new data")


def quick_test():
    """
    Quick test to verify everything works
    Use this for debugging
    """
    print("🧪 Running Quick Test")
    print("-" * 60)
    
    # Create minimal test data
    events_df = pd.DataFrame({
        'date': pd.date_range('2024-10-01', periods=10, freq='D'),
        'ticker': ['HOOD'] * 10,
        'event_type': ['earnings'] * 10,
        'sentiment': np.random.uniform(-1, 1, 10)
    })
    
    prices_df = pd.DataFrame({
        'date': pd.date_range('2024-09-01', periods=60, freq='D'),
        'ticker': ['HOOD'] * 60,
        'close': 100 * (1 + np.random.randn(60).cumsum() * 0.02),
        'volume': np.random.randint(1e6, 10e6, 60)
    })
    
    # Initialize analyzer
    analyzer = EventImpactAnalyzer()
    analyzer.load_historical_data(events_df, prices_df)
    
    # Test basic functionality
    print("✓ Data loaded")
    
    # Calculate risk metrics
    risk = analyzer.calculate_risk_metrics('HOOD', 'earnings')
    print(f"✓ Risk calculation works: VaR = {risk['var_95']:.2%}")
    
    print("\n✅ All systems operational!")


if __name__ == "__main__":
    import sys
    
    print("\n")
    print("╔═══════════════════════════════════════════════════════╗")
    print("║   MARKET INTELLIGENCE ENGINE - CORRELATION ANALYSIS   ║")
    print("╚═══════════════════════════════════════════════════════╝")
    print("\nSelect mode:")
    print("  1. Run full analysis (requires real data)")
    print("  2. Run quick test (uses sample data)")
    print("  3. Just show me the code structure")
    
    choice = input("\nEnter choice (1-3): ").strip()
    
    if choice == "1":
        print("\n⚠️  Running full analysis requires real data in ./data/ directory")
        confirm = input("Continue? (y/n): ").strip().lower()
        if confirm == 'y':
            example_workflow()
    elif choice == "2":
        quick_test()
    else:
        print("\n📁 Code Structure:")
        print("""
        correlation_engine.py      → Core analysis logic (EVENT STUDY, CORRELATIONS)
        statistical_analysis.py    → Stats utilities (HYPOTHESIS TESTING, RISK METRICS)
        data_loader.py            → Data pipeline (LOADING, FEATURES, VALIDATION)
        main.py                   → This file (ORCHESTRATION)
        
        YOUR WORKFLOW:
        1. Get historical event + price data (CSV format)
        2. Run: python main.py
        3. Review correlation_report.json
        4. Use predictions in your dashboard
        
        YOUR FOCUS (the quant parts):
        ✓ correlation_engine.py    - Fill in TODO methods
        ✓ statistical_analysis.py  - Understand each test
        ✓ Risk calculations        - VaR, CVaR, Sharpe, etc.
        
        DELEGATE TO ENGINEER:
        → Data collection APIs (Layer 1)
        → Dashboard frontend (Layer 4)
        → Production deployment
        """)
    
    print("\n🎓 Learning Value for Quant/Risk Roles:")
    print("   • Event study methodology (used by every quant desk)")
    print("   • Statistical hypothesis testing (interview favorite)")
    print("   • Correlation vs causation (critical thinking)")
    print("   • Risk metrics calculation (VaR, CVaR, Sharpe)")
    print("   • Time-series analysis (bread and butter)")
    print("\n💼 This project directly translates to:")
    print("   • Market Risk Analyst roles")
    print("   • Quantitative Research positions")
    print("   • Data-driven trading strategies")
    print("\n")
