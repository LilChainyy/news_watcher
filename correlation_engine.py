"""
Correlation Engine - Event Impact Analysis
Your core quant work: Learning hidden relationships between news events and price movements
"""

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.cluster import KMeans
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')


class EventImpactAnalyzer:
    """
    Analyzes historical correlations between news events and stock price movements
    This is your event study methodology implementation
    """
    
    def __init__(self, lookback_days: int = 180):
        """
        Args:
            lookback_days: How many days of history to analyze (default 6 months)
        """
        self.lookback_days = lookback_days
        self.event_history = pd.DataFrame()
        self.price_history = pd.DataFrame()
        self.correlation_matrix = None
        self.significant_correlations = {}
        self.detailed_impacts = pd.DataFrame()  # Store detailed impact analysis
        
    def load_historical_data(self, events_df: pd.DataFrame, prices_df: pd.DataFrame):
        """
        Load historical news events and price data
        
        Args:
            events_df: DataFrame with columns ['date', 'ticker', 'event_type', 'sentiment', 'entities']
            prices_df: DataFrame with columns ['date', 'ticker', 'close', 'volume']
        """
        self.event_history = events_df
        self.price_history = prices_df
        
    def calculate_price_impact(self, event_date: datetime, ticker: str, 
                              window_hours: int = 24) -> Dict[str, float]:
        """
        Calculate price movement following an event
        This is your basic event study calculation
        
        IMPLEMENTATION NOTES:
        - Gets price data before and after event
        - Calculates returns at multiple time horizons
        - Measures volume changes (surge often precedes big moves)
        - Tracks volatility changes
        
        Returns:
            Dict with 'return_1h', 'return_4h', 'return_24h', 'volume_change'
        """
        impact = {
            'return_1h': 0.0,
            'return_4h': 0.0, 
            'return_24h': 0.0,
            'volume_change': 0.0,
            'volatility_change': 0.0
        }
        
        # Filter price data for this ticker
        ticker_prices = self.price_history[self.price_history['ticker'] == ticker].copy()
        
        if len(ticker_prices) == 0:
            return impact
        
        # Ensure date column is datetime
        if not pd.api.types.is_datetime64_any_dtype(ticker_prices['date']):
            ticker_prices['date'] = pd.to_datetime(ticker_prices['date'])
        
        ticker_prices = ticker_prices.sort_values('date')
        
        # Get price just before event
        pre_event = ticker_prices[ticker_prices['date'] <= event_date]
        if len(pre_event) == 0:
            return impact
        
        pre_event_price = pre_event.iloc[-1]['close']
        pre_event_volume = pre_event.iloc[-1]['volume']
        
        # Calculate pre-event volatility (5-day rolling std of returns)
        if len(pre_event) >= 6:
            pre_returns = pre_event['close'].pct_change().tail(5)
            pre_volatility = pre_returns.std()
        else:
            pre_volatility = 0
        
        # Get prices at different time windows after event
        # For daily data, we approximate hours with days
        time_windows = {
            'return_1h': timedelta(hours=4),   # Intraday (use 4 hours for daily data)
            'return_4h': timedelta(days=1),    # Next day
            'return_24h': timedelta(days=1)    # 1 trading day
        }
        
        for key, window in time_windows.items():
            target_date = event_date + window
            post_event = ticker_prices[ticker_prices['date'] >= target_date]
            
            if len(post_event) > 0:
                post_event_price = post_event.iloc[0]['close']
                # Calculate return
                impact[key] = (post_event_price - pre_event_price) / pre_event_price
        
        # Volume change (comparing to pre-event average)
        post_event_1d = ticker_prices[
            (ticker_prices['date'] > event_date) & 
            (ticker_prices['date'] <= event_date + timedelta(days=1))
        ]
        
        if len(post_event_1d) > 0:
            post_event_volume = post_event_1d.iloc[0]['volume']
            impact['volume_change'] = (post_event_volume - pre_event_volume) / pre_event_volume
            
            # Volatility change
            post_event_5d = ticker_prices[
                (ticker_prices['date'] > event_date) & 
                (ticker_prices['date'] <= event_date + timedelta(days=5))
            ]
            
            if len(post_event_5d) >= 2:
                post_returns = post_event_5d['close'].pct_change().dropna()
                post_volatility = post_returns.std()
                
                if pre_volatility > 0:
                    impact['volatility_change'] = (post_volatility - pre_volatility) / pre_volatility
        
        return impact
    
    def build_correlation_network(self) -> pd.DataFrame:
        """
        Build correlation matrix between different types of events and affected tickers
        This is where you discover hidden relationships like CoreWeave → Nebius
        
        IMPLEMENTATION NOTES:
        - For each event, calculate impact on ALL tickers (not just the primary one)
        - This reveals indirect correlations
        - Build a matrix of event-type x ticker impacts
        - Calculate correlations across this matrix
        
        Returns:
            DataFrame: Correlation matrix of event impacts
        """
        print("Building event correlation network...")
        
        # Step 1: Calculate all event impacts
        impact_matrix = []
        
        # Get unique tickers from price history
        all_tickers = self.price_history['ticker'].unique()
        
        for idx, event in self.event_history.iterrows():
            event_date = pd.to_datetime(event['date'])
            primary_ticker = event['ticker']
            event_type = event['event_type']
            
            # Calculate impact on ALL tickers in watchlist (this finds hidden correlations)
            for ticker in all_tickers:
                impact = self.calculate_price_impact(event_date, ticker)
                
                # Only record if we have valid data
                if impact['return_24h'] != 0.0 or len(impact_matrix) == 0:
                    impact_matrix.append({
                        'event_id': idx,
                        'event_date': event_date,
                        'event_ticker': primary_ticker,
                        'event_type': event_type,
                        'affected_ticker': ticker,
                        'return_1h': impact['return_1h'],
                        'return_4h': impact['return_4h'],
                        'return_24h': impact['return_24h'],
                        'volume_change': impact['volume_change'],
                        'is_direct': ticker == primary_ticker
                    })
        
        if len(impact_matrix) == 0:
            print("Warning: No impact data collected. Check data alignment.")
            return pd.DataFrame()
        
        impact_df = pd.DataFrame(impact_matrix)
        
        # Step 2: Calculate correlation matrix
        # Create a pivot table: rows = events, columns = tickers, values = returns
        
        # Create unique event identifier
        impact_df['event_key'] = (
            impact_df['event_ticker'] + '_' + 
            impact_df['event_type'] + '_' + 
            impact_df['event_date'].astype(str)
        )
        
        # Pivot to get event x ticker matrix of returns
        correlation_pivot = impact_df.pivot_table(
            index='event_key',
            columns='affected_ticker',
            values='return_24h',
            aggfunc='mean'
        )
        
        # Calculate correlation matrix between tickers
        # This shows which tickers tend to move together when events happen
        if len(correlation_pivot) > 1:
            self.correlation_matrix = correlation_pivot.corr()
        else:
            self.correlation_matrix = pd.DataFrame()
        
        print(f"✓ Analyzed {len(impact_df)} event-ticker pairs")
        print(f"✓ Found {len(all_tickers)} tickers with price data")
        
        # Store the detailed impact data for later analysis
        self.detailed_impacts = impact_df
        
        return self.correlation_matrix
    
    def detect_indirect_correlations(self, significance_level: float = 0.05) -> Dict:
        """
        Find statistically significant INDIRECT correlations
        Example: CoreWeave news → Nebius price movement (even when Nebius not mentioned)
        
        IMPLEMENTATION NOTES:
        - For each ticker's events, measure impact on OTHER tickers
        - Use statistical testing to filter out noise
        - Only report relationships with p-value < significance_level
        - Require minimum sample size for reliability
        
        This is advanced correlation detection - your money skill
        """
        print("Detecting indirect market relationships...")
        
        if not hasattr(self, 'detailed_impacts') or len(self.detailed_impacts) == 0:
            print("Warning: Must run build_correlation_network() first")
            return {}
        
        indirect_correlations = {}
        
        # Group by event ticker to analyze cross-ticker impacts
        event_tickers = self.detailed_impacts['event_ticker'].unique()
        
        for event_ticker in event_tickers:
            # Get all events for this ticker
            ticker_events = self.detailed_impacts[
                self.detailed_impacts['event_ticker'] == event_ticker
            ]
            
            # Look at impacts on OTHER tickers (indirect relationships)
            affected_tickers = ticker_events['affected_ticker'].unique()
            
            for other_ticker in affected_tickers:
                if other_ticker == event_ticker:
                    continue  # Skip direct relationships
                
                # Get all returns for this ticker when event_ticker has news
                cross_impacts = ticker_events[
                    ticker_events['affected_ticker'] == other_ticker
                ]['return_24h'].values
                
                # Filter out zeros (missing data)
                cross_impacts = cross_impacts[cross_impacts != 0]
                
                if len(cross_impacts) < 5:  # Need minimum sample size
                    continue
                
                # Statistical testing: Is mean return significantly different from 0?
                from scipy import stats
                
                # One-sample t-test against 0
                t_stat, p_value = stats.ttest_1samp(cross_impacts, 0)
                
                # Check if significant and has meaningful magnitude
                avg_impact = np.mean(cross_impacts)
                
                if p_value < significance_level and abs(avg_impact) > 0.005:  # >0.5% avg impact
                    
                    relationship_key = f"{event_ticker} → {other_ticker}"
                    
                    # Calculate additional statistics
                    std_dev = np.std(cross_impacts)
                    confidence = 1 - p_value
                    
                    # Effect size (Cohen's d)
                    cohens_d = avg_impact / std_dev if std_dev > 0 else 0
                    
                    indirect_correlations[relationship_key] = {
                        'avg_impact': avg_impact,
                        'std_dev': std_dev,
                        'p_value': p_value,
                        't_statistic': t_stat,
                        'sample_size': len(cross_impacts),
                        'confidence': confidence,
                        'effect_size': cohens_d,
                        'min_impact': np.min(cross_impacts),
                        'max_impact': np.max(cross_impacts),
                        'median_impact': np.median(cross_impacts)
                    }
        
        print(f"✓ Found {len(indirect_correlations)} significant indirect relationships")
        
        self.significant_correlations = indirect_correlations
        return indirect_correlations
    
    def cluster_event_types(self, n_clusters: int = 5) -> Dict:
        """
        Group similar events together based on their market impact patterns
        Example: 'GPU shortage', 'chip delay', 'supply chain issue' → same cluster
        
        IMPLEMENTATION NOTES:
        - Creates feature vectors for each event (impact profile across tickers)
        - Uses K-means clustering to group similar events
        - Helps identify event categories with similar market reactions
        
        Uses K-means clustering on impact vectors
        """
        print(f"Clustering events into {n_clusters} groups...")
        
        if not hasattr(self, 'detailed_impacts') or len(self.detailed_impacts) == 0:
            print("Warning: Must run build_correlation_network() first")
            return {}
        
        # Step 1: Create feature vectors for each event
        # Each event is represented by its impact across all tickers
        event_features = []
        event_labels = []
        
        for event_id in self.detailed_impacts['event_id'].unique():
            event_data = self.detailed_impacts[self.detailed_impacts['event_id'] == event_id]
            
            # Create feature vector: [return on ticker1, return on ticker2, ...]
            feature_vector = event_data.sort_values('affected_ticker')['return_24h'].values
            
            if len(feature_vector) > 0:
                event_features.append(feature_vector)
                
                # Store metadata
                event_info = event_data.iloc[0]
                event_labels.append({
                    'event_id': event_id,
                    'event_ticker': event_info['event_ticker'],
                    'event_type': event_info['event_type'],
                    'event_date': event_info['event_date']
                })
        
        if len(event_features) < n_clusters:
            print(f"Warning: Only {len(event_features)} events, need at least {n_clusters} for clustering")
            return {}
        
        # Pad vectors to same length (in case some events have missing ticker data)
        max_len = max(len(v) for v in event_features)
        feature_matrix = np.array([
            np.pad(v, (0, max_len - len(v)), constant_values=0) 
            for v in event_features
        ])
        
        # Step 2: Run K-means clustering
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler
        
        # Standardize features
        scaler = StandardScaler()
        feature_matrix_scaled = scaler.fit_transform(feature_matrix)
        
        # Fit K-means
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(feature_matrix_scaled)
        
        # Step 3: Analyze clusters
        event_clusters = {}
        
        for cluster_id in range(n_clusters):
            cluster_events = [
                event_labels[i] for i, label in enumerate(cluster_labels) 
                if label == cluster_id
            ]
            
            if len(cluster_events) == 0:
                continue
            
            # Get common characteristics
            event_types = [e['event_type'] for e in cluster_events]
            most_common_type = max(set(event_types), key=event_types.count)
            
            # Calculate average impact profile for this cluster
            cluster_impacts = feature_matrix[cluster_labels == cluster_id]
            avg_impact_profile = np.mean(cluster_impacts, axis=0)
            
            event_clusters[f"Cluster_{cluster_id}"] = {
                'size': len(cluster_events),
                'dominant_event_type': most_common_type,
                'event_types': list(set(event_types)),
                'avg_market_impact': float(np.mean(np.abs(avg_impact_profile))),
                'max_impact': float(np.max(np.abs(avg_impact_profile))),
                'events': cluster_events[:5]  # Sample events
            }
        
        print(f"✓ Created {len(event_clusters)} clusters")
        for name, info in event_clusters.items():
            print(f"  {name}: {info['size']} events, dominant type: {info['dominant_event_type']}")
        
        return event_clusters
    
    def calculate_risk_metrics(self, ticker: str, event_type: str) -> Dict:
        """
        Calculate risk metrics for a ticker given an event type
        Beta, volatility, max drawdown, etc.
        
        This is your risk management component
        """
        # Filter events for this ticker and type
        relevant_events = self.event_history[
            (self.event_history['ticker'] == ticker) & 
            (self.event_history['event_type'] == event_type)
        ]
        
        # Calculate return distribution
        returns = []
        for idx, event in relevant_events.iterrows():
            impact = self.calculate_price_impact(event['date'], ticker)
            returns.append(impact['return_24h'])
        
        returns = np.array(returns)
        
        risk_metrics = {
            'mean_return': np.mean(returns),
            'volatility': np.std(returns),
            'max_drawdown': np.min(returns),
            'max_gain': np.max(returns),
            'var_95': np.percentile(returns, 5),  # Value at Risk
            'cvar_95': np.mean(returns[returns <= np.percentile(returns, 5)]),  # Conditional VaR
            'sharpe_ratio': np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0
        }
        
        return risk_metrics
    
    def predict_impact(self, new_event: Dict) -> Dict:
        """
        Given a new event, predict its impact using historical correlations
        
        Args:
            new_event: Dict with 'ticker', 'event_type', 'entities'
            
        Returns:
            Predicted impact on all tickers in watchlist with confidence scores
        """
        predictions = {}
        
        event_ticker = new_event['ticker']
        event_type = new_event['event_type']
        
        # 1. Direct impact prediction (historical average for this event type)
        direct_risk = self.calculate_risk_metrics(event_ticker, event_type)
        predictions[event_ticker] = {
            'predicted_return': direct_risk['mean_return'],
            'confidence': 0.8,  # High confidence for direct impact
            'risk_range': (direct_risk['var_95'], direct_risk['max_gain']),
            'type': 'direct'
        }
        
        # 2. Indirect impact predictions (using correlation network)
        for correlation_key, stats_info in self.significant_correlations.items():
            if correlation_key.startswith(event_ticker + " →"):
                affected_ticker = correlation_key.split(" → ")[1]
                
                predictions[affected_ticker] = {
                    'predicted_return': stats_info['avg_impact'],
                    'confidence': stats_info['confidence'],
                    'p_value': stats_info['p_value'],
                    'type': 'indirect'
                }
        
        return predictions


class SectorContagionAnalyzer:
    """
    Analyzes how events spread across sectors
    Example: Bank failure → all financial stocks drop
    """
    
    def __init__(self):
        self.sector_map = {}  # ticker → sector mapping
        self.contagion_network = None
        
    def load_sector_data(self, ticker_sector_map: Dict[str, str]):
        """Load mapping of tickers to sectors"""
        self.sector_map = ticker_sector_map
        
    def calculate_sector_beta(self, ticker: str, sector_event_returns: pd.Series) -> float:
        """
        Calculate how much a ticker moves relative to sector events
        
        Beta > 1: More volatile than sector average
        Beta < 1: Less volatile than sector average
        """
        # TODO: Implement sector beta calculation
        # This is standard finance beta calculation: Cov(ticker, sector) / Var(sector)
        
        beta = 0.0
        return beta
    
    def detect_contagion_patterns(self) -> Dict:
        """
        Identify contagion patterns: when event in one company affects whole sector
        
        Returns:
            Dict of contagion relationships with strength scores
        """
        contagion_patterns = {}
        
        # Your implementation:
        # For each sector, identify events that cause sector-wide movements
        
        return contagion_patterns


class TimeSeriesAnalyzer:
    """
    Analyzes timing of market reactions
    How long does it take for news to be priced in?
    """
    
    def analyze_reaction_timing(self, event_date: datetime, ticker: str) -> Dict:
        """
        Break down price reaction into time windows
        
        Returns:
            Dict with returns at different time intervals
        """
        timing_analysis = {
            'immediate_15m': 0.0,  # HFT reaction
            'early_1h': 0.0,       # Algo traders
            'intraday_4h': 0.0,    # Day traders
            'full_day_24h': 0.0,   # Full price discovery
            'sustained_1w': 0.0    # Long-term impact
        }
        
        # TODO: Implement time-decay analysis
        # This tells you optimal entry/exit timing
        
        return timing_analysis
    
    def calculate_information_decay(self, events: List[Dict]) -> pd.Series:
        """
        How quickly does news get priced in?
        Useful for timing trades
        """
        # TODO: Implement exponential decay model
        pass


# Example usage and testing
if __name__ == "__main__":
    print("Correlation Engine - Event Impact Analysis")
    print("=" * 50)
    
    # Initialize analyzer
    analyzer = EventImpactAnalyzer(lookback_days=180)
    
    # TODO: Load your historical data
    # events_df = pd.read_csv('historical_events.csv')
    # prices_df = pd.read_csv('historical_prices.csv')
    # analyzer.load_historical_data(events_df, prices_df)
    
    # Build correlation network
    # correlation_matrix = analyzer.build_correlation_network()
    # print("\nCorrelation Matrix:")
    # print(correlation_matrix)
    
    # Detect indirect correlations
    # indirect = analyzer.detect_indirect_correlations()
    # print("\nIndirect Correlations Found:")
    # for key, value in indirect.items():
    #     print(f"{key}: {value['avg_impact']:.2%} impact (p={value['p_value']:.4f})")
    
    # Test prediction
    # new_event = {
    #     'ticker': 'CRWV',
    #     'event_type': 'infrastructure_failure',
    #     'entities': ['CoreWeave', 'GPU']
    # }
    # predictions = analyzer.predict_impact(new_event)
    # print("\nPredicted Impacts:")
    # for ticker, pred in predictions.items():
    #     print(f"{ticker}: {pred['predicted_return']:.2%} (confidence: {pred['confidence']:.0%})")
    
    print("\n✅ Skeleton ready - fill in the TODOs with your implementations")
