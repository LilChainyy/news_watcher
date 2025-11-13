"""
Data Loading and Preprocessing
Handles loading historical events and price data, cleaning, and feature engineering
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import json


class DataLoader:
    """
    Load and preprocess historical data for correlation analysis
    """
    
    def __init__(self, data_dir: str = "./data"):
        self.data_dir = data_dir
        self.events_df = None
        self.prices_df = None
        
    def load_events(self, filepath: Optional[str] = None) -> pd.DataFrame:
        """
        Load historical news events
        
        Expected CSV columns:
        - date: datetime of event
        - ticker: primary stock ticker mentioned
        - event_type: category (earnings, product_launch, regulatory, etc.)
        - headline: raw news headline
        - sentiment: -1 to 1 score
        - entities: list of related companies/products
        - source: where news came from
        """
        if filepath is None:
            filepath = f"{self.data_dir}/historical_events.csv"
        
        # TODO: Load your actual data
        # self.events_df = pd.read_csv(filepath, parse_dates=['date'])
        
        # For now, create sample structure
        self.events_df = pd.DataFrame({
            'date': pd.date_range('2024-05-01', periods=100, freq='D'),
            'ticker': np.random.choice(['HOOD', 'AMD', 'NEBIUS', 'CRWV'], 100),
            'event_type': np.random.choice(['earnings', 'product', 'regulatory'], 100),
            'headline': ['Sample headline'] * 100,
            'sentiment': np.random.uniform(-1, 1, 100),
            'entities': [['entity1', 'entity2']] * 100,
            'source': ['NewsAPI'] * 100
        })
        
        return self.events_df
    
    def load_prices(self, filepath: Optional[str] = None) -> pd.DataFrame:
        """
        Load historical price data
        
        Expected CSV columns:
        - date: datetime
        - ticker: stock symbol
        - open, high, low, close: OHLC prices
        - volume: trading volume
        """
        if filepath is None:
            filepath = f"{self.data_dir}/historical_prices.csv"
        
        # TODO: Load your actual data
        # self.prices_df = pd.read_csv(filepath, parse_dates=['date'])
        
        # For now, create sample structure
        dates = pd.date_range('2024-01-01', periods=200, freq='D')
        tickers = ['HOOD', 'AMD', 'NEBIUS', 'CRWV', 'SPY']
        
        data = []
        for ticker in tickers:
            base_price = np.random.uniform(50, 200)
            prices = base_price * (1 + np.random.randn(len(dates)).cumsum() * 0.02)
            
            for i, date in enumerate(dates):
                data.append({
                    'date': date,
                    'ticker': ticker,
                    'open': prices[i] * 0.99,
                    'high': prices[i] * 1.02,
                    'low': prices[i] * 0.98,
                    'close': prices[i],
                    'volume': np.random.randint(1000000, 10000000)
                })
        
        self.prices_df = pd.DataFrame(data)
        return self.prices_df
    
    def get_price_at_time(self, ticker: str, target_date: datetime) -> Optional[float]:
        """Get closing price for ticker at specific date"""
        if self.prices_df is None:
            return None
        
        price_row = self.prices_df[
            (self.prices_df['ticker'] == ticker) & 
            (self.prices_df['date'] == target_date)
        ]
        
        return price_row['close'].iloc[0] if len(price_row) > 0 else None
    
    def get_price_window(self, ticker: str, start_date: datetime, 
                        end_date: datetime) -> pd.DataFrame:
        """Get OHLCV data for ticker in date range"""
        if self.prices_df is None:
            return pd.DataFrame()
        
        return self.prices_df[
            (self.prices_df['ticker'] == ticker) &
            (self.prices_df['date'] >= start_date) &
            (self.prices_df['date'] <= end_date)
        ].sort_values('date')


class FeatureEngineering:
    """
    Create features from raw data for better correlation detection
    """
    
    @staticmethod
    def calculate_returns(prices_df: pd.DataFrame, periods: List[int] = [1, 5, 20]) -> pd.DataFrame:
        """
        Calculate returns over different periods
        
        Args:
            prices_df: DataFrame with 'ticker', 'date', 'close'
            periods: List of periods (in days) to calculate returns
        
        Returns:
            DataFrame with return columns added
        """
        result = prices_df.copy()
        
        for period in periods:
            result[f'return_{period}d'] = result.groupby('ticker')['close'].pct_change(period)
        
        return result
    
    @staticmethod
    def calculate_volatility_features(prices_df: pd.DataFrame, 
                                     windows: List[int] = [5, 20, 60]) -> pd.DataFrame:
        """
        Calculate rolling volatility measures
        """
        result = prices_df.copy()
        
        for window in windows:
            # Realized volatility
            result[f'volatility_{window}d'] = (
                result.groupby('ticker')['close']
                .pct_change()
                .rolling(window)
                .std()
                .reset_index(level=0, drop=True)
            )
            
            # High-low volatility (Parkinson estimator)
            result[f'hl_volatility_{window}d'] = (
                result.groupby('ticker')
                .apply(lambda x: (np.log(x['high'] / x['low']) ** 2).rolling(window).mean())
                .reset_index(level=0, drop=True)
            )
        
        return result
    
    @staticmethod
    def calculate_volume_features(prices_df: pd.DataFrame) -> pd.DataFrame:
        """
        Volume-based features (often surge before major moves)
        """
        result = prices_df.copy()
        
        # Volume moving averages
        for window in [5, 20]:
            result[f'volume_ma_{window}'] = (
                result.groupby('ticker')['volume']
                .rolling(window)
                .mean()
                .reset_index(level=0, drop=True)
            )
        
        # Volume ratio (current vs average)
        result['volume_ratio'] = result['volume'] / result['volume_ma_20']
        
        # Dollar volume
        result['dollar_volume'] = result['close'] * result['volume']
        
        return result
    
    @staticmethod
    def add_technical_indicators(prices_df: pd.DataFrame) -> pd.DataFrame:
        """
        Add technical indicators that might correlate with event impacts
        """
        result = prices_df.copy()
        
        # RSI (Relative Strength Index)
        def calculate_rsi(prices, period=14):
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            return 100 - (100 / (1 + rs))
        
        result['rsi'] = (
            result.groupby('ticker')['close']
            .transform(lambda x: calculate_rsi(x))
        )
        
        # Bollinger Bands
        for window in [20]:
            ma = result.groupby('ticker')['close'].transform(lambda x: x.rolling(window).mean())
            std = result.groupby('ticker')['close'].transform(lambda x: x.rolling(window).std())
            
            result[f'bb_upper_{window}'] = ma + (std * 2)
            result[f'bb_lower_{window}'] = ma - (std * 2)
            result[f'bb_position_{window}'] = (result['close'] - result[f'bb_lower_{window}']) / (result[f'bb_upper_{window}'] - result[f'bb_lower_{window}'])
        
        return result
    
    @staticmethod
    def engineer_event_features(events_df: pd.DataFrame) -> pd.DataFrame:
        """
        Create features from event data
        """
        result = events_df.copy()
        
        # Time-based features
        result['hour'] = pd.to_datetime(result['date']).dt.hour
        result['day_of_week'] = pd.to_datetime(result['date']).dt.dayofweek
        result['is_trading_hours'] = result['hour'].between(9, 16)
        result['is_after_hours'] = ~result['is_trading_hours']
        
        # Event clustering (how many events for this ticker recently?)
        result = result.sort_values('date')
        result['events_last_7d'] = (
            result.groupby('ticker')['date']
            .transform(lambda x: x.rolling('7D').count())
        )
        
        # Sentiment momentum
        result['sentiment_ma_7d'] = (
            result.groupby('ticker')['sentiment']
            .transform(lambda x: x.rolling(7, min_periods=1).mean())
        )
        
        return result


class DataValidator:
    """
    Validate data quality before running correlation analysis
    """
    
    @staticmethod
    def check_data_completeness(df: pd.DataFrame, required_columns: List[str]) -> Dict:
        """Check if all required columns exist and have data"""
        issues = []
        
        for col in required_columns:
            if col not in df.columns:
                issues.append(f"Missing column: {col}")
            elif df[col].isna().sum() > len(df) * 0.1:  # >10% missing
                issues.append(f"Column {col} has {df[col].isna().sum()} missing values")
        
        return {
            'is_valid': len(issues) == 0,
            'issues': issues,
            'missing_rate': df.isna().sum() / len(df)
        }
    
    @staticmethod
    def check_date_alignment(events_df: pd.DataFrame, prices_df: pd.DataFrame) -> Dict:
        """Check if event dates align with available price data"""
        event_dates = set(events_df['date'].dt.date)
        price_dates = set(prices_df['date'].dt.date)
        
        missing_dates = event_dates - price_dates
        
        return {
            'has_coverage': len(missing_dates) == 0,
            'coverage_rate': 1 - (len(missing_dates) / len(event_dates)),
            'missing_dates': sorted(list(missing_dates))[:10]  # Show first 10
        }
    
    @staticmethod
    def detect_outliers(df: pd.DataFrame, column: str, 
                       method: str = 'iqr', threshold: float = 3.0) -> pd.Series:
        """
        Detect outliers that might skew correlation analysis
        
        Args:
            method: 'iqr' (Interquartile Range) or 'zscore'
        """
        if method == 'iqr':
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            
            outliers = (df[column] < lower_bound) | (df[column] > upper_bound)
            
        else:  # z-score
            z_scores = np.abs(stats.zscore(df[column].dropna()))
            outliers = z_scores > threshold
        
        return outliers


class DataPipeline:
    """
    Complete pipeline: load → validate → engineer features → output clean data
    """
    
    def __init__(self, data_dir: str = "./data"):
        self.loader = DataLoader(data_dir)
        self.validator = DataValidator()
        self.feature_engineer = FeatureEngineering()
        
    def run_pipeline(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Execute full data preparation pipeline
        
        Returns:
            (events_df, prices_df) both with features added
        """
        print("🔄 Starting data pipeline...")
        
        # Step 1: Load data
        print("  Loading events...")
        events_df = self.loader.load_events()
        print(f"  ✓ Loaded {len(events_df)} events")
        
        print("  Loading prices...")
        prices_df = self.loader.load_prices()
        print(f"  ✓ Loaded {len(prices_df)} price records")
        
        # Step 2: Validate
        print("\n  Validating data quality...")
        
        events_validation = self.validator.check_data_completeness(
            events_df, 
            ['date', 'ticker', 'event_type', 'sentiment']
        )
        
        prices_validation = self.validator.check_data_completeness(
            prices_df,
            ['date', 'ticker', 'open', 'high', 'low', 'close', 'volume']
        )
        
        if not events_validation['is_valid']:
            print(f"  ⚠️  Event data issues: {events_validation['issues']}")
        
        if not prices_validation['is_valid']:
            print(f"  ⚠️  Price data issues: {prices_validation['issues']}")
        
        alignment = self.validator.check_date_alignment(events_df, prices_df)
        print(f"  ✓ Date coverage: {alignment['coverage_rate']:.1%}")
        
        # Step 3: Feature engineering
        print("\n  Engineering features...")
        
        prices_df = self.feature_engineer.calculate_returns(prices_df)
        prices_df = self.feature_engineer.calculate_volatility_features(prices_df)
        prices_df = self.feature_engineer.calculate_volume_features(prices_df)
        prices_df = self.feature_engineer.add_technical_indicators(prices_df)
        
        events_df = self.feature_engineer.engineer_event_features(events_df)
        
        print(f"  ✓ Added {len(prices_df.columns) - 7} price features")
        print(f"  ✓ Added {len(events_df.columns) - 6} event features")
        
        # Step 4: Detect outliers
        print("\n  Checking for outliers...")
        return_outliers = self.validator.detect_outliers(prices_df, 'return_1d')
        print(f"  Found {return_outliers.sum()} outlier returns (will flag for review)")
        
        print("\n✅ Pipeline complete!")
        return events_df, prices_df


# Example usage
if __name__ == "__main__":
    print("Data Loading and Preprocessing Pipeline")
    print("=" * 50)
    
    # Initialize pipeline
    pipeline = DataPipeline(data_dir="./data")
    
    # Run full pipeline
    events_df, prices_df = pipeline.run_pipeline()
    
    print("\n📊 Sample Event Data:")
    print(events_df.head())
    
    print("\n📈 Sample Price Data:")
    print(prices_df.head())
    
    print("\n💡 Next Steps:")
    print("  1. Replace sample data with your actual CSV files")
    print("  2. Feed this data into correlation_engine.py")
    print("  3. Start discovering hidden market relationships!")
