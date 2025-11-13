"""
Statistical Analysis Utilities
Advanced statistical methods for validating correlations and testing hypotheses
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class StatisticalTest:
    """Results from statistical hypothesis test"""
    test_name: str
    statistic: float
    p_value: float
    is_significant: bool
    confidence_level: float
    interpretation: str


class CorrelationValidator:
    """
    Validates whether correlations are statistically significant or just noise
    This prevents you from trading on false patterns
    """
    
    def __init__(self, significance_level: float = 0.05):
        """
        Args:
            significance_level: Alpha level for hypothesis tests (default 5%)
        """
        self.alpha = significance_level
        
    def test_correlation_significance(self, returns_x: np.ndarray, 
                                     returns_y: np.ndarray) -> StatisticalTest:
        """
        Test if correlation between two return series is statistically significant
        
        H0: No correlation (rho = 0)
        H1: Correlation exists (rho ≠ 0)
        """
        # Calculate Pearson correlation
        corr_coef, p_value = stats.pearsonr(returns_x, returns_y)
        
        is_significant = p_value < self.alpha
        
        interpretation = f"Correlation of {corr_coef:.3f} is "
        interpretation += "SIGNIFICANT" if is_significant else "NOT significant"
        interpretation += f" at {100*(1-self.alpha):.0f}% confidence level"
        
        return StatisticalTest(
            test_name="Pearson Correlation Test",
            statistic=corr_coef,
            p_value=p_value,
            is_significant=is_significant,
            confidence_level=1 - p_value,
            interpretation=interpretation
        )
    
    def test_event_impact(self, returns_during_event: np.ndarray, 
                         returns_baseline: np.ndarray) -> StatisticalTest:
        """
        Test if returns during events are significantly different from baseline
        
        Uses two-sample t-test to compare distributions
        """
        t_stat, p_value = stats.ttest_ind(returns_during_event, returns_baseline)
        
        is_significant = p_value < self.alpha
        
        mean_diff = np.mean(returns_during_event) - np.mean(returns_baseline)
        
        interpretation = f"Event returns differ by {mean_diff:.2%} from baseline. "
        interpretation += "This difference is " + ("SIGNIFICANT" if is_significant else "NOT significant")
        
        return StatisticalTest(
            test_name="Two-Sample T-Test",
            statistic=t_stat,
            p_value=p_value,
            is_significant=is_significant,
            confidence_level=1 - p_value,
            interpretation=interpretation
        )
    
    def test_stationarity(self, time_series: np.ndarray) -> StatisticalTest:
        """
        Test if time series is stationary (mean/variance constant over time)
        Uses Augmented Dickey-Fuller test
        
        Important: Non-stationary data can create spurious correlations
        """
        from statsmodels.tsa.stattools import adfuller
        
        result = adfuller(time_series)
        
        adf_stat = result[0]
        p_value = result[1]
        is_stationary = p_value < self.alpha
        
        interpretation = "Time series is " + ("STATIONARY" if is_stationary else "NON-STATIONARY")
        interpretation += " - correlations are " + ("reliable" if is_stationary else "potentially spurious")
        
        return StatisticalTest(
            test_name="Augmented Dickey-Fuller Test",
            statistic=adf_stat,
            p_value=p_value,
            is_significant=is_stationary,
            confidence_level=1 - p_value,
            interpretation=interpretation
        )
    
    def bonferroni_correction(self, p_values: List[float]) -> List[bool]:
        """
        Adjust for multiple hypothesis testing
        
        When testing many correlations, some will appear significant by chance
        Bonferroni correction prevents false discoveries
        
        Returns:
            List of bools indicating which tests remain significant after correction
        """
        adjusted_alpha = self.alpha / len(p_values)
        
        return [p < adjusted_alpha for p in p_values]
    
    def false_discovery_rate(self, p_values: List[float], 
                            q_value: float = 0.05) -> List[bool]:
        """
        Benjamini-Hochberg procedure for controlling false discovery rate
        Less conservative than Bonferroni, better for exploratory analysis
        """
        n = len(p_values)
        sorted_indices = np.argsort(p_values)
        sorted_p_values = np.array(p_values)[sorted_indices]
        
        # Find largest k where p(k) <= (k/n) * q
        threshold = np.arange(1, n + 1) / n * q_value
        significant = sorted_p_values <= threshold
        
        # Create result array in original order
        result = np.zeros(n, dtype=bool)
        if significant.any():
            max_k = np.where(significant)[0][-1]
            result[sorted_indices[:max_k + 1]] = True
        
        return result.tolist()


class RiskMetricsCalculator:
    """
    Calculate various risk metrics for event-driven trading
    """
    
    @staticmethod
    def calculate_var(returns: np.ndarray, confidence_level: float = 0.95) -> float:
        """
        Value at Risk: Maximum expected loss at given confidence level
        
        Example: VaR(95%) = -2.5% means "95% of the time, loss won't exceed 2.5%"
        """
        return np.percentile(returns, (1 - confidence_level) * 100)
    
    @staticmethod
    def calculate_cvar(returns: np.ndarray, confidence_level: float = 0.95) -> float:
        """
        Conditional Value at Risk (Expected Shortfall)
        Average loss in the worst (1-confidence_level)% of cases
        
        More informative than VaR for tail risk
        """
        var = RiskMetricsCalculator.calculate_var(returns, confidence_level)
        return np.mean(returns[returns <= var])
    
    @staticmethod
    def calculate_max_drawdown(prices: np.ndarray) -> Dict[str, float]:
        """
        Maximum peak-to-trough decline
        
        Returns:
            Dict with max_drawdown, peak_idx, trough_idx, recovery_idx
        """
        cumulative = np.maximum.accumulate(prices)
        drawdown = (prices - cumulative) / cumulative
        
        max_dd_idx = np.argmin(drawdown)
        max_dd = drawdown[max_dd_idx]
        
        # Find the peak before the trough
        peak_idx = np.argmax(prices[:max_dd_idx + 1]) if max_dd_idx > 0 else 0
        
        # Find recovery point (if any)
        recovery_idx = None
        if max_dd_idx < len(prices) - 1:
            recovery_prices = prices[max_dd_idx + 1:]
            peak_price = prices[peak_idx]
            recovery_mask = recovery_prices >= peak_price
            if recovery_mask.any():
                recovery_idx = max_dd_idx + 1 + np.where(recovery_mask)[0][0]
        
        return {
            'max_drawdown': max_dd,
            'peak_idx': peak_idx,
            'trough_idx': max_dd_idx,
            'recovery_idx': recovery_idx,
            'recovery_time': recovery_idx - peak_idx if recovery_idx else None
        }
    
    @staticmethod
    def calculate_sharpe_ratio(returns: np.ndarray, risk_free_rate: float = 0.0) -> float:
        """
        Risk-adjusted return metric
        Higher is better (>1 is good, >2 is excellent)
        """
        excess_returns = returns - risk_free_rate
        return np.mean(excess_returns) / np.std(excess_returns) if np.std(excess_returns) > 0 else 0
    
    @staticmethod
    def calculate_sortino_ratio(returns: np.ndarray, risk_free_rate: float = 0.0) -> float:
        """
        Like Sharpe but only penalizes downside volatility
        Better for asymmetric return distributions
        """
        excess_returns = returns - risk_free_rate
        downside_returns = excess_returns[excess_returns < 0]
        
        if len(downside_returns) == 0 or np.std(downside_returns) == 0:
            return float('inf') if np.mean(excess_returns) > 0 else 0
        
        return np.mean(excess_returns) / np.std(downside_returns)
    
    @staticmethod
    def calculate_beta(asset_returns: np.ndarray, market_returns: np.ndarray) -> float:
        """
        Systematic risk: How much asset moves relative to market
        
        Beta = 1: Moves with market
        Beta > 1: More volatile than market
        Beta < 1: Less volatile than market
        """
        covariance = np.cov(asset_returns, market_returns)[0, 1]
        market_variance = np.var(market_returns)
        
        return covariance / market_variance if market_variance > 0 else 0


class EventStudyAnalysis:
    """
    Formal event study methodology
    This is what academic researchers and quants use to analyze event impacts
    """
    
    def __init__(self, estimation_window: int = 120, event_window: Tuple[int, int] = (-1, 1)):
        """
        Args:
            estimation_window: Days before event to estimate normal returns
            event_window: (start, end) days around event to measure abnormal returns
        """
        self.estimation_window = estimation_window
        self.event_window = event_window
        
    def calculate_abnormal_returns(self, 
                                   asset_prices: pd.Series,
                                   market_prices: pd.Series,
                                   event_date: pd.Timestamp) -> Dict:
        """
        Calculate abnormal returns using market model
        
        Abnormal Return = Actual Return - Expected Return
        Where Expected Return is based on asset's historical relationship with market
        """
        # Step 1: Estimation period (before event)
        estimation_end = event_date - pd.Timedelta(days=2)
        estimation_start = estimation_end - pd.Timedelta(days=self.estimation_window)
        
        estimation_asset = asset_prices.loc[estimation_start:estimation_end].pct_change().dropna()
        estimation_market = market_prices.loc[estimation_start:estimation_end].pct_change().dropna()
        
        # Step 2: Estimate market model parameters (alpha, beta)
        # Returns_asset = alpha + beta * Returns_market + epsilon
        from scipy.stats import linregress
        
        slope, intercept, r_value, p_value, std_err = linregress(
            estimation_market, estimation_asset
        )
        
        alpha = intercept
        beta = slope
        
        # Step 3: Calculate abnormal returns in event window
        event_start = event_date + pd.Timedelta(days=self.event_window[0])
        event_end = event_date + pd.Timedelta(days=self.event_window[1])
        
        event_asset_returns = asset_prices.loc[event_start:event_end].pct_change().dropna()
        event_market_returns = market_prices.loc[event_start:event_end].pct_change().dropna()
        
        # Expected returns based on market model
        expected_returns = alpha + beta * event_market_returns
        
        # Abnormal returns
        abnormal_returns = event_asset_returns - expected_returns
        
        # Cumulative abnormal return (CAR)
        car = abnormal_returns.sum()
        
        # Standard error and t-test
        residuals = estimation_asset - (alpha + beta * estimation_market)
        std_error = np.std(residuals)
        
        car_std_error = std_error * np.sqrt(len(abnormal_returns))
        t_stat = car / car_std_error
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=len(residuals) - 2))
        
        return {
            'alpha': alpha,
            'beta': beta,
            'r_squared': r_value ** 2,
            'abnormal_returns': abnormal_returns,
            'cumulative_abnormal_return': car,
            'car_t_statistic': t_stat,
            'car_p_value': p_value,
            'is_significant': p_value < 0.05
        }
    
    def cross_sectional_analysis(self, 
                                multiple_events: pd.DataFrame,
                                asset_prices: pd.DataFrame,
                                market_prices: pd.Series) -> pd.DataFrame:
        """
        Analyze multiple events across different stocks
        Aggregates abnormal returns to find general patterns
        """
        results = []
        
        for idx, event in multiple_events.iterrows():
            ticker = event['ticker']
            event_date = event['date']
            
            if ticker not in asset_prices.columns:
                continue
            
            ar_analysis = self.calculate_abnormal_returns(
                asset_prices[ticker],
                market_prices,
                event_date
            )
            
            results.append({
                'ticker': ticker,
                'event_date': event_date,
                'event_type': event.get('event_type', 'unknown'),
                'car': ar_analysis['cumulative_abnormal_return'],
                'car_p_value': ar_analysis['car_p_value'],
                'is_significant': ar_analysis['is_significant']
            })
        
        return pd.DataFrame(results)


class VolatilityAnalyzer:
    """
    Analyze volatility patterns around events
    """
    
    @staticmethod
    def realized_volatility(returns: np.ndarray, window: int = 20) -> np.ndarray:
        """
        Rolling realized volatility (standard deviation of returns)
        """
        return pd.Series(returns).rolling(window).std().values
    
    @staticmethod
    def parkinson_volatility(high_prices: np.ndarray, 
                            low_prices: np.ndarray,
                            window: int = 20) -> np.ndarray:
        """
        Parkinson volatility estimator (uses high-low range)
        More efficient than close-to-close volatility
        """
        hl_ratio = np.log(high_prices / low_prices) ** 2
        return pd.Series(hl_ratio).rolling(window).mean().apply(
            lambda x: np.sqrt(x / (4 * np.log(2)))
        ).values
    
    @staticmethod
    def volatility_regime_detection(returns: np.ndarray, 
                                    n_regimes: int = 2) -> np.ndarray:
        """
        Detect volatility regimes using Gaussian Mixture Model
        Useful for adapting strategy to market conditions
        """
        from sklearn.mixture import GaussianMixture
        
        volatility = pd.Series(returns).rolling(20).std().dropna().values.reshape(-1, 1)
        
        gmm = GaussianMixture(n_components=n_regimes, random_state=42)
        regimes = gmm.fit_predict(volatility)
        
        return regimes


# Example usage
if __name__ == "__main__":
    print("Statistical Analysis Utilities")
    print("=" * 50)
    
    # Simulate some data for testing
    np.random.seed(42)
    
    # Create fake return series
    baseline_returns = np.random.normal(0.001, 0.02, 100)  # Normal market
    event_returns = np.random.normal(0.015, 0.03, 30)     # Returns during events
    
    # Test correlation validator
    validator = CorrelationValidator(significance_level=0.05)
    
    print("\n1. Testing event impact significance:")
    result = validator.test_event_impact(event_returns, baseline_returns)
    print(f"   {result.interpretation}")
    print(f"   P-value: {result.p_value:.4f}")
    
    # Test risk metrics
    print("\n2. Calculating risk metrics:")
    print(f"   VaR (95%): {RiskMetricsCalculator.calculate_var(event_returns):.2%}")
    print(f"   CVaR (95%): {RiskMetricsCalculator.calculate_cvar(event_returns):.2%}")
    print(f"   Sharpe Ratio: {RiskMetricsCalculator.calculate_sharpe_ratio(event_returns):.2f}")
    print(f"   Sortino Ratio: {RiskMetricsCalculator.calculate_sortino_ratio(event_returns):.2f}")
    
    # Test multiple comparison correction
    print("\n3. Testing multiple hypothesis correction:")
    p_values = [0.01, 0.03, 0.05, 0.08, 0.15]
    significant = validator.bonferroni_correction(p_values)
    print(f"   Original p-values: {p_values}")
    print(f"   Significant after Bonferroni: {significant}")
    
    fdr_significant = validator.false_discovery_rate(p_values)
    print(f"   Significant after FDR: {fdr_significant}")
    
    print("\n✅ Statistical utilities ready for your correlation engine")
