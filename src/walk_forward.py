"""
Walk-Forward Optimization Module.

Implements walk-forward analysis to prevent overfitting by:
1. Splitting data into rolling train/test windows
2. Optimizing parameters on training data
3. Validating on out-of-sample test data
4. Aggregating results to assess strategy robustness
"""
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional, Callable
from datetime import datetime, timedelta

from .backtester import run_backtest, BacktestResult
from .strategy import StrategyConfig


@dataclass
class WalkForwardWindow:
    """Represents a single walk-forward window."""
    window_num: int
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime
    train_bars: int
    test_bars: int
    # Optimization results
    best_config: Optional[Dict[str, Any]] = None
    train_result: Optional[BacktestResult] = None
    test_result: Optional[BacktestResult] = None


@dataclass
class WalkForwardResult:
    """Results from walk-forward optimization."""
    windows: List[WalkForwardWindow]
    aggregate_metrics: Dict[str, float]
    robustness_score: float

    @property
    def total_windows(self) -> int:
        return len(self.windows)

    @property
    def valid_windows(self) -> int:
        return sum(1 for w in self.windows if w.test_result and w.test_result.total_trades > 0)


def split_data_windows(
    df: pd.DataFrame,
    train_pct: float = 0.7,
    n_windows: int = 5,
    min_train_bars: int = 100,
    min_test_bars: int = 30,
) -> List[Tuple[pd.DataFrame, pd.DataFrame, int]]:
    """
    Split data into rolling train/test windows.

    Args:
        df: Full dataframe with OHLCV data
        train_pct: Percentage of each window for training (0.5-0.9)
        n_windows: Number of walk-forward windows
        min_train_bars: Minimum bars required for training
        min_test_bars: Minimum bars required for testing

    Returns:
        List of (train_df, test_df, window_num) tuples
    """
    total_bars = len(df)

    # Calculate window size
    window_size = total_bars // n_windows

    if window_size < (min_train_bars + min_test_bars):
        # Reduce number of windows if data is insufficient
        window_size = min_train_bars + min_test_bars
        n_windows = max(1, total_bars // window_size)

    windows = []

    for i in range(n_windows):
        start_idx = i * window_size
        end_idx = min(start_idx + window_size, total_bars)

        if end_idx - start_idx < (min_train_bars + min_test_bars):
            continue

        window_df = df.iloc[start_idx:end_idx]

        # Split into train/test
        train_size = int(len(window_df) * train_pct)
        test_size = len(window_df) - train_size

        if train_size < min_train_bars or test_size < min_test_bars:
            continue

        train_df = window_df.iloc[:train_size]
        test_df = window_df.iloc[train_size:]

        windows.append((train_df, test_df, i + 1))

    return windows


def optimize_single_window(
    train_df: pd.DataFrame,
    base_config: StrategyConfig,
    param_grid: Dict[str, List[Any]],
    metric: str = "sharpe_ratio",
    min_trades: int = 5,
) -> Tuple[Dict[str, Any], BacktestResult]:
    """
    Optimize parameters on a single training window.

    Args:
        train_df: Training data
        base_config: Base configuration
        param_grid: Dictionary of parameter names to test values
        metric: Metric to optimize ("sharpe_ratio", "profit_factor", "total_pnl", "win_rate")
        min_trades: Minimum trades required for valid result

    Returns:
        Tuple of (best_params, best_result)
    """
    from itertools import product

    best_params = {}
    best_result = None
    best_metric_value = float('-inf')

    # Generate all parameter combinations
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())

    for values in product(*param_values):
        # Create config with these parameters
        config_dict = base_config.__dict__.copy()
        test_params = dict(zip(param_names, values))
        config_dict.update(test_params)

        try:
            config = StrategyConfig(**config_dict)
            result = run_backtest(train_df, config)

            if result.total_trades >= min_trades:
                # Get metric value
                metric_value = getattr(result, metric, 0)

                if metric_value > best_metric_value:
                    best_metric_value = metric_value
                    best_params = test_params
                    best_result = result
        except Exception:
            continue

    return best_params, best_result


def run_walk_forward(
    df: pd.DataFrame,
    base_config: StrategyConfig,
    param_grid: Dict[str, List[Any]],
    train_pct: float = 0.7,
    n_windows: int = 5,
    metric: str = "sharpe_ratio",
    min_trades: int = 5,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
) -> WalkForwardResult:
    """
    Run walk-forward optimization.

    Args:
        df: Full dataframe with OHLCV data
        base_config: Base strategy configuration
        param_grid: Dictionary of parameter names to test values
        train_pct: Percentage of each window for training
        n_windows: Number of walk-forward windows
        metric: Metric to optimize
        min_trades: Minimum trades per window
        progress_callback: Optional callback for progress updates

    Returns:
        WalkForwardResult with all windows and aggregate metrics
    """
    # Split data into windows
    windows_data = split_data_windows(
        df, train_pct=train_pct, n_windows=n_windows
    )

    if not windows_data:
        raise ValueError("Insufficient data for walk-forward analysis")

    results = []

    for i, (train_df, test_df, window_num) in enumerate(windows_data):
        if progress_callback:
            progress_callback(i + 1, len(windows_data), f"Window {window_num}")

        # Create window object
        window = WalkForwardWindow(
            window_num=window_num,
            train_start=train_df.index[0].to_pydatetime(),
            train_end=train_df.index[-1].to_pydatetime(),
            test_start=test_df.index[0].to_pydatetime(),
            test_end=test_df.index[-1].to_pydatetime(),
            train_bars=len(train_df),
            test_bars=len(test_df),
        )

        # Optimize on training data
        best_params, train_result = optimize_single_window(
            train_df, base_config, param_grid, metric, min_trades
        )

        window.best_config = best_params
        window.train_result = train_result

        # Validate on test data
        if best_params and train_result:
            config_dict = base_config.__dict__.copy()
            config_dict.update(best_params)

            try:
                test_config = StrategyConfig(**config_dict)
                test_result = run_backtest(test_df, test_config)
                window.test_result = test_result
            except Exception:
                pass

        results.append(window)

    # Calculate aggregate metrics
    aggregate_metrics = calculate_aggregate_metrics(results)
    robustness_score = calculate_robustness_score(results)

    return WalkForwardResult(
        windows=results,
        aggregate_metrics=aggregate_metrics,
        robustness_score=robustness_score,
    )


def calculate_aggregate_metrics(windows: List[WalkForwardWindow]) -> Dict[str, float]:
    """Calculate aggregate metrics across all windows."""
    valid_windows = [w for w in windows if w.test_result and w.test_result.total_trades > 0]

    if not valid_windows:
        return {
            "avg_trades": 0,
            "avg_win_rate": 0,
            "avg_pnl": 0,
            "avg_sharpe": 0,
            "avg_profit_factor": 0,
            "total_pnl": 0,
            "consistency": 0,
        }

    # Calculate averages from test results
    trades = [w.test_result.total_trades for w in valid_windows]
    win_rates = [w.test_result.win_rate for w in valid_windows]
    pnls = [w.test_result.total_pnl for w in valid_windows]
    sharpes = [w.test_result.sharpe_ratio for w in valid_windows]
    profit_factors = [w.test_result.profit_factor for w in valid_windows]

    # Consistency = % of profitable windows
    profitable_windows = sum(1 for pnl in pnls if pnl > 0)
    consistency = profitable_windows / len(valid_windows) if valid_windows else 0

    return {
        "avg_trades": np.mean(trades),
        "avg_win_rate": np.mean(win_rates),
        "avg_pnl": np.mean(pnls),
        "avg_sharpe": np.mean(sharpes),
        "avg_profit_factor": np.mean(profit_factors),
        "total_pnl": sum(pnls),
        "consistency": consistency,
    }


def calculate_robustness_score(windows: List[WalkForwardWindow]) -> float:
    """
    Calculate a robustness score (0-100) based on:
    - Consistency of profitability across windows
    - Similarity of train vs test performance
    - Overall performance metrics
    """
    valid_windows = [w for w in windows if w.test_result and w.train_result]

    if not valid_windows:
        return 0.0

    scores = []

    # 1. Consistency score (40% weight) - % of profitable test windows
    test_pnls = [w.test_result.total_pnl for w in valid_windows]
    profitable_pct = sum(1 for pnl in test_pnls if pnl > 0) / len(valid_windows)
    consistency_score = profitable_pct * 40
    scores.append(consistency_score)

    # 2. Train/Test similarity (30% weight) - how close is test to train performance
    ratios = []
    for w in valid_windows:
        if w.train_result.total_pnl != 0:
            ratio = w.test_result.total_pnl / w.train_result.total_pnl
            ratios.append(min(ratio, 1.0))  # Cap at 1.0 (test can't be "better" than train for robustness)

    if ratios:
        avg_ratio = np.mean(ratios)
        similarity_score = max(0, avg_ratio) * 30
    else:
        similarity_score = 0
    scores.append(similarity_score)

    # 3. Win rate stability (15% weight) - low std dev in win rates
    test_win_rates = [w.test_result.win_rate for w in valid_windows]
    if len(test_win_rates) > 1:
        win_rate_std = np.std(test_win_rates)
        # Lower std = higher score
        stability_score = max(0, (0.3 - win_rate_std) / 0.3) * 15
    else:
        stability_score = 7.5
    scores.append(stability_score)

    # 4. Sharpe consistency (15% weight)
    test_sharpes = [w.test_result.sharpe_ratio for w in valid_windows]
    positive_sharpe_pct = sum(1 for s in test_sharpes if s > 0) / len(valid_windows)
    sharpe_score = positive_sharpe_pct * 15
    scores.append(sharpe_score)

    return min(100, sum(scores))


def format_walk_forward_results(result: WalkForwardResult) -> pd.DataFrame:
    """Format walk-forward results as a DataFrame."""
    data = []

    for w in result.windows:
        row = {
            "Window": w.window_num,
            "Train Period": f"{w.train_start.strftime('%Y-%m-%d')} to {w.train_end.strftime('%Y-%m-%d')}",
            "Test Period": f"{w.test_start.strftime('%Y-%m-%d')} to {w.test_end.strftime('%Y-%m-%d')}",
            "Train Bars": w.train_bars,
            "Test Bars": w.test_bars,
        }

        if w.train_result:
            row["Train Trades"] = w.train_result.total_trades
            row["Train Win%"] = f"{w.train_result.win_rate:.1%}"
            row["Train PnL"] = f"${w.train_result.total_pnl:,.2f}"
        else:
            row["Train Trades"] = "-"
            row["Train Win%"] = "-"
            row["Train PnL"] = "-"

        if w.test_result:
            row["Test Trades"] = w.test_result.total_trades
            row["Test Win%"] = f"{w.test_result.win_rate:.1%}"
            row["Test PnL"] = f"${w.test_result.total_pnl:,.2f}"
            row["Test Sharpe"] = f"{w.test_result.sharpe_ratio:.2f}"
        else:
            row["Test Trades"] = "-"
            row["Test Win%"] = "-"
            row["Test PnL"] = "-"
            row["Test Sharpe"] = "-"

        data.append(row)

    return pd.DataFrame(data)
