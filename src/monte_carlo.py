"""
Monte Carlo Simulation Module.

Provides statistical analysis of strategy performance through:
1. Trade resampling - randomly reorder trades to test robustness
2. Confidence intervals - calculate likely range of outcomes
3. Risk analysis - estimate worst-case scenarios
"""
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

from .backtester import BacktestResult


@dataclass
class MonteCarloResult:
    """Results from Monte Carlo simulation."""
    n_simulations: int
    original_pnl: float
    original_max_dd: float

    # PnL statistics
    pnl_mean: float
    pnl_median: float
    pnl_std: float
    pnl_ci_lower: float  # Lower confidence bound
    pnl_ci_upper: float  # Upper confidence bound
    pnl_worst: float
    pnl_best: float

    # Win rate statistics
    win_rate_mean: float
    win_rate_ci_lower: float
    win_rate_ci_upper: float

    # Drawdown statistics
    max_dd_mean: float
    max_dd_worst: float
    max_dd_best: float
    max_dd_ci_lower: float
    max_dd_ci_upper: float

    # Risk metrics
    risk_of_ruin: float  # % of simulations with > 50% drawdown
    probability_positive: float  # % of simulations with positive PnL

    # Raw data for visualization
    pnl_distribution: List[float]
    dd_distribution: List[float]


def run_monte_carlo(
    result: BacktestResult,
    n_simulations: int = 1000,
    confidence_level: float = 0.95,
    initial_capital: float = None,
) -> MonteCarloResult:
    """
    Run Monte Carlo simulation by resampling trades.

    Args:
        result: BacktestResult containing trades
        n_simulations: Number of simulations to run
        confidence_level: Confidence interval (0.90, 0.95, 0.99)
        initial_capital: Initial capital (defaults to config value)

    Returns:
        MonteCarloResult with statistics
    """
    if not result.trades:
        raise ValueError("No trades to simulate")

    # Extract trade PnLs
    trade_pnls = np.array([t.pnl for t in result.trades])
    n_trades = len(trade_pnls)

    if initial_capital is None:
        initial_capital = result.config.initial_capital

    # Run simulations
    sim_pnls = []
    sim_win_rates = []
    sim_max_dds = []

    for _ in range(n_simulations):
        # Resample trades with replacement
        resampled_pnls = np.random.choice(trade_pnls, size=n_trades, replace=True)

        # Calculate metrics for this simulation
        total_pnl = resampled_pnls.sum()
        win_rate = (resampled_pnls > 0).sum() / n_trades

        # Calculate drawdown
        cumulative = np.cumsum(resampled_pnls) + initial_capital
        running_max = np.maximum.accumulate(cumulative)
        drawdown = cumulative - running_max
        max_dd = drawdown.min()
        max_dd_pct = (max_dd / running_max[np.argmin(drawdown)]) * 100 if running_max[np.argmin(drawdown)] > 0 else 0

        sim_pnls.append(total_pnl)
        sim_win_rates.append(win_rate)
        sim_max_dds.append(abs(max_dd_pct))

    # Calculate statistics
    alpha = (1 - confidence_level) / 2

    # PnL stats
    pnl_mean = np.mean(sim_pnls)
    pnl_median = np.median(sim_pnls)
    pnl_std = np.std(sim_pnls)
    pnl_ci_lower = np.percentile(sim_pnls, alpha * 100)
    pnl_ci_upper = np.percentile(sim_pnls, (1 - alpha) * 100)
    pnl_worst = np.min(sim_pnls)
    pnl_best = np.max(sim_pnls)

    # Win rate stats
    win_rate_mean = np.mean(sim_win_rates)
    win_rate_ci_lower = np.percentile(sim_win_rates, alpha * 100)
    win_rate_ci_upper = np.percentile(sim_win_rates, (1 - alpha) * 100)

    # Drawdown stats
    max_dd_mean = np.mean(sim_max_dds)
    max_dd_worst = np.max(sim_max_dds)
    max_dd_best = np.min(sim_max_dds)
    max_dd_ci_lower = np.percentile(sim_max_dds, alpha * 100)
    max_dd_ci_upper = np.percentile(sim_max_dds, (1 - alpha) * 100)

    # Risk metrics
    risk_of_ruin = sum(1 for dd in sim_max_dds if dd > 50) / n_simulations
    probability_positive = sum(1 for pnl in sim_pnls if pnl > 0) / n_simulations

    return MonteCarloResult(
        n_simulations=n_simulations,
        original_pnl=result.total_pnl,
        original_max_dd=abs(result.max_drawdown_pct),
        pnl_mean=pnl_mean,
        pnl_median=pnl_median,
        pnl_std=pnl_std,
        pnl_ci_lower=pnl_ci_lower,
        pnl_ci_upper=pnl_ci_upper,
        pnl_worst=pnl_worst,
        pnl_best=pnl_best,
        win_rate_mean=win_rate_mean,
        win_rate_ci_lower=win_rate_ci_lower,
        win_rate_ci_upper=win_rate_ci_upper,
        max_dd_mean=max_dd_mean,
        max_dd_worst=max_dd_worst,
        max_dd_best=max_dd_best,
        max_dd_ci_lower=max_dd_ci_lower,
        max_dd_ci_upper=max_dd_ci_upper,
        risk_of_ruin=risk_of_ruin,
        probability_positive=probability_positive,
        pnl_distribution=sim_pnls,
        dd_distribution=sim_max_dds,
    )


def create_equity_paths(
    result: BacktestResult,
    n_simulations: int = 100,
    initial_capital: float = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create multiple equity curve paths for visualization.

    Args:
        result: BacktestResult containing trades
        n_simulations: Number of paths to generate
        initial_capital: Starting capital

    Returns:
        Tuple of (paths_array, mean_path)
        paths_array shape: (n_simulations, n_trades+1)
    """
    if not result.trades:
        raise ValueError("No trades to simulate")

    trade_pnls = np.array([t.pnl for t in result.trades])
    n_trades = len(trade_pnls)

    if initial_capital is None:
        initial_capital = result.config.initial_capital

    # Generate paths
    paths = np.zeros((n_simulations, n_trades + 1))
    paths[:, 0] = initial_capital

    for i in range(n_simulations):
        resampled = np.random.choice(trade_pnls, size=n_trades, replace=True)
        paths[i, 1:] = initial_capital + np.cumsum(resampled)

    mean_path = np.mean(paths, axis=0)

    return paths, mean_path


def get_var_cvar(
    sim_pnls: List[float],
    confidence_level: float = 0.95,
) -> Tuple[float, float]:
    """
    Calculate Value at Risk (VaR) and Conditional VaR (CVaR).

    Args:
        sim_pnls: List of simulated PnLs
        confidence_level: Confidence level (e.g., 0.95 for 95%)

    Returns:
        Tuple of (VaR, CVaR)
        VaR: Maximum expected loss at confidence level
        CVaR: Average of losses worse than VaR
    """
    alpha = 1 - confidence_level
    var = np.percentile(sim_pnls, alpha * 100)
    cvar = np.mean([pnl for pnl in sim_pnls if pnl <= var])
    return var, cvar


def format_monte_carlo_summary(mc_result: MonteCarloResult, confidence_level: float = 0.95) -> Dict[str, str]:
    """Format Monte Carlo results for display."""
    return {
        "simulations": f"{mc_result.n_simulations:,}",
        "original_pnl": f"${mc_result.original_pnl:,.2f}",
        "pnl_range": f"${mc_result.pnl_ci_lower:,.2f} to ${mc_result.pnl_ci_upper:,.2f}",
        "pnl_expected": f"${mc_result.pnl_mean:,.2f} ± ${mc_result.pnl_std:,.2f}",
        "win_rate_range": f"{mc_result.win_rate_ci_lower:.1%} to {mc_result.win_rate_ci_upper:.1%}",
        "max_dd_expected": f"{mc_result.max_dd_mean:.1f}%",
        "max_dd_worst": f"{mc_result.max_dd_worst:.1f}%",
        "risk_of_ruin": f"{mc_result.risk_of_ruin:.1%}",
        "probability_positive": f"{mc_result.probability_positive:.1%}",
        "confidence_level": f"{confidence_level:.0%}",
    }
