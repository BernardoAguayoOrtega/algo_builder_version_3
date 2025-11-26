"""
Optimizer module - automated parameter optimization for strategy configurations.
Tests all combinations of entries, exits, filters, sessions, and days to find the best configuration.
"""
import pandas as pd
import numpy as np
from dataclasses import dataclass, replace
from typing import List, Dict, Optional, Callable, Tuple
from itertools import product
import concurrent.futures
from .strategy import StrategyConfig
from .backtester import run_backtest, BacktestResult


@dataclass
class OptimizationSettings:
    """Settings for what to optimize."""
    # Entry patterns
    optimize_entries: bool = False
    entry_options: List[Tuple[bool, bool, bool]] = None  # (sacudida, engulfing, climatic)

    # Exit configurations
    optimize_exits: bool = False
    tp_ratio_options: List[float] = None  # e.g., [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    n_bars_options: List[int] = None  # e.g., [3, 5, 7, 10]

    # MA Filter options - which values to test
    optimize_ma_filter: bool = True
    ma_filter_options: List[str] = None  # Which MA filters to test

    # Session options - which combinations to test
    optimize_sessions: bool = True
    session_options: List[Tuple[bool, bool, bool]] = None  # (london, ny, tokyo)

    # Day options
    optimize_days: bool = False
    day_options: List[Tuple[bool, bool, bool, bool, bool, bool, bool]] = None

    def __post_init__(self):
        # Default entry options: all combinations where at least one is True
        if self.entry_options is None:
            self.entry_options = [
                combo for combo in product([True, False], repeat=3)
                if any(combo)
            ]
        # Default TP ratios
        if self.tp_ratio_options is None:
            self.tp_ratio_options = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
        # Default N bars
        if self.n_bars_options is None:
            self.n_bars_options = [3, 5, 7, 10, 15]
        # Default MA filter options (all three)
        if self.ma_filter_options is None:
            self.ma_filter_options = ["Sin filtro", "Alcista (MM50>200)", "Bajista (MM50<200)"]
        # Default session options (all combinations where at least one is True)
        if self.session_options is None:
            self.session_options = [
                combo for combo in product([True, False], repeat=3)
                if any(combo)
            ]
        # Default day options (all combinations where at least one is True)
        if self.day_options is None:
            self.day_options = [
                combo for combo in product([True, False], repeat=7)
                if any(combo)
            ]


@dataclass
class OptimizationResult:
    """Results from a single optimization run."""
    config: StrategyConfig
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    total_pnl: float
    profit_factor: float
    max_drawdown_pct: float
    sharpe_ratio: float
    avg_bars_held: float
    mar_ratio: float  # Mean Annual Return / Max Drawdown

    # Config summary for display
    config_summary: str


def calculate_mar_ratio(result: BacktestResult, df: pd.DataFrame) -> float:
    """
    Calculate MAR ratio (Mean Annual Return / Max Drawdown).

    MAR = Annualized Return / Max Drawdown (absolute)

    Args:
        result: BacktestResult from backtest
        df: DataFrame with price data (for date range calculation)

    Returns:
        MAR ratio (higher is better)
    """
    if result.max_drawdown_pct == 0 or result.total_trades == 0:
        return 0.0

    # Calculate time span in years
    date_range = df.index[-1] - df.index[0]
    years = date_range.days / 365.25

    if years <= 0:
        return 0.0

    # Annualized return
    total_return_pct = (result.total_pnl / result.config.initial_capital) * 100
    annualized_return = total_return_pct / years

    # MAR ratio (absolute drawdown for division)
    mar = annualized_return / abs(result.max_drawdown_pct)

    return mar


def get_config_summary(config: StrategyConfig) -> str:
    """Generate a human-readable summary of the configuration."""
    parts = []

    # Entry patterns
    entries = []
    if config.use_sacudida:
        entries.append("Sac")
    if config.use_engulfing:
        entries.append("Eng")
    if config.use_climatic_volume:
        entries.append("Clim")
    if entries:
        parts.append(f"E:{'+'.join(entries)}")

    # Exits
    exits = []
    if config.use_sl:
        exits.append("SL")
    if config.use_tp_ratio:
        exits.append(f"TP:{config.tp_ratio}")
    if config.use_n_bars_exit:
        exits.append(f"N:{config.n_bars_exit}")
    if exits:
        parts.append(f"X:{'+'.join(exits)}")

    # MA Filter
    if config.ma_filter == "Sin filtro":
        parts.append("MA:Off")
    elif config.ma_filter == "Alcista (MM50>200)":
        parts.append("MA:Bull")
    else:
        parts.append("MA:Bear")

    # Sessions
    sessions = []
    if config.use_london:
        sessions.append("L")
    if config.use_newyork:
        sessions.append("NY")
    if config.use_tokyo:
        sessions.append("T")
    if sessions:
        parts.append(f"Sess:{'+'.join(sessions)}")
    else:
        parts.append("Sess:All")

    # Days
    days_enabled = sum([
        config.trade_monday, config.trade_tuesday, config.trade_wednesday,
        config.trade_thursday, config.trade_friday, config.trade_saturday,
        config.trade_sunday
    ])

    if days_enabled == 7:
        parts.append("Days:All")
    else:
        day_abbr = []
        if config.trade_monday: day_abbr.append("M")
        if config.trade_tuesday: day_abbr.append("Tu")
        if config.trade_wednesday: day_abbr.append("W")
        if config.trade_thursday: day_abbr.append("Th")
        if config.trade_friday: day_abbr.append("F")
        if config.trade_saturday: day_abbr.append("Sa")
        if config.trade_sunday: day_abbr.append("Su")
        parts.append(f"Days:{'+'.join(day_abbr)}")

    return " | ".join(parts)


def generate_parameter_combinations(
    base_config: StrategyConfig,
    settings: OptimizationSettings,
    max_combinations: Optional[int] = None,
) -> List[StrategyConfig]:
    """
    Generate parameter combinations for optimization based on settings.

    Args:
        base_config: Base configuration to modify
        settings: OptimizationSettings specifying what to optimize
        max_combinations: Maximum number of combinations

    Returns:
        List of StrategyConfig objects to test
    """
    # Build options for each dimension
    # Entry patterns
    if settings.optimize_entries:
        entry_options = settings.entry_options
    else:
        entry_options = [(
            base_config.use_sacudida,
            base_config.use_engulfing,
            base_config.use_climatic_volume
        )]

    # Exit configurations - build combinations
    exit_options = []
    if settings.optimize_exits:
        # Generate exit combinations: (use_tp, tp_ratio, use_n_bars, n_bars)
        # Always keep SL on
        for tp_ratio in settings.tp_ratio_options:
            # TP only
            exit_options.append((True, tp_ratio, False, base_config.n_bars_exit))

        for n_bars in settings.n_bars_options:
            # N-bars only
            exit_options.append((False, base_config.tp_ratio, True, n_bars))

        # Both TP and N-bars combinations
        for tp_ratio in settings.tp_ratio_options:
            for n_bars in settings.n_bars_options:
                exit_options.append((True, tp_ratio, True, n_bars))
    else:
        exit_options = [(
            base_config.use_tp_ratio,
            base_config.tp_ratio,
            base_config.use_n_bars_exit,
            base_config.n_bars_exit
        )]

    # MA filter
    if settings.optimize_ma_filter:
        ma_options = settings.ma_filter_options
    else:
        ma_options = [base_config.ma_filter]

    # Sessions
    if settings.optimize_sessions:
        session_options = settings.session_options
    else:
        session_options = [(base_config.use_london, base_config.use_newyork, base_config.use_tokyo)]

    # Days
    if settings.optimize_days:
        day_options = settings.day_options
    else:
        day_options = [(
            base_config.trade_monday, base_config.trade_tuesday, base_config.trade_wednesday,
            base_config.trade_thursday, base_config.trade_friday, base_config.trade_saturday,
            base_config.trade_sunday
        )]

    # Generate all combinations
    configs = []
    for entries, exits, ma, sessions, days in product(
        entry_options, exit_options, ma_options, session_options, day_options
    ):
        sacudida, engulfing, climatic = entries
        use_tp, tp_ratio, use_n_bars, n_bars = exits
        london, newyork, tokyo = sessions
        mon, tue, wed, thu, fri, sat, sun = days

        new_config = replace(
            base_config,
            # Entries
            use_sacudida=sacudida,
            use_engulfing=engulfing,
            use_climatic_volume=climatic,
            # Exits
            use_tp_ratio=use_tp,
            tp_ratio=tp_ratio,
            use_n_bars_exit=use_n_bars,
            n_bars_exit=n_bars,
            # Filters
            ma_filter=ma,
            use_london=london,
            use_newyork=newyork,
            use_tokyo=tokyo,
            trade_monday=mon,
            trade_tuesday=tue,
            trade_wednesday=wed,
            trade_thursday=thu,
            trade_friday=fri,
            trade_saturday=sat,
            trade_sunday=sun,
        )
        configs.append(new_config)

    # Apply max_combinations limit if specified
    if max_combinations is not None and len(configs) > max_combinations:
        # Prioritize diversity: sample evenly from the list
        step = len(configs) // max_combinations
        configs = configs[::step][:max_combinations]

    return configs


def generate_parameter_combinations_legacy(
    base_config: StrategyConfig,
    level: str = "medium",
    max_combinations: Optional[int] = None,
) -> List[StrategyConfig]:
    """
    Legacy function for backward compatibility.
    Generate parameter combinations based on optimization level.

    Levels:
    - quick: MA filter only (3 combinations)
    - medium: MA filter + sessions (~21 combinations)
    - full: MA filter + sessions + days (~2,667 combinations)
    """
    settings = OptimizationSettings(
        optimize_entries=False,
        optimize_exits=False,
        optimize_ma_filter=True,
        optimize_sessions=(level in ["medium", "full"]),
        optimize_days=(level == "full"),
    )
    return generate_parameter_combinations(base_config, settings, max_combinations)


def run_single_optimization(
    df: pd.DataFrame,
    config: StrategyConfig,
) -> OptimizationResult:
    """
    Run a single backtest and return optimization result.

    Args:
        df: DataFrame with price data
        config: Strategy configuration

    Returns:
        OptimizationResult with all metrics
    """
    result = run_backtest(df, config)
    mar = calculate_mar_ratio(result, df)

    return OptimizationResult(
        config=config,
        total_trades=result.total_trades,
        winning_trades=result.winning_trades,
        losing_trades=result.losing_trades,
        win_rate=result.win_rate,
        total_pnl=result.total_pnl,
        profit_factor=result.profit_factor,
        max_drawdown_pct=result.max_drawdown_pct,
        sharpe_ratio=result.sharpe_ratio,
        avg_bars_held=result.avg_bars_held,
        mar_ratio=mar,
        config_summary=get_config_summary(config),
    )


def run_optimization(
    df: pd.DataFrame,
    base_config: StrategyConfig,
    settings: Optional[OptimizationSettings] = None,
    max_combinations: Optional[int] = None,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
    min_trades: int = 10,
) -> List[OptimizationResult]:
    """
    Run parameter optimization over all combinations.

    Args:
        df: DataFrame with price data
        base_config: Base configuration with fixed parameters (risk, etc.)
        settings: OptimizationSettings specifying what to optimize
        max_combinations: Maximum combinations to test
        progress_callback: Optional callback(current, total, message) for progress
        min_trades: Minimum trades required to include result

    Returns:
        List of OptimizationResult sorted by MAR ratio (descending)
    """
    # Use default settings if not provided
    if settings is None:
        settings = OptimizationSettings()

    # Generate configurations
    configs = generate_parameter_combinations(base_config, settings, max_combinations)
    total = len(configs)

    if progress_callback:
        progress_callback(0, total, f"Testing {total} configurations...")

    results = []

    for i, config in enumerate(configs):
        if progress_callback:
            progress_callback(i + 1, total, f"Testing {i + 1}/{total}")

        try:
            opt_result = run_single_optimization(df, config)

            # Filter by minimum trades
            if opt_result.total_trades >= min_trades:
                results.append(opt_result)
        except Exception:
            # Skip failed configurations
            continue

    # Sort by MAR ratio (descending)
    results.sort(key=lambda x: x.mar_ratio, reverse=True)

    return results


def results_to_dataframe(results: List[OptimizationResult]) -> pd.DataFrame:
    """
    Convert optimization results to a DataFrame for display.

    Args:
        results: List of OptimizationResult

    Returns:
        DataFrame with all metrics
    """
    if not results:
        return pd.DataFrame()

    data = []
    for i, r in enumerate(results, 1):
        data.append({
            "Rank": i,
            "Configuration": r.config_summary,
            "Trades": r.total_trades,
            "Win Rate": f"{r.win_rate:.1%}",
            "Net P&L": f"${r.total_pnl:,.2f}",
            "Profit Factor": f"{r.profit_factor:.2f}",
            "Max DD %": f"{r.max_drawdown_pct:.1f}%",
            "Sharpe": f"{r.sharpe_ratio:.2f}",
            "MAR": f"{r.mar_ratio:.2f}",
            "Avg Bars": f"{r.avg_bars_held:.1f}",
        })

    return pd.DataFrame(data)


def estimate_combinations(settings: OptimizationSettings) -> int:
    """Estimate the number of combinations based on settings."""
    count = 1

    if settings.optimize_entries:
        count *= len(settings.entry_options)  # 7 by default

    if settings.optimize_exits:
        # TP only + N-bars only + both
        tp_count = len(settings.tp_ratio_options)  # 6
        n_bars_count = len(settings.n_bars_options)  # 5
        count *= (tp_count + n_bars_count + tp_count * n_bars_count)  # 6 + 5 + 30 = 41

    if settings.optimize_ma_filter:
        count *= len(settings.ma_filter_options)

    if settings.optimize_sessions:
        count *= len(settings.session_options)

    if settings.optimize_days:
        count *= len(settings.day_options)

    return count


def get_settings_description(settings: OptimizationSettings) -> str:
    """Get a human-readable description of what will be optimized."""
    parts = []
    if settings.optimize_entries:
        parts.append("Entries")
    if settings.optimize_exits:
        parts.append("Exits")
    if settings.optimize_ma_filter:
        parts.append("MA Filter")
    if settings.optimize_sessions:
        parts.append("Sessions")
    if settings.optimize_days:
        parts.append("Days")

    return " + ".join(parts) if parts else "None"


def get_level_info(level: str) -> Tuple[int, str]:
    """
    Get information about an optimization level (legacy).

    Args:
        level: 'quick', 'medium', or 'full'

    Returns:
        Tuple of (approximate_combinations, description)
    """
    if level == "quick":
        return 3, "MA filter only (3 combinations)"
    elif level == "medium":
        return 21, "MA filter + sessions (21 combinations)"
    else:  # full
        return 2667, "MA filter + sessions + days (~2,667 combinations)"
