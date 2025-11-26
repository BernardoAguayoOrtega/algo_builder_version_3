"""
Backtester module - simulates strategy execution on historical data.
Exact port of TradingView strategy execution model.

Key behaviors to match:
1. Stop orders: entry triggers when price crosses the stop price (can persist multiple bars)
2. Market orders (climatic volume): entry at next bar open
3. New signals REPLACE pending orders (TradingView behavior)
4. SL/TP: checked on each bar after entry
5. Order cancellation: if price hits SL level before stop order fills
6. N-bars exit: close position after N bars
7. Single position at a time (pyramiding=1)
"""
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import List, Optional
from .strategy import StrategyConfig, generate_signals


@dataclass
class Trade:
    """Represents a completed trade."""
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    direction: str  # 'long' or 'short'
    entry_price: float
    exit_price: float
    sl: float
    tp: Optional[float]
    size: float
    pnl: float
    pnl_percent: float
    exit_reason: str  # 'sl', 'tp', 'n_bars', 'end'
    bars_held: int


@dataclass
class BacktestResult:
    """Results from a backtest run."""
    trades: List[Trade]
    equity_curve: pd.Series
    signals_df: pd.DataFrame  # DataFrame with signals for visualization
    config: StrategyConfig

    # Summary metrics (calculated after init)
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    total_pnl: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_pct: float = 0.0
    profit_factor: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    avg_rr: float = 0.0
    sharpe_ratio: float = 0.0
    avg_bars_held: float = 0.0

    def calculate_metrics(self):
        """Calculate summary statistics from trades."""
        if not self.trades:
            return

        self.total_trades = len(self.trades)
        self.winning_trades = sum(1 for t in self.trades if t.pnl > 0)
        self.losing_trades = sum(1 for t in self.trades if t.pnl <= 0)
        self.win_rate = self.winning_trades / self.total_trades if self.total_trades > 0 else 0

        self.total_pnl = sum(t.pnl for t in self.trades)
        self.avg_bars_held = np.mean([t.bars_held for t in self.trades])

        # Profit factor
        gross_profit = sum(t.pnl for t in self.trades if t.pnl > 0)
        gross_loss = abs(sum(t.pnl for t in self.trades if t.pnl < 0))
        self.profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        # Average win/loss
        wins = [t.pnl for t in self.trades if t.pnl > 0]
        losses = [t.pnl for t in self.trades if t.pnl < 0]
        self.avg_win = np.mean(wins) if wins else 0
        self.avg_loss = np.mean(losses) if losses else 0
        self.avg_rr = abs(self.avg_win / self.avg_loss) if self.avg_loss != 0 else float('inf')

        # Drawdown from equity curve
        if len(self.equity_curve) > 1:
            peak = self.equity_curve.cummax()
            drawdown = self.equity_curve - peak
            self.max_drawdown = drawdown.min()
            self.max_drawdown_pct = (drawdown / peak).min() * 100

            # Sharpe ratio (annualized, assuming daily returns)
            returns = self.equity_curve.pct_change().dropna()
            if len(returns) > 0 and returns.std() > 0:
                # Annualize based on approximate trading days
                self.sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252)


def calculate_position_size(
    config: StrategyConfig,
    entry_price: float,
    sl_price: float,
    equity: float,
    point_value: float = 1.0,
) -> float:
    """
    Calculate position size based on risk management settings.

    Pine Script (lines 141-158):
        f_floor_step(x, step) => step > 0 ? math.floor(x / step) * step : x

        f_calc_qty(_entry, _sl, _risk_money) =>
            dist  = math.abs(_entry - _sl)
            valid = not na(dist) and dist > 0 and not na(_risk_money) and _risk_money > 0
            raw   = valid ? (_risk_money / (dist * syminfo.pointvalue)) : na
            f_floor_step(raw, qty_step)
    """
    if config.risk_type == "fixed_size":
        size = config.fixed_size
    elif config.risk_type == "fixed_money":
        dist = abs(entry_price - sl_price)
        if dist > 0 and point_value > 0:
            size = config.fixed_risk_money / (dist * point_value)
        else:
            size = 0
    else:  # percent_equity
        risk_money = equity * (config.risk_percent / 100)
        dist = abs(entry_price - sl_price)
        if dist > 0 and point_value > 0:
            size = risk_money / (dist * point_value)
        else:
            size = 0

    # Apply qty_step (floor to step)
    if config.qty_step > 0:
        size = np.floor(size / config.qty_step) * config.qty_step

    # Check minimum
    if size < config.qty_min:
        size = 0

    return size


def run_backtest(df: pd.DataFrame, config: StrategyConfig) -> BacktestResult:
    """
    Run backtest on historical data.

    Simulates TradingView's strategy execution model:
    1. Signals generated on bar close
    2. Stop orders persist until filled, cancelled (SL hit), or replaced by new signal
    3. Market orders (climatic volume): fill at next bar open
    4. New signals REPLACE existing pending orders
    5. Exits: SL/TP checked each bar, N-bars exit at bar close

    Args:
        df: DataFrame with OHLCV data
        config: Strategy configuration

    Returns:
        BacktestResult with trades and metrics
    """
    # Generate signals
    signals_df = generate_signals(df, config)

    trades: List[Trade] = []
    equity = config.initial_capital
    equity_history = [equity]
    equity_times = [signals_df.index[0]]

    # Position state
    position = None  # {'direction', 'entry_price', 'sl', 'tp', 'size', 'entry_time', 'entry_bar', 'bars'}

    # Pending order state (for stop orders) - persists until filled/cancelled/replaced
    pending_order = None  # {'direction', 'stop_price', 'sl', 'tp', 'signal_bar', 'is_climatic'}

    for i in range(1, len(signals_df)):  # Start from 1 to have previous bar
        row = signals_df.iloc[i]
        prev_row = signals_df.iloc[i - 1]
        time = signals_df.index[i]
        bar_open = row["open"]
        bar_high = row["high"]
        bar_low = row["low"]
        bar_close = row["close"]

        # =====================================================================
        # 1. If in position, check for exit FIRST
        # =====================================================================
        if position is not None:
            exit_price = None
            exit_reason = None

            if position["direction"] == "long":
                # Check SL (Pine Script uses stop order for exit)
                if config.use_sl and bar_low <= position["sl"]:
                    exit_price = min(bar_open, position["sl"])  # Fill at SL or gap down
                    exit_reason = "sl"
                # Check TP
                elif config.use_tp_ratio and position["tp"] is not None and bar_high >= position["tp"]:
                    exit_price = max(bar_open, position["tp"])  # Fill at TP or gap up
                    exit_reason = "tp"
                # Check N bars exit
                elif config.use_n_bars_exit and position["bars"] >= config.n_bars_exit:
                    exit_price = bar_close
                    exit_reason = "n_bars"
            else:  # short
                # Check SL
                if config.use_sl and bar_high >= position["sl"]:
                    exit_price = max(bar_open, position["sl"])  # Fill at SL or gap up
                    exit_reason = "sl"
                # Check TP
                elif config.use_tp_ratio and position["tp"] is not None and bar_low <= position["tp"]:
                    exit_price = min(bar_open, position["tp"])  # Fill at TP or gap down
                    exit_reason = "tp"
                # Check N bars exit
                elif config.use_n_bars_exit and position["bars"] >= config.n_bars_exit:
                    exit_price = bar_close
                    exit_reason = "n_bars"

            if exit_price is not None:
                # Calculate P&L
                if position["direction"] == "long":
                    pnl = (exit_price - position["entry_price"]) * position["size"]
                else:
                    pnl = (position["entry_price"] - exit_price) * position["size"]

                # Apply commission (entry + exit)
                pnl -= config.commission * 2

                # Apply slippage
                slippage_cost = config.slippage * config.pip_size * position["size"] * 2
                pnl -= slippage_cost

                pnl_pct = (pnl / equity) * 100 if equity > 0 else 0

                trade = Trade(
                    entry_time=position["entry_time"],
                    exit_time=time,
                    direction=position["direction"],
                    entry_price=position["entry_price"],
                    exit_price=exit_price,
                    sl=position["sl"],
                    tp=position["tp"],
                    size=position["size"],
                    pnl=pnl,
                    pnl_percent=pnl_pct,
                    exit_reason=exit_reason,
                    bars_held=position["bars"],
                )
                trades.append(trade)

                equity += pnl
                equity_history.append(equity)
                equity_times.append(time)

                position = None
            else:
                # Still in position, increment bar counter
                position["bars"] += 1

        # =====================================================================
        # 2. Check for new signals - handles REVERSALS
        # Pine Script behavior:
        #   - strategy.entry('Long', ...) when short -> closes short, opens long
        #   - strategy.entry('Short', ...) when long -> closes long, opens short
        #   - New signals REPLACE pending orders
        # =====================================================================
        new_signal = None

        # Check signals from previous bar
        # In Pine Script: if long_ok and strategy.position_size <= 0 (can be 0 or negative/short)
        # This means long signals fire when flat OR short
        if prev_row["long_signal"]:
            # Can fire if flat or SHORT (position_size <= 0)
            can_fire = position is None or position["direction"] == "short"
            if can_fire:
                new_signal = {
                    "direction": "long",
                    "stop_price": prev_row["entry_long"],
                    "sl": prev_row["sl_long"],
                    "tp": prev_row["tp_long"] if config.use_tp_ratio else None,
                    "signal_bar": i - 1,
                    "is_climatic": prev_row["is_climatic_long"],
                }
        elif prev_row["short_signal"]:
            # Can fire if flat or LONG (position_size >= 0)
            can_fire = position is None or position["direction"] == "long"
            if can_fire:
                new_signal = {
                    "direction": "short",
                    "stop_price": prev_row["entry_short"],
                    "sl": prev_row["sl_short"],
                    "tp": prev_row["tp_short"] if config.use_tp_ratio else None,
                    "signal_bar": i - 1,
                    "is_climatic": prev_row["is_climatic_short"],
                }

        # If new signal fires while in opposite position -> REVERSAL
        if new_signal is not None and position is not None:
            if new_signal["direction"] != position["direction"]:
                # Close current position at market (bar open)
                exit_price = bar_open

                if position["direction"] == "long":
                    pnl = (exit_price - position["entry_price"]) * position["size"]
                else:
                    pnl = (position["entry_price"] - exit_price) * position["size"]

                pnl -= config.commission * 2
                pnl -= config.slippage * config.pip_size * position["size"] * 2
                pnl_pct = (pnl / equity) * 100 if equity > 0 else 0

                trade = Trade(
                    entry_time=position["entry_time"],
                    exit_time=time,
                    direction=position["direction"],
                    entry_price=position["entry_price"],
                    exit_price=exit_price,
                    sl=position["sl"],
                    tp=position["tp"],
                    size=position["size"],
                    pnl=pnl,
                    pnl_percent=pnl_pct,
                    exit_reason="reversal",
                    bars_held=position["bars"],
                )
                trades.append(trade)

                equity += pnl
                equity_history.append(equity)
                equity_times.append(time)

                position = None

        # Replace pending order with new signal (TradingView behavior)
        if new_signal is not None:
            pending_order = new_signal

        # =====================================================================
        # 3. Check if pending order should be cancelled (SL hit before entry)
        # Pine Script (lines 285-287):
        #   if strategy.position_size <= 0 and low < stoplossL: strategy.cancel('Long Entry')
        #   if strategy.position_size >= 0 and high > stoplossS: strategy.cancel('Short Entry')
        # =====================================================================
        if pending_order is not None and position is None:
            if pending_order["direction"] == "long":
                # Cancel if price hit SL before entry
                if bar_low <= pending_order["sl"]:
                    pending_order = None
            else:  # short
                if bar_high >= pending_order["sl"]:
                    pending_order = None

        # =====================================================================
        # 4. Check if pending order fills
        # =====================================================================
        if pending_order is not None and position is None:
            fill_price = None

            if pending_order["is_climatic"]:
                # Market order: fill at this bar's open
                fill_price = bar_open
            else:
                # Stop order: fill if price crosses stop level
                if pending_order["direction"] == "long":
                    if bar_high >= pending_order["stop_price"]:
                        # Long stop order triggered - fill at stop price or open (whichever is worse)
                        fill_price = max(bar_open, pending_order["stop_price"])
                else:  # short
                    if bar_low <= pending_order["stop_price"]:
                        # Short stop order triggered - fill at stop price or open (whichever is worse)
                        fill_price = min(bar_open, pending_order["stop_price"])

            if fill_price is not None:
                # Calculate position size at fill time
                size = calculate_position_size(
                    config,
                    fill_price,
                    pending_order["sl"],
                    equity,
                )

                if size > 0:
                    position = {
                        "direction": pending_order["direction"],
                        "entry_price": fill_price,
                        "sl": pending_order["sl"],
                        "tp": pending_order["tp"],
                        "size": size,
                        "entry_time": time,
                        "entry_bar": i,
                        "bars": 0,
                    }

                # Order filled (or rejected due to size), clear pending
                pending_order = None

    # =========================================================================
    # Close any remaining position at last bar
    # =========================================================================
    if position is not None:
        last_row = signals_df.iloc[-1]
        last_time = signals_df.index[-1]
        exit_price = last_row["close"]

        if position["direction"] == "long":
            pnl = (exit_price - position["entry_price"]) * position["size"]
        else:
            pnl = (position["entry_price"] - exit_price) * position["size"]

        pnl -= config.commission * 2
        pnl -= config.slippage * config.pip_size * position["size"] * 2
        pnl_pct = (pnl / equity) * 100 if equity > 0 else 0

        trade = Trade(
            entry_time=position["entry_time"],
            exit_time=last_time,
            direction=position["direction"],
            entry_price=position["entry_price"],
            exit_price=exit_price,
            sl=position["sl"],
            tp=position["tp"],
            size=position["size"],
            pnl=pnl,
            pnl_percent=pnl_pct,
            exit_reason="end",
            bars_held=position["bars"],
        )
        trades.append(trade)
        equity += pnl
        equity_history.append(equity)
        equity_times.append(last_time)

    # Build equity curve
    equity_curve = pd.Series(equity_history, index=equity_times)

    # Calculate result
    result = BacktestResult(
        trades=trades,
        equity_curve=equity_curve,
        signals_df=signals_df,
        config=config,
    )
    result.calculate_metrics()

    return result
