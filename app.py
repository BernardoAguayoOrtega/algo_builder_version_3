"""
Algo Strategy Builder - Streamlit App
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

from src.data import fetch_data, get_symbol_info, get_available_timeframes, TIMEFRAME_LIMITS
from src.strategy import StrategyConfig
from src.backtester import run_backtest, BacktestResult
from src.optimizer import (
    run_optimization, results_to_dataframe, OptimizationSettings,
    estimate_combinations, get_settings_description
)


# Page config
st.set_page_config(
    page_title="Algo Strategy Builder",
    page_icon="📈",
    layout="wide",
)

st.title("📈 Algo Strategy Builder")


# =============================================================================
# Sidebar - Data Configuration
# =============================================================================
st.sidebar.header("📊 Data")

# Symbol input
symbol = st.sidebar.text_input(
    "Symbol",
    value="SPY",
    help="Yahoo Finance symbol (e.g., SPY, AAPL, EURUSD=X, BTC-USD)"
)

# Timeframe selector
timeframes = get_available_timeframes()
timeframe = st.sidebar.selectbox(
    "Timeframe",
    options=timeframes,
    index=timeframes.index("1d"),
)

# Date range - allow selection back to 2000
col1, col2 = st.sidebar.columns(2)
limit = TIMEFRAME_LIMITS.get(timeframe)
default_days = min(limit * 2, 365) if limit else 365 * 2

# Show data availability hint for intraday timeframes
if limit:
    st.sidebar.caption(f"Note: {timeframe} data limited to last {limit} days")

with col1:
    start_date = st.date_input(
        "Start",
        value=datetime.now() - timedelta(days=default_days),
        min_value=datetime(2000, 1, 1),
    )
with col2:
    end_date = st.date_input("End", value=datetime.now())

# Load button
load_button = st.sidebar.button("🔄 Load Data", type="primary", use_container_width=True)

# =============================================================================
# Sidebar - Strategy Configuration
# =============================================================================
st.sidebar.header("⚙️ Strategy")

# Direction
st.sidebar.subheader("Sentido")
col1, col2 = st.sidebar.columns(2)
with col1:
    trade_longs = st.checkbox("Largos", value=True)
with col2:
    trade_shorts = st.checkbox("Cortos", value=True)

# Entry Patterns
st.sidebar.subheader("Entradas")
use_sacudida = st.sidebar.checkbox("Sacudida", value=True, help="Shakeout pattern")
use_engulfing = st.sidebar.checkbox("Envolvente", value=True, help="Engulfing pattern")
use_climatic = st.sidebar.checkbox("Volumen Climático", value=False, help="Climatic volume")

# Entry/SL Configuration
st.sidebar.subheader("Configuración Básica")
col1, col2 = st.sidebar.columns(2)
with col1:
    entry_pips = st.number_input("Pips entrada", min_value=0, value=1, step=1)
with col2:
    sl_pips = st.number_input("Pips SL", min_value=0, value=1, step=1)

pip_size = st.sidebar.number_input(
    "Tamaño pip",
    min_value=0.0001,
    value=0.01,
    step=0.0001,
    format="%.4f",
    help="0.0001 forex, 0.01 stocks, 1.0 crypto",
)

# Exit Configuration
st.sidebar.subheader("Salidas")
use_sl = st.sidebar.checkbox("SL Original", value=True)
use_tp = st.sidebar.checkbox("TP por Ratio", value=True)
if use_tp:
    tp_ratio = st.sidebar.slider("Ratio TP", min_value=0.5, max_value=5.0, value=1.0, step=0.1)
else:
    tp_ratio = 1.0

use_n_bars = st.sidebar.checkbox("Salida N velas", value=False)
if use_n_bars:
    n_bars = st.sidebar.number_input("N velas", min_value=1, value=5, step=1)
else:
    n_bars = 5

# Filters
st.sidebar.subheader("Filtros")
ma_filter = st.sidebar.selectbox(
    "Cruce MM 50/200",
    options=["Sin filtro", "Alcista (MM50>200)", "Bajista (MM50<200)"],
)

# Sessions
st.sidebar.subheader("Sesiones")
col1, col2, col3 = st.sidebar.columns(3)
with col1:
    use_london = st.checkbox("London", value=True)
with col2:
    use_newyork = st.checkbox("NY", value=True)
with col3:
    use_tokyo = st.checkbox("Tokyo", value=True)

# Days of week
with st.sidebar.expander("Días de la semana"):
    col1, col2 = st.columns(2)
    with col1:
        trade_mon = st.checkbox("Lun", value=True)
        trade_tue = st.checkbox("Mar", value=True)
        trade_wed = st.checkbox("Mié", value=True)
        trade_thu = st.checkbox("Jue", value=True)
    with col2:
        trade_fri = st.checkbox("Vie", value=True)
        trade_sat = st.checkbox("Sáb", value=True)
        trade_sun = st.checkbox("Dom", value=True)

# Risk Management
st.sidebar.subheader("Gestión de Riesgo")
risk_type = st.sidebar.selectbox(
    "Tipo",
    options=["fixed_size", "fixed_money", "percent_equity"],
    format_func=lambda x: {"fixed_size": "Tamaño fijo", "fixed_money": "Riesgo $ fijo", "percent_equity": "% Equity"}[x],
)

if risk_type == "fixed_size":
    fixed_size = st.sidebar.number_input("Tamaño", min_value=0.01, value=1.0, step=0.01)
    fixed_risk = 100.0
    risk_pct = 1.0
elif risk_type == "fixed_money":
    fixed_size = 1.0
    fixed_risk = st.sidebar.number_input("Riesgo $", min_value=1.0, value=100.0, step=10.0)
    risk_pct = 1.0
else:
    fixed_size = 1.0
    fixed_risk = 100.0
    risk_pct = st.sidebar.slider("Riesgo %", min_value=0.1, max_value=10.0, value=1.0, step=0.1)

initial_capital = st.sidebar.number_input("Capital inicial", min_value=1000, value=100000, step=1000)
commission = st.sidebar.number_input("Comisión", min_value=0.0, value=1.5, step=0.5)

# Build config
config = StrategyConfig(
    entry_pips=entry_pips,
    sl_pips=sl_pips,
    pip_size=pip_size,
    trade_longs=trade_longs,
    trade_shorts=trade_shorts,
    use_sacudida=use_sacudida,
    use_engulfing=use_engulfing,
    use_climatic_volume=use_climatic,
    use_sl=use_sl,
    use_tp_ratio=use_tp,
    tp_ratio=tp_ratio,
    use_n_bars_exit=use_n_bars,
    n_bars_exit=n_bars,
    ma_filter=ma_filter,
    use_london=use_london,
    use_newyork=use_newyork,
    use_tokyo=use_tokyo,
    trade_monday=trade_mon,
    trade_tuesday=trade_tue,
    trade_wednesday=trade_wed,
    trade_thursday=trade_thu,
    trade_friday=trade_fri,
    trade_saturday=trade_sat,
    trade_sunday=trade_sun,
    risk_type=risk_type,
    fixed_size=fixed_size,
    fixed_risk_money=fixed_risk,
    risk_percent=risk_pct,
    initial_capital=initial_capital,
    commission=commission,
)

# Run Backtest button
run_backtest_btn = st.sidebar.button("🚀 Run Backtest", type="primary", use_container_width=True)


# =============================================================================
# Helper Functions
# =============================================================================

def create_chart_with_trades(df: pd.DataFrame, result: BacktestResult, symbol: str, timeframe: str) -> go.Figure:
    """Create candlestick chart with trade markers."""
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.75, 0.25],
        subplot_titles=(f"{symbol} - {timeframe}", "Volume"),
    )

    # Candlestick
    fig.add_trace(
        go.Candlestick(
            x=df.index, open=df["open"], high=df["high"], low=df["low"], close=df["close"],
            name="Price", increasing_line_color="#26a69a", decreasing_line_color="#ef5350",
        ),
        row=1, col=1,
    )

    # SMAs
    if len(df) >= 50:
        sma50 = df["close"].rolling(window=50).mean()
        fig.add_trace(go.Scatter(x=df.index, y=sma50, mode="lines", name="SMA 50",
                                  line=dict(color="#2196f3", width=1)), row=1, col=1)
    if len(df) >= 200:
        sma200 = df["close"].rolling(window=200).mean()
        fig.add_trace(go.Scatter(x=df.index, y=sma200, mode="lines", name="SMA 200",
                                  line=dict(color="#ff9800", width=1)), row=1, col=1)

    # Volume
    colors = ["#26a69a" if c >= o else "#ef5350" for o, c in zip(df["open"], df["close"])]
    fig.add_trace(go.Bar(x=df.index, y=df["volume"], name="Volume", marker_color=colors, opacity=0.7), row=2, col=1)

    # Trade markers
    for trade in result.trades:
        # Entry marker
        entry_color = "#26a69a" if trade.direction == "long" else "#ef5350"
        entry_symbol = "triangle-up" if trade.direction == "long" else "triangle-down"
        fig.add_trace(go.Scatter(
            x=[trade.entry_time], y=[trade.entry_price],
            mode="markers", marker=dict(symbol=entry_symbol, size=12, color=entry_color),
            name=f"{trade.direction} entry", showlegend=False,
        ), row=1, col=1)

        # Exit marker
        exit_color = "#4caf50" if trade.pnl > 0 else "#f44336"
        fig.add_trace(go.Scatter(
            x=[trade.exit_time], y=[trade.exit_price],
            mode="markers", marker=dict(symbol="x", size=10, color=exit_color),
            name=f"exit ({trade.exit_reason})", showlegend=False,
        ), row=1, col=1)

    fig.update_layout(
        height=700, template="plotly_dark", xaxis_rangeslider_visible=False,
        showlegend=True, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=50, r=50, t=80, b=50),
    )
    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)

    return fig


def create_equity_chart(result: BacktestResult) -> go.Figure:
    """Create equity curve chart."""
    fig = go.Figure()
    color = "#4caf50" if result.total_pnl > 0 else "#f44336"

    # Calculate min/max for proper y-axis range
    eq_min = result.equity_curve.min()
    eq_max = result.equity_curve.max()
    eq_range = eq_max - eq_min

    # Add some padding to the range
    y_min = eq_min - eq_range * 0.1 if eq_range > 0 else eq_min * 0.99
    y_max = eq_max + eq_range * 0.1 if eq_range > 0 else eq_max * 1.01

    fig.add_trace(go.Scatter(
        x=result.equity_curve.index, y=result.equity_curve.values,
        mode="lines+markers", name="Equity",
        line=dict(color=color, width=2),
        marker=dict(size=4),
    ))

    # Add initial capital reference line
    fig.add_hline(y=result.config.initial_capital, line_dash="dash",
                  line_color="gray", annotation_text="Initial Capital")

    fig.update_layout(
        height=350, template="plotly_dark",
        title=f"Equity Curve (P&L: ${result.total_pnl:,.2f})",
        xaxis_title="Date", yaxis_title="Equity ($)",
        margin=dict(l=50, r=50, t=50, b=50),
        yaxis=dict(range=[y_min, y_max]),  # Auto-scale to show changes
    )
    return fig


# =============================================================================
# Main Content
# =============================================================================

# Load data
if load_button:
    with st.spinner(f"Fetching {symbol} data..."):
        try:
            progress_container = st.empty()
            def update_progress(current, total, message):
                progress_container.text(f"📡 {message}")

            result = fetch_data(
                symbol=symbol, timeframe=timeframe,
                start_date=datetime.combine(start_date, datetime.min.time()),
                end_date=datetime.combine(end_date, datetime.max.time()),
                progress_callback=update_progress,
            )
            progress_container.empty()

            df = result.df
            st.session_state["df"] = df
            st.session_state["symbol"] = symbol
            st.session_state["timeframe"] = timeframe
            st.session_state["symbol_info"] = get_symbol_info(symbol)
            st.session_state.pop("result", None)  # Clear previous backtest

            st.success(f"✅ Loaded {len(df):,} bars from {df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')}")

            # Show warning if data range differs from requested
            if result.message:
                st.warning(result.message)
        except Exception as e:
            st.error(f"❌ Error: {e}")

# Run backtest
if run_backtest_btn and "df" in st.session_state:
    with st.spinner("Running backtest..."):
        try:
            # Use applied optimization config if available, otherwise use sidebar config
            backtest_config = st.session_state.get("applied_opt_config", config)
            result = run_backtest(st.session_state["df"], backtest_config)
            st.session_state["result"] = result
            st.session_state["config"] = backtest_config
        except Exception as e:
            st.error(f"❌ Backtest error: {e}")

# Display results
if "df" in st.session_state:
    df = st.session_state["df"]
    symbol = st.session_state["symbol"]
    timeframe = st.session_state["timeframe"]

    # Tab navigation using selectbox to persist selection
    tab_names = ["📈 Chart", "📊 Results", "📋 Trades", "🔬 Optimizer"]

    # Get current tab from session state
    if "active_tab" not in st.session_state:
        st.session_state["active_tab"] = "📈 Chart"

    selected_tab = st.radio(
        "Navigation",
        tab_names,
        index=tab_names.index(st.session_state["active_tab"]),
        horizontal=True,
        key="tab_selector",
        label_visibility="collapsed",
    )
    st.session_state["active_tab"] = selected_tab

    st.divider()

    # Chart tab
    if selected_tab == "📈 Chart":
        if "result" in st.session_state:
            fig = create_chart_with_trades(df, st.session_state["result"], symbol, timeframe)
        else:
            # Simple chart without trades
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03,
                               row_heights=[0.75, 0.25], subplot_titles=(f"{symbol} - {timeframe}", "Volume"))
            fig.add_trace(go.Candlestick(x=df.index, open=df["open"], high=df["high"], low=df["low"], close=df["close"],
                                          name="Price", increasing_line_color="#26a69a", decreasing_line_color="#ef5350"), row=1, col=1)
            if len(df) >= 50:
                sma50 = df["close"].rolling(window=50).mean()
                fig.add_trace(go.Scatter(x=df.index, y=sma50, mode="lines", name="SMA 50", line=dict(color="#2196f3", width=1)), row=1, col=1)
            if len(df) >= 200:
                sma200 = df["close"].rolling(window=200).mean()
                fig.add_trace(go.Scatter(x=df.index, y=sma200, mode="lines", name="SMA 200", line=dict(color="#ff9800", width=1)), row=1, col=1)
            colors = ["#26a69a" if c >= o else "#ef5350" for o, c in zip(df["open"], df["close"])]
            fig.add_trace(go.Bar(x=df.index, y=df["volume"], name="Volume", marker_color=colors, opacity=0.7), row=2, col=1)
            fig.update_layout(height=700, template="plotly_dark", xaxis_rangeslider_visible=False,
                             showlegend=True, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        st.plotly_chart(fig, use_container_width=True)

    # Results tab
    elif selected_tab == "📊 Results":
        if "result" in st.session_state:
            result = st.session_state["result"]

            # Metrics
            st.subheader("Performance Summary")
            col1, col2, col3, col4, col5, col6 = st.columns(6)
            with col1:
                st.metric("Total Trades", result.total_trades)
            with col2:
                st.metric("Win Rate", f"{result.win_rate:.1%}")
            with col3:
                delta_color = "normal" if result.total_pnl >= 0 else "inverse"
                st.metric("Total P&L", f"${result.total_pnl:,.2f}")
            with col4:
                st.metric("Profit Factor", f"{result.profit_factor:.2f}")
            with col5:
                st.metric("Max Drawdown", f"{result.max_drawdown_pct:.1f}%")
            with col6:
                st.metric("Sharpe Ratio", f"{result.sharpe_ratio:.2f}")

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Winning", result.winning_trades)
            with col2:
                st.metric("Losing", result.losing_trades)
            with col3:
                st.metric("Avg Win", f"${result.avg_win:,.2f}")
            with col4:
                st.metric("Avg Loss", f"${result.avg_loss:,.2f}")

            # Equity curve
            st.subheader("Equity Curve")
            eq_fig = create_equity_chart(result)
            st.plotly_chart(eq_fig, use_container_width=True)
        else:
            st.info("👈 Click **Run Backtest** to see results")

    # Trades tab
    elif selected_tab == "📋 Trades":
        if "result" in st.session_state and st.session_state["result"].trades:
            result = st.session_state["result"]
            trades_data = [{
                "Entry": t.entry_time.strftime("%Y-%m-%d %H:%M"),
                "Exit": t.exit_time.strftime("%Y-%m-%d %H:%M"),
                "Dir": t.direction.upper(),
                "Entry $": f"{t.entry_price:.4f}",
                "Exit $": f"{t.exit_price:.4f}",
                "Size": t.size,
                "P&L": f"${t.pnl:.2f}",
                "P&L %": f"{t.pnl_percent:.2f}%",
                "Reason": t.exit_reason.upper(),
                "Bars": t.bars_held,
            } for t in result.trades]

            trades_df = pd.DataFrame(trades_data)
            st.dataframe(trades_df, use_container_width=True, hide_index=True)

            # CSV Export
            csv_data = trades_df.to_csv(index=False)
            symbol = st.session_state.get("symbol", "trades")
            timeframe = st.session_state.get("timeframe", "")
            filename = f"{symbol}_{timeframe}_trades.csv"

            st.download_button(
                label="📥 Download Trades CSV",
                data=csv_data,
                file_name=filename,
                mime="text/csv",
            )
        else:
            st.info("No trades to display")

    # Optimizer tab
    elif selected_tab == "🔬 Optimizer":
        st.subheader("Parameter Optimization")
        st.markdown("""
        Test all combinations of **entries**, **exits**, **filters**, **sessions**, and **days**
        to find the best configuration. Results are ranked by **MAR ratio** (Mean Annual Return / Max Drawdown).
        """)

        # What to optimize - Main toggles (use session state keys to persist)
        st.markdown("##### What to Optimize")
        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            opt_entries = st.checkbox("Entries", value=False, help="Test entry pattern combinations", key="opt_entries_main")
        with col2:
            opt_exits = st.checkbox("Exits", value=False, help="Test TP ratios and N-bars exit", key="opt_exits_main")
        with col3:
            opt_ma = st.checkbox("MA Filter", value=True, help="Test MA 50/200 filter options", key="opt_ma_main")
        with col4:
            opt_sessions = st.checkbox("Sessions", value=True, help="Test session combinations", key="opt_sessions_main")
        with col5:
            opt_days = st.checkbox("Days", value=False, help="Test day of week combinations", key="opt_days_main")

        # Detailed options in expanders
        # Entry pattern options
        if opt_entries:
            with st.expander("Entry Patterns to Test", expanded=True):
                col1, col2, col3 = st.columns(3)
                with col1:
                    entry_sacudida = st.checkbox("Sacudida", value=True, key="opt_sac")
                with col2:
                    entry_engulfing = st.checkbox("Engulfing", value=True, key="opt_eng")
                with col3:
                    entry_climatic = st.checkbox("Climatic Vol", value=True, key="opt_clim")

                # Generate entry combinations based on selected patterns
                selected_entries = []
                if entry_sacudida:
                    selected_entries.append(0)
                if entry_engulfing:
                    selected_entries.append(1)
                if entry_climatic:
                    selected_entries.append(2)

                # Build all combinations of selected entries
                from itertools import combinations
                entry_options = []
                for r in range(1, len(selected_entries) + 1):
                    for combo in combinations(selected_entries, r):
                        opt = [False, False, False]
                        for idx in combo:
                            opt[idx] = True
                        entry_options.append(tuple(opt))

                if not entry_options:
                    entry_options = [(True, True, False)]  # Default fallback
        else:
            entry_options = None

        # Exit options
        if opt_exits:
            with st.expander("Exit Options to Test", expanded=True):
                col1, col2 = st.columns(2)
                with col1:
                    tp_ratios_str = st.text_input(
                        "TP Ratios",
                        value="0.5, 1.0, 1.5, 2.0, 2.5, 3.0",
                        help="Comma-separated TP ratios"
                    )
                    tp_ratios = [float(x.strip()) for x in tp_ratios_str.split(",") if x.strip()]
                with col2:
                    n_bars_str = st.text_input(
                        "N-bars exit",
                        value="3, 5, 7, 10, 15",
                        help="Comma-separated N-bars values"
                    )
                    n_bars_opts = [int(x.strip()) for x in n_bars_str.split(",") if x.strip()]
        else:
            tp_ratios = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
            n_bars_opts = [3, 5, 7, 10, 15]

        # MA Filter options
        if opt_ma:
            with st.expander("MA Filter Options", expanded=True):
                col1, col2, col3 = st.columns(3)
                with col1:
                    ma_sin_filtro = st.checkbox("Sin filtro", value=True, key="ma_off")
                with col2:
                    ma_alcista = st.checkbox("Alcista (MM50>200)", value=True, key="ma_bull")
                with col3:
                    ma_bajista = st.checkbox("Bajista (MM50<200)", value=True, key="ma_bear")

                ma_filter_options = []
                if ma_sin_filtro:
                    ma_filter_options.append("Sin filtro")
                if ma_alcista:
                    ma_filter_options.append("Alcista (MM50>200)")
                if ma_bajista:
                    ma_filter_options.append("Bajista (MM50<200)")

                if not ma_filter_options:
                    ma_filter_options = ["Sin filtro"]  # Default fallback
        else:
            ma_filter_options = None

        # Session options
        if opt_sessions:
            with st.expander("Session Options", expanded=True):
                col1, col2, col3 = st.columns(3)
                with col1:
                    sess_london = st.checkbox("London", value=True, key="sess_l")
                with col2:
                    sess_ny = st.checkbox("New York", value=True, key="sess_ny")
                with col3:
                    sess_tokyo = st.checkbox("Tokyo", value=True, key="sess_t")

                # Generate session combinations
                selected_sessions = []
                if sess_london:
                    selected_sessions.append(0)
                if sess_ny:
                    selected_sessions.append(1)
                if sess_tokyo:
                    selected_sessions.append(2)

                from itertools import combinations
                session_options = []
                for r in range(1, len(selected_sessions) + 1):
                    for combo in combinations(selected_sessions, r):
                        opt = [False, False, False]
                        for idx in combo:
                            opt[idx] = True
                        session_options.append(tuple(opt))

                if not session_options:
                    session_options = [(True, True, True)]  # Default fallback
        else:
            session_options = None

        # Day options
        if opt_days:
            with st.expander("Day Options", expanded=True):
                st.markdown("Select which days to include in optimization:")
                col1, col2, col3, col4, col5, col6, col7 = st.columns(7)
                with col1:
                    day_mon = st.checkbox("Mon", value=True, key="day_m")
                with col2:
                    day_tue = st.checkbox("Tue", value=True, key="day_tu")
                with col3:
                    day_wed = st.checkbox("Wed", value=True, key="day_w")
                with col4:
                    day_thu = st.checkbox("Thu", value=True, key="day_th")
                with col5:
                    day_fri = st.checkbox("Fri", value=True, key="day_f")
                with col6:
                    day_sat = st.checkbox("Sat", value=False, key="day_sa")
                with col7:
                    day_sun = st.checkbox("Sun", value=False, key="day_su")

                # Generate day combinations
                selected_days = []
                if day_mon:
                    selected_days.append(0)
                if day_tue:
                    selected_days.append(1)
                if day_wed:
                    selected_days.append(2)
                if day_thu:
                    selected_days.append(3)
                if day_fri:
                    selected_days.append(4)
                if day_sat:
                    selected_days.append(5)
                if day_sun:
                    selected_days.append(6)

                from itertools import combinations
                day_options = []
                for r in range(1, len(selected_days) + 1):
                    for combo in combinations(selected_days, r):
                        opt = [False, False, False, False, False, False, False]
                        for idx in combo:
                            opt[idx] = True
                        day_options.append(tuple(opt))

                if not day_options:
                    day_options = [(True, True, True, True, True, False, False)]  # Default: weekdays
        else:
            day_options = None

        # Build optimization settings
        opt_settings = OptimizationSettings(
            optimize_entries=opt_entries,
            entry_options=entry_options,
            optimize_exits=opt_exits,
            tp_ratio_options=tp_ratios,
            n_bars_options=n_bars_opts,
            optimize_ma_filter=opt_ma,
            ma_filter_options=ma_filter_options,
            optimize_sessions=opt_sessions,
            session_options=session_options,
            optimize_days=opt_days,
            day_options=day_options,
        )

        # Estimate combinations
        estimated_combos = estimate_combinations(opt_settings)
        settings_desc = get_settings_description(opt_settings)

        # Limit combinations option
        col1, col2 = st.columns(2)
        with col1:
            use_max_combos = st.checkbox(
                f"Limit combinations",
                value=estimated_combos > 500,
            )
        with col2:
            if use_max_combos:
                max_combos = st.number_input(
                    "Max combinations",
                    min_value=1,
                    max_value=10000,
                    value=min(500, estimated_combos),
                    step=50,
                )
            else:
                max_combos = None

        min_trades = st.slider(
            "Minimum trades required",
            min_value=1,
            max_value=100,
            value=10,
            help="Filter out configurations with fewer trades than this",
        )

        # Show estimate
        final_combos = max_combos if max_combos else estimated_combos
        st.info(f"**Optimizing:** {settings_desc} | **Estimated:** ~{estimated_combos:,} combinations → Testing: {final_combos:,}")

        # Run optimization button
        run_opt_btn = st.button("🚀 Run Optimization", type="primary", use_container_width=True)

        if run_opt_btn:
            with st.spinner("Running optimization..."):
                progress_bar = st.progress(0)
                status_text = st.empty()

                def update_opt_progress(current, total, message):
                    progress_bar.progress(current / total if total > 0 else 0)
                    status_text.text(message)

                try:
                    opt_results = run_optimization(
                        df=st.session_state["df"],
                        base_config=config,
                        settings=opt_settings,
                        max_combinations=max_combos,
                        progress_callback=update_opt_progress,
                        min_trades=min_trades,
                    )

                    st.session_state["opt_results"] = opt_results
                    progress_bar.empty()
                    status_text.empty()

                    if opt_results:
                        st.success(f"Optimization complete! Found {len(opt_results)} valid configurations.")
                    else:
                        st.warning("No configurations met the minimum trade requirement.")
                except Exception as e:
                    st.error(f"Optimization error: {e}")
                    progress_bar.empty()
                    status_text.empty()

        # Display results
        if "opt_results" in st.session_state and st.session_state["opt_results"]:
            opt_results = st.session_state["opt_results"]

            st.subheader("Top Configurations")

            # Convert to dataframe for display
            results_df = results_to_dataframe(opt_results)
            st.dataframe(results_df, use_container_width=True, hide_index=True)

            # CSV Export for optimization results
            csv_opt = results_df.to_csv(index=False)
            symbol = st.session_state.get("symbol", "opt")
            filename_opt = f"{symbol}_optimization_results.csv"

            st.download_button(
                label="📥 Download Optimization Results CSV",
                data=csv_opt,
                file_name=filename_opt,
                mime="text/csv",
            )

            # Apply best configuration
            st.subheader("Apply Configuration")
            selected_rank = st.selectbox(
                "Select configuration to apply",
                options=list(range(len(opt_results))),
                format_func=lambda i: f"#{i+1}: {opt_results[i].config_summary} (MAR: {opt_results[i].mar_ratio:.2f})",
            )

            if st.button("✅ Apply Selected Configuration"):
                selected_config = opt_results[selected_rank].config
                st.session_state["applied_opt_config"] = selected_config
                st.success(f"Configuration #{selected_rank + 1} applied! Click 'Run Backtest' to see detailed results.")
                st.info(f"**Applied:** {opt_results[selected_rank].config_summary}")

else:
    st.info("👈 Configure data source and click **Load Data** to start")
    with st.expander("📋 Sample Symbols"):
        st.markdown("""
        **Stocks**: `AAPL`, `MSFT`, `SPY`, `QQQ`
        **Forex**: `EURUSD=X`, `GBPUSD=X`, `USDJPY=X`
        **Crypto**: `BTC-USD`, `ETH-USD`
        **Futures**: `ES=F`, `NQ=F`, `GC=F`
        """)
