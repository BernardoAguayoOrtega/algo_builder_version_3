"""
Algo Strategy Builder - Streamlit App

Flow:
1. Upload Pine Script (required) → extract all inputs dynamically
2. Load market data
3. Run backtest / optimize with extracted config
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
from src.pine_parser import parse_universal, get_universal_summary, InputType
from src.pine_parser.exceptions import ValidationError


# Page config
st.set_page_config(
    page_title="Algo Strategy Builder",
    page_icon="📈",
    layout="wide",
)

st.title("📈 Algo Strategy Builder")


# =============================================================================
# Helper: Get Pine input value
# =============================================================================
def get_pine_val(var_name, default=None):
    """Get value from Pine inputs."""
    pine_vals = st.session_state.get("pine_input_values", {})
    return pine_vals.get(var_name, default)


# =============================================================================
# Helper: Build StrategyConfig from Pine inputs (intelligent mapping)
# =============================================================================
def find_pine_input(keywords: list, input_type=None, default=None):
    """
    Find a Pine input by matching keywords in var_name or label.
    Searches through all extracted inputs for semantic matches.

    Args:
        keywords: List of keywords to search for (case-insensitive)
        input_type: Optional InputType to filter by
        default: Default value if no match found

    Returns:
        The matched input value, or default if not found
    """
    pine_vals = st.session_state.get("pine_input_values", {})
    parsed = st.session_state.get("universal_parsed")

    if not parsed:
        return default

    # Search through all inputs
    for var_name, inp in parsed.inputs.items():
        # Check if input type matches (if specified)
        if input_type and inp.input_type != input_type:
            continue

        # Search in var_name and label
        search_text = f"{var_name} {inp.label}".lower()

        for keyword in keywords:
            if keyword.lower() in search_text:
                return pine_vals.get(var_name, inp.default_value)

    return default


def build_config_from_pine() -> StrategyConfig:
    """
    Build StrategyConfig from extracted Pine Script inputs using intelligent mapping.

    This function uses semantic matching to map Pine inputs to StrategyConfig fields,
    allowing it to work with ANY Pine Script, not just specific variable names.

    Matching strategy:
    1. First try exact variable name matches (backward compatible)
    2. Then try keyword-based semantic matching
    3. Fall back to sensible defaults
    """
    parsed = st.session_state.get("universal_parsed")

    # Helper for exact match with fallback to keyword search
    def get_val(exact_name: str, keywords: list, input_type=None, default=None):
        """Try exact match first, then keyword search."""
        # Try exact match
        exact_val = get_pine_val(exact_name)
        if exact_val is not None:
            return exact_val
        # Try keyword search
        return find_pine_input(keywords, input_type, default)

    # ===== Risk Type Mapping =====
    risk_type_raw = get_val(
        "tipo_gestion",
        ["gestion", "risk_type", "position_size", "sizing"],
        InputType.STRING,
        "Tamaño fijo"
    )
    risk_type_map = {
        "Tamaño fijo": "fixed_size",
        "Riesgo monetario fijo": "fixed_risk_money",
        "Riesgo % equity": "risk_percent",
        "Fixed Size": "fixed_size",
        "Fixed Risk": "fixed_risk_money",
        "Risk Percent": "risk_percent",
    }
    risk_type = risk_type_map.get(risk_type_raw, "fixed_size")

    # ===== Build Config with Intelligent Matching =====
    return StrategyConfig(
        # Basic config
        entry_pips=int(get_val("entry_pip", ["entry", "pip", "entrada"], InputType.INT, 1)),
        sl_pips=int(get_val("sl_pip", ["sl", "stop", "loss"], InputType.INT, 1)),
        pip_size=0.01,  # User can adjust in sidebar

        # Direction
        trade_longs=get_val("operar_largos", ["long", "largo", "buy", "compra"], InputType.BOOL, True),
        trade_shorts=get_val("operar_cortos", ["short", "corto", "sell", "venta"], InputType.BOOL, True),

        # Entry patterns - default to both if no specific pattern inputs found
        use_sacudida=get_val("usar_patron_sacudida", ["sacudida", "shakeout"], InputType.BOOL, True),
        use_engulfing=get_val("usar_patron_envolvente", ["envolvente", "engulf"], InputType.BOOL, True),
        use_climatic_volume=get_val("usar_patron_vol_climatico", ["climatico", "climatic", "volume"], InputType.BOOL, False),

        # Exits
        use_sl=get_val("usar_sl_original", ["usar_sl", "use_sl", "stop_loss"], InputType.BOOL, True),
        use_tp_ratio=get_val("usar_salida_tp_ratio", ["tp_ratio", "take_profit", "target"], InputType.BOOL, True),
        tp_ratio=float(get_val("target_ratio", ["ratio", "reward", "target", "tp"], InputType.FLOAT, 1.5)),
        use_n_bars_exit=get_val("usar_salida_n_velas", ["n_bar", "bar_exit", "velas"], InputType.BOOL, False),
        n_bars_exit=int(get_val("n_velas_salida", ["n_velas", "bars_exit", "exit_bars"], InputType.INT, 5)),

        # Filters - detect trend filter from various formats
        ma_filter=_detect_ma_filter(),

        # Sessions - default all enabled
        use_london=get_val("usarLondon", ["london", "londres"], InputType.BOOL, True),
        use_newyork=get_val("usarNewYork", ["new_york", "newyork", "ny"], InputType.BOOL, True),
        use_tokyo=get_val("usarTokio", ["tokyo", "tokio", "asia"], InputType.BOOL, True),

        # Days - default weekdays enabled
        trade_monday=get_val("operarLunes", ["monday", "lunes", "mon"], InputType.BOOL, True),
        trade_tuesday=get_val("operarMartes", ["tuesday", "martes", "tue"], InputType.BOOL, True),
        trade_wednesday=get_val("operarMiercoles", ["wednesday", "miercoles", "wed"], InputType.BOOL, True),
        trade_thursday=get_val("operarJueves", ["thursday", "jueves", "thu"], InputType.BOOL, True),
        trade_friday=get_val("operarViernes", ["friday", "viernes", "fri"], InputType.BOOL, True),
        trade_saturday=get_val("operarSabado", ["saturday", "sabado", "sat"], InputType.BOOL, False),
        trade_sunday=get_val("operarDomingo", ["sunday", "domingo", "sun"], InputType.BOOL, False),

        # Risk management
        risk_type=risk_type,
        fixed_size=float(get_val("tamano_fijo_qty", ["fixed_size", "qty", "quantity", "size"], InputType.FLOAT, 1.0)),
        fixed_risk_money=float(get_val("riesgo_monetario", ["risk_money", "riesgo_monetario"], InputType.FLOAT, 100.0)),
        risk_percent=float(get_val("porc_riesgo_equity", ["risk_percent", "equity", "porc"], InputType.FLOAT, 1.0)),

        # Backtest settings (from strategy() call)
        initial_capital=float(parsed.initial_capital or 10000),
        commission=float(parsed.commission or 0.0),
    )


def _detect_ma_filter() -> str:
    """
    Detect MA filter setting from Pine inputs.
    Handles both dropdown options and boolean trend filters.
    """
    # Try exact Spanish dropdown
    filtro = get_pine_val("filtro_mm50200")
    if filtro:
        return filtro

    # Try boolean trend filter (like simple_pine.txt's "use_trend")
    use_trend = find_pine_input(["trend", "tendencia", "filter", "filtro", "sma"], InputType.BOOL)
    if use_trend is not None:
        # If trend filter is enabled, default to bullish (uptrend)
        return "Alcista (MM50>200)" if use_trend else "Sin filtro"

    return "Sin filtro"


# =============================================================================
# STEP 1: Import Pine Script (REQUIRED)
# =============================================================================
st.markdown("### 📄 Step 1: Import Pine Script")

if "universal_parsed" not in st.session_state:
    st.info("Upload your TradingView Pine Script (.txt or .pine) to extract the strategy configuration.")

    uploaded_file = st.file_uploader(
        "Choose a Pine Script file",
        type=['txt', 'pine'],
        help="Upload your TradingView Pine Script strategy file",
        key="pine_uploader"
    )

    if uploaded_file is not None:
        content = uploaded_file.read().decode('utf-8')

        with st.spinner("Parsing Pine Script..."):
            try:
                parsed = parse_universal(content)
                summary = get_universal_summary(parsed)
                st.session_state["universal_parsed"] = parsed
                st.session_state["universal_summary"] = summary

                # Initialize input values from defaults
                input_values = {}
                for var_name, inp in parsed.inputs.items():
                    input_values[var_name] = inp.default_value
                st.session_state["pine_input_values"] = input_values

                st.rerun()

            except Exception as e:
                st.error(f"Error parsing file: {e}")

    st.stop()  # Don't show rest of app until Pine Script is loaded

# Pine Script is loaded - show strategy info
parsed = st.session_state["universal_parsed"]
summary = st.session_state["universal_summary"]

with st.expander(f"📄 **{parsed.name}** (v{parsed.version}) - {parsed.total_inputs} inputs", expanded=False):
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Inputs", parsed.total_inputs)
    with col2:
        st.metric("Functions", len(parsed.functions))
    with col3:
        st.metric("Indicators", len(parsed.indicators))
    with col4:
        if st.button("🗑️ Clear & Upload New"):
            for key in ["universal_parsed", "universal_summary", "pine_input_values", "df", "result"]:
                st.session_state.pop(key, None)
            st.rerun()

    if parsed.indicators:
        st.markdown(f"**Indicators detected:** {', '.join(parsed.indicators)}")


# =============================================================================
# Sidebar - Dynamic Strategy Configuration from Pine Script
# =============================================================================
st.sidebar.header("⚙️ Strategy Inputs")
st.sidebar.caption(f"From: {parsed.name}")

# Get current input values
input_values = st.session_state.get("pine_input_values", {})

# Generate inputs dynamically by group
for group_name, var_names in parsed.input_groups.items():
    # Clean up group name for display
    display_name = group_name.replace("grp", "").replace("group", "").strip()
    if display_name.startswith("***"):
        display_name = display_name.strip("*").strip()
    if not display_name or display_name == "Ungrouped":
        display_name = "Other"

    with st.sidebar.expander(f"**{display_name}**", expanded=True):
        for var_name in var_names:
            inp = parsed.inputs[var_name]

            # Create appropriate input widget based on type
            if inp.input_type == InputType.BOOL:
                input_values[var_name] = st.checkbox(
                    inp.label,
                    value=input_values.get(var_name, inp.default_value),
                    key=f"sidebar_{var_name}",
                    help=inp.tooltip
                )
            elif inp.input_type == InputType.INT:
                input_values[var_name] = st.number_input(
                    inp.label,
                    value=int(input_values.get(var_name, inp.default_value)),
                    min_value=int(inp.min_val) if inp.min_val else None,
                    max_value=int(inp.max_val) if inp.max_val else None,
                    step=int(inp.step) if inp.step else 1,
                    key=f"sidebar_{var_name}",
                    help=inp.tooltip
                )
            elif inp.input_type == InputType.FLOAT:
                input_values[var_name] = st.number_input(
                    inp.label,
                    value=float(input_values.get(var_name, inp.default_value)),
                    min_value=float(inp.min_val) if inp.min_val else None,
                    max_value=float(inp.max_val) if inp.max_val else None,
                    step=float(inp.step) if inp.step else 0.1,
                    key=f"sidebar_{var_name}",
                    help=inp.tooltip
                )
            elif inp.input_type == InputType.STRING:
                if inp.options:
                    current_val = input_values.get(var_name, inp.default_value)
                    idx = inp.options.index(current_val) if current_val in inp.options else 0
                    input_values[var_name] = st.selectbox(
                        inp.label,
                        options=inp.options,
                        index=idx,
                        key=f"sidebar_{var_name}",
                        help=inp.tooltip
                    )
                else:
                    input_values[var_name] = st.text_input(
                        inp.label,
                        value=input_values.get(var_name, inp.default_value),
                        key=f"sidebar_{var_name}",
                        help=inp.tooltip
                    )

# Update session state
st.session_state["pine_input_values"] = input_values

# Additional backtester settings
st.sidebar.markdown("---")
st.sidebar.subheader("Backtester Settings")
pip_size = st.sidebar.number_input(
    "Pip Size",
    min_value=0.0001,
    value=0.01,
    step=0.0001,
    format="%.4f",
    help="0.0001 forex, 0.01 stocks, 1.0 crypto",
)


# =============================================================================
# Sidebar - Data Loading
# =============================================================================
st.sidebar.markdown("---")
st.sidebar.header("📊 Load Data")

symbol = st.sidebar.text_input(
    "Symbol",
    value="SPY",
    help="Yahoo Finance symbol (e.g., SPY, AAPL, EURUSD=X, BTC-USD)"
)

timeframes = get_available_timeframes()
timeframe = st.sidebar.selectbox(
    "Timeframe",
    options=timeframes,
    index=timeframes.index("1d"),
)

# Date range
col1, col2 = st.sidebar.columns(2)
limit = TIMEFRAME_LIMITS.get(timeframe)
default_days = min(limit * 2, 365) if limit else 365 * 2

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

load_button = st.sidebar.button("🔄 Load Data", type="primary")

# Run backtest button
st.sidebar.markdown("---")
run_backtest_btn = st.sidebar.button("🚀 Run Backtest", type="primary")


# =============================================================================
# Chart Functions
# =============================================================================
def create_chart_with_trades(df: pd.DataFrame, result: BacktestResult, symbol: str, timeframe: str):
    """Create candlestick chart with trade markers."""
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.03,
                        row_heights=[0.6, 0.2, 0.2],
                        subplot_titles=(f"{symbol} - {timeframe}", "Volume", "Equity"))

    # Candlesticks
    fig.add_trace(go.Candlestick(x=df.index, open=df["open"], high=df["high"], low=df["low"], close=df["close"],
                                  name="Price", increasing_line_color="#26a69a", decreasing_line_color="#ef5350"), row=1, col=1)

    # SMAs
    if len(df) >= 50:
        sma50 = df["close"].rolling(window=50).mean()
        fig.add_trace(go.Scatter(x=df.index, y=sma50, mode="lines", name="SMA 50",
                                line=dict(color="#2196f3", width=1)), row=1, col=1)
    if len(df) >= 200:
        sma200 = df["close"].rolling(window=200).mean()
        fig.add_trace(go.Scatter(x=df.index, y=sma200, mode="lines", name="SMA 200",
                                line=dict(color="#ff9800", width=1)), row=1, col=1)

    # Trade markers
    for trade in result.trades:
        color = "#26a69a" if trade.direction == "long" else "#ef5350"
        marker = "triangle-up" if trade.direction == "long" else "triangle-down"
        fig.add_trace(go.Scatter(x=[trade.entry_time], y=[trade.entry_price], mode="markers",
                                marker=dict(symbol=marker, size=12, color=color), name=f"Entry {trade.direction}",
                                showlegend=False), row=1, col=1)
        exit_color = "#26a69a" if trade.pnl > 0 else "#ef5350"
        fig.add_trace(go.Scatter(x=[trade.exit_time], y=[trade.exit_price], mode="markers",
                                marker=dict(symbol="x", size=10, color=exit_color), name="Exit",
                                showlegend=False), row=1, col=1)

    # Volume
    colors = ["#26a69a" if c >= o else "#ef5350" for o, c in zip(df["open"], df["close"])]
    fig.add_trace(go.Bar(x=df.index, y=df["volume"], name="Volume", marker_color=colors, opacity=0.7), row=2, col=1)

    # Equity curve
    fig.add_trace(go.Scatter(x=result.equity_curve.index, y=result.equity_curve.values, mode="lines",
                            name="Equity", line=dict(color="#2196f3", width=2)), row=3, col=1)

    fig.update_layout(height=800, template="plotly_dark", xaxis_rangeslider_visible=False,
                     showlegend=True, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    return fig


def create_equity_chart(result: BacktestResult):
    """Create standalone equity curve chart."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=result.equity_curve.index, y=result.equity_curve.values, mode="lines",
                            name="Equity", fill="tozeroy", line=dict(color="#2196f3", width=2)))
    fig.update_layout(height=300, template="plotly_dark", title="Equity Curve",
                     xaxis_title="Date", yaxis_title="Equity ($)")
    return fig


# =============================================================================
# Main Content - Load Data
# =============================================================================
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
            st.session_state.pop("result", None)

            st.success(f"✅ Loaded {len(df):,} bars from {df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')}")

            if result.message:
                st.warning(result.message)
        except Exception as e:
            st.error(f"❌ Error: {e}")

# Build config and run backtest
if run_backtest_btn and "df" in st.session_state:
    with st.spinner("Running backtest..."):
        try:
            # Build config from Pine inputs
            config = build_config_from_pine()
            config.pip_size = pip_size  # Add user-specified pip size

            backtest_result = run_backtest(st.session_state["df"], config)
            st.session_state["result"] = backtest_result
            st.session_state["config"] = config
        except Exception as e:
            st.error(f"❌ Backtest error: {e}")


# =============================================================================
# Display Results
# =============================================================================
if "df" in st.session_state:
    df = st.session_state["df"]
    symbol = st.session_state["symbol"]
    timeframe = st.session_state["timeframe"]

    # Tab navigation
    tab_names = ["📈 Chart", "📊 Results", "📋 Trades", "🔬 Optimizer"]

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
        st.plotly_chart(fig)

    # Results tab
    elif selected_tab == "📊 Results":
        if "result" in st.session_state:
            result = st.session_state["result"]

            st.subheader("Performance Summary")
            col1, col2, col3, col4, col5, col6 = st.columns(6)
            with col1:
                st.metric("Total Trades", result.total_trades)
            with col2:
                st.metric("Win Rate", f"{result.win_rate:.1%}")
            with col3:
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

            st.subheader("Equity Curve")
            eq_fig = create_equity_chart(result)
            st.plotly_chart(eq_fig)
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
            st.dataframe(trades_df, hide_index=True)

            csv_data = trades_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Trades CSV",
                data=csv_data,
                file_name=f"{symbol}_{timeframe}_trades.csv",
                mime="text/csv",
            )
        else:
            st.info("No trades to display")

    # Optimizer tab
    elif selected_tab == "🔬 Optimizer":
        st.subheader("Parameter Optimization")
        st.markdown("""
        Test combinations of **entries**, **exits**, **filters**, **sessions**, and **days**
        to find the best configuration. Results ranked by **MAR ratio**.
        """)

        st.markdown("##### What to Optimize")
        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            opt_entries = st.checkbox("Entries", value=False, key="opt_entries_main")
        with col2:
            opt_exits = st.checkbox("Exits", value=False, key="opt_exits_main")
        with col3:
            opt_ma = st.checkbox("MA Filter", value=True, key="opt_ma_main")
        with col4:
            opt_sessions = st.checkbox("Sessions", value=True, key="opt_sessions_main")
        with col5:
            opt_days = st.checkbox("Days", value=False, key="opt_days_main")

        # Entry options
        if opt_entries:
            with st.expander("Entry Patterns to Test", expanded=True):
                col1, col2, col3 = st.columns(3)
                with col1:
                    entry_sacudida = st.checkbox("Sacudida", value=True, key="opt_sac")
                with col2:
                    entry_engulfing = st.checkbox("Engulfing", value=True, key="opt_eng")
                with col3:
                    entry_climatic = st.checkbox("Climatic Vol", value=True, key="opt_clim")

                from itertools import combinations
                selected_entries = []
                if entry_sacudida: selected_entries.append(0)
                if entry_engulfing: selected_entries.append(1)
                if entry_climatic: selected_entries.append(2)

                entry_options = []
                for r in range(1, len(selected_entries) + 1):
                    for combo in combinations(selected_entries, r):
                        opt = [False, False, False]
                        for idx in combo:
                            opt[idx] = True
                        entry_options.append(tuple(opt))
                if not entry_options:
                    entry_options = [(True, True, False)]
        else:
            entry_options = None

        # Exit options
        if opt_exits:
            with st.expander("Exit Options to Test", expanded=True):
                col1, col2 = st.columns(2)
                with col1:
                    tp_ratios_str = st.text_input("TP Ratios", value="0.5, 1.0, 1.5, 2.0, 2.5, 3.0")
                    tp_ratios = [float(x.strip()) for x in tp_ratios_str.split(",") if x.strip()]
                with col2:
                    n_bars_str = st.text_input("N-bars exit", value="3, 5, 7, 10, 15")
                    n_bars_opts = [int(x.strip()) for x in n_bars_str.split(",") if x.strip()]
        else:
            tp_ratios = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
            n_bars_opts = [3, 5, 7, 10, 15]

        # MA options
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
                if ma_sin_filtro: ma_filter_options.append("Sin filtro")
                if ma_alcista: ma_filter_options.append("Alcista (MM50>200)")
                if ma_bajista: ma_filter_options.append("Bajista (MM50<200)")
                if not ma_filter_options: ma_filter_options = ["Sin filtro"]
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

                from itertools import combinations
                selected_sessions = []
                if sess_london: selected_sessions.append(0)
                if sess_ny: selected_sessions.append(1)
                if sess_tokyo: selected_sessions.append(2)

                session_options = []
                for r in range(1, len(selected_sessions) + 1):
                    for combo in combinations(selected_sessions, r):
                        opt = [False, False, False]
                        for idx in combo:
                            opt[idx] = True
                        session_options.append(tuple(opt))
                if not session_options: session_options = [(True, True, True)]
        else:
            session_options = None

        # Day options
        if opt_days:
            with st.expander("Day Options", expanded=True):
                cols = st.columns(7)
                day_vars = []
                day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
                defaults = [True, True, True, True, True, False, False]
                for i, (col, name, default) in enumerate(zip(cols, day_names, defaults)):
                    with col:
                        day_vars.append(st.checkbox(name, value=default, key=f"day_{i}"))

                from itertools import combinations
                selected_days = [i for i, v in enumerate(day_vars) if v]
                day_options = []
                for r in range(1, len(selected_days) + 1):
                    for combo in combinations(selected_days, r):
                        opt = [False] * 7
                        for idx in combo:
                            opt[idx] = True
                        day_options.append(tuple(opt))
                if not day_options: day_options = [(True, True, True, True, True, False, False)]
        else:
            day_options = None

        # Build settings
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

        estimated_combos = estimate_combinations(opt_settings)
        settings_desc = get_settings_description(opt_settings)

        col1, col2 = st.columns(2)
        with col1:
            use_max_combos = st.checkbox("Limit combinations", value=estimated_combos > 500)
        with col2:
            if use_max_combos:
                max_combos = st.number_input("Max", min_value=1, max_value=10000,
                                            value=min(500, estimated_combos), step=50)
            else:
                max_combos = None

        min_trades = st.slider("Minimum trades", min_value=1, max_value=100, value=10)

        final_combos = max_combos if max_combos else estimated_combos
        st.info(f"**{settings_desc}** | ~{estimated_combos:,} combinations → Testing: {final_combos:,}")

        run_opt_btn = st.button("🚀 Run Optimization", type="primary")

        if run_opt_btn:
            with st.spinner("Running optimization..."):
                progress_bar = st.progress(0)
                status_text = st.empty()

                def update_opt_progress(current, total, message):
                    progress_bar.progress(current / total if total > 0 else 0)
                    status_text.text(message)

                try:
                    # Build base config from Pine
                    config = build_config_from_pine()
                    config.pip_size = pip_size

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
                        st.success(f"Found {len(opt_results)} valid configurations!")
                    else:
                        st.warning("No configurations met the minimum trade requirement.")
                except Exception as e:
                    st.error(f"Error: {e}")
                    progress_bar.empty()
                    status_text.empty()

        # Display results
        if "opt_results" in st.session_state and st.session_state["opt_results"]:
            opt_results = st.session_state["opt_results"]

            st.subheader("Top Configurations")
            results_df = results_to_dataframe(opt_results)
            st.dataframe(results_df, hide_index=True)

            st.download_button(
                label="📥 Download Results CSV",
                data=results_df.to_csv(index=False),
                file_name=f"{symbol}_optimization.csv",
                mime="text/csv",
            )

            st.subheader("Apply Configuration")
            selected_rank = st.selectbox(
                "Select configuration",
                options=list(range(len(opt_results))),
                format_func=lambda i: f"#{i+1}: {opt_results[i].config_summary} (MAR: {opt_results[i].mar_ratio:.2f})",
            )

            if st.button("✅ Apply Selected"):
                selected_config = opt_results[selected_rank].config
                st.session_state["applied_opt_config"] = selected_config
                st.success(f"Configuration #{selected_rank + 1} applied!")

else:
    # No data loaded yet
    st.markdown("### 📊 Step 2: Load Data")
    st.info("👈 Use the sidebar to load market data, then run backtest")

    with st.expander("📋 Sample Symbols"):
        st.markdown("""
        **Stocks**: `AAPL`, `MSFT`, `SPY`, `QQQ`
        **Forex**: `EURUSD=X`, `GBPUSD=X`, `USDJPY=X`
        **Crypto**: `BTC-USD`, `ETH-USD`
        **Futures**: `ES=F`, `NQ=F`, `GC=F`
        """)
