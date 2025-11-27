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
from src.optimizer import run_optimization, results_to_dataframe
from src.pine_parser import parse_universal, get_universal_summary, InputType
from src.pine_parser.exceptions import ValidationError
from src.dynamic_optimizer import (
    create_optimizable_params, DynamicOptimizationSettings, OptimizableParam,
    generate_dynamic_combinations, get_param_summary
)


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

    # Optimizer tab - Dynamic based on Pine Script inputs
    elif selected_tab == "🔬 Optimizer":
        st.subheader("Parameter Optimization")
        st.markdown(f"""
        Optimize parameters from **{parsed.name}**. Select which inputs to vary
        and the values to test. Results ranked by **MAR ratio**.
        """)

        # Initialize dynamic optimizer params if not exists
        if "opt_params" not in st.session_state:
            st.session_state["opt_params"] = create_optimizable_params(parsed)

        opt_params = st.session_state["opt_params"]

        # Group inputs by type for better organization
        bool_inputs = {k: v for k, v in opt_params.items() if v.input_type == InputType.BOOL}
        float_inputs = {k: v for k, v in opt_params.items() if v.input_type == InputType.FLOAT}
        int_inputs = {k: v for k, v in opt_params.items() if v.input_type == InputType.INT}
        string_inputs = {k: v for k, v in opt_params.items() if v.input_type == InputType.STRING}

        st.markdown("##### Select Parameters to Optimize")

        # Boolean inputs (checkboxes - test True/False)
        if bool_inputs:
            with st.expander(f"**Boolean Inputs** ({len(bool_inputs)} available)", expanded=True):
                cols = st.columns(min(3, len(bool_inputs)))
                for i, (var_name, param) in enumerate(bool_inputs.items()):
                    with cols[i % len(cols)]:
                        param.enabled = st.checkbox(
                            param.label,
                            value=param.enabled,
                            key=f"opt_bool_{var_name}",
                            help=f"Test both True and False for '{param.label}'"
                        )

        # Float inputs (numeric ranges)
        if float_inputs:
            with st.expander(f"**Float Inputs** ({len(float_inputs)} available)", expanded=True):
                for var_name, param in float_inputs.items():
                    col1, col2 = st.columns([1, 3])
                    with col1:
                        param.enabled = st.checkbox(
                            param.label,
                            value=param.enabled,
                            key=f"opt_float_cb_{var_name}"
                        )
                    with col2:
                        if param.enabled:
                            values_str = st.text_input(
                                f"Values for {param.label}",
                                value=", ".join(str(v) for v in param.values_to_test),
                                key=f"opt_float_vals_{var_name}",
                                help="Comma-separated values to test"
                            )
                            try:
                                param.values_to_test = [float(x.strip()) for x in values_str.split(",") if x.strip()]
                            except ValueError:
                                st.error(f"Invalid values for {param.label}")

        # Integer inputs (numeric ranges)
        if int_inputs:
            with st.expander(f"**Integer Inputs** ({len(int_inputs)} available)", expanded=True):
                for var_name, param in int_inputs.items():
                    col1, col2 = st.columns([1, 3])
                    with col1:
                        param.enabled = st.checkbox(
                            param.label,
                            value=param.enabled,
                            key=f"opt_int_cb_{var_name}"
                        )
                    with col2:
                        if param.enabled:
                            values_str = st.text_input(
                                f"Values for {param.label}",
                                value=", ".join(str(v) for v in param.values_to_test),
                                key=f"opt_int_vals_{var_name}",
                                help="Comma-separated values to test"
                            )
                            try:
                                param.values_to_test = [int(x.strip()) for x in values_str.split(",") if x.strip()]
                            except ValueError:
                                st.error(f"Invalid values for {param.label}")

        # String inputs with options (dropdowns)
        if string_inputs:
            with st.expander(f"**String Inputs** ({len(string_inputs)} available)", expanded=True):
                for var_name, param in string_inputs.items():
                    inp = parsed.inputs[var_name]
                    if inp.options:
                        col1, col2 = st.columns([1, 3])
                        with col1:
                            param.enabled = st.checkbox(
                                param.label,
                                value=param.enabled,
                                key=f"opt_str_cb_{var_name}"
                            )
                        with col2:
                            if param.enabled:
                                selected_options = st.multiselect(
                                    f"Options for {param.label}",
                                    options=inp.options,
                                    default=param.values_to_test if param.values_to_test else inp.options,
                                    key=f"opt_str_opts_{var_name}"
                                )
                                param.values_to_test = selected_options if selected_options else inp.options

        # Build dynamic settings
        dyn_settings = DynamicOptimizationSettings(params=opt_params)
        estimated_combos = dyn_settings.estimate_combinations()
        enabled_params = dyn_settings.get_enabled_params()
        enabled_names = [p.label for p in enabled_params.values()]

        st.markdown("---")
        st.markdown("##### Optimization Settings")
        col1, col2, col3 = st.columns(3)
        with col1:
            max_iterations = st.number_input(
                "Max Iterations",
                min_value=10,
                max_value=50000,
                value=min(500, max(10, estimated_combos)),
                step=50,
                help="Maximum number of parameter combinations to test"
            )
        with col2:
            min_trades = st.number_input(
                "Min Trades",
                min_value=1,
                max_value=100,
                value=10,
                help="Minimum trades required for a valid result"
            )
        with col3:
            st.metric("Total Combinations", f"{estimated_combos:,}")

        # Summary
        if enabled_names:
            st.info(f"**Optimizing:** {', '.join(enabled_names)} → Testing: {min(max_iterations, estimated_combos):,} combinations")
        else:
            st.warning("Select at least one parameter to optimize")

        run_opt_btn = st.button("🚀 Run Optimization", type="primary", disabled=not enabled_names)

        if run_opt_btn and enabled_names:
            with st.spinner("Running optimization..."):
                progress_bar = st.progress(0)
                status_text = st.empty()

                try:
                    # Get current input values
                    base_values = st.session_state.get("pine_input_values", {})

                    # Generate all combinations
                    combinations = generate_dynamic_combinations(
                        base_values, dyn_settings, max_iterations
                    )

                    total = len(combinations)
                    results = []

                    for i, combo_values in enumerate(combinations):
                        progress_bar.progress((i + 1) / total)
                        status_text.text(f"Testing {i + 1}/{total}")

                        # Update session state with combo values temporarily
                        st.session_state["pine_input_values"] = combo_values

                        # Build config and run backtest
                        config = build_config_from_pine()
                        config.pip_size = pip_size

                        result = run_backtest(st.session_state["df"], config)

                        if result.total_trades >= min_trades:
                            # Calculate MAR ratio
                            df_data = st.session_state["df"]
                            date_range = df_data.index[-1] - df_data.index[0]
                            years = date_range.days / 365.25
                            if years > 0 and result.max_drawdown_pct != 0:
                                total_return_pct = (result.total_pnl / config.initial_capital) * 100
                                annualized_return = total_return_pct / years
                                mar = annualized_return / abs(result.max_drawdown_pct)
                            else:
                                mar = 0.0

                            results.append({
                                "values": combo_values.copy(),
                                "summary": get_param_summary(combo_values, enabled_params),
                                "trades": result.total_trades,
                                "win_rate": result.win_rate,
                                "pnl": result.total_pnl,
                                "profit_factor": result.profit_factor,
                                "max_dd": result.max_drawdown_pct,
                                "sharpe": result.sharpe_ratio,
                                "mar": mar,
                            })

                    # Restore original values
                    st.session_state["pine_input_values"] = base_values

                    # Sort by MAR
                    results.sort(key=lambda x: x["mar"], reverse=True)
                    st.session_state["dyn_opt_results"] = results

                    progress_bar.empty()
                    status_text.empty()

                    if results:
                        st.success(f"Found {len(results)} valid configurations!")
                    else:
                        st.warning("No configurations met the minimum trade requirement.")

                except Exception as e:
                    st.error(f"Error: {e}")
                    progress_bar.empty()
                    status_text.empty()
                    # Restore original values
                    st.session_state["pine_input_values"] = base_values

        # Display dynamic optimization results
        if "dyn_opt_results" in st.session_state and st.session_state["dyn_opt_results"]:
            results = st.session_state["dyn_opt_results"]

            st.subheader("Top Configurations")

            # Build results dataframe
            results_data = []
            for i, r in enumerate(results[:50], 1):  # Top 50
                results_data.append({
                    "Rank": i,
                    "Configuration": r["summary"],
                    "Trades": r["trades"],
                    "Win Rate": f"{r['win_rate']:.1%}",
                    "Net P&L": f"${r['pnl']:,.2f}",
                    "Profit Factor": f"{r['profit_factor']:.2f}",
                    "Max DD %": f"{r['max_dd']:.1f}%",
                    "Sharpe": f"{r['sharpe']:.2f}",
                    "MAR": f"{r['mar']:.2f}",
                })

            results_df = pd.DataFrame(results_data)
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
                options=list(range(min(50, len(results)))),
                format_func=lambda i: f"#{i+1}: {results[i]['summary']} (MAR: {results[i]['mar']:.2f})",
            )

            if st.button("✅ Apply Selected"):
                selected_values = results[selected_rank]["values"]
                st.session_state["pine_input_values"] = selected_values
                st.success(f"Configuration #{selected_rank + 1} applied! Sidebar values updated.")

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
