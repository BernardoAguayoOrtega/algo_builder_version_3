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
import pickle
import hashlib
import time

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
from src.monte_carlo import run_monte_carlo, MonteCarloResult, format_monte_carlo_summary
from src.walk_forward import run_walk_forward, format_walk_forward_results, WalkForwardResult


# =============================================================================
# Session Persistence Functions
# =============================================================================
def export_session_state() -> bytes:
    """
    Export current session state to bytes for download.
    Saves: Pine Script data, input values, data, results, optimization results.
    """
    export_keys = [
        "universal_parsed", "universal_summary", "pine_input_values",
        "df", "result", "symbol", "timeframe", "symbol_info",
        "opt_params", "dyn_opt_results", "config"
    ]
    export_data = {}
    for key in export_keys:
        if key in st.session_state:
            export_data[key] = st.session_state[key]

    export_data["_export_timestamp"] = datetime.now().isoformat()
    export_data["_export_version"] = "1.0"

    return pickle.dumps(export_data)


def import_session_state(data: bytes) -> bool:
    """
    Import session state from bytes.
    Returns True if successful, False otherwise.
    """
    try:
        import_data = pickle.loads(data)

        # Validate version
        if import_data.get("_export_version") != "1.0":
            return False

        # Import all keys
        for key, value in import_data.items():
            if not key.startswith("_"):
                st.session_state[key] = value

        return True
    except Exception:
        return False


def get_session_filename() -> str:
    """Generate a descriptive filename for session export."""
    parsed = st.session_state.get("universal_parsed")
    symbol = st.session_state.get("symbol", "unknown")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")

    if parsed:
        strategy_name = parsed.name.replace(" ", "_")[:20]
        return f"session_{strategy_name}_{symbol}_{timestamp}.pkl"
    return f"session_{symbol}_{timestamp}.pkl"


# =============================================================================
# Performance Caching Functions
# =============================================================================
@st.cache_data(ttl=3600, show_spinner=False)
def cached_fetch_data(symbol: str, timeframe: str, start_ts: float, end_ts: float):
    """
    Cached data fetching - avoids re-fetching the same data.
    Uses timestamps instead of datetime for hashability.
    """
    start_date = datetime.fromtimestamp(start_ts)
    end_date = datetime.fromtimestamp(end_ts)

    result = fetch_data(
        symbol=symbol,
        timeframe=timeframe,
        start_date=start_date,
        end_date=end_date,
    )
    return result


def get_config_hash(config: StrategyConfig) -> str:
    """Generate a hash for a StrategyConfig for caching purposes."""
    config_str = str(config.__dict__)
    return hashlib.md5(config_str.encode()).hexdigest()[:8]


@st.cache_data(ttl=600, show_spinner=False)
def cached_backtest(_df_hash: str, config_dict: dict, _df):
    """
    Cached backtest execution.
    Uses df hash and config dict for cache key.
    """
    config = StrategyConfig(**config_dict)
    return run_backtest(_df, config)


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

    col_upload, col_session = st.columns([2, 1])

    with col_upload:
        uploaded_file = st.file_uploader(
            "Choose a Pine Script file",
            type=['txt', 'pine'],
            help="Upload your TradingView Pine Script strategy file",
            key="pine_uploader"
        )

    with col_session:
        st.markdown("**Or load a saved session:**")
        session_file = st.file_uploader(
            "Load Session (.pkl)",
            type=['pkl'],
            help="Resume a previously saved session",
            key="initial_session_uploader"
        )
        if session_file:
            if import_session_state(session_file.read()):
                st.success("Session restored!")
                st.rerun()
            else:
                st.error("Invalid session file")

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

                # Reset optimizer params for new file
                st.session_state.pop("opt_params", None)
                st.session_state.pop("dyn_opt_results", None)

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
            # Clear all strategy-related state including optimizer
            keys_to_clear = [
                "universal_parsed", "universal_summary", "pine_input_values",
                "df", "result", "opt_params", "dyn_opt_results", "opt_results"
            ]
            for key in keys_to_clear:
                st.session_state.pop(key, None)
            st.rerun()

    if parsed.indicators:
        st.markdown(f"**Indicators detected:** {', '.join(parsed.indicators)}")


# =============================================================================
# Sidebar - Session Management
# =============================================================================
st.sidebar.markdown("---")
st.sidebar.subheader("💾 Session")

# Export session
session_col1, session_col2 = st.sidebar.columns(2)
with session_col1:
    if st.button("📤 Export", help="Save your current session"):
        session_data = export_session_state()
        st.session_state["_export_ready"] = session_data
        st.session_state["_export_filename"] = get_session_filename()

# Show download button if export is ready
if "_export_ready" in st.session_state:
    with session_col2:
        st.download_button(
            "📥 Download",
            data=st.session_state["_export_ready"],
            file_name=st.session_state["_export_filename"],
            mime="application/octet-stream",
            help="Download session file"
        )

# Import session
session_upload = st.sidebar.file_uploader(
    "📂 Load Session",
    type=['pkl'],
    help="Load a previously saved session",
    key="session_uploader"
)
if session_upload:
    if import_session_state(session_upload.read()):
        st.sidebar.success("Session loaded!")
        st.rerun()
    else:
        st.sidebar.error("Invalid session file")


# =============================================================================
# Sidebar - Dynamic Strategy Configuration from Pine Script
# =============================================================================
st.sidebar.markdown("---")
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
# Chart Functions (Cached)
# =============================================================================
@st.cache_data(ttl=300, show_spinner=False)
def create_chart_with_trades_cached(df_hash: str, trades_data: list, equity_index: list, equity_values: list, symbol: str, timeframe: str, _df):
    """
    Create candlestick chart with trade markers.
    Cached version - uses hash for cache key.
    """
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.03,
                        row_heights=[0.6, 0.2, 0.2],
                        subplot_titles=(f"{symbol} - {timeframe}", "Volume", "Equity"))

    # Candlesticks
    fig.add_trace(go.Candlestick(x=_df.index, open=_df["open"], high=_df["high"], low=_df["low"], close=_df["close"],
                                  name="Price", increasing_line_color="#26a69a", decreasing_line_color="#ef5350"), row=1, col=1)

    # SMAs
    if len(_df) >= 50:
        sma50 = _df["close"].rolling(window=50).mean()
        fig.add_trace(go.Scatter(x=_df.index, y=sma50, mode="lines", name="SMA 50",
                                line=dict(color="#2196f3", width=1)), row=1, col=1)
    if len(_df) >= 200:
        sma200 = _df["close"].rolling(window=200).mean()
        fig.add_trace(go.Scatter(x=_df.index, y=sma200, mode="lines", name="SMA 200",
                                line=dict(color="#ff9800", width=1)), row=1, col=1)

    # Trade markers from serialized data
    for trade in trades_data:
        color = "#26a69a" if trade["direction"] == "long" else "#ef5350"
        marker = "triangle-up" if trade["direction"] == "long" else "triangle-down"
        fig.add_trace(go.Scatter(x=[trade["entry_time"]], y=[trade["entry_price"]], mode="markers",
                                marker=dict(symbol=marker, size=12, color=color), name=f"Entry {trade['direction']}",
                                showlegend=False), row=1, col=1)
        exit_color = "#26a69a" if trade["pnl"] > 0 else "#ef5350"
        fig.add_trace(go.Scatter(x=[trade["exit_time"]], y=[trade["exit_price"]], mode="markers",
                                marker=dict(symbol="x", size=10, color=exit_color), name="Exit",
                                showlegend=False), row=1, col=1)

    # Volume
    colors = ["#26a69a" if c >= o else "#ef5350" for o, c in zip(_df["open"], _df["close"])]
    fig.add_trace(go.Bar(x=_df.index, y=_df["volume"], name="Volume", marker_color=colors, opacity=0.7), row=2, col=1)

    # Equity curve
    fig.add_trace(go.Scatter(x=equity_index, y=equity_values, mode="lines",
                            name="Equity", line=dict(color="#2196f3", width=2)), row=3, col=1)

    fig.update_layout(height=800, template="plotly_dark", xaxis_rangeslider_visible=False,
                     showlegend=True, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    return fig


def create_chart_with_trades(df: pd.DataFrame, result: BacktestResult, symbol: str, timeframe: str):
    """Wrapper to call cached function with serialized data."""
    # Serialize trades for caching
    trades_data = [
        {
            "direction": t.direction,
            "entry_time": t.entry_time,
            "entry_price": t.entry_price,
            "exit_time": t.exit_time,
            "exit_price": t.exit_price,
            "pnl": t.pnl
        }
        for t in result.trades
    ]

    # Create hash for cache key
    df_hash = hashlib.md5(pd.util.hash_pandas_object(df).values.tobytes()).hexdigest()[:8]
    trades_hash = hashlib.md5(str(trades_data).encode()).hexdigest()[:8]
    cache_key = f"{df_hash}_{trades_hash}"

    return create_chart_with_trades_cached(
        df_hash=cache_key,
        trades_data=trades_data,
        equity_index=list(result.equity_curve.index),
        equity_values=list(result.equity_curve.values),
        symbol=symbol,
        timeframe=timeframe,
        _df=df
    )


@st.cache_data(ttl=300, show_spinner=False)
def create_equity_chart_cached(equity_index: list, equity_values: list):
    """Create standalone equity curve chart (cached)."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=equity_index, y=equity_values, mode="lines",
                            name="Equity", fill="tozeroy", line=dict(color="#2196f3", width=2)))
    fig.update_layout(height=300, template="plotly_dark", title="Equity Curve",
                     xaxis_title="Date", yaxis_title="Equity ($)")
    return fig


def create_equity_chart(result: BacktestResult):
    """Wrapper for cached equity chart."""
    return create_equity_chart_cached(
        equity_index=list(result.equity_curve.index),
        equity_values=list(result.equity_curve.values)
    )


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
    tab_names = ["📈 Chart", "📊 Results", "📋 Trades", "🔬 Optimizer", "🌐 Multi-Symbol", "🎲 Advanced"]

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

        # Initialize dynamic optimizer params if not exists
        if "opt_params" not in st.session_state:
            st.session_state["opt_params"] = create_optimizable_params(parsed)

        opt_params = st.session_state["opt_params"]

        # Organize params by their Pine Script group
        params_by_group = {}
        for var_name, param in opt_params.items():
            inp = parsed.inputs.get(var_name)
            group = inp.group if inp and inp.group else "Other"
            # Clean up group name
            display_group = group.replace("grp", "").replace("group", "").strip()
            if display_group.startswith("***"):
                display_group = display_group.strip("*").strip()
            if not display_group:
                display_group = "Other"
            if display_group not in params_by_group:
                params_by_group[display_group] = []
            params_by_group[display_group].append((var_name, param))

        # Calculate combinations for each param
        def get_param_combos(param):
            return len(param.values_to_test) if param.values_to_test else 1

        # ===== Quick Presets =====
        st.markdown("##### Quick Presets")
        preset_cols = st.columns(5)

        def apply_preset(preset_func):
            """Apply a preset and update both opt_params and widget state."""
            for var_name, param in opt_params.items():
                enabled = preset_func(var_name, param)
                param.enabled = enabled
                # Also update the widget state key
                widget_key = f"opt_{var_name}"
                st.session_state[widget_key] = enabled

        with preset_cols[0]:
            if st.button("🎯 Key Params", help="Optimize most impactful: TP Ratio, MA Filter"):
                def key_params_preset(var_name, param):
                    label_lower = param.label.lower()
                    return any(kw in label_lower for kw in ["ratio", "filtro", "filter", "trend"])
                apply_preset(key_params_preset)
                st.rerun()

        with preset_cols[1]:
            if st.button("📊 All Numeric", help="Optimize all float and int parameters"):
                def numeric_preset(var_name, param):
                    return param.input_type in [InputType.FLOAT, InputType.INT]
                apply_preset(numeric_preset)
                st.rerun()

        with preset_cols[2]:
            if st.button("🔀 All Options", help="Optimize all dropdown options"):
                def options_preset(var_name, param):
                    inp = parsed.inputs.get(var_name)
                    return param.input_type == InputType.STRING and inp and bool(inp.options)
                apply_preset(options_preset)
                st.rerun()

        with preset_cols[3]:
            if st.button("✅ Select All", help="Enable all parameters"):
                apply_preset(lambda vn, p: True)
                st.rerun()

        with preset_cols[4]:
            if st.button("❌ Clear All", help="Disable all parameters"):
                apply_preset(lambda vn, p: False)
                st.rerun()

        st.markdown("---")

        # ===== Parameters by Group =====
        st.markdown("##### Select Parameters to Optimize")

        # Priority groups (show expanded), others collapsed
        priority_groups = ["Salidas", "Filtros", "Exits", "Filters", "Other"]

        for group_name, group_params in params_by_group.items():
            is_priority = any(pg.lower() in group_name.lower() for pg in priority_groups)
            total_count = len(group_params)

            # Calculate enabled count and combinations using session state
            enabled_count = 0
            group_combos = 1
            for vn, p in group_params:
                is_enabled = st.session_state.get(f"opt_{vn}", p.enabled)
                if is_enabled:
                    enabled_count += 1
                    group_combos *= get_param_combos(p)

            header = f"**{group_name}** ({enabled_count}/{total_count} selected)"
            if enabled_count > 0:
                header += f" → {group_combos:,} combinations"

            with st.expander(header, expanded=is_priority and total_count <= 6):
                for var_name, param in group_params:
                    inp = parsed.inputs.get(var_name)
                    combos = get_param_combos(param)
                    widget_key = f"opt_{var_name}"

                    # Get current enabled state (prefer session state if set by preset)
                    current_enabled = st.session_state.get(widget_key, param.enabled)

                    # Different UI based on type
                    if param.input_type == InputType.BOOL:
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            param.enabled = st.checkbox(
                                f"{param.label}",
                                value=current_enabled,
                                key=widget_key,
                                help=f"Test True/False (×2 combinations)"
                            )
                        with col2:
                            if param.enabled:
                                st.caption("True, False")

                    elif param.input_type == InputType.FLOAT:
                        col1, col2 = st.columns([1, 2])
                        with col1:
                            param.enabled = st.checkbox(
                                param.label,
                                value=current_enabled,
                                key=widget_key
                            )
                        with col2:
                            default_vals = ", ".join(f"{v:.2f}" for v in param.values_to_test[:5])
                            if len(param.values_to_test) > 5:
                                default_vals += "..."
                            if param.enabled:
                                values_str = st.text_input(
                                    "Values",
                                    value=", ".join(str(v) for v in param.values_to_test),
                                    key=f"opt_vals_{var_name}",
                                    label_visibility="collapsed"
                                )
                                try:
                                    param.values_to_test = [float(x.strip()) for x in values_str.split(",") if x.strip()]
                                except ValueError:
                                    st.error("Invalid")
                            else:
                                st.caption(f"💡 {default_vals}")

                    elif param.input_type == InputType.INT:
                        col1, col2 = st.columns([1, 2])
                        with col1:
                            param.enabled = st.checkbox(
                                param.label,
                                value=current_enabled,
                                key=widget_key
                            )
                        with col2:
                            default_vals = ", ".join(str(v) for v in param.values_to_test[:5])
                            if len(param.values_to_test) > 5:
                                default_vals += "..."
                            if param.enabled:
                                values_str = st.text_input(
                                    "Values",
                                    value=", ".join(str(v) for v in param.values_to_test),
                                    key=f"opt_vals_{var_name}",
                                    label_visibility="collapsed"
                                )
                                try:
                                    param.values_to_test = [int(x.strip()) for x in values_str.split(",") if x.strip()]
                                except ValueError:
                                    st.error("Invalid")
                            else:
                                st.caption(f"💡 {default_vals}")

                    elif param.input_type == InputType.STRING and inp and inp.options:
                        col1, col2 = st.columns([1, 2])
                        with col1:
                            param.enabled = st.checkbox(
                                param.label,
                                value=current_enabled,
                                key=widget_key
                            )
                        with col2:
                            if param.enabled:
                                selected = st.multiselect(
                                    "Options",
                                    options=inp.options,
                                    default=param.values_to_test if param.values_to_test else inp.options,
                                    key=f"opt_opts_{var_name}",
                                    label_visibility="collapsed"
                                )
                                param.values_to_test = selected if selected else inp.options
                            else:
                                st.caption(f"💡 {len(inp.options)} options")

        # Build dynamic settings
        dyn_settings = DynamicOptimizationSettings(params=opt_params)
        estimated_combos = dyn_settings.estimate_combinations()
        enabled_params = dyn_settings.get_enabled_params()
        enabled_names = [p.label for p in enabled_params.values()]

        st.markdown("---")
        st.markdown("##### Optimization Settings")

        # Warning for too many combinations
        MAX_REASONABLE_COMBOS = 100000
        too_many_combos = estimated_combos > MAX_REASONABLE_COMBOS

        if too_many_combos:
            st.error(f"⚠️ **Too many combinations!** ({estimated_combos:,}) - Please select fewer parameters or reduce test values. Max recommended: {MAX_REASONABLE_COMBOS:,}")

        col1, col2, col3 = st.columns(3)
        with col1:
            max_iterations = st.number_input(
                "Max Iterations",
                min_value=10,
                max_value=10000,
                value=min(500, max(10, estimated_combos if estimated_combos < 10000 else 500)),
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
            if estimated_combos > 1000000:
                st.metric("Total Combinations", f"{estimated_combos/1e6:.1f}M", delta="Too many!", delta_color="inverse")
            elif estimated_combos > 1000:
                st.metric("Total Combinations", f"{estimated_combos/1e3:.1f}K")
            else:
                st.metric("Total Combinations", f"{estimated_combos:,}")

        # Summary
        if enabled_names:
            names_display = ", ".join(enabled_names[:5])
            if len(enabled_names) > 5:
                names_display += f" +{len(enabled_names)-5} more"
            st.info(f"**Optimizing:** {names_display} → Testing: {min(max_iterations, estimated_combos):,} combinations")
        else:
            st.warning("Select at least one parameter to optimize")

        # Disable button if too many combos or no params
        can_run = enabled_names and not too_many_combos
        run_opt_btn = st.button("🚀 Run Optimization", type="primary", disabled=not can_run)

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

    # Multi-Symbol Batch Backtesting tab
    elif selected_tab == "🌐 Multi-Symbol":
        st.subheader("Multi-Symbol Batch Backtesting")
        st.markdown("Test your strategy across multiple symbols simultaneously.")

        # Preset symbol groups
        symbol_presets = {
            "US Indices": "SPY\nQQQ\nIWM\nDIA",
            "Tech Giants": "AAPL\nMSFT\nGOOG\nAMZN\nNVDA\nMETA",
            "Forex Majors": "EURUSD=X\nGBPUSD=X\nUSDJPY=X\nAUDUSD=X",
            "Crypto": "BTC-USD\nETH-USD\nSOL-USD\nADA-USD",
            "Commodities": "GC=F\nSI=F\nCL=F\nNG=F",
        }

        col1, col2 = st.columns([2, 1])

        with col1:
            # Symbol input
            default_symbols = st.session_state.get("batch_symbols", "SPY\nQQQ\nIWM")
            symbols_text = st.text_area(
                "Symbols (one per line)",
                value=default_symbols,
                height=150,
                help="Enter Yahoo Finance symbols, one per line"
            )
            st.session_state["batch_symbols"] = symbols_text

        with col2:
            st.markdown("**Quick Presets:**")
            for preset_name, preset_symbols in symbol_presets.items():
                if st.button(f"📋 {preset_name}", key=f"preset_{preset_name}"):
                    st.session_state["batch_symbols"] = preset_symbols
                    st.rerun()

        # Parse symbols
        symbols = [s.strip().upper() for s in symbols_text.split('\n') if s.strip()]

        if symbols:
            st.info(f"**{len(symbols)} symbols selected:** {', '.join(symbols[:10])}{'...' if len(symbols) > 10 else ''}")

        # Settings
        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        with col1:
            batch_min_trades = st.number_input(
                "Min Trades per Symbol",
                min_value=1,
                value=5,
                help="Minimum trades required for valid result"
            )
        with col2:
            use_same_dates = st.checkbox(
                "Use loaded date range",
                value=True,
                help="Use the same date range as the main data"
            )
        with col3:
            st.metric("Symbols to Test", len(symbols))

        # Run batch backtest
        run_batch = st.button("🚀 Run Batch Backtest", type="primary", disabled=len(symbols) == 0)

        if run_batch and symbols:
            batch_results = []
            progress_bar = st.progress(0)
            status_text = st.empty()
            start_time = time.time()

            # Get config from Pine inputs
            config = build_config_from_pine()
            config.pip_size = pip_size

            for i, sym in enumerate(symbols):
                elapsed = time.time() - start_time
                if i > 0:
                    eta = (elapsed / i) * (len(symbols) - i)
                    eta_str = f"ETA: {int(eta)}s"
                else:
                    eta_str = ""

                progress_bar.progress((i + 1) / len(symbols))
                status_text.text(f"Testing {sym} ({i + 1}/{len(symbols)}) {eta_str}")

                try:
                    # Fetch data for this symbol
                    if use_same_dates and "df" in st.session_state:
                        # Use same date range
                        df_main = st.session_state["df"]
                        fetch_result = fetch_data(
                            symbol=sym,
                            timeframe=timeframe,
                            start_date=df_main.index[0].to_pydatetime(),
                            end_date=df_main.index[-1].to_pydatetime(),
                        )
                    else:
                        # Use default range
                        fetch_result = fetch_data(
                            symbol=sym,
                            timeframe=timeframe,
                            start_date=datetime.now() - timedelta(days=365),
                            end_date=datetime.now(),
                        )

                    df_sym = fetch_result.df

                    if len(df_sym) < 50:
                        batch_results.append({
                            "Symbol": sym,
                            "Status": "⚠️ Insufficient data",
                            "Trades": 0,
                            "Win Rate": "-",
                            "Net P&L": "-",
                            "Profit Factor": "-",
                            "Max DD %": "-",
                            "Sharpe": "-",
                        })
                        continue

                    # Run backtest
                    bt_result = run_backtest(df_sym, config)

                    if bt_result.total_trades >= batch_min_trades:
                        batch_results.append({
                            "Symbol": sym,
                            "Status": "✅ OK",
                            "Trades": bt_result.total_trades,
                            "Win Rate": f"{bt_result.win_rate:.1%}",
                            "Net P&L": f"${bt_result.total_pnl:,.2f}",
                            "Profit Factor": f"{bt_result.profit_factor:.2f}",
                            "Max DD %": f"{bt_result.max_drawdown_pct:.1f}%",
                            "Sharpe": f"{bt_result.sharpe_ratio:.2f}",
                        })
                    else:
                        batch_results.append({
                            "Symbol": sym,
                            "Status": f"⚠️ Only {bt_result.total_trades} trades",
                            "Trades": bt_result.total_trades,
                            "Win Rate": f"{bt_result.win_rate:.1%}" if bt_result.total_trades > 0 else "-",
                            "Net P&L": f"${bt_result.total_pnl:,.2f}",
                            "Profit Factor": f"{bt_result.profit_factor:.2f}",
                            "Max DD %": f"{bt_result.max_drawdown_pct:.1f}%",
                            "Sharpe": "-",
                        })

                except Exception as e:
                    batch_results.append({
                        "Symbol": sym,
                        "Status": f"❌ Error: {str(e)[:30]}",
                        "Trades": 0,
                        "Win Rate": "-",
                        "Net P&L": "-",
                        "Profit Factor": "-",
                        "Max DD %": "-",
                        "Sharpe": "-",
                    })

            progress_bar.empty()
            status_text.empty()

            total_time = time.time() - start_time
            st.success(f"Completed {len(symbols)} symbols in {total_time:.1f}s")

            # Store results
            st.session_state["batch_results"] = batch_results

        # Display results
        if "batch_results" in st.session_state and st.session_state["batch_results"]:
            batch_results = st.session_state["batch_results"]

            st.subheader("Batch Results")

            # Summary metrics
            valid_results = [r for r in batch_results if r["Status"] == "✅ OK"]
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Symbols Tested", len(batch_results))
            with col2:
                st.metric("Valid Results", len(valid_results))
            with col3:
                if valid_results:
                    avg_win_rate = sum(float(r["Win Rate"].rstrip('%')) for r in valid_results) / len(valid_results)
                    st.metric("Avg Win Rate", f"{avg_win_rate:.1f}%")
            with col4:
                if valid_results:
                    total_pnl = sum(float(r["Net P&L"].replace('$', '').replace(',', '')) for r in valid_results)
                    st.metric("Total P&L", f"${total_pnl:,.2f}")

            # Results table
            results_df = pd.DataFrame(batch_results)
            st.dataframe(results_df, hide_index=True, use_container_width=True)

            # Download button
            st.download_button(
                "📥 Download Batch Results CSV",
                data=results_df.to_csv(index=False),
                file_name=f"batch_results_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv"
            )

    # Advanced Analysis tab (Monte Carlo + Walk-Forward)
    elif selected_tab == "🎲 Advanced":
        st.subheader("Advanced Analysis")

        if "result" not in st.session_state or st.session_state["result"].total_trades < 5:
            st.warning("Run a backtest with at least 5 trades first to enable advanced analysis.")
        else:
            result = st.session_state["result"]

            # Sub-tabs for different analyses
            analysis_type = st.radio(
                "Analysis Type",
                ["🎲 Monte Carlo Simulation", "📊 Walk-Forward Optimization"],
                horizontal=True,
                label_visibility="collapsed"
            )

            st.markdown("---")

            # ============ MONTE CARLO SIMULATION ============
            if analysis_type == "🎲 Monte Carlo Simulation":
                st.markdown("### Monte Carlo Simulation")
                st.markdown("""
                Monte Carlo simulation tests strategy robustness by randomly resampling trades
                to estimate the range of possible outcomes and risk metrics.
                """)

                col1, col2, col3 = st.columns(3)
                with col1:
                    mc_simulations = st.number_input(
                        "Number of Simulations",
                        min_value=100,
                        max_value=10000,
                        value=1000,
                        step=100,
                        help="More simulations = more accurate but slower"
                    )
                with col2:
                    mc_confidence = st.selectbox(
                        "Confidence Level",
                        options=[0.90, 0.95, 0.99],
                        index=1,
                        format_func=lambda x: f"{x:.0%}"
                    )
                with col3:
                    st.metric("Original Trades", result.total_trades)

                run_mc = st.button("🎲 Run Monte Carlo", type="primary")

                if run_mc:
                    with st.spinner(f"Running {mc_simulations:,} simulations..."):
                        try:
                            mc_result = run_monte_carlo(
                                result,
                                n_simulations=mc_simulations,
                                confidence_level=mc_confidence
                            )
                            st.session_state["mc_result"] = mc_result
                        except Exception as e:
                            st.error(f"Error: {e}")

                # Display results
                if "mc_result" in st.session_state:
                    mc = st.session_state["mc_result"]

                    st.subheader("Results Summary")

                    # Key metrics
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        delta_pnl = mc.pnl_mean - mc.original_pnl
                        st.metric(
                            "Expected P&L",
                            f"${mc.pnl_mean:,.0f}",
                            delta=f"${delta_pnl:+,.0f} vs actual"
                        )
                    with col2:
                        st.metric(
                            "P&L Range (CI)",
                            f"${mc.pnl_ci_lower:,.0f} to ${mc.pnl_ci_upper:,.0f}"
                        )
                    with col3:
                        st.metric(
                            "Prob. Positive",
                            f"{mc.probability_positive:.1%}",
                            delta="Good" if mc.probability_positive > 0.7 else "Risky",
                            delta_color="normal" if mc.probability_positive > 0.7 else "inverse"
                        )
                    with col4:
                        st.metric(
                            "Risk of Ruin",
                            f"{mc.risk_of_ruin:.1%}",
                            delta="Low" if mc.risk_of_ruin < 0.05 else "High",
                            delta_color="normal" if mc.risk_of_ruin < 0.05 else "inverse"
                        )

                    # More detailed metrics
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Win Rate Range", f"{mc.win_rate_ci_lower:.1%} - {mc.win_rate_ci_upper:.1%}")
                    with col2:
                        st.metric("Expected Max DD", f"{mc.max_dd_mean:.1f}%")
                    with col3:
                        st.metric("Worst Case DD", f"{mc.max_dd_worst:.1f}%")
                    with col4:
                        st.metric("Worst Case P&L", f"${mc.pnl_worst:,.0f}")

                    # Distribution charts
                    st.subheader("P&L Distribution")
                    fig_pnl = go.Figure()
                    fig_pnl.add_trace(go.Histogram(
                        x=mc.pnl_distribution,
                        nbinsx=50,
                        name="Simulated P&L",
                        marker_color="#2196f3",
                        opacity=0.7
                    ))
                    # Add vertical line for original PnL
                    fig_pnl.add_vline(
                        x=mc.original_pnl,
                        line_dash="dash",
                        line_color="red",
                        annotation_text="Actual"
                    )
                    fig_pnl.add_vline(
                        x=mc.pnl_ci_lower,
                        line_dash="dot",
                        line_color="orange",
                        annotation_text=f"CI Lower"
                    )
                    fig_pnl.add_vline(
                        x=mc.pnl_ci_upper,
                        line_dash="dot",
                        line_color="green",
                        annotation_text=f"CI Upper"
                    )
                    fig_pnl.update_layout(
                        height=300,
                        template="plotly_dark",
                        title=f"P&L Distribution ({mc.n_simulations:,} simulations)",
                        xaxis_title="P&L ($)",
                        yaxis_title="Frequency"
                    )
                    st.plotly_chart(fig_pnl, use_container_width=True)

                    # Drawdown distribution
                    st.subheader("Maximum Drawdown Distribution")
                    fig_dd = go.Figure()
                    fig_dd.add_trace(go.Histogram(
                        x=mc.dd_distribution,
                        nbinsx=50,
                        name="Max Drawdown %",
                        marker_color="#ef5350",
                        opacity=0.7
                    ))
                    fig_dd.add_vline(
                        x=mc.original_max_dd,
                        line_dash="dash",
                        line_color="yellow",
                        annotation_text="Actual"
                    )
                    fig_dd.update_layout(
                        height=300,
                        template="plotly_dark",
                        title="Maximum Drawdown Distribution",
                        xaxis_title="Max Drawdown (%)",
                        yaxis_title="Frequency"
                    )
                    st.plotly_chart(fig_dd, use_container_width=True)

            # ============ WALK-FORWARD OPTIMIZATION ============
            else:
                st.markdown("### Walk-Forward Optimization")
                st.markdown("""
                Walk-forward analysis splits data into train/test windows to prevent overfitting.
                Parameters are optimized on training data, then validated on out-of-sample test data.
                """)

                col1, col2, col3 = st.columns(3)
                with col1:
                    wf_windows = st.number_input(
                        "Number of Windows",
                        min_value=2,
                        max_value=10,
                        value=5,
                        help="More windows = more robust but requires more data"
                    )
                with col2:
                    wf_train_pct = st.slider(
                        "Train %",
                        min_value=50,
                        max_value=90,
                        value=70,
                        help="Percentage of each window for training"
                    ) / 100
                with col3:
                    wf_metric = st.selectbox(
                        "Optimize For",
                        options=["sharpe_ratio", "profit_factor", "total_pnl", "win_rate"],
                        format_func=lambda x: x.replace("_", " ").title()
                    )

                # Simple parameter grid (based on current config)
                st.markdown("##### Parameters to Optimize")
                st.caption("Walk-forward will test these parameter variations in each window")

                col1, col2 = st.columns(2)
                with col1:
                    wf_optimize_tp = st.checkbox("TP Ratio", value=True)
                    wf_tp_values = st.text_input(
                        "TP Ratio values",
                        value="1.0, 1.5, 2.0, 2.5",
                        disabled=not wf_optimize_tp
                    )
                with col2:
                    wf_optimize_sl = st.checkbox("SL Pips", value=False)
                    wf_sl_values = st.text_input(
                        "SL Pips values",
                        value="5, 10, 15, 20",
                        disabled=not wf_optimize_sl
                    )

                # Build param grid
                param_grid = {}
                if wf_optimize_tp:
                    try:
                        param_grid["tp_ratio"] = [float(x.strip()) for x in wf_tp_values.split(",")]
                    except:
                        st.error("Invalid TP Ratio values")
                if wf_optimize_sl:
                    try:
                        param_grid["sl_pips"] = [int(x.strip()) for x in wf_sl_values.split(",")]
                    except:
                        st.error("Invalid SL Pips values")

                run_wf = st.button(
                    "📊 Run Walk-Forward",
                    type="primary",
                    disabled=len(param_grid) == 0
                )

                if run_wf and param_grid:
                    with st.spinner("Running walk-forward analysis..."):
                        try:
                            # Get base config
                            config = build_config_from_pine()
                            config.pip_size = pip_size

                            progress_bar = st.progress(0)
                            status_text = st.empty()

                            def update_progress(current, total, msg):
                                progress_bar.progress(current / total)
                                status_text.text(f"Window {current}/{total}: {msg}")

                            wf_result = run_walk_forward(
                                df,
                                config,
                                param_grid,
                                train_pct=wf_train_pct,
                                n_windows=wf_windows,
                                metric=wf_metric,
                                min_trades=3,
                                progress_callback=update_progress
                            )

                            progress_bar.empty()
                            status_text.empty()

                            st.session_state["wf_result"] = wf_result
                            st.success("Walk-forward analysis complete!")

                        except Exception as e:
                            st.error(f"Error: {e}")

                # Display results
                if "wf_result" in st.session_state:
                    wf = st.session_state["wf_result"]

                    st.subheader("Walk-Forward Results")

                    # Summary metrics
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Total Windows", wf.total_windows)
                    with col2:
                        st.metric("Valid Windows", wf.valid_windows)
                    with col3:
                        st.metric(
                            "Robustness Score",
                            f"{wf.robustness_score:.0f}/100",
                            delta="Robust" if wf.robustness_score >= 60 else "Needs work",
                            delta_color="normal" if wf.robustness_score >= 60 else "inverse"
                        )
                    with col4:
                        st.metric(
                            "Consistency",
                            f"{wf.aggregate_metrics['consistency']:.0%}",
                            help="% of windows with positive test P&L"
                        )

                    # Aggregate metrics
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Avg Test Trades", f"{wf.aggregate_metrics['avg_trades']:.0f}")
                    with col2:
                        st.metric("Avg Test Win Rate", f"{wf.aggregate_metrics['avg_win_rate']:.1%}")
                    with col3:
                        st.metric("Avg Test P&L", f"${wf.aggregate_metrics['avg_pnl']:,.2f}")
                    with col4:
                        st.metric("Total Test P&L", f"${wf.aggregate_metrics['total_pnl']:,.2f}")

                    # Detailed results table
                    st.subheader("Window Details")
                    wf_df = format_walk_forward_results(wf)
                    st.dataframe(wf_df, hide_index=True, use_container_width=True)

                    # Download
                    st.download_button(
                        "📥 Download Walk-Forward Results",
                        data=wf_df.to_csv(index=False),
                        file_name=f"walk_forward_{symbol}_{datetime.now().strftime('%Y%m%d')}.csv",
                        mime="text/csv"
                    )

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
