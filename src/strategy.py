"""
Strategy module - Python implementation of Pine Script patterns and filters.
Exact port of Algo Strategy Builder patterns for matching TradingView results.

Pine Script Reference: strategy_builder/Algo_Strategy_Builder.pine
"""
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import List


@dataclass
class StrategyConfig:
    """
    Strategy configuration matching Pine Script inputs.
    Maps directly to the inputs in Algo_Strategy_Builder.pine
    """
    # ===== Configuración Básica =====
    entry_pips: int = 1      # Pips para entrada
    sl_pips: int = 1         # Pips para Stop Loss
    pip_size: float = 0.0001 # For forex; adjust for other assets

    # ===== Sentido =====
    trade_longs: bool = True   # operar_largos
    trade_shorts: bool = True  # operar_cortos

    # ===== Entradas (Patrones) =====
    use_sacudida: bool = True           # usar_patron_sacudida
    use_engulfing: bool = True          # usar_patron_envolvente
    use_climatic_volume: bool = False   # usar_patron_vol_climatico

    # ===== Salidas =====
    use_sl: bool = True              # usar_sl_original
    use_tp_ratio: bool = True        # usar_salida_tp_ratio
    tp_ratio: float = 1.0            # target_ratio
    use_n_bars_exit: bool = False    # usar_salida_n_velas
    n_bars_exit: int = 5             # n_velas_salida

    # ===== Filtros =====
    # filtro_mm50200: 'Sin filtro', 'Alcista (MM50>200)', 'Bajista (MM50<200)'
    ma_filter: str = "Sin filtro"

    # ===== Sesiones de negociación =====
    use_london: bool = True    # usarLondon   (01:00-08:15 UTC)
    use_newyork: bool = True   # usarNewYork  (08:15-15:45 UTC)
    use_tokyo: bool = True     # usarTokio    (15:45-01:00 UTC)

    # ===== Días de la semana =====
    trade_monday: bool = True     # operarLunes
    trade_tuesday: bool = True    # operarMartes
    trade_wednesday: bool = True  # operarMiercoles
    trade_thursday: bool = True   # operarJueves
    trade_friday: bool = True     # operarViernes
    trade_saturday: bool = True   # operarSabado
    trade_sunday: bool = True     # operarDomingo

    # ===== Gestión de riesgo =====
    # tipo_gestion: 'Tamaño fijo', 'Riesgo monetario fijo', 'Riesgo % equity'
    risk_type: str = "fixed_size"
    fixed_size: float = 1.0           # tamano_fijo_qty
    fixed_risk_money: float = 100.0   # riesgo_monetario
    risk_percent: float = 1.0         # porc_riesgo_equity
    qty_step: float = 1.0             # qty_step
    qty_min: float = 1.0              # qty_min

    # ===== Backtest settings (from strategy() call) =====
    initial_capital: float = 100000.0
    commission: float = 1.5   # commission_value (cash per contract)
    slippage: int = 1         # slippage (in ticks)


# =============================================================================
# PATTERN DETECTION
# Exact port of Pine Script patterns
# =============================================================================

def detect_sacudida_long(df: pd.DataFrame) -> pd.Series:
    """
    Sacudida (Shakeout) LONG pattern.

    Pine Script (lines 92-97):
        sacudida_long_condition() =>
            vela2_bajista        = close[1] < open[1]
            vela2_rompe_minimo   = low[1]   < low[2]
            vela3_alcista        = close    > open
            vela3_confirmacion   = close    > low[2]
            vela2_bajista and vela2_rompe_minimo and vela3_alcista and vela3_confirmacion

    Logic:
        - Bar[1] (previous) is bearish and breaks below bar[2] low
        - Bar[0] (current) is bullish and closes above bar[2] low
    """
    vela2_bajista = df["close"].shift(1) < df["open"].shift(1)
    vela2_rompe_minimo = df["low"].shift(1) < df["low"].shift(2)
    vela3_alcista = df["close"] > df["open"]
    vela3_confirmacion = df["close"] > df["low"].shift(2)

    return vela2_bajista & vela2_rompe_minimo & vela3_alcista & vela3_confirmacion


def detect_sacudida_short(df: pd.DataFrame) -> pd.Series:
    """
    Sacudida (Shakeout) SHORT pattern.

    Pine Script (lines 99-104):
        sacudida_short_condition() =>
            vela2_alcista        = close[1] > open[1]
            vela2_rompe_maximo   = high[1]  > high[2]
            vela3_bajista        = close    < open
            vela3_confirmacion   = close    < high[2]
            vela2_alcista and vela2_rompe_maximo and vela3_bajista and vela3_confirmacion

    Logic:
        - Bar[1] (previous) is bullish and breaks above bar[2] high
        - Bar[0] (current) is bearish and closes below bar[2] high
    """
    vela2_alcista = df["close"].shift(1) > df["open"].shift(1)
    vela2_rompe_maximo = df["high"].shift(1) > df["high"].shift(2)
    vela3_bajista = df["close"] < df["open"]
    vela3_confirmacion = df["close"] < df["high"].shift(2)

    return vela2_alcista & vela2_rompe_maximo & vela3_bajista & vela3_confirmacion


def detect_engulfing_bullish(df: pd.DataFrame) -> pd.Series:
    """
    Bullish Engulfing pattern.

    Pine Script (lines 107-112):
        bullEngulf() =>
            VelaAlcista = close > open
            VelaBajistaPrev = close[1] < open[1]
            cierra_sobre_ap1 = close >= open[1]
            abre_bajo_c1     = open  <= close[1]
            VelaAlcista and VelaBajistaPrev and cierra_sobre_ap1 and abre_bajo_c1
    """
    vela_alcista = df["close"] > df["open"]
    vela_bajista_prev = df["close"].shift(1) < df["open"].shift(1)
    cierra_sobre_ap1 = df["close"] >= df["open"].shift(1)
    abre_bajo_c1 = df["open"] <= df["close"].shift(1)

    return vela_alcista & vela_bajista_prev & cierra_sobre_ap1 & abre_bajo_c1


def detect_engulfing_bearish(df: pd.DataFrame) -> pd.Series:
    """
    Bearish Engulfing pattern.

    Pine Script (lines 114-119):
        bearEngulf() =>
            VelaBajista = close < open
            VelaAlcistaPrev = close[1] > open[1]
            cierra_bajo_ap1  = close <= open[1]
            abre_sobre_c1    = open  >= close[1]
            VelaBajista and VelaAlcistaPrev and cierra_bajo_ap1 and abre_sobre_c1
    """
    vela_bajista = df["close"] < df["open"]
    vela_alcista_prev = df["close"].shift(1) > df["open"].shift(1)
    cierra_bajo_ap1 = df["close"] <= df["open"].shift(1)
    abre_sobre_c1 = df["open"] >= df["close"].shift(1)

    return vela_bajista & vela_alcista_prev & cierra_bajo_ap1 & abre_sobre_c1


def detect_climatic_volume(df: pd.DataFrame) -> pd.Series:
    """
    Climatic Volume detection.

    Pine Script (lines 122-123):
        volMA20      = ta.sma(volume, 20)
        volClimatico = not na(volMA20) and volume > volMA20 * 1.75

    Volume is considered "climatic" when it's > 1.75x the 20-period average.
    """
    vol_ma20 = df["volume"].rolling(window=20).mean()
    return df["volume"] > vol_ma20 * 1.75


def detect_climatic_volume_long(df: pd.DataFrame) -> pd.Series:
    """
    Climatic Volume LONG: climatic volume + bullish candle.

    Pine Script (line 124):
        clim_long_raw = volClimatico and close > open
    """
    climatic = detect_climatic_volume(df)
    bullish = df["close"] > df["open"]
    return climatic & bullish


def detect_climatic_volume_short(df: pd.DataFrame) -> pd.Series:
    """
    Climatic Volume SHORT: climatic volume + bearish candle.

    Pine Script (line 125):
        clim_short_raw = volClimatico and close < open
    """
    climatic = detect_climatic_volume(df)
    bearish = df["close"] < df["open"]
    return climatic & bearish


# =============================================================================
# FILTERS
# =============================================================================

def calculate_sma(series: pd.Series, period: int) -> pd.Series:
    """Calculate Simple Moving Average."""
    return series.rolling(window=period).mean()


def apply_ma_filter(df: pd.DataFrame, filter_type: str) -> pd.Series:
    """
    Apply MA 50/200 filter.

    Pine Script (lines 134-136):
        sma50  = ta.sma(close, 50)
        sma200 = ta.sma(close, 200)
        filtro_ok_mm = filtro_mm50200 == 'Sin filtro' ? true :
                       filtro_mm50200 == 'Alcista (MM50>200)' ? (sma50 > sma200) :
                       (sma50 < sma200)
    """
    if filter_type == "Sin filtro":
        return pd.Series(True, index=df.index)

    sma50 = calculate_sma(df["close"], 50)
    sma200 = calculate_sma(df["close"], 200)

    if filter_type == "Alcista (MM50>200)":
        return sma50 > sma200
    else:  # Bajista (MM50<200)
        return sma50 < sma200


def apply_day_filter(df: pd.DataFrame, config: StrategyConfig) -> pd.Series:
    """
    Filter by day of week.

    Pine Script (lines 66-76):
        operarHoy() => checks dayofweek against enabled days

    Note: Pine Script dayofweek: monday=2, sunday=1
          Python weekday(): monday=0, sunday=6
    """
    day_map = {
        0: config.trade_monday,
        1: config.trade_tuesday,
        2: config.trade_wednesday,
        3: config.trade_thursday,
        4: config.trade_friday,
        5: config.trade_saturday,
        6: config.trade_sunday,
    }
    return pd.Series([day_map.get(d, True) for d in df.index.dayofweek], index=df.index)


def apply_session_filter(df: pd.DataFrame, config: StrategyConfig) -> pd.Series:
    """
    Filter by trading session.

    Pine Script (lines 79-82):
        enHorario1 = not na(time(timeframe.period, '0100-0815')) // London
        enHorario2 = not na(time(timeframe.period, '0815-1545')) // New York
        enHorario3 = not na(time(timeframe.period, '1545-0100')) // Tokyo
        filtro_ok_horario = (usarLondon and enHorario1) or (usarNewYork and enHorario2) or (usarTokio and enHorario3)

    Note: Times are in exchange timezone. Using minute-precision for exact match.
    """
    # Convert to minutes since midnight for precise session boundaries
    try:
        minutes = df.index.hour * 60 + df.index.minute
    except AttributeError:
        # If index is not datetime, return all True
        return pd.Series(True, index=df.index)

    # Session definitions (exact minute boundaries as per Pine Script)
    # London: 01:00 - 08:15 (60 to 495 minutes)
    london = (minutes >= 60) & (minutes < 495)

    # New York: 08:15 - 15:45 (495 to 945 minutes)
    newyork = (minutes >= 495) & (minutes < 945)

    # Tokyo: 15:45 - 01:00 (945 to 1440 OR 0 to 60 minutes)
    tokyo = (minutes >= 945) | (minutes < 60)

    session_ok = pd.Series(False, index=df.index)
    if config.use_london:
        session_ok = session_ok | london
    if config.use_newyork:
        session_ok = session_ok | newyork
    if config.use_tokyo:
        session_ok = session_ok | tokyo

    # If no sessions enabled, allow all
    if not (config.use_london or config.use_newyork or config.use_tokyo):
        return pd.Series(True, index=df.index)

    return session_ok


# =============================================================================
# SIGNAL GENERATION
# =============================================================================

def generate_signals(df: pd.DataFrame, config: StrategyConfig) -> pd.DataFrame:
    """
    Generate entry signals based on patterns and filters.

    Pine Script (lines 165-176):
        bull_signal_raw = (usar_patron_sacudida and sacudida_long_condition()) or
                          (usar_patron_envolvente and bullEngulf()) or
                          (usar_patron_vol_climatico and clim_long_raw)

        bear_signal_raw = (usar_patron_sacudida and sacudida_short_condition()) or
                          (usar_patron_envolvente and bearEngulf()) or
                          (usar_patron_vol_climatico and clim_short_raw)

        long_ok  = operar_largos and operarHoy() and filtro_ok_horario and filtro_ok_mm and bull_signal_raw
        short_ok = operar_cortos and operarHoy() and filtro_ok_horario and filtro_ok_mm and bear_signal_raw

    Returns:
        DataFrame with signal columns added
    """
    result = df.copy()

    # ===== Detect patterns =====
    bull_patterns = pd.Series(False, index=df.index)
    bear_patterns = pd.Series(False, index=df.index)

    # Track which pattern triggered (for order type determination)
    climatic_long = pd.Series(False, index=df.index)
    climatic_short = pd.Series(False, index=df.index)

    if config.use_sacudida:
        bull_patterns = bull_patterns | detect_sacudida_long(df)
        bear_patterns = bear_patterns | detect_sacudida_short(df)

    if config.use_engulfing:
        bull_patterns = bull_patterns | detect_engulfing_bullish(df)
        bear_patterns = bear_patterns | detect_engulfing_bearish(df)

    if config.use_climatic_volume:
        climatic_long = detect_climatic_volume_long(df)
        climatic_short = detect_climatic_volume_short(df)
        bull_patterns = bull_patterns | climatic_long
        bear_patterns = bear_patterns | climatic_short

    # ===== Apply filters =====
    ma_ok = apply_ma_filter(df, config.ma_filter)
    day_ok = apply_day_filter(df, config)
    session_ok = apply_session_filter(df, config)

    filters_ok = ma_ok & day_ok & session_ok

    # ===== Final signals =====
    # long_ok = operar_largos and operarHoy() and filtro_ok_horario and filtro_ok_mm and bull_signal_raw
    long_signal = config.trade_longs & bull_patterns & filters_ok
    short_signal = config.trade_shorts & bear_patterns & filters_ok

    # ===== Calculate entry/exit prices =====
    pip = config.pip_size

    # SL base levels (Pine Script lines 196-197):
    # sllow  = ta.lowest(low,  2)   // base SL largos
    # slhigh = ta.highest(high, 2)  // base SL cortos
    sl_low = df["low"].rolling(window=2).min()
    sl_high = df["high"].rolling(window=2).max()

    # Entry prices (Pine Script lines 206-209, 243-246):
    # For climatic volume: market order at close
    # For other patterns: stop order above high (long) / below low (short)
    entry_long = np.where(
        climatic_long,
        df["close"],  # Market order
        df["high"] + config.entry_pips * pip  # Stop order
    )
    entry_short = np.where(
        climatic_short,
        df["close"],  # Market order
        df["low"] - config.entry_pips * pip  # Stop order
    )

    # Stop loss (Pine Script lines 211, 248):
    # sl_theo_L = sllow - sl_pip * GetPipSize()
    # sl_theo_S = slhigh + sl_pip * GetPipSize()
    sl_long = sl_low - config.sl_pips * pip
    sl_short = sl_high + config.sl_pips * pip

    # Take profit (Pine Script lines 217-218, 254-255):
    # riesgoL = entry_price_L - sl_theo_L
    # targetL := entry_price_L + riesgoL * target_ratio
    risk_long = entry_long - sl_long
    risk_short = sl_short - entry_short
    tp_long = entry_long + risk_long * config.tp_ratio
    tp_short = entry_short - risk_short * config.tp_ratio

    # ===== Store results =====
    result["long_signal"] = long_signal
    result["short_signal"] = short_signal
    result["entry_long"] = entry_long
    result["entry_short"] = entry_short
    result["sl_long"] = sl_long
    result["sl_short"] = sl_short
    result["tp_long"] = tp_long
    result["tp_short"] = tp_short
    result["is_climatic_long"] = climatic_long
    result["is_climatic_short"] = climatic_short

    # Add indicators for visualization
    result["sma50"] = calculate_sma(df["close"], 50)
    result["sma200"] = calculate_sma(df["close"], 200)
    result["vol_ma20"] = df["volume"].rolling(window=20).mean()

    return result
