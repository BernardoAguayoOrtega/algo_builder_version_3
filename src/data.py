"""
Data module for fetching market data from Yahoo Finance.
Handles chunked fetching for intraday data and resampling for 4h timeframe.
"""
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from typing import Optional, List, Tuple
import time


# Yahoo Finance limits per timeframe (in days FROM TODAY - not just span)
# These are hard limits on how far back you can request data
TIMEFRAME_LIMITS = {
    "1m": 7,
    "5m": 60,
    "15m": 60,
    "30m": 60,
    "1h": 730,
    "1d": None,   # Unlimited
    "1wk": None,
    "1mo": None,
}

# Chunk sizes for fetching (smaller than limits to be safe)
CHUNK_SIZES = {
    "1m": 6,      # Fetch 6 days at a time
    "5m": 55,     # Fetch 55 days at a time
    "15m": 55,
    "30m": 55,
    "1h": 700,    # Fetch 700 days at a time
}

# Map our timeframes to Yahoo Finance intervals
YAHOO_INTERVALS = {
    "1m": "1m",
    "5m": "5m",
    "15m": "15m",
    "30m": "30m",
    "1h": "1h",
    "4h": "1h",   # Fetch 1h, resample to 4h
    "1d": "1d",
    "1wk": "1wk",
    "1mo": "1mo",
}

# Delay between chunk requests (seconds) to avoid rate limiting
REQUEST_DELAY = 0.5


def get_chunk_ranges_backward(
    start_date: datetime,
    end_date: datetime,
    max_days: int,
) -> List[Tuple[datetime, datetime]]:
    """
    Split a date range into chunks, working BACKWARD from end_date.
    This is critical for Yahoo Finance which limits how far back you can go from TODAY.

    Args:
        start_date: Start of full range
        end_date: End of full range
        max_days: Maximum days per chunk

    Returns:
        List of (chunk_start, chunk_end) tuples, ordered from oldest to newest
    """
    chunks = []
    current_end = end_date

    while current_end > start_date:
        current_start = max(current_end - timedelta(days=max_days), start_date)
        chunks.append((current_start, current_end))
        current_end = current_start

    # Reverse so we fetch from oldest to newest (for progress display)
    chunks.reverse()
    return chunks


def fetch_chunk(
    symbol: str,
    interval: str,
    start: datetime,
    end: datetime,
) -> pd.DataFrame:
    """
    Fetch a single chunk of data from Yahoo Finance.

    Args:
        symbol: Ticker symbol
        interval: Yahoo Finance interval string
        start: Chunk start date
        end: Chunk end date

    Returns:
        DataFrame with OHLCV data
    """
    ticker = yf.Ticker(symbol)
    df = ticker.history(start=start, end=end, interval=interval)
    return df


def resample_to_4h(df: pd.DataFrame) -> pd.DataFrame:
    """
    Resample 1h data to 4h candles.

    Args:
        df: DataFrame with 1h OHLCV data

    Returns:
        DataFrame with 4h OHLCV data
    """
    if df.empty:
        return df

    resampled = df.resample("4h").agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }).dropna()

    return resampled


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize column names to lowercase.

    Args:
        df: DataFrame from Yahoo Finance

    Returns:
        DataFrame with lowercase column names
    """
    df = df.rename(columns={
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Volume": "volume",
    })

    # Keep only OHLCV columns
    cols_to_keep = ["open", "high", "low", "close", "volume"]
    available_cols = [c for c in cols_to_keep if c in df.columns]

    return df[available_cols]


def fetch_data(
    symbol: str,
    timeframe: str = "1d",
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    progress_callback=None,
) -> pd.DataFrame:
    """
    Fetch OHLCV data from Yahoo Finance with automatic chunking.

    Args:
        symbol: Ticker symbol (e.g., 'AAPL', 'EURUSD=X', 'BTC-USD')
        timeframe: Candle timeframe ('1m', '5m', '15m', '30m', '1h', '4h', '1d', '1wk', '1mo')
        start_date: Start date for data
        end_date: End date for data
        progress_callback: Optional callback(current, total, message) for progress updates

    Returns:
        DataFrame with columns: open, high, low, close, volume
    """
    if timeframe not in YAHOO_INTERVALS:
        raise ValueError(f"Invalid timeframe: {timeframe}. Use one of {list(YAHOO_INTERVALS.keys())}")

    yahoo_interval = YAHOO_INTERVALS[timeframe]
    needs_resample = timeframe == "4h"

    # Get the limit for this timeframe (use 1h limit for 4h since we fetch 1h)
    limit_key = "1h" if needs_resample else timeframe
    max_days_back = TIMEFRAME_LIMITS.get(limit_key)
    chunk_size = CHUNK_SIZES.get(limit_key)

    # Set default date range
    now = datetime.now()
    if end_date is None:
        end_date = now
    if start_date is None:
        if max_days_back:
            start_date = end_date - timedelta(days=max_days_back)
        else:
            start_date = end_date - timedelta(days=365 * 5)  # 5 years default for daily+

    # Enforce Yahoo's hard limit: can't request data older than max_days_back from TODAY
    if max_days_back is not None:
        earliest_allowed = now - timedelta(days=max_days_back)
        if start_date < earliest_allowed:
            start_date = earliest_allowed
            if progress_callback:
                progress_callback(0, 1, f"Note: {timeframe} data limited to last {max_days_back} days")

    # For intraday timeframes, use chunking to avoid timeouts
    if chunk_size is not None:
        total_days = (end_date - start_date).days
        if total_days > chunk_size:
            # Fetch in chunks
            chunks = get_chunk_ranges_backward(start_date, end_date, chunk_size)
            all_data = []

            for i, (chunk_start, chunk_end) in enumerate(chunks):
                if progress_callback:
                    progress_callback(i + 1, len(chunks), f"Fetching chunk {i + 1}/{len(chunks)}")

                try:
                    chunk_df = fetch_chunk(symbol, yahoo_interval, chunk_start, chunk_end)
                    if not chunk_df.empty:
                        all_data.append(chunk_df)
                except Exception as e:
                    # Skip failed chunks but continue
                    if progress_callback:
                        progress_callback(i + 1, len(chunks), f"Chunk {i + 1} failed, continuing...")

                # Delay between requests to avoid rate limiting
                if i < len(chunks) - 1:
                    time.sleep(REQUEST_DELAY)

            if not all_data:
                raise ValueError(f"No data found for {symbol}")

            # Merge chunks
            df = pd.concat(all_data)
            df = df[~df.index.duplicated(keep='first')]
            df = df.sort_index()
        else:
            # Single request for small ranges
            if progress_callback:
                progress_callback(1, 1, "Fetching data...")
            df = fetch_chunk(symbol, yahoo_interval, start_date, end_date)
    else:
        # Daily+ timeframes: single request
        if progress_callback:
            progress_callback(1, 1, "Fetching data...")
        df = fetch_chunk(symbol, yahoo_interval, start_date, end_date)

    if df.empty:
        raise ValueError(f"No data found for {symbol}")

    # Standardize column names
    df = standardize_columns(df)

    # Resample to 4h if needed
    if needs_resample:
        if progress_callback:
            progress_callback(1, 1, "Resampling to 4h...")
        df = resample_to_4h(df)

    return df


def get_symbol_info(symbol: str) -> dict:
    """
    Get basic info about a symbol.

    Args:
        symbol: Ticker symbol

    Returns:
        Dictionary with symbol info
    """
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        return {
            "symbol": symbol,
            "name": info.get("longName", info.get("shortName", symbol)),
            "type": info.get("quoteType", "Unknown"),
            "currency": info.get("currency", "USD"),
            "exchange": info.get("exchange", "Unknown"),
        }
    except Exception:
        return {
            "symbol": symbol,
            "name": symbol,
            "type": "Unknown",
            "currency": "USD",
            "exchange": "Unknown",
        }


def get_available_timeframes() -> List[str]:
    """Return list of available timeframes."""
    return list(YAHOO_INTERVALS.keys())
