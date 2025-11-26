# Algo Strategy Builder - Implementation Plan

## Goal
Personal backtesting app that replicates TradingView/Pine Script results exactly, using Yahoo Finance data.

## Phase 1: Data Loading + Charts (MVP) ✅ COMPLETE

### 1.1 Data Module
- [x] Yahoo Finance data fetcher (OHLCV)
- [x] Support all timeframes (1m to 1mo)
- [x] **Chunked fetching for intraday** (backward chunking from today)
- [x] **Resample 1h → 4h** (Yahoo doesn't support 4h natively)
- [x] Rate limit handling (delays between requests)
- [x] Data validation and cleaning
- [x] Merge chunks and handle gaps

#### Yahoo Finance Data Limits
| Timeframe | Max per Request | Strategy |
|-----------|-----------------|----------|
| 1m | 7 days | Chunk into 6-day requests |
| 5m | 60 days | Chunk into 55-day requests |
| 15m | 60 days | Chunk into 55-day requests |
| 30m | 60 days | Chunk into 55-day requests |
| 1h | 730 days | Chunk into 700-day requests |
| 4h | N/A | Fetch 1h, resample to 4h |
| 1d, 1wk, 1mo | Unlimited | Single request |

### 1.2 Streamlit UI - Data Tab
- [x] Symbol input (stocks, forex, crypto, futures)
- [x] Timeframe selector
- [x] Date range picker
- [x] Load data button
- [x] Display data stats (bars loaded, date range)

### 1.3 Charts
- [x] Interactive candlestick chart (Plotly)
- [x] Volume subplot
- [x] Basic indicators overlay (SMA 50/200 for MA filter)
- [x] Zoom/pan functionality

---

## Phase 2: Strategy Engine (Exact Pine Script Match) ✅ COMPLETE

### 2.1 Pattern Detection
- [x] Sacudida (Shakeout) - long & short
- [x] Envolvente (Engulfing) - bullish & bearish
- [x] Volumen Climático - long & short

### 2.2 Filters
- [x] MA 50/200 crossover filter
- [x] Session filter (London/NY/Tokyo)
- [x] Day of week filter

### 2.3 Order Logic (Critical for exact match)
- [x] Stop orders vs market orders (climatic volume = market)
- [x] Entry price calculation (high + pips / low - pips)
- [x] SL calculation (lowest low 2 bars / highest high 2 bars)
- [x] TP calculation (entry + risk * ratio)
- [x] Order cancellation if SL hit before entry

---

## Phase 3: Backtester ✅ COMPLETE

### 3.1 Position Management
- [x] Single position at a time (pyramiding=1)
- [x] Position sizing (fixed, fixed $, % equity)
- [x] Commission and slippage

### 3.2 Exit Logic
- [x] Stop loss execution
- [x] Take profit execution
- [x] N-bars exit
- [x] Proper order of checks (SL before TP on same bar)

### 3.3 Results
- [x] Trade list with all details
- [x] Equity curve
- [x] Performance metrics (win rate, PF, drawdown, Sharpe, etc.)

---

## Phase 4: Validation & Polish

### 4.1 Validation
- [ ] Compare specific trades with TradingView
- [ ] Document any differences and why
- [ ] Handle edge cases

### 4.2 UI Polish
- [x] Trade markers on chart
- [x] Export trades to CSV
- [ ] Save/load configurations

### 4.3 Parameter Optimization ✅ COMPLETE
- [x] Optimizer module (src/optimizer.py)
- [x] **Entry patterns optimization** (Sacudida, Engulfing, Climatic - all combinations)
- [x] **Exit configurations optimization** (TP ratios, N-bars exit)
- [x] Test all combinations of filters/sessions/days
- [x] MAR ratio ranking (Mean Annual Return / Max Drawdown)
- [x] Flexible optimization settings (choose what to optimize)
- [x] Custom max combinations limit
- [x] Apply best configuration to backtest
- [x] Export optimization results to CSV

---

## File Structure
```
algo_builder_version_3/
├── .venv/                 # uv virtual environment
├── app.py                 # Streamlit main app
├── requirements.txt
├── PLAN.md
├── src/
│   ├── __init__.py
│   ├── data.py           # Yahoo Finance loader with chunking
│   ├── strategy.py       # Pattern detection & filters
│   ├── backtester.py     # Trade simulation
│   └── optimizer.py      # Parameter optimization
└── strategy_builder/
    ├── Algo_Strategy_Builder.pine  # Reference Pine Script
    ├── Proceso creación estrategias.txt
    └── PROMPS.txt
```

---

## Current Status
- [x] Phase 1 - Data Loading + Charts ✅
- [x] Phase 2 - Strategy Engine ✅
- [x] Phase 3 - Backtester ✅
- [ ] Phase 4 - Validation & Polish (in progress)

## Running the App
```bash
cd algo_builder_version_3
source .venv/bin/activate
streamlit run app.py
```
