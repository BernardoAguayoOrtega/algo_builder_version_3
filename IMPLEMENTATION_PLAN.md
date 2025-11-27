# Algo Strategy Builder - Implementation Plan

## Overview
Enhancement roadmap to transform the current backtesting app into a production-ready algorithmic trading platform.

---

## Phase 1: Quick Wins (High Impact, Low Effort)

### 1.1 Session State Persistence
**Priority:** HIGH | **Effort:** 2 hours

Save and restore user sessions to prevent data loss.

**Features:**
- [ ] Export session to downloadable `.pkl` file
- [ ] Import session from file
- [ ] Auto-save to browser localStorage (optional)

**Files to modify:**
- `app.py` - Add session management functions and UI

---

### 1.2 Performance Caching
**Priority:** HIGH | **Effort:** 1 hour

Cache expensive operations for 3-5x faster UI.

**Features:**
- [ ] Cache chart creation with `@st.cache_data`
- [ ] Cache data fetching
- [ ] Cache backtest results by config hash

**Files to modify:**
- `app.py` - Add caching decorators
- `src/data.py` - Cache fetch operations

---

## Phase 2: Medium Effort Enhancements

### 2.1 Multi-Symbol Batch Backtesting
**Priority:** HIGH | **Effort:** 4 hours

Test strategies across multiple symbols simultaneously.

**Features:**
- [ ] Batch mode toggle in sidebar
- [ ] Multi-line symbol input
- [ ] Parallel fetching and backtesting
- [ ] Comparative results table
- [ ] Export batch results to CSV

**Files to modify:**
- `app.py` - Add batch mode UI and logic

---

### 2.2 Walk-Forward Optimization
**Priority:** HIGH | **Effort:** 6 hours

Prevent overfitting with rolling train/test windows.

**Features:**
- [ ] Split data into train/test windows
- [ ] Optimize on train, validate on test
- [ ] Display out-of-sample performance
- [ ] Robustness score calculation

**Files to create:**
- `src/walk_forward.py` - Walk-forward logic

**Files to modify:**
- `app.py` - Add walk-forward tab/section

---

### 2.3 Monte Carlo Simulation
**Priority:** MEDIUM | **Effort:** 4 hours

Show confidence intervals and worst-case scenarios.

**Features:**
- [ ] Resample trades randomly (1,000+ simulations)
- [ ] Calculate PnL confidence intervals
- [ ] Show worst-case drawdown (95% confidence)
- [ ] Visualize distribution charts

**Files to create:**
- `src/monte_carlo.py` - Simulation logic

**Files to modify:**
- `app.py` - Add Monte Carlo section in Results tab

---

### 2.4 Parameter Sensitivity Analysis
**Priority:** MEDIUM | **Effort:** 3 hours

Show which parameters have the most impact.

**Features:**
- [ ] Calculate impact score per parameter
- [ ] Rank parameters by importance
- [ ] Show optimal value for each
- [ ] Heatmap visualization

**Files to modify:**
- `app.py` - Add sensitivity analysis after optimization

---

## Phase 3: Strategic Enhancements

### 3.1 Database Backend
**Priority:** MEDIUM | **Effort:** 6 hours

Persist results for historical tracking.

**Features:**
- [ ] SQLite database for local storage
- [ ] Save all backtest runs automatically
- [ ] Save individual trades
- [ ] History browser tab
- [ ] Compare historical runs

**Files to create:**
- `src/database.py` - Database models and queries

**Files to modify:**
- `app.py` - Add history tab and auto-save

---

### 3.2 Real-Time Alerts (Future)
**Priority:** LOW | **Effort:** 8 hours

Monitor live markets and send alerts.

**Features:**
- [ ] Live data streaming
- [ ] Signal detection loop
- [ ] Email/SMS notifications
- [ ] Webhook integration

---

## Implementation Order

```
Phase 1 - COMPLETED:
├── 1.1 Session Persistence ✅
├── 1.2 Performance Caching ✅
└── 2.1 Multi-Symbol Batch ✅

Phase 2 - COMPLETED:
├── 2.2 Walk-Forward Optimization ✅
├── 2.3 Monte Carlo Simulation ✅
└── 2.4 Parameter Sensitivity ✅ (integrated in Optimizer)

Phase 3 - PENDING:
├── 3.1 Database Backend
└── Polish & Testing
```

---

## Implemented Features Summary

### Session Management (sidebar)
- **Export Session** - Save current state to `.pkl` file
- **Load Session** - Restore previous work from file
- Works from initial upload screen too

### Performance Caching
- Data fetching cached for 1 hour
- Chart creation cached for 5 minutes
- Backtest results cached by config hash

### Multi-Symbol Batch Backtesting (new tab)
- Test strategy across multiple symbols
- Quick presets: US Indices, Tech Giants, Forex, Crypto, Commodities
- Summary metrics and CSV export

### Advanced Analysis (new tab)
- **Monte Carlo Simulation**
  - P&L distribution with confidence intervals
  - Probability of positive returns
  - Risk of ruin calculation
  - Max drawdown distribution

- **Walk-Forward Optimization**
  - Train/test window splitting
  - Out-of-sample validation
  - Robustness score (0-100)
  - Consistency tracking

---

## New Files Created

- `src/monte_carlo.py` - Monte Carlo simulation engine
- `src/walk_forward.py` - Walk-forward optimization logic

## App Running

Access at: **http://localhost:8503**
