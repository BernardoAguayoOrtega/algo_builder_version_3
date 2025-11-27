# Pine Script Parser Feature - Implementation Plan

**Status:** ✅ APPROVED by Senior Architect (with modifications)

## Overview

Allow users to upload a `.txt` file containing Pine Script code, parse it, and automatically configure the backtesting system with the extracted strategy parameters.

## Goal

**NOT** to execute Pine Script directly, but to:
1. Parse the Pine Script file
2. Extract known patterns, filters, and parameters
3. Map them to our existing `StrategyConfig` dataclass
4. Auto-fill the UI configuration
5. Run backtest with our proven Python engine

## Why This Approach?

| Full Pine Runtime | Our Approach: Config Extractor |
|-------------------|-------------------------------|
| Must handle ALL Pine features | Only handle known patterns |
| Real-time interpretation | One-time extraction |
| User expects perfection | User can review/adjust |
| 10/10 difficulty | 5-6/10 difficulty |
| Ongoing maintenance nightmare | Extensible pattern library |

## Architecture

```
┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│  Pine Script     │     │  Pattern         │     │  StrategyConfig  │
│  .txt file       │ ──► │  Extractor       │ ──► │  (Python)        │
│                  │     │  (regex-based)   │     │                  │
└──────────────────┘     └──────────────────┘     └──────────────────┘
                                                          │
                                                          ▼
                                                  ┌──────────────────┐
                                                  │  Existing        │
                                                  │  Backtester      │
                                                  └──────────────────┘
```

## What We Can Extract

### From Reference: `Algo Strategy Builder.txt`

#### 1. Entry Patterns
```pine
// Sacudida (Shakeout)
sacudidaLong = close > open and close > high[1] and low < low[1] and low < low[2]
sacudidaShort = close < open and close < low[1] and high > high[1] and high > high[2]

// Envolvente (Engulfing)
envolventeLong = close > open and close > high[1] and open <= low[1]
envolventeShort = close < open and close < low[1] and open >= high[1]

// Volumen Climático (Climatic Volume)
volumenClimaticoLong = close > open and volume > volume[1] * 1.5 and low == ta.lowest(low, 10)
volumenClimaticoShort = close < open and volume > volume[1] * 1.5 and high == ta.highest(high, 10)
```

**Extraction:** Regex to detect pattern names → map to `use_sacudida`, `use_engulfing`, `use_climatic_volume`

#### 2. Direction
```pine
usarLargos = input.bool(true, "Operar Largos")
usarCortos = input.bool(true, "Operar Cortos")
```

**Extraction:** `input.bool` with "Largos"/"Cortos" → `trade_longs`, `trade_shorts`

#### 3. Entry/Exit Parameters
```pine
pipsEntrada = input.int(1, "Pips entrada")
pipsSL = input.int(1, "Pips SL")
tamañoPip = input.float(0.01, "Tamaño pip")
ratioTP = input.float(1.0, "Ratio TP")
useNBarsExit = input.bool(false, "Salida N velas")
nBarsExit = input.int(5, "N velas")
```

**Extraction:** `input.int`/`input.float`/`input.bool` → corresponding config fields

#### 4. MA Filter
```pine
filtroMA = input.string("Sin filtro", "Filtro MA", options=["Sin filtro", "Alcista", "Bajista"])
ma50 = ta.sma(close, 50)
ma200 = ta.sma(close, 200)
```

**Extraction:** MA filter option → `ma_filter`

#### 5. Session Filter
```pine
sesionLondres = input.bool(true, "Londres")
sesionNY = input.bool(true, "Nueva York")
sesionTokyo = input.bool(true, "Tokyo")

time_london = time(timeframe.period, "0100-0815")
time_ny = time(timeframe.period, "0815-1545")
time_tokyo = time(timeframe.period, "1545-0100")
```

**Extraction:** Session booleans → `use_london`, `use_newyork`, `use_tokyo`

#### 6. Day Filter
```pine
tradeLunes = input.bool(true, "Lunes")
tradeMartes = input.bool(true, "Martes")
// ... etc
```

**Extraction:** Day booleans → `trade_monday`, `trade_tuesday`, etc.

#### 7. Risk Management
```pine
tipoTamaño = input.string("Tamaño fijo", options=["Tamaño fijo", "Riesgo fijo $", "% Equity"])
tamaño = input.float(1.0, "Tamaño")
capitalInicial = input.float(100000, "Capital inicial")
comision = input.float(1.5, "Comisión")
```

**Extraction:** Position sizing params → `position_type`, `position_size`, `initial_capital`, `commission`

## What We Cannot Extract (Limitations)

| Feature | Reason | Workaround |
|---------|--------|------------|
| `request.security()` | Multi-timeframe data | Not supported, warn user |
| Custom indicators | Requires full Pine interpreter | Add to pattern library manually |
| Complex loops | State management | Not supported |
| Drawing objects | Not relevant to backtesting | Ignore |
| Arrays/matrices | Complex data structures | Not supported |

## Implementation Plan

### Phase 1: Core Parser (Week 1)

#### File: `src/pine_parser.py`

```python
@dataclass
class ParsedStrategy:
    """Result of parsing a Pine Script file."""
    # What was successfully extracted
    name: str
    version: int

    # Entry patterns
    use_sacudida: Optional[bool] = None
    use_engulfing: Optional[bool] = None
    use_climatic_volume: Optional[bool] = None

    # Direction
    trade_longs: Optional[bool] = None
    trade_shorts: Optional[bool] = None

    # Parameters
    entry_pips: Optional[int] = None
    sl_pips: Optional[int] = None
    pip_size: Optional[float] = None
    tp_ratio: Optional[float] = None
    n_bars_exit: Optional[int] = None
    use_n_bars_exit: Optional[bool] = None

    # Filters
    ma_filter: Optional[str] = None
    use_london: Optional[bool] = None
    use_newyork: Optional[bool] = None
    use_tokyo: Optional[bool] = None
    trade_monday: Optional[bool] = None
    # ... etc

    # Risk
    position_type: Optional[str] = None
    position_size: Optional[float] = None
    initial_capital: Optional[float] = None
    commission: Optional[float] = None

    # Parsing metadata
    warnings: List[str] = field(default_factory=list)
    unsupported_features: List[str] = field(default_factory=list)
    confidence_score: float = 0.0  # 0-1, how much was successfully parsed


def parse_pine_script(content: str) -> ParsedStrategy:
    """Parse Pine Script content and extract strategy configuration."""
    pass


def parsed_to_config(parsed: ParsedStrategy, base_config: StrategyConfig) -> StrategyConfig:
    """Convert parsed strategy to StrategyConfig, filling gaps with base_config."""
    pass
```

#### Parsing Approach

1. **Input detection** - Regex for `input.bool`, `input.int`, `input.float`, `input.string`
2. **Pattern detection** - Look for known pattern names (sacudida, envolvente, etc.)
3. **Filter detection** - Look for MA, session, day filter patterns
4. **Parameter extraction** - Extract default values from inputs

### Phase 2: UI Integration (Week 2)

#### New Tab or Section in Optimizer

```
┌─────────────────────────────────────────────────────────┐
│  📄 Import Pine Script                                  │
│                                                         │
│  [Choose File] or drag & drop .txt file                 │
│                                                         │
│  ─────────────────────────────────────────────────────  │
│                                                         │
│  🔍 Parse Results:                                      │
│                                                         │
│  ✅ Detected Patterns:                                  │
│     • Sacudida (Long & Short)                          │
│     • Envolvente (Long & Short)                        │
│     • Volumen Climático (Long & Short)                 │
│                                                         │
│  ✅ Detected Filters:                                   │
│     • MA 50/200 filter                                 │
│     • Session filter (London, NY, Tokyo)               │
│     • Day of week filter                               │
│                                                         │
│  ✅ Detected Parameters:                                │
│     • Entry pips: 1                                    │
│     • SL pips: 1                                       │
│     • TP ratio: 1.0                                    │
│     • Position size: Fixed 1.0                         │
│                                                         │
│  ⚠️  Warnings:                                          │
│     • N-bars exit detected but disabled by default     │
│                                                         │
│  Confidence Score: 95%                                  │
│                                                         │
│  [Apply to Configuration]  [Show Raw Extraction]        │
└─────────────────────────────────────────────────────────┘
```

### Phase 3: Pattern Library Extension (Ongoing)

Create extensible pattern definitions:

```python
# src/pine_patterns.py

ENTRY_PATTERNS = {
    "sacudida": {
        "pine_regex": r"sacudida\w*\s*=.*close\s*[><].*open.*high\[1\].*low\[1\]",
        "config_field": "use_sacudida",
        "description": "Shakeout pattern",
    },
    "engulfing": {
        "pine_regex": r"envolvente\w*\s*=.*close\s*[><].*open.*high\[1\]",
        "config_field": "use_engulfing",
        "description": "Engulfing pattern",
    },
    # Easy to add more patterns here
}

INDICATOR_PATTERNS = {
    "ma_crossover": {
        "pine_regex": r"ta\.sma\(close,\s*50\).*ta\.sma\(close,\s*200\)",
        "config_field": "ma_filter",
        "description": "MA 50/200 crossover filter",
    },
    # RSI, MACD, etc. can be added later
}
```

## Success Criteria

1. **Parse reference file with 90%+ accuracy**
   - All inputs correctly extracted
   - All known patterns detected
   - Warnings for unsupported features

2. **Generate valid StrategyConfig**
   - Backtest runs without errors
   - Results match manual configuration

3. **Clear user feedback**
   - Show what was detected
   - Show what couldn't be parsed
   - Allow manual override

## Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| Pine Script syntax varies widely | Focus on common patterns, show warnings for unknown |
| Users expect 100% compatibility | Clear messaging: "extraction tool, not Pine interpreter" |
| Maintenance burden | Pattern library is data-driven, easy to extend |
| Edge cases break parser | Graceful degradation, always allow manual config |

## Timeline

| Phase | Duration | Deliverable |
|-------|----------|-------------|
| Phase 1: Core Parser | 1 week | `src/pine_parser.py` with tests |
| Phase 2: UI Integration | 1 week | Upload UI, parse results display |
| Phase 3: Polish | 3-4 days | Error handling, edge cases |
| **Total** | **~2.5 weeks** | Working feature |

## Future Extensions

Once core parser works:

1. **More entry patterns** - RSI oversold, MACD crossover, Bollinger breakout
2. **More indicators** - ATR-based stops, volume filters
3. **Strategy templates** - Common strategies pre-defined
4. **Export to Pine** - Generate Pine Script from StrategyConfig (reverse direction)

---

## Senior Architect Review Notes

### Required Modifications (Must Do)

1. **Add Tokenizer Pre-processing**
   - Strip comments (`//` and `/* */`)
   - Normalize whitespace (multi-line to single line)
   - Preserve string literals
   - This prevents 80% of regex edge cases

2. **Field Mapping Layer**
   ```python
   RISK_TYPE_MAP = {
       "Tamaño fijo": "fixed_size",
       "Riesgo monetario fijo": "fixed_risk_money",
       "Riesgo % equity": "risk_percent",
   }
   ```

3. **Comprehensive Test Suite**
   - Test reference file (90%+ accuracy)
   - Test edge cases (comments, multi-line, special chars)
   - Test error recovery (incomplete/invalid files)

4. **Error Recovery & Confidence Scoring**
   - Calculate confidence based on fields extracted vs expected
   - Graceful degradation when fields missing
   - Clear error messages

5. **Split Parser into Classes**
   ```
   src/pine_parser/
   ├── tokenizer.py      # Clean input
   ├── extractors.py     # InputExtractor, PatternDetector, FilterDetector
   ├── converters.py     # ParsedStrategy → StrategyConfig
   └── parser.py         # Orchestrator
   ```

### Revised Architecture

```
src/pine_parser/
├── __init__.py
├── parser.py              # Main parse_pine_script() function
├── tokenizer.py           # Clean Pine Script (comments, whitespace)
├── extractors.py          # InputExtractor, PatternDetector, FilterDetector
├── converters.py          # parsed_to_config(), mapping logic
├── patterns.py            # Pattern definitions (INPUT_PATTERNS dict)
└── exceptions.py          # ParsingError, ValidationError

tests/pine_parser/
├── test_tokenizer.py
├── test_extractors.py
├── test_parser.py
├── test_converters.py
└── fixtures/
    ├── reference.pine     # Reference file
    ├── edge_cases.pine    # Tricky syntax
    └── invalid.pine       # Should fail gracefully
```

### Edge Cases to Handle

1. Comments: `// commented input`
2. Multi-line inputs: `input.int(\n  1,\n  "title"\n)`
3. String literals with parens: `"Pips (1-10)"`
4. Disabled blocks: `/* ... */`
5. Renamed variables: `use_sacudida_pattern` vs `usar_patron_sacudida`
6. Missing inputs: fallback to defaults

### Success Probability

- Current plan: **70%** (works but many edge case bugs)
- With modifications: **90%** (robust, extensible, professional)
