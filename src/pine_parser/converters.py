"""
Pine Script Converters - Convert parsed data to StrategyConfig.

This module handles the conversion of extracted Pine Script data
to our StrategyConfig dataclass, including value mappings.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple
from .patterns import INPUT_PATTERNS, RISK_TYPE_MAP
from .exceptions import ValidationError


@dataclass
class ParsedStrategy:
    """
    Result of parsing a Pine Script file.

    Contains all extracted values plus metadata about the parsing process.
    """
    # ===== Metadata =====
    name: str = "Untitled Strategy"
    version: int = 0

    # ===== Configuracion Basica =====
    entry_pips: Optional[int] = None
    sl_pips: Optional[int] = None

    # ===== Sentido =====
    trade_longs: Optional[bool] = None
    trade_shorts: Optional[bool] = None

    # ===== Entradas (Patterns) =====
    use_sacudida: Optional[bool] = None
    use_engulfing: Optional[bool] = None
    use_climatic_volume: Optional[bool] = None

    # ===== Salidas (Exits) =====
    use_sl: Optional[bool] = None
    use_tp_ratio: Optional[bool] = None
    tp_ratio: Optional[float] = None
    use_n_bars_exit: Optional[bool] = None
    n_bars_exit: Optional[int] = None

    # ===== Filtros =====
    ma_filter: Optional[str] = None

    # ===== Sesiones =====
    use_london: Optional[bool] = None
    use_newyork: Optional[bool] = None
    use_tokyo: Optional[bool] = None

    # ===== Dias =====
    trade_monday: Optional[bool] = None
    trade_tuesday: Optional[bool] = None
    trade_wednesday: Optional[bool] = None
    trade_thursday: Optional[bool] = None
    trade_friday: Optional[bool] = None
    trade_saturday: Optional[bool] = None
    trade_sunday: Optional[bool] = None

    # ===== Gestion de Riesgo =====
    risk_type: Optional[str] = None
    fixed_size: Optional[float] = None
    fixed_risk_money: Optional[float] = None
    risk_percent: Optional[float] = None
    qty_step: Optional[float] = None
    qty_min: Optional[float] = None

    # ===== Backtest Settings =====
    initial_capital: Optional[float] = None
    commission: Optional[float] = None
    slippage: Optional[int] = None

    # ===== Parsing Metadata =====
    warnings: List[str] = field(default_factory=list)
    unsupported_features: List[str] = field(default_factory=list)
    fields_extracted: List[str] = field(default_factory=list)
    fields_missing: List[str] = field(default_factory=list)

    # ===== Pattern/Filter Detection =====
    has_sacudida: bool = False
    has_engulfing: bool = False
    has_climatic_volume: bool = False
    has_ma_filter: bool = False
    has_session_filter: bool = False
    has_day_filter: bool = False

    @property
    def confidence_score(self) -> float:
        """
        Calculate confidence score based on extraction success.

        Returns:
            Float between 0 and 1 indicating parsing confidence
        """
        total_fields = len(INPUT_PATTERNS)
        extracted = len(self.fields_extracted)

        if total_fields == 0:
            return 0.0

        return min(extracted / total_fields, 1.0)

    @property
    def total_fields(self) -> int:
        """Total number of fields we tried to extract."""
        return len(INPUT_PATTERNS)

    @property
    def extracted_count(self) -> int:
        """Number of fields successfully extracted."""
        return len(self.fields_extracted)


def create_parsed_strategy(
    inputs: Dict[str, Any],
    patterns: Dict[str, bool],
    filters: Dict[str, bool],
    metadata: Dict[str, Any]
) -> ParsedStrategy:
    """
    Create a ParsedStrategy from extraction results.

    Args:
        inputs: Dictionary of extracted input values
        patterns: Dictionary of detected patterns
        filters: Dictionary of detected filters
        metadata: Strategy metadata (name, version, etc.)

    Returns:
        ParsedStrategy instance
    """
    parsed = ParsedStrategy()

    # Apply metadata
    parsed.version = metadata.get('version', 0)
    strategy_params = metadata.get('strategy_params', {})
    parsed.name = strategy_params.get('name', 'Untitled Strategy')
    parsed.initial_capital = strategy_params.get('initial_capital')
    parsed.commission = strategy_params.get('commission')
    parsed.slippage = strategy_params.get('slippage')

    # Apply detected patterns/filters
    parsed.has_sacudida = patterns.get('has_sacudida', False)
    parsed.has_engulfing = patterns.get('has_engulfing', False)
    parsed.has_climatic_volume = patterns.get('has_climatic_volume', False)
    parsed.has_ma_filter = filters.get('has_ma_filter', False)
    parsed.has_session_filter = filters.get('has_session_filter', False)
    parsed.has_day_filter = filters.get('has_day_filter', False)

    # Track which fields were extracted
    fields_extracted = []
    fields_missing = []

    # Apply extracted inputs
    for pattern_key, pattern_def in INPUT_PATTERNS.items():
        config_field = pattern_def['config_field']

        if config_field in inputs:
            value = inputs[config_field]

            # Handle risk_type mapping
            if config_field == 'risk_type' and value in RISK_TYPE_MAP:
                value = RISK_TYPE_MAP[value]

            setattr(parsed, config_field, value)
            fields_extracted.append(config_field)
        else:
            fields_missing.append(config_field)

    parsed.fields_extracted = fields_extracted
    parsed.fields_missing = fields_missing

    # Add warnings
    if parsed.version < 5:
        parsed.warnings.append(f"Pine Script v{parsed.version} detected - may have compatibility issues")

    if not parsed.has_sacudida and not parsed.has_engulfing and not parsed.has_climatic_volume:
        parsed.warnings.append("No standard entry patterns detected - manual configuration may be needed")

    return parsed


def parsed_to_config(
    parsed: ParsedStrategy,
    base_config: Optional[Any] = None
) -> Tuple[Any, List[str]]:
    """
    Convert ParsedStrategy to StrategyConfig.

    Missing fields are filled from base_config or defaults.

    Args:
        parsed: ParsedStrategy instance
        base_config: Optional StrategyConfig to use for missing values

    Returns:
        Tuple of (StrategyConfig, list of warnings/notes)

    Raises:
        ValidationError: If parsed strategy is too incomplete
    """
    # Import here to avoid circular imports
    from ..strategy import StrategyConfig

    # Check confidence threshold
    if parsed.confidence_score < 0.3:
        raise ValidationError(
            f"Low confidence ({parsed.confidence_score:.0%}). "
            f"Only {parsed.extracted_count}/{parsed.total_fields} fields extracted. "
            "This file may not be compatible with the parser."
        )

    # Start with base config or defaults
    if base_config:
        config = StrategyConfig(**base_config.__dict__)
    else:
        config = StrategyConfig()

    notes = []

    # Apply all extracted values
    field_mapping = [
        ('entry_pips', 'entry_pips'),
        ('sl_pips', 'sl_pips'),
        ('trade_longs', 'trade_longs'),
        ('trade_shorts', 'trade_shorts'),
        ('use_sacudida', 'use_sacudida'),
        ('use_engulfing', 'use_engulfing'),
        ('use_climatic_volume', 'use_climatic_volume'),
        ('use_sl', 'use_sl'),
        ('use_tp_ratio', 'use_tp_ratio'),
        ('tp_ratio', 'tp_ratio'),
        ('use_n_bars_exit', 'use_n_bars_exit'),
        ('n_bars_exit', 'n_bars_exit'),
        ('ma_filter', 'ma_filter'),
        ('use_london', 'use_london'),
        ('use_newyork', 'use_newyork'),
        ('use_tokyo', 'use_tokyo'),
        ('trade_monday', 'trade_monday'),
        ('trade_tuesday', 'trade_tuesday'),
        ('trade_wednesday', 'trade_wednesday'),
        ('trade_thursday', 'trade_thursday'),
        ('trade_friday', 'trade_friday'),
        ('trade_saturday', 'trade_saturday'),
        ('trade_sunday', 'trade_sunday'),
        ('risk_type', 'risk_type'),
        ('fixed_size', 'fixed_size'),
        ('fixed_risk_money', 'fixed_risk_money'),
        ('risk_percent', 'risk_percent'),
        ('qty_step', 'qty_step'),
        ('qty_min', 'qty_min'),
        ('initial_capital', 'initial_capital'),
        ('commission', 'commission'),
        ('slippage', 'slippage'),
    ]

    for parsed_field, config_field in field_mapping:
        parsed_value = getattr(parsed, parsed_field, None)
        if parsed_value is not None:
            setattr(config, config_field, parsed_value)
        else:
            notes.append(f"Using default for {config_field}")

    # Add any warnings from parsing
    notes.extend(parsed.warnings)

    return config, notes


def get_extraction_summary(parsed: ParsedStrategy) -> Dict[str, Any]:
    """
    Generate a summary of what was extracted for display.

    Args:
        parsed: ParsedStrategy instance

    Returns:
        Dictionary with extraction summary
    """
    summary = {
        'name': parsed.name,
        'version': parsed.version,
        'confidence': f"{parsed.confidence_score:.0%}",
        'fields_extracted': parsed.extracted_count,
        'fields_total': parsed.total_fields,
        'patterns_detected': {
            'Sacudida': parsed.has_sacudida,
            'Engulfing': parsed.has_engulfing,
            'Climatic Volume': parsed.has_climatic_volume,
        },
        'filters_detected': {
            'MA 50/200': parsed.has_ma_filter,
            'Sessions': parsed.has_session_filter,
            'Days': parsed.has_day_filter,
        },
        'warnings': parsed.warnings,
        'unsupported': parsed.unsupported_features,
    }

    # Group extracted values by category
    categories = {
        'Basic Config': ['entry_pips', 'sl_pips'],
        'Direction': ['trade_longs', 'trade_shorts'],
        'Entry Patterns': ['use_sacudida', 'use_engulfing', 'use_climatic_volume'],
        'Exits': ['use_sl', 'use_tp_ratio', 'tp_ratio', 'use_n_bars_exit', 'n_bars_exit'],
        'Filters': ['ma_filter'],
        'Sessions': ['use_london', 'use_newyork', 'use_tokyo'],
        'Days': ['trade_monday', 'trade_tuesday', 'trade_wednesday',
                 'trade_thursday', 'trade_friday', 'trade_saturday', 'trade_sunday'],
        'Risk Management': ['risk_type', 'fixed_size', 'fixed_risk_money',
                           'risk_percent', 'qty_step', 'qty_min'],
        'Backtest': ['initial_capital', 'commission', 'slippage'],
    }

    extracted_by_category = {}
    for category, fields in categories.items():
        cat_values = {}
        for fld in fields:
            val = getattr(parsed, fld, None)
            if val is not None:
                cat_values[fld] = val
        if cat_values:
            extracted_by_category[category] = cat_values

    summary['extracted_values'] = extracted_by_category

    return summary
