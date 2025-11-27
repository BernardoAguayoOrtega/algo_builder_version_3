"""
Pine Script Parser - Extract strategy configuration from Pine Script files.

This module parses Pine Script (.txt) files and extracts strategy parameters
to auto-configure the backtesting system.

Two parsing modes:
1. Pattern-based (original): Maps known Spanish inputs to config fields
2. Universal: Extracts ALL inputs dynamically without mappings

Usage:
    # Pattern-based (for known script format)
    from src.pine_parser import parse_pine_script, parsed_to_config

    # Universal (for any Pine Script)
    from src.pine_parser import parse_universal, get_universal_summary
"""

from .parser import parse_pine_script
from .converters import parsed_to_config, ParsedStrategy
from .exceptions import PineParserError, ValidationError
from .universal import (
    parse_universal,
    get_universal_summary,
    UniversalParsedStrategy,
    ExtractedInput,
    InputType,
)

__all__ = [
    # Pattern-based parser
    "parse_pine_script",
    "parsed_to_config",
    "ParsedStrategy",
    # Universal parser
    "parse_universal",
    "get_universal_summary",
    "UniversalParsedStrategy",
    "ExtractedInput",
    "InputType",
    # Exceptions
    "PineParserError",
    "ValidationError",
]
