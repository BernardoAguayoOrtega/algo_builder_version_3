"""
Custom exceptions for Pine Script parser.
"""


class PineParserError(Exception):
    """Base exception for Pine parser errors."""
    pass


class ValidationError(PineParserError):
    """Raised when parsed strategy fails validation."""
    pass


class TokenizationError(PineParserError):
    """Raised when tokenization fails."""
    pass


class ExtractionError(PineParserError):
    """Raised when extraction of a specific field fails."""
    pass
