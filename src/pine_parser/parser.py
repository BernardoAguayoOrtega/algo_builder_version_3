"""
Pine Script Parser - Main orchestrator for parsing Pine Script files.

Usage:
    from src.pine_parser import parse_pine_script

    with open("strategy.pine") as f:
        content = f.read()

    parsed = parse_pine_script(content)
    print(f"Confidence: {parsed.confidence_score:.0%}")
    print(f"Extracted: {parsed.fields_extracted}")
"""

from typing import Optional
from .tokenizer import tokenize
from .extractors import InputExtractor, PatternDetector, FilterDetector
from .converters import ParsedStrategy, create_parsed_strategy, get_extraction_summary
from .exceptions import PineParserError


def parse_pine_script(content: str) -> ParsedStrategy:
    """
    Parse Pine Script content and extract strategy configuration.

    This is the main entry point for the parser. It:
    1. Tokenizes the content (removes comments, normalizes whitespace)
    2. Extracts input declarations (input.bool, input.int, etc.)
    3. Detects patterns (sacudida, engulfing, climatic volume)
    4. Detects filters (MA, sessions, days)
    5. Creates a ParsedStrategy with all extracted data

    Args:
        content: Raw Pine Script content as string

    Returns:
        ParsedStrategy with extracted configuration and metadata

    Raises:
        PineParserError: If parsing fails critically
    """
    try:
        # Step 1: Tokenize
        cleaned_content, metadata = tokenize(content)

        # Step 2: Extract inputs
        input_extractor = InputExtractor()
        inputs = input_extractor.extract(cleaned_content)

        # Step 3: Detect patterns
        pattern_detector = PatternDetector()
        patterns = pattern_detector.detect(content)  # Use original for pattern detection

        # Step 4: Detect filters
        filter_detector = FilterDetector()
        filters = filter_detector.detect(content)  # Use original for filter detection

        # Step 5: Create ParsedStrategy
        parsed = create_parsed_strategy(inputs, patterns, filters, metadata)

        return parsed

    except PineParserError:
        raise
    except Exception as e:
        raise PineParserError(f"Failed to parse Pine Script: {e}")


def parse_and_summarize(content: str) -> dict:
    """
    Parse Pine Script and return a summary suitable for display.

    Convenience function that combines parsing with summary generation.

    Args:
        content: Raw Pine Script content

    Returns:
        Dictionary with parsing results and summary
    """
    parsed = parse_pine_script(content)
    summary = get_extraction_summary(parsed)

    return {
        'parsed': parsed,
        'summary': summary,
    }


def validate_pine_script(content: str) -> dict:
    """
    Validate Pine Script content without full parsing.

    Quick check to see if the content looks like valid Pine Script.

    Args:
        content: Content to validate

    Returns:
        Dictionary with validation results
    """
    issues = []
    warnings = []

    # Check for version directive
    if '//@version=' not in content:
        issues.append("No version directive found (//@version=N)")

    # Check for strategy() call
    if 'strategy(' not in content:
        issues.append("No strategy() declaration found - may be an indicator, not a strategy")

    # Check for at least one input
    if 'input.' not in content:
        warnings.append("No input declarations found - configuration may not be extractable")

    # Check for known patterns
    has_patterns = any([
        'sacudida' in content.lower(),
        'engulf' in content.lower(),
        'volumen' in content.lower() and 'climatico' in content.lower(),
    ])

    if not has_patterns:
        warnings.append("No recognized entry patterns found - may need manual configuration")

    return {
        'valid': len(issues) == 0,
        'issues': issues,
        'warnings': warnings,
    }


# CLI interface for quick testing
if __name__ == "__main__":
    import sys
    import json

    if len(sys.argv) < 2:
        print("Usage: python -m src.pine_parser.parser <file.pine>")
        sys.exit(1)

    filepath = sys.argv[1]

    try:
        with open(filepath, 'r') as f:
            content = f.read()

        # Validate first
        validation = validate_pine_script(content)
        print("\n=== Validation ===")
        print(f"Valid: {validation['valid']}")
        if validation['issues']:
            print(f"Issues: {validation['issues']}")
        if validation['warnings']:
            print(f"Warnings: {validation['warnings']}")

        # Parse
        result = parse_and_summarize(content)
        parsed = result['parsed']
        summary = result['summary']

        print("\n=== Parse Results ===")
        print(f"Name: {summary['name']}")
        print(f"Version: {summary['version']}")
        print(f"Confidence: {summary['confidence']}")
        print(f"Fields extracted: {summary['fields_extracted']}/{summary['fields_total']}")

        print("\n=== Patterns Detected ===")
        for name, detected in summary['patterns_detected'].items():
            status = "✓" if detected else "✗"
            print(f"  {status} {name}")

        print("\n=== Filters Detected ===")
        for name, detected in summary['filters_detected'].items():
            status = "✓" if detected else "✗"
            print(f"  {status} {name}")

        print("\n=== Extracted Values ===")
        for category, values in summary['extracted_values'].items():
            print(f"\n{category}:")
            for field, value in values.items():
                print(f"  {field}: {value}")

        if summary['warnings']:
            print("\n=== Warnings ===")
            for warning in summary['warnings']:
                print(f"  ⚠ {warning}")

    except FileNotFoundError:
        print(f"Error: File not found: {filepath}")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
