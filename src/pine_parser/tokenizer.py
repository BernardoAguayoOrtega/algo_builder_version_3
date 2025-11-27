"""
Pine Script Tokenizer - Pre-process Pine Script for reliable parsing.

This module cleans Pine Script content to make regex-based extraction reliable:
1. Remove single-line comments (// ...)
2. Remove multi-line comments (/* ... */)
3. Normalize multi-line statements to single lines
4. Preserve string literals
"""

import re
from typing import Tuple, List
from .exceptions import TokenizationError


def remove_single_line_comments(content: str) -> str:
    """
    Remove single-line comments (// ...) while preserving strings.

    Handles:
        // This is a comment
        x = 1 // inline comment
        s = "string with // inside"  // actual comment
    """
    result = []
    i = 0
    in_string = False
    string_char = None

    while i < len(content):
        char = content[i]

        # Handle string start/end
        if char in ('"', "'") and (i == 0 or content[i-1] != '\\'):
            if not in_string:
                in_string = True
                string_char = char
            elif char == string_char:
                in_string = False
                string_char = None
            result.append(char)
            i += 1
            continue

        # Check for // comment (only outside strings)
        if not in_string and char == '/' and i + 1 < len(content) and content[i + 1] == '/':
            # Skip until end of line
            while i < len(content) and content[i] != '\n':
                i += 1
            continue

        result.append(char)
        i += 1

    return ''.join(result)


def remove_multiline_comments(content: str) -> str:
    """
    Remove multi-line comments (/* ... */).

    Handles nested comments and preserves strings.
    """
    result = []
    i = 0
    in_string = False
    string_char = None

    while i < len(content):
        char = content[i]

        # Handle string start/end
        if char in ('"', "'") and (i == 0 or content[i-1] != '\\'):
            if not in_string:
                in_string = True
                string_char = char
            elif char == string_char:
                in_string = False
                string_char = None
            result.append(char)
            i += 1
            continue

        # Check for /* comment start (only outside strings)
        if not in_string and char == '/' and i + 1 < len(content) and content[i + 1] == '*':
            # Skip until */
            i += 2
            while i < len(content) - 1:
                if content[i] == '*' and content[i + 1] == '/':
                    i += 2
                    break
                i += 1
            continue

        result.append(char)
        i += 1

    return ''.join(result)


def normalize_multiline_statements(content: str) -> str:
    """
    Normalize multi-line statements to single lines for easier parsing.

    Handles:
        entry_pip = input.int(
            1,
            'Title',
            minval = 1
        )

    Becomes:
        entry_pip = input.int(1, 'Title', minval = 1)
    """
    lines = content.split('\n')
    result = []
    buffer = []
    paren_depth = 0
    bracket_depth = 0
    in_string = False
    string_char = None

    for line in lines:
        stripped = line.strip()

        # Skip empty lines if not in a continuation
        if not stripped and not buffer:
            result.append('')
            continue

        # Count parens/brackets to detect multi-line statements
        for char in line:
            # Track strings
            if char in ('"', "'"):
                if not in_string:
                    in_string = True
                    string_char = char
                elif char == string_char:
                    in_string = False
                    string_char = None
                continue

            if not in_string:
                if char == '(':
                    paren_depth += 1
                elif char == ')':
                    paren_depth -= 1
                elif char == '[':
                    bracket_depth += 1
                elif char == ']':
                    bracket_depth -= 1

        buffer.append(stripped)

        # If balanced, flush buffer
        if paren_depth <= 0 and bracket_depth <= 0:
            combined = ' '.join(buffer)
            # Clean up extra whitespace
            combined = re.sub(r'\s+', ' ', combined)
            result.append(combined)
            buffer = []
            paren_depth = 0
            bracket_depth = 0

    # Handle any remaining buffer
    if buffer:
        combined = ' '.join(buffer)
        combined = re.sub(r'\s+', ' ', combined)
        result.append(combined)

    return '\n'.join(result)


def extract_version(content: str) -> int:
    """
    Extract Pine Script version from //@version=N directive.

    Returns:
        Version number (e.g., 6) or 0 if not found
    """
    match = re.search(r'//@version=(\d+)', content)
    if match:
        return int(match.group(1))
    return 0


def extract_strategy_params(content: str) -> dict:
    """
    Extract parameters from strategy() call.

    Example:
        strategy('Name', overlay=true, initial_capital=100000, commission_value=1.5)

    Returns:
        Dictionary of extracted parameters
    """
    params = {}

    # Find strategy() call
    match = re.search(r"strategy\s*\([^)]+\)", content, re.DOTALL)
    if not match:
        return params

    strategy_call = match.group(0)

    # Extract initial_capital
    cap_match = re.search(r'initial_capital\s*=\s*([\d.]+)', strategy_call)
    if cap_match:
        params['initial_capital'] = float(cap_match.group(1))

    # Extract commission_value
    comm_match = re.search(r'commission_value\s*=\s*([\d.]+)', strategy_call)
    if comm_match:
        params['commission'] = float(comm_match.group(1))

    # Extract slippage
    slip_match = re.search(r'slippage\s*=\s*(\d+)', strategy_call)
    if slip_match:
        params['slippage'] = int(slip_match.group(1))

    # Extract strategy name
    name_match = re.search(r"strategy\s*\(\s*['\"]([^'\"]+)['\"]", strategy_call)
    if name_match:
        params['name'] = name_match.group(1)

    return params


def tokenize(content: str) -> Tuple[str, dict]:
    """
    Main tokenization function - clean Pine Script for parsing.

    Args:
        content: Raw Pine Script content

    Returns:
        Tuple of (cleaned_content, metadata)

    Metadata includes:
        - version: Pine Script version
        - strategy_params: Parameters from strategy() call
    """
    try:
        # Extract metadata before cleaning
        version = extract_version(content)
        strategy_params = extract_strategy_params(content)

        # Clean content
        cleaned = remove_multiline_comments(content)
        cleaned = remove_single_line_comments(cleaned)
        cleaned = normalize_multiline_statements(cleaned)

        # Remove empty lines
        lines = [line for line in cleaned.split('\n') if line.strip()]
        cleaned = '\n'.join(lines)

        metadata = {
            'version': version,
            'strategy_params': strategy_params,
        }

        return cleaned, metadata

    except Exception as e:
        raise TokenizationError(f"Failed to tokenize Pine Script: {e}")


def get_input_lines(content: str) -> List[str]:
    """
    Extract all lines containing input.* declarations.

    Args:
        content: Tokenized Pine Script content

    Returns:
        List of lines containing input declarations
    """
    lines = []
    for line in content.split('\n'):
        if re.search(r'input\.(bool|int|float|string)\s*\(', line):
            lines.append(line)
    return lines
