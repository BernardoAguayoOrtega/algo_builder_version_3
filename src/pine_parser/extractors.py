"""
Pine Script Extractors - Extract specific data from tokenized Pine Script.

This module contains classes that extract different types of data:
- InputExtractor: Extract input.bool/int/float/string declarations
- PatternDetector: Detect which trading patterns are used
- FilterDetector: Detect which filters are enabled
"""

import re
from typing import Dict, Any, List, Optional, Tuple
from .patterns import INPUT_PATTERNS, get_pattern_by_pine_name, get_pattern_by_label


class InputExtractor:
    """
    Extract input.* declarations from Pine Script.

    Handles:
        entry_pip = input.int(1, 'Pips para entrada', minval=1, group='Config')
        operar_largos = input.bool(true, 'Largos', group=grpStrat)
        target_ratio = input.float(1.0, 'Ratio TP', minval=0.1, step=0.1)
        filtro_mm = input.string(defval='Sin filtro', title='MA Filter', options=[...])
    """

    # Regex patterns for different input types
    BOOL_PATTERN = re.compile(
        r"(\w+)\s*=\s*input\.bool\s*\(\s*(true|false)\s*,\s*['\"]([^'\"]+)['\"]",
        re.IGNORECASE
    )

    INT_PATTERN = re.compile(
        r"(\w+)\s*=\s*input\.int\s*\(\s*(\d+)\s*,\s*['\"]([^'\"]+)['\"]",
        re.IGNORECASE
    )

    FLOAT_PATTERN = re.compile(
        r"(\w+)\s*=\s*input\.float\s*\(\s*([\d.]+)\s*,\s*['\"]([^'\"]+)['\"]",
        re.IGNORECASE
    )

    STRING_PATTERN = re.compile(
        r"(\w+)\s*=\s*input\.string\s*\([^)]*(?:defval\s*=\s*['\"]([^'\"]+)['\"]|['\"]([^'\"]+)['\"])[^)]*title\s*=\s*['\"]([^'\"]+)['\"]",
        re.IGNORECASE
    )

    # Alternative string pattern (defval first)
    STRING_PATTERN_ALT = re.compile(
        r"(\w+)\s*=\s*input\.string\s*\(\s*defval\s*=\s*['\"]([^'\"]+)['\"].*?title\s*=\s*['\"]([^'\"]+)['\"]",
        re.IGNORECASE | re.DOTALL
    )

    def extract(self, content: str) -> Dict[str, Any]:
        """
        Extract all input declarations from Pine Script content.

        Args:
            content: Tokenized Pine Script content

        Returns:
            Dictionary mapping config field names to extracted values
        """
        extracted = {}
        lines = content.split('\n')

        for line in lines:
            # Try each input type
            result = self._extract_bool(line)
            if result:
                extracted.update(result)
                continue

            result = self._extract_int(line)
            if result:
                extracted.update(result)
                continue

            result = self._extract_float(line)
            if result:
                extracted.update(result)
                continue

            result = self._extract_string(line)
            if result:
                extracted.update(result)

        return extracted

    def _extract_bool(self, line: str) -> Optional[Dict[str, bool]]:
        """Extract input.bool declaration."""
        match = self.BOOL_PATTERN.search(line)
        if not match:
            return None

        var_name, value, label = match.groups()
        value = value.lower() == 'true'

        # Find which config field this maps to
        config_field = self._find_config_field(var_name, label, 'bool')
        if config_field:
            return {config_field: value}

        return None

    def _extract_int(self, line: str) -> Optional[Dict[str, int]]:
        """Extract input.int declaration."""
        match = self.INT_PATTERN.search(line)
        if not match:
            return None

        var_name, value, label = match.groups()
        value = int(value)

        config_field = self._find_config_field(var_name, label, 'int')
        if config_field:
            return {config_field: value}

        return None

    def _extract_float(self, line: str) -> Optional[Dict[str, float]]:
        """Extract input.float declaration."""
        match = self.FLOAT_PATTERN.search(line)
        if not match:
            return None

        var_name, value, label = match.groups()
        value = float(value)

        config_field = self._find_config_field(var_name, label, 'float')
        if config_field:
            return {config_field: value}

        return None

    def _extract_string(self, line: str) -> Optional[Dict[str, str]]:
        """Extract input.string declaration."""
        # Try main pattern
        match = self.STRING_PATTERN_ALT.search(line)
        if match:
            var_name = match.group(1)
            value = match.group(2)
            label = match.group(3)

            config_field = self._find_config_field(var_name, label, 'string')
            if config_field:
                return {config_field: value}

        return None

    def _find_config_field(self, var_name: str, label: str, input_type: str) -> Optional[str]:
        """
        Find the StrategyConfig field that matches this input.

        First tries to match by Pine variable name, then by label.
        """
        # Try matching by variable name
        pattern = get_pattern_by_pine_name(var_name)
        if pattern and pattern['input_type'] == input_type:
            return pattern['config_field']

        # Try matching by label
        pattern = get_pattern_by_label(label)
        if pattern and pattern['input_type'] == input_type:
            return pattern['config_field']

        return None


class PatternDetector:
    """
    Detect which trading patterns are defined in the Pine Script.

    Looks for pattern function definitions:
        - sacudida_long_condition()
        - sacudida_short_condition()
        - bullEngulf()
        - bearEngulf()
        - volClimatico / clim_long_raw / clim_short_raw
    """

    # Pattern detection regexes
    SACUDIDA_PATTERN = re.compile(
        r'sacudida.*condition|sacudidaLong|sacudidaShort',
        re.IGNORECASE
    )

    ENGULFING_PATTERN = re.compile(
        r'bullEngulf|bearEngulf|envolvente',
        re.IGNORECASE
    )

    CLIMATIC_PATTERN = re.compile(
        r'volClimatico|clim_long|clim_short|vol_climatico',
        re.IGNORECASE
    )

    def detect(self, content: str) -> Dict[str, bool]:
        """
        Detect which patterns are present in the Pine Script.

        Args:
            content: Pine Script content

        Returns:
            Dictionary of detected patterns (presence, not enabled/disabled status)
        """
        return {
            'has_sacudida': bool(self.SACUDIDA_PATTERN.search(content)),
            'has_engulfing': bool(self.ENGULFING_PATTERN.search(content)),
            'has_climatic_volume': bool(self.CLIMATIC_PATTERN.search(content)),
        }


class FilterDetector:
    """
    Detect which filters are defined in the Pine Script.

    Looks for:
        - MA 50/200 filter (sma50, sma200, filtro_mm)
        - Session filters (time(), enHorario)
        - Day of week filters (dayofweek)
    """

    MA_FILTER_PATTERN = re.compile(
        r'ta\.sma\(close,\s*50\)|sma50|filtro.*mm.*200|ma50.*ma200',
        re.IGNORECASE
    )

    SESSION_PATTERN = re.compile(
        r"time\(timeframe\.period,\s*['\"][\d-]+['\"]\)|enHorario|filtro.*horario",
        re.IGNORECASE
    )

    DAY_PATTERN = re.compile(
        r'dayofweek|operarHoy|operarLunes|operarMartes',
        re.IGNORECASE
    )

    def detect(self, content: str) -> Dict[str, bool]:
        """
        Detect which filters are present in the Pine Script.

        Args:
            content: Pine Script content

        Returns:
            Dictionary of detected filters
        """
        return {
            'has_ma_filter': bool(self.MA_FILTER_PATTERN.search(content)),
            'has_session_filter': bool(self.SESSION_PATTERN.search(content)),
            'has_day_filter': bool(self.DAY_PATTERN.search(content)),
        }


class SessionTimeExtractor:
    """
    Extract session time definitions from Pine Script.

    Looks for time() calls like:
        time(timeframe.period, '0100-0815')  // London
        time(timeframe.period, '0815-1545')  // New York
    """

    TIME_PATTERN = re.compile(
        r"time\(timeframe\.period,\s*['\"](\d{4}-\d{4})['\"]\)",
        re.IGNORECASE
    )

    def extract(self, content: str) -> List[str]:
        """Extract all session time strings."""
        matches = self.TIME_PATTERN.findall(content)
        return matches


def extract_all(content: str) -> Tuple[Dict[str, Any], Dict[str, bool], Dict[str, bool]]:
    """
    Convenience function to run all extractors.

    Args:
        content: Tokenized Pine Script content

    Returns:
        Tuple of (inputs, patterns, filters)
    """
    input_extractor = InputExtractor()
    pattern_detector = PatternDetector()
    filter_detector = FilterDetector()

    inputs = input_extractor.extract(content)
    patterns = pattern_detector.detect(content)
    filters = filter_detector.detect(content)

    return inputs, patterns, filters
