"""
Universal Pine Script Parser - Extract all inputs and logic from ANY Pine Script.

Unlike the pattern-based parser, this module extracts everything dynamically
without requiring predefined mappings. Users can then map extracted inputs
to backtester parameters through the UI.
"""

import re
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple
from enum import Enum


class InputType(Enum):
    BOOL = "bool"
    INT = "int"
    FLOAT = "float"
    STRING = "string"
    SOURCE = "source"
    TIMEFRAME = "timeframe"
    COLOR = "color"


@dataclass
class ExtractedInput:
    """A single input extracted from Pine Script."""
    var_name: str
    input_type: InputType
    default_value: Any
    label: str
    group: Optional[str] = None
    options: Optional[List[str]] = None  # For string inputs with options
    min_val: Optional[float] = None
    max_val: Optional[float] = None
    step: Optional[float] = None
    tooltip: Optional[str] = None


@dataclass
class StrategyCall:
    """A strategy.entry() or strategy.exit() call."""
    call_type: str  # "entry" or "exit"
    id: str  # The trade ID
    direction: Optional[str] = None  # "long" or "short"
    condition_var: Optional[str] = None  # Variable name used in the when condition
    raw_line: str = ""


@dataclass
class FunctionDefinition:
    """A function definition in Pine Script."""
    name: str
    body: str
    returns_bool: bool = False


@dataclass
class UniversalParsedStrategy:
    """Result of universal Pine Script parsing."""
    # Metadata
    name: str = "Untitled Strategy"
    version: int = 0

    # Strategy settings from strategy() call
    initial_capital: Optional[float] = None
    commission: Optional[float] = None
    slippage: Optional[int] = None
    pyramiding: Optional[int] = None

    # All extracted inputs (key = var_name)
    inputs: Dict[str, ExtractedInput] = field(default_factory=dict)

    # Grouped inputs for UI display
    input_groups: Dict[str, List[str]] = field(default_factory=dict)

    # Entry/exit calls
    strategy_calls: List[StrategyCall] = field(default_factory=list)

    # Function definitions (potential entry conditions)
    functions: Dict[str, FunctionDefinition] = field(default_factory=dict)

    # Detected indicators
    indicators: List[str] = field(default_factory=list)

    # Raw content for reference
    raw_content: str = ""

    @property
    def total_inputs(self) -> int:
        return len(self.inputs)

    @property
    def entry_calls(self) -> List[StrategyCall]:
        return [c for c in self.strategy_calls if c.call_type == "entry"]

    @property
    def exit_calls(self) -> List[StrategyCall]:
        return [c for c in self.strategy_calls if c.call_type == "exit"]


class UniversalInputExtractor:
    """
    Extract ALL input declarations from Pine Script without predefined mappings.
    """

    # Generic patterns for all input types
    PATTERNS = {
        InputType.BOOL: re.compile(
            r"(\w+)\s*=\s*input\.bool\s*\(\s*(true|false)\s*,\s*['\"]([^'\"]+)['\"]"
            r"(?:[^)]*group\s*=\s*(?:(\w+)|['\"]([^'\"]+)['\"]))?",
            re.IGNORECASE
        ),
        InputType.INT: re.compile(
            r"(\w+)\s*=\s*input\.int\s*\(\s*(-?\d+)\s*,\s*['\"]([^'\"]+)['\"]"
            r"(?:[^)]*group\s*=\s*(?:(\w+)|['\"]([^'\"]+)['\"]))?",
            re.IGNORECASE
        ),
        InputType.FLOAT: re.compile(
            r"(\w+)\s*=\s*input\.float\s*\(\s*(-?[\d.]+)\s*,\s*['\"]([^'\"]+)['\"]"
            r"(?:[^)]*group\s*=\s*(?:(\w+)|['\"]([^'\"]+)['\"]))?",
            re.IGNORECASE
        ),
    }

    # String pattern is more complex due to options
    STRING_PATTERN = re.compile(
        r"(\w+)\s*=\s*input\.string\s*\("
        r"[^)]*defval\s*=\s*['\"]([^'\"]+)['\"]"
        r"[^)]*title\s*=\s*['\"]([^'\"]+)['\"]"
        r"(?:[^)]*group\s*=\s*(?:(\w+)|['\"]([^'\"]+)['\"]))?",
        re.IGNORECASE | re.DOTALL
    )

    # Extract options from string inputs
    OPTIONS_PATTERN = re.compile(
        r"options\s*=\s*\[([^\]]+)\]",
        re.IGNORECASE
    )

    # Extract minval, maxval, step
    MINVAL_PATTERN = re.compile(r"minval\s*=\s*(-?[\d.]+)", re.IGNORECASE)
    MAXVAL_PATTERN = re.compile(r"maxval\s*=\s*(-?[\d.]+)", re.IGNORECASE)
    STEP_PATTERN = re.compile(r"step\s*=\s*(-?[\d.]+)", re.IGNORECASE)
    TOOLTIP_PATTERN = re.compile(r"tooltip\s*=\s*['\"]([^'\"]+)['\"]", re.IGNORECASE)

    def extract(self, content: str) -> Tuple[Dict[str, ExtractedInput], Dict[str, List[str]]]:
        """
        Extract all inputs from Pine Script.

        Returns:
            Tuple of (inputs dict, groups dict)
        """
        inputs = {}
        groups = {}

        # Find all input lines
        for line in content.split('\n'):
            extracted = self._extract_from_line(line)
            if extracted:
                inputs[extracted.var_name] = extracted

                # Add to group
                group_name = extracted.group or "Ungrouped"
                if group_name not in groups:
                    groups[group_name] = []
                groups[group_name].append(extracted.var_name)

        return inputs, groups

    def _extract_from_line(self, line: str) -> Optional[ExtractedInput]:
        """Try to extract an input from a single line."""

        # Try bool
        match = self.PATTERNS[InputType.BOOL].search(line)
        if match:
            var_name = match.group(1)
            value = match.group(2).lower() == 'true'
            label = match.group(3)
            group = match.group(4) or match.group(5)
            return ExtractedInput(
                var_name=var_name,
                input_type=InputType.BOOL,
                default_value=value,
                label=label,
                group=group
            )

        # Try int
        match = self.PATTERNS[InputType.INT].search(line)
        if match:
            var_name = match.group(1)
            value = int(match.group(2))
            label = match.group(3)
            group = match.group(4) or match.group(5)

            # Extract constraints
            minval = self._extract_number(self.MINVAL_PATTERN, line)
            maxval = self._extract_number(self.MAXVAL_PATTERN, line)
            step = self._extract_number(self.STEP_PATTERN, line)
            tooltip = self._extract_string(self.TOOLTIP_PATTERN, line)

            return ExtractedInput(
                var_name=var_name,
                input_type=InputType.INT,
                default_value=value,
                label=label,
                group=group,
                min_val=minval,
                max_val=maxval,
                step=step,
                tooltip=tooltip
            )

        # Try float
        match = self.PATTERNS[InputType.FLOAT].search(line)
        if match:
            var_name = match.group(1)
            value = float(match.group(2))
            label = match.group(3)
            group = match.group(4) or match.group(5)

            minval = self._extract_number(self.MINVAL_PATTERN, line)
            maxval = self._extract_number(self.MAXVAL_PATTERN, line)
            step = self._extract_number(self.STEP_PATTERN, line)
            tooltip = self._extract_string(self.TOOLTIP_PATTERN, line)

            return ExtractedInput(
                var_name=var_name,
                input_type=InputType.FLOAT,
                default_value=value,
                label=label,
                group=group,
                min_val=minval,
                max_val=maxval,
                step=step,
                tooltip=tooltip
            )

        # Try string
        match = self.STRING_PATTERN.search(line)
        if match:
            var_name = match.group(1)
            value = match.group(2)
            label = match.group(3)
            group = match.group(4) or match.group(5)

            # Extract options
            options = None
            opt_match = self.OPTIONS_PATTERN.search(line)
            if opt_match:
                opts_str = opt_match.group(1)
                options = [o.strip().strip("'\"") for o in opts_str.split(',')]

            return ExtractedInput(
                var_name=var_name,
                input_type=InputType.STRING,
                default_value=value,
                label=label,
                group=group,
                options=options
            )

        return None

    def _extract_number(self, pattern: re.Pattern, line: str) -> Optional[float]:
        match = pattern.search(line)
        return float(match.group(1)) if match else None

    def _extract_string(self, pattern: re.Pattern, line: str) -> Optional[str]:
        match = pattern.search(line)
        return match.group(1) if match else None


class StrategyCallExtractor:
    """Extract strategy.entry() and strategy.exit() calls."""

    ENTRY_PATTERN = re.compile(
        r"strategy\.entry\s*\(\s*['\"]([^'\"]+)['\"]"
        r"\s*,\s*strategy\.(long|short)"
        r"(?:[^)]*when\s*=\s*(\w+))?",
        re.IGNORECASE
    )

    EXIT_PATTERN = re.compile(
        r"strategy\.exit\s*\(\s*['\"]([^'\"]+)['\"]"
        r"\s*,\s*['\"]([^'\"]+)['\"]",
        re.IGNORECASE
    )

    CLOSE_PATTERN = re.compile(
        r"strategy\.close\s*\(\s*['\"]([^'\"]+)['\"]",
        re.IGNORECASE
    )

    def extract(self, content: str) -> List[StrategyCall]:
        """Extract all strategy calls."""
        calls = []

        for line in content.split('\n'):
            # Entry calls
            for match in self.ENTRY_PATTERN.finditer(line):
                calls.append(StrategyCall(
                    call_type="entry",
                    id=match.group(1),
                    direction=match.group(2).lower(),
                    condition_var=match.group(3),
                    raw_line=line.strip()
                ))

            # Exit calls
            for match in self.EXIT_PATTERN.finditer(line):
                calls.append(StrategyCall(
                    call_type="exit",
                    id=match.group(1),
                    raw_line=line.strip()
                ))

            # Close calls
            for match in self.CLOSE_PATTERN.finditer(line):
                calls.append(StrategyCall(
                    call_type="close",
                    id=match.group(1),
                    raw_line=line.strip()
                ))

        return calls


class FunctionExtractor:
    """Extract function definitions that might be entry conditions."""

    # Pattern for function definitions
    FUNC_PATTERN = re.compile(
        r"(\w+)\s*\(\s*\)\s*=>\s*\n?((?:.*\n)*?)(?=\n\w|\n//|\nstrategy|\Z)",
        re.MULTILINE
    )

    # Simpler single-line function
    FUNC_SIMPLE = re.compile(
        r"(\w+)\s*\(\s*\)\s*=>\s*(.+?)(?:\n|$)"
    )

    def extract(self, content: str) -> Dict[str, FunctionDefinition]:
        """Extract function definitions."""
        functions = {}

        # Multi-line functions
        for match in self.FUNC_PATTERN.finditer(content):
            name = match.group(1)
            body = match.group(2).strip()
            if body:
                functions[name] = FunctionDefinition(
                    name=name,
                    body=body,
                    returns_bool=self._likely_returns_bool(body)
                )

        # Single-line functions
        for match in self.FUNC_SIMPLE.finditer(content):
            name = match.group(1)
            if name not in functions:  # Don't override multi-line
                body = match.group(2).strip()
                functions[name] = FunctionDefinition(
                    name=name,
                    body=body,
                    returns_bool=self._likely_returns_bool(body)
                )

        return functions

    def _likely_returns_bool(self, body: str) -> bool:
        """Check if function likely returns boolean."""
        keywords = ['and', 'or', '>', '<', '>=', '<=', '==', '!=', 'true', 'false']
        return any(kw in body.lower() for kw in keywords)


class IndicatorDetector:
    """Detect which indicators are used in the script."""

    INDICATORS = {
        'sma': r'ta\.sma\s*\(',
        'ema': r'ta\.ema\s*\(',
        'rsi': r'ta\.rsi\s*\(',
        'macd': r'ta\.macd\s*\(',
        'bb': r'ta\.bb\s*\(',
        'atr': r'ta\.atr\s*\(',
        'stoch': r'ta\.stoch\s*\(',
        'adx': r'ta\.dmi\s*\(',
        'vwap': r'ta\.vwap\s*\(',
        'volume_sma': r'ta\.sma\s*\(\s*volume',
        'crossover': r'ta\.crossover\s*\(',
        'crossunder': r'ta\.crossunder\s*\(',
    }

    def detect(self, content: str) -> List[str]:
        """Detect which indicators are used."""
        found = []
        for name, pattern in self.INDICATORS.items():
            if re.search(pattern, content, re.IGNORECASE):
                found.append(name)
        return found


class MetadataExtractor:
    """Extract strategy metadata from strategy() call."""

    STRATEGY_PATTERN = re.compile(
        r"strategy\s*\(\s*['\"]([^'\"]+)['\"]",
        re.IGNORECASE
    )

    INITIAL_CAPITAL = re.compile(r"initial_capital\s*=\s*(\d+)", re.IGNORECASE)
    COMMISSION = re.compile(r"commission_value\s*=\s*([\d.]+)", re.IGNORECASE)
    SLIPPAGE = re.compile(r"slippage\s*=\s*(\d+)", re.IGNORECASE)
    PYRAMIDING = re.compile(r"pyramiding\s*=\s*(\d+)", re.IGNORECASE)

    VERSION_PATTERN = re.compile(r"//@version=(\d+)")

    def extract(self, content: str) -> Dict[str, Any]:
        """Extract strategy metadata."""
        metadata = {
            'name': 'Untitled Strategy',
            'version': 0,
            'initial_capital': None,
            'commission': None,
            'slippage': None,
            'pyramiding': None,
        }

        # Version
        match = self.VERSION_PATTERN.search(content)
        if match:
            metadata['version'] = int(match.group(1))

        # Strategy name
        match = self.STRATEGY_PATTERN.search(content)
        if match:
            metadata['name'] = match.group(1)

        # Settings
        match = self.INITIAL_CAPITAL.search(content)
        if match:
            metadata['initial_capital'] = float(match.group(1))

        match = self.COMMISSION.search(content)
        if match:
            metadata['commission'] = float(match.group(1))

        match = self.SLIPPAGE.search(content)
        if match:
            metadata['slippage'] = int(match.group(1))

        match = self.PYRAMIDING.search(content)
        if match:
            metadata['pyramiding'] = int(match.group(1))

        return metadata


def parse_universal(content: str) -> UniversalParsedStrategy:
    """
    Parse any Pine Script and extract all available information.

    Args:
        content: Raw Pine Script content

    Returns:
        UniversalParsedStrategy with all extracted data
    """
    # Extract metadata
    metadata_extractor = MetadataExtractor()
    metadata = metadata_extractor.extract(content)

    # Extract inputs
    input_extractor = UniversalInputExtractor()
    inputs, groups = input_extractor.extract(content)

    # Extract strategy calls
    call_extractor = StrategyCallExtractor()
    calls = call_extractor.extract(content)

    # Extract functions
    func_extractor = FunctionExtractor()
    functions = func_extractor.extract(content)

    # Detect indicators
    indicator_detector = IndicatorDetector()
    indicators = indicator_detector.detect(content)

    return UniversalParsedStrategy(
        name=metadata['name'],
        version=metadata['version'],
        initial_capital=metadata['initial_capital'],
        commission=metadata['commission'],
        slippage=metadata['slippage'],
        pyramiding=metadata['pyramiding'],
        inputs=inputs,
        input_groups=groups,
        strategy_calls=calls,
        functions=functions,
        indicators=indicators,
        raw_content=content,
    )


def get_universal_summary(parsed: UniversalParsedStrategy) -> Dict[str, Any]:
    """Generate a summary for UI display."""
    return {
        'name': parsed.name,
        'version': parsed.version,
        'total_inputs': parsed.total_inputs,
        'groups': list(parsed.input_groups.keys()),
        'entry_conditions': len(parsed.entry_calls),
        'exit_conditions': len(parsed.exit_calls),
        'functions': len(parsed.functions),
        'indicators': parsed.indicators,
        'strategy_settings': {
            'initial_capital': parsed.initial_capital,
            'commission': parsed.commission,
            'slippage': parsed.slippage,
            'pyramiding': parsed.pyramiding,
        }
    }
