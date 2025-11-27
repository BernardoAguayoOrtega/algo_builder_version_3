"""
Tests for Universal Pine Script Parser.

These tests verify that the universal parser correctly extracts ALL inputs
from any Pine Script file without requiring predefined mappings.
"""

import pytest
from pathlib import Path

from src.pine_parser.universal import (
    parse_universal,
    get_universal_summary,
    UniversalInputExtractor,
    StrategyCallExtractor,
    FunctionExtractor,
    IndicatorDetector,
    MetadataExtractor,
    InputType,
)


# Get the fixtures directory
FIXTURES_DIR = Path(__file__).parent / "fixtures"


class TestUniversalInputExtractor:
    """Tests for the UniversalInputExtractor class."""

    def test_extract_bool_input(self):
        """Boolean inputs should be extracted with correct type."""
        content = "operar_largos = input.bool(true, 'Largos', group=grpStrat)"
        extractor = UniversalInputExtractor()
        inputs, groups = extractor.extract(content)

        assert 'operar_largos' in inputs
        assert inputs['operar_largos'].input_type == InputType.BOOL
        assert inputs['operar_largos'].default_value is True
        assert inputs['operar_largos'].label == 'Largos'

    def test_extract_int_input(self):
        """Integer inputs should be extracted with constraints."""
        content = "entry_pip = input.int(5, 'Pips entrada', minval=1, group='Config')"
        extractor = UniversalInputExtractor()
        inputs, groups = extractor.extract(content)

        assert 'entry_pip' in inputs
        assert inputs['entry_pip'].input_type == InputType.INT
        assert inputs['entry_pip'].default_value == 5
        assert inputs['entry_pip'].min_val == 1.0

    def test_extract_float_input(self):
        """Float inputs should be extracted with constraints."""
        content = "target_ratio = input.float(1.5, 'Ratio TP', minval=0.1, step=0.1)"
        extractor = UniversalInputExtractor()
        inputs, groups = extractor.extract(content)

        assert 'target_ratio' in inputs
        assert inputs['target_ratio'].input_type == InputType.FLOAT
        assert inputs['target_ratio'].default_value == 1.5
        assert inputs['target_ratio'].min_val == 0.1
        assert inputs['target_ratio'].step == 0.1

    def test_extract_string_input_with_options(self):
        """String inputs with options should have options list."""
        content = "filtro_mm = input.string(defval='Sin filtro', title='MA Filter', options=['Sin filtro','Alcista'])"
        extractor = UniversalInputExtractor()
        inputs, groups = extractor.extract(content)

        assert 'filtro_mm' in inputs
        assert inputs['filtro_mm'].input_type == InputType.STRING
        assert inputs['filtro_mm'].default_value == 'Sin filtro'
        assert inputs['filtro_mm'].options == ['Sin filtro', 'Alcista']

    def test_extract_groups_correctly(self):
        """Inputs should be grouped by their group parameter."""
        content = """
        a = input.bool(true, 'A', group='Group1')
        b = input.bool(false, 'B', group='Group1')
        c = input.int(5, 'C', group='Group2')
        """
        extractor = UniversalInputExtractor()
        inputs, groups = extractor.extract(content)

        assert 'Group1' in groups
        assert 'Group2' in groups
        assert 'a' in groups['Group1']
        assert 'b' in groups['Group1']
        assert 'c' in groups['Group2']


class TestStrategyCallExtractor:
    """Tests for the StrategyCallExtractor class."""

    def test_extract_entry_long(self):
        """Long entry calls should be extracted."""
        content = "strategy.entry('Long Entry', strategy.long, qty=1)"
        extractor = StrategyCallExtractor()
        calls = extractor.extract(content)

        assert len(calls) == 1
        assert calls[0].call_type == 'entry'
        assert calls[0].id == 'Long Entry'
        assert calls[0].direction == 'long'

    def test_extract_entry_short(self):
        """Short entry calls should be extracted."""
        content = "strategy.entry('Short Entry', strategy.short, qty=1)"
        extractor = StrategyCallExtractor()
        calls = extractor.extract(content)

        assert len(calls) == 1
        assert calls[0].call_type == 'entry'
        assert calls[0].direction == 'short'

    def test_extract_close(self):
        """Close calls should be extracted."""
        content = "strategy.close('Long Entry', comment='TP')"
        extractor = StrategyCallExtractor()
        calls = extractor.extract(content)

        assert len(calls) == 1
        assert calls[0].call_type == 'close'
        assert calls[0].id == 'Long Entry'


class TestFunctionExtractor:
    """Tests for the FunctionExtractor class."""

    def test_extract_simple_function(self):
        """Simple functions should be extracted."""
        content = "bullEngulf() => close > open and close[1] < open[1]"
        extractor = FunctionExtractor()
        functions = extractor.extract(content)

        assert 'bullEngulf' in functions
        assert functions['bullEngulf'].returns_bool is True

    def test_detect_boolean_functions(self):
        """Functions returning boolean should be detected."""
        content = """
        myCondition() =>
            a = close > open
            b = high > high[1]
            a and b
        """
        extractor = FunctionExtractor()
        functions = extractor.extract(content)

        assert 'myCondition' in functions
        assert functions['myCondition'].returns_bool is True


class TestIndicatorDetector:
    """Tests for the IndicatorDetector class."""

    def test_detect_sma(self):
        """SMA indicator should be detected."""
        content = "sma50 = ta.sma(close, 50)"
        detector = IndicatorDetector()
        indicators = detector.detect(content)

        assert 'sma' in indicators

    def test_detect_rsi(self):
        """RSI indicator should be detected."""
        content = "rsiValue = ta.rsi(close, 14)"
        detector = IndicatorDetector()
        indicators = detector.detect(content)

        assert 'rsi' in indicators

    def test_detect_crossover(self):
        """Crossover should be detected."""
        content = "bullCross = ta.crossover(sma50, sma200)"
        detector = IndicatorDetector()
        indicators = detector.detect(content)

        assert 'crossover' in indicators


class TestMetadataExtractor:
    """Tests for the MetadataExtractor class."""

    def test_extract_version(self):
        """Version directive should be extracted."""
        content = "//@version=6\nstrategy('Test')"
        extractor = MetadataExtractor()
        metadata = extractor.extract(content)

        assert metadata['version'] == 6

    def test_extract_strategy_name(self):
        """Strategy name should be extracted."""
        content = "//@version=6\nstrategy('My Strategy', overlay=true)"
        extractor = MetadataExtractor()
        metadata = extractor.extract(content)

        assert metadata['name'] == 'My Strategy'

    def test_extract_initial_capital(self):
        """Initial capital should be extracted."""
        content = "strategy('Test', initial_capital=50000)"
        extractor = MetadataExtractor()
        metadata = extractor.extract(content)

        assert metadata['initial_capital'] == 50000.0

    def test_extract_commission(self):
        """Commission should be extracted."""
        content = "strategy('Test', commission_value=2.5)"
        extractor = MetadataExtractor()
        metadata = extractor.extract(content)

        assert metadata['commission'] == 2.5


class TestParseUniversal:
    """Integration tests for the full parse_universal function."""

    @pytest.fixture
    def reference_content(self):
        """Load the reference Pine Script file."""
        filepath = FIXTURES_DIR / "reference.pine"
        if not filepath.exists():
            pytest.skip("Reference file not found")
        with open(filepath) as f:
            return f.read()

    def test_parse_reference_file(self, reference_content):
        """Reference file should parse correctly."""
        parsed = parse_universal(reference_content)

        assert parsed.name == "Algo Strategy Builder"
        assert parsed.version == 6
        assert parsed.total_inputs == 29

    def test_parse_extracts_all_input_types(self, reference_content):
        """All input types should be represented."""
        parsed = parse_universal(reference_content)

        input_types = {inp.input_type for inp in parsed.inputs.values()}
        assert InputType.BOOL in input_types
        assert InputType.INT in input_types
        assert InputType.FLOAT in input_types
        assert InputType.STRING in input_types

    def test_parse_extracts_strategy_settings(self, reference_content):
        """Strategy settings should be extracted."""
        parsed = parse_universal(reference_content)

        assert parsed.initial_capital == 100000.0
        assert parsed.commission == 1.5
        assert parsed.slippage == 1

    def test_parse_extracts_functions(self, reference_content):
        """Entry pattern functions should be extracted."""
        parsed = parse_universal(reference_content)

        # Should find the entry condition functions
        assert any('sacudida' in name.lower() for name in parsed.functions.keys())
        assert any('engulf' in name.lower() for name in parsed.functions.keys())

    def test_parse_extracts_indicators(self, reference_content):
        """Used indicators should be detected."""
        parsed = parse_universal(reference_content)

        assert 'sma' in parsed.indicators

    def test_summary_generation(self, reference_content):
        """Summary should be generated correctly."""
        parsed = parse_universal(reference_content)
        summary = get_universal_summary(parsed)

        assert summary['name'] == "Algo Strategy Builder"
        assert summary['version'] == 6
        assert summary['total_inputs'] == 29
        assert len(summary['groups']) > 0


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_content(self):
        """Empty content should return empty results."""
        parsed = parse_universal("")

        assert parsed.total_inputs == 0
        assert parsed.version == 0

    def test_no_inputs(self):
        """Content without inputs should parse."""
        content = """
        //@version=6
        strategy('No Inputs')
        entry = close > open
        """
        parsed = parse_universal(content)

        assert parsed.name == "No Inputs"
        assert parsed.version == 6
        assert parsed.total_inputs == 0

    def test_minimal_script(self):
        """Minimal valid script should parse."""
        content = """
        //@version=6
        strategy('Minimal')
        """
        parsed = parse_universal(content)

        assert parsed.name == "Minimal"
        assert parsed.version == 6


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
