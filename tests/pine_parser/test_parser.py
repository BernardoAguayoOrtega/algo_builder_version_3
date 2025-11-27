"""
Tests for Pine Script Parser.

These tests verify that the parser correctly extracts configuration
from Pine Script files.
"""

import pytest
import os
from pathlib import Path

from src.pine_parser import parse_pine_script, parsed_to_config, ParsedStrategy
from src.pine_parser.tokenizer import (
    tokenize,
    remove_single_line_comments,
    remove_multiline_comments,
    normalize_multiline_statements,
    extract_version,
)
from src.pine_parser.extractors import InputExtractor, PatternDetector, FilterDetector
from src.pine_parser.exceptions import PineParserError, ValidationError


# Get the fixtures directory
FIXTURES_DIR = Path(__file__).parent / "fixtures"


class TestTokenizer:
    """Tests for the tokenizer module."""

    def test_remove_single_line_comments(self):
        """Single line comments should be removed."""
        content = """
        entry_pip = input.int(1, 'Title') // this is a comment
        // full line comment
        sl_pip = input.int(2, 'SL')
        """
        result = remove_single_line_comments(content)

        assert "this is a comment" not in result
        assert "full line comment" not in result
        assert "entry_pip" in result
        assert "sl_pip" in result

    def test_preserve_strings_with_slashes(self):
        """Strings containing // should not be treated as comments."""
        content = """
        title = "Path: //server/folder" // actual comment
        """
        result = remove_single_line_comments(content)

        assert "Path: //server/folder" in result
        assert "actual comment" not in result

    def test_remove_multiline_comments(self):
        """Multiline comments should be removed."""
        content = """
        entry_pip = input.int(1, 'Title')
        /* this is a
        multiline comment */
        sl_pip = input.int(2, 'SL')
        """
        result = remove_multiline_comments(content)

        assert "multiline comment" not in result
        assert "entry_pip" in result
        assert "sl_pip" in result

    def test_normalize_multiline_statements(self):
        """Multi-line statements should be normalized to single lines."""
        content = """
        entry_pip = input.int(
            1,
            'Title',
            minval = 1
        )
        """
        result = normalize_multiline_statements(content)

        # Should be on one line now
        assert "input.int( 1, 'Title', minval = 1 )" in result

    def test_extract_version(self):
        """Version directive should be extracted."""
        content = "//@version=6\nstrategy('Test')"
        version = extract_version(content)
        assert version == 6

    def test_extract_version_missing(self):
        """Missing version should return 0."""
        content = "strategy('Test')"
        version = extract_version(content)
        assert version == 0


class TestInputExtractor:
    """Tests for the InputExtractor class."""

    def test_extract_bool_true(self):
        """Boolean true inputs should be extracted."""
        content = "operar_largos = input.bool(true, 'Largos', group=grpStrat)"
        extractor = InputExtractor()
        result = extractor.extract(content)

        assert 'trade_longs' in result
        assert result['trade_longs'] is True

    def test_extract_bool_false(self):
        """Boolean false inputs should be extracted."""
        content = "usar_patron_vol_climatico = input.bool(false, 'Volumen climático', group=grpPat)"
        extractor = InputExtractor()
        result = extractor.extract(content)

        assert 'use_climatic_volume' in result
        assert result['use_climatic_volume'] is False

    def test_extract_int(self):
        """Integer inputs should be extracted."""
        content = "entry_pip = input.int(1, 'Pips para entrada', minval=1, group='Config')"
        extractor = InputExtractor()
        result = extractor.extract(content)

        assert 'entry_pips' in result
        assert result['entry_pips'] == 1

    def test_extract_float(self):
        """Float inputs should be extracted."""
        content = "target_ratio = input.float(1.5, 'Ratio Take Profit', minval=0.1)"
        extractor = InputExtractor()
        result = extractor.extract(content)

        assert 'tp_ratio' in result
        assert result['tp_ratio'] == 1.5

    def test_extract_string(self):
        """String inputs should be extracted."""
        content = "filtro_mm50200 = input.string(defval='Sin filtro', title='Cruce MM 50/200', options=['Sin filtro'])"
        extractor = InputExtractor()
        result = extractor.extract(content)

        assert 'ma_filter' in result
        assert result['ma_filter'] == 'Sin filtro'


class TestPatternDetector:
    """Tests for the PatternDetector class."""

    def test_detect_sacudida(self):
        """Sacudida pattern should be detected."""
        content = """
        sacudida_long_condition() =>
            vela2_bajista = close[1] < open[1]
        """
        detector = PatternDetector()
        result = detector.detect(content)

        assert result['has_sacudida'] is True

    def test_detect_engulfing(self):
        """Engulfing pattern should be detected."""
        content = """
        bullEngulf() =>
            VelaAlcista = close > open
        """
        detector = PatternDetector()
        result = detector.detect(content)

        assert result['has_engulfing'] is True

    def test_detect_climatic_volume(self):
        """Climatic volume pattern should be detected."""
        content = """
        volClimatico = volume > volMA20 * 1.75
        clim_long_raw = volClimatico and close > open
        """
        detector = PatternDetector()
        result = detector.detect(content)

        assert result['has_climatic_volume'] is True


class TestFilterDetector:
    """Tests for the FilterDetector class."""

    def test_detect_ma_filter(self):
        """MA filter should be detected."""
        content = """
        sma50 = ta.sma(close, 50)
        sma200 = ta.sma(close, 200)
        filtro_ok_mm = sma50 > sma200
        """
        detector = FilterDetector()
        result = detector.detect(content)

        assert result['has_ma_filter'] is True

    def test_detect_session_filter(self):
        """Session filter should be detected."""
        content = """
        enHorario1 = not na(time(timeframe.period, '0100-0815'))
        """
        detector = FilterDetector()
        result = detector.detect(content)

        assert result['has_session_filter'] is True

    def test_detect_day_filter(self):
        """Day filter should be detected."""
        content = """
        operarHoy() =>
            dayofweek == dayofweek.monday and operarLunes
        """
        detector = FilterDetector()
        result = detector.detect(content)

        assert result['has_day_filter'] is True


class TestParseReferenceFile:
    """Tests using the actual reference Pine Script file."""

    @pytest.fixture
    def reference_content(self):
        """Load the reference Pine Script file."""
        filepath = FIXTURES_DIR / "reference.pine"
        if not filepath.exists():
            pytest.skip("Reference file not found")
        with open(filepath) as f:
            return f.read()

    def test_parse_reference_file_confidence(self, reference_content):
        """Reference file should parse with high confidence."""
        parsed = parse_pine_script(reference_content)

        assert parsed.confidence_score >= 0.9, f"Expected >=90% confidence, got {parsed.confidence_score:.0%}"

    def test_parse_reference_file_version(self, reference_content):
        """Version should be extracted correctly."""
        parsed = parse_pine_script(reference_content)

        assert parsed.version == 6

    def test_parse_reference_file_name(self, reference_content):
        """Strategy name should be extracted."""
        parsed = parse_pine_script(reference_content)

        assert parsed.name == "Algo Strategy Builder"

    def test_parse_reference_file_basic_config(self, reference_content):
        """Basic config values should be extracted."""
        parsed = parse_pine_script(reference_content)

        assert parsed.entry_pips == 1
        assert parsed.sl_pips == 1

    def test_parse_reference_file_entry_patterns(self, reference_content):
        """Entry pattern settings should be extracted."""
        parsed = parse_pine_script(reference_content)

        assert parsed.use_sacudida is True
        assert parsed.use_engulfing is True
        assert parsed.use_climatic_volume is False

    def test_parse_reference_file_exits(self, reference_content):
        """Exit settings should be extracted."""
        parsed = parse_pine_script(reference_content)

        assert parsed.use_sl is True
        assert parsed.use_tp_ratio is True
        assert parsed.tp_ratio == 1.0
        assert parsed.use_n_bars_exit is False
        assert parsed.n_bars_exit == 5

    def test_parse_reference_file_filters(self, reference_content):
        """Filter settings should be extracted."""
        parsed = parse_pine_script(reference_content)

        assert parsed.ma_filter == "Sin filtro"

    def test_parse_reference_file_sessions(self, reference_content):
        """Session settings should be extracted."""
        parsed = parse_pine_script(reference_content)

        assert parsed.use_london is True
        assert parsed.use_newyork is True
        assert parsed.use_tokyo is True

    def test_parse_reference_file_days(self, reference_content):
        """Day settings should be extracted."""
        parsed = parse_pine_script(reference_content)

        assert parsed.trade_monday is True
        assert parsed.trade_tuesday is True
        assert parsed.trade_wednesday is True
        assert parsed.trade_thursday is True
        assert parsed.trade_friday is True
        assert parsed.trade_saturday is True
        assert parsed.trade_sunday is True

    def test_parse_reference_file_risk(self, reference_content):
        """Risk management settings should be extracted."""
        parsed = parse_pine_script(reference_content)

        assert parsed.risk_type == "fixed_size"
        assert parsed.fixed_size == 1.0
        assert parsed.fixed_risk_money == 100.0
        assert parsed.risk_percent == 1.0

    def test_parse_reference_file_backtest(self, reference_content):
        """Backtest settings should be extracted."""
        parsed = parse_pine_script(reference_content)

        assert parsed.initial_capital == 100000.0
        assert parsed.commission == 1.5
        assert parsed.slippage == 1

    def test_parsed_to_config(self, reference_content):
        """ParsedStrategy should convert to StrategyConfig."""
        parsed = parse_pine_script(reference_content)
        config, notes = parsed_to_config(parsed)

        assert config.entry_pips == 1
        assert config.sl_pips == 1
        assert config.use_sacudida is True
        assert config.use_engulfing is True
        assert config.use_climatic_volume is False
        assert config.tp_ratio == 1.0
        assert config.ma_filter == "Sin filtro"
        assert config.initial_capital == 100000.0


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_content(self):
        """Empty content should return low confidence."""
        parsed = parse_pine_script("")
        assert parsed.confidence_score == 0.0

    def test_invalid_content(self):
        """Invalid content should parse with low confidence."""
        content = "This is not Pine Script at all"
        parsed = parse_pine_script(content)

        assert parsed.confidence_score < 0.5

    def test_low_confidence_conversion_fails(self):
        """Low confidence parsing should fail conversion."""
        content = "Random text without any inputs"
        parsed = parse_pine_script(content)

        with pytest.raises(ValidationError):
            parsed_to_config(parsed)

    def test_partial_content(self):
        """Partial Pine Script should extract what it can."""
        content = """
        //@version=6
        strategy('Test')
        entry_pip = input.int(5, 'Pips para entrada')
        """
        parsed = parse_pine_script(content)

        assert parsed.version == 6
        assert parsed.name == "Test"
        assert parsed.entry_pips == 5
        # Other fields should be None
        assert parsed.sl_pips is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
