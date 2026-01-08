"""
Unit tests for the text helpers module.

Run with: uv run pytest tests/test_utils_text_helpers.py -v
"""

import pytest

from src.utils.text_helpers import (
    truncate_text,
    sanitize_filename,
    format_duration,
    capitalize_sentences,
    remove_extra_whitespace,
    extract_numbers,
)


class TestTruncateText:
    """Tests for truncate_text function."""

    def test_short_text_unchanged(self):
        """Text shorter than max_length should be unchanged."""
        assert truncate_text("Hello", 10) == "Hello"

    def test_exact_length_unchanged(self):
        """Text exactly at max_length should be unchanged."""
        assert truncate_text("Hello", 5) == "Hello"

    def test_long_text_truncated(self):
        """Long text should be truncated with suffix."""
        assert truncate_text("Hello World", 8) == "Hello..."

    def test_custom_suffix(self):
        """Custom suffix should be used."""
        assert truncate_text("Hello World", 9, suffix="~") == "Hello Wo~"

    def test_empty_string(self):
        """Empty string should return empty."""
        assert truncate_text("", 10) == ""

    def test_zero_max_length(self):
        """Zero max_length should return empty."""
        assert truncate_text("Hello", 0) == ""


class TestSanitizeFilename:
    """Tests for sanitize_filename function."""

    def test_clean_filename_unchanged(self):
        """Clean filename should be unchanged."""
        assert sanitize_filename("document.txt") == "document.txt"

    def test_invalid_chars_replaced(self):
        """Invalid characters should be replaced."""
        result = sanitize_filename("my:file/name?.txt")
        assert ":" not in result
        assert "/" not in result
        assert "?" not in result

    def test_empty_string_returns_unnamed(self):
        """Empty string should return 'unnamed'."""
        assert sanitize_filename("") == "unnamed"

    def test_max_length_enforced(self):
        """Filename should be truncated to max_length."""
        long_name = "a" * 300
        result = sanitize_filename(long_name, max_length=100)
        assert len(result) <= 100


class TestFormatDuration:
    """Tests for format_duration function."""

    def test_seconds_only(self):
        """Duration under 60 seconds."""
        assert format_duration(45.5) == "45.5s"

    def test_minutes_and_seconds(self):
        """Duration with minutes."""
        assert format_duration(125.5) == "2m 5.5s"

    def test_hours_minutes_seconds(self):
        """Duration with hours."""
        assert format_duration(3661.5) == "1h 1m 1.5s"

    def test_zero_duration(self):
        """Zero duration."""
        assert format_duration(0) == "0.0s"

    def test_negative_duration(self):
        """Negative duration should have minus sign."""
        result = format_duration(-45)
        assert result.startswith("-")

    def test_custom_precision(self):
        """Custom decimal precision."""
        assert format_duration(1.234, precision=2) == "1.23s"


class TestCapitalizeSentences:
    """Tests for capitalize_sentences function."""

    def test_single_sentence(self):
        """Single sentence should be capitalized."""
        assert capitalize_sentences("hello world.") == "Hello world."

    def test_multiple_sentences(self):
        """Multiple sentences should all be capitalized."""
        result = capitalize_sentences("hello. world. test.")
        assert result == "Hello. World. Test."

    def test_question_marks(self):
        """Question marks should trigger capitalization."""
        result = capitalize_sentences("hello? world?")
        assert result == "Hello? World?"

    def test_empty_string(self):
        """Empty string should return empty."""
        assert capitalize_sentences("") == ""

    def test_already_capitalized(self):
        """Already capitalized text should be unchanged."""
        assert capitalize_sentences("Hello. World.") == "Hello. World."


class TestRemoveExtraWhitespace:
    """Tests for remove_extra_whitespace function."""

    def test_multiple_spaces(self):
        """Multiple spaces should become single space."""
        assert remove_extra_whitespace("hello   world") == "hello world"

    def test_leading_trailing_spaces(self):
        """Leading/trailing spaces should be removed."""
        assert remove_extra_whitespace("  hello  ") == "hello"

    def test_tabs_and_newlines(self):
        """Tabs and newlines should be normalized."""
        assert remove_extra_whitespace("hello\t\nworld") == "hello world"

    def test_empty_string(self):
        """Empty string should return empty."""
        assert remove_extra_whitespace("") == ""


class TestExtractNumbers:
    """Tests for extract_numbers function."""

    def test_integers(self):
        """Should extract integers."""
        assert extract_numbers("I have 3 apples") == [3.0]

    def test_floats(self):
        """Should extract floats."""
        assert extract_numbers("Price is 19.99") == [19.99]

    def test_multiple_numbers(self):
        """Should extract multiple numbers."""
        assert extract_numbers("1 and 2 and 3") == [1.0, 2.0, 3.0]

    def test_negative_numbers(self):
        """Should extract negative numbers."""
        assert extract_numbers("Temperature is -5 degrees") == [-5.0]

    def test_no_numbers(self):
        """Should return empty list if no numbers."""
        assert extract_numbers("no numbers here") == []

    def test_empty_string(self):
        """Empty string should return empty list."""
        assert extract_numbers("") == []