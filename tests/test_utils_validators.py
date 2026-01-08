"""
Unit tests for the validators module.

Run with: uv run pytest tests/test_utils_validators.py -v
"""

import pytest

from src.utils.validators import (
    ValidationError,
    validate_positive_number,
    validate_non_empty_string,
    validate_in_range,
    validate_list_not_empty,
    validate_dict_has_keys,
)


class TestValidatePositiveNumber:
    """Tests for validate_positive_number function."""

    def test_positive_number_passes(self):
        """Positive numbers should pass."""
        assert validate_positive_number(5) == 5
        assert validate_positive_number(0.1) == 0.1

    def test_zero_fails_by_default(self):
        """Zero should fail by default."""
        with pytest.raises(ValidationError):
            validate_positive_number(0)

    def test_zero_passes_when_allowed(self):
        """Zero should pass when allow_zero=True."""
        assert validate_positive_number(0, allow_zero=True) == 0

    def test_negative_number_fails(self):
        """Negative numbers should fail."""
        with pytest.raises(ValidationError) as exc_info:
            validate_positive_number(-5, field_name="count")
        assert "count" in str(exc_info.value)

    def test_non_number_fails(self):
        """Non-numbers should fail."""
        with pytest.raises(ValidationError):
            validate_positive_number("5")


class TestValidateNonEmptyString:
    """Tests for validate_non_empty_string function."""

    def test_valid_string_passes(self):
        """Non-empty strings should pass."""
        assert validate_non_empty_string("hello") == "hello"

    def test_empty_string_fails(self):
        """Empty strings should fail."""
        with pytest.raises(ValidationError):
            validate_non_empty_string("")

    def test_whitespace_only_fails(self):
        """Whitespace-only strings should fail."""
        with pytest.raises(ValidationError):
            validate_non_empty_string("   ")

    def test_whitespace_passes_when_strip_false(self):
        """Whitespace should pass when strip=False."""
        assert validate_non_empty_string("   ", strip=False) == "   "

    def test_non_string_fails(self):
        """Non-strings should fail."""
        with pytest.raises(ValidationError):
            validate_non_empty_string(123)

    def test_field_name_in_error(self):
        """Field name should appear in error message."""
        with pytest.raises(ValidationError) as exc_info:
            validate_non_empty_string("", field_name="username")
        assert "username" in str(exc_info.value)


class TestValidateInRange:
    """Tests for validate_in_range function."""

    def test_value_in_range_passes(self):
        """Value within range should pass."""
        assert validate_in_range(5, min_value=0, max_value=10) == 5

    def test_value_at_min_passes(self):
        """Value at minimum should pass (inclusive)."""
        assert validate_in_range(0, min_value=0, max_value=10) == 0

    def test_value_at_max_passes(self):
        """Value at maximum should pass (inclusive)."""
        assert validate_in_range(10, min_value=0, max_value=10) == 10

    def test_value_below_min_fails(self):
        """Value below minimum should fail."""
        with pytest.raises(ValidationError):
            validate_in_range(-1, min_value=0, max_value=10)

    def test_value_above_max_fails(self):
        """Value above maximum should fail."""
        with pytest.raises(ValidationError):
            validate_in_range(11, min_value=0, max_value=10)

    def test_no_min_allows_low_values(self):
        """No minimum should allow any low value."""
        assert validate_in_range(-1000, max_value=10) == -1000

    def test_no_max_allows_high_values(self):
        """No maximum should allow any high value."""
        assert validate_in_range(1000, min_value=0) == 1000

    def test_exclusive_bounds(self):
        """Exclusive bounds should exclude boundary values."""
        with pytest.raises(ValidationError):
            validate_in_range(0, min_value=0, inclusive=False)


class TestValidateListNotEmpty:
    """Tests for validate_list_not_empty function."""

    def test_non_empty_list_passes(self):
        """Non-empty list should pass."""
        assert validate_list_not_empty([1, 2, 3]) == [1, 2, 3]

    def test_empty_list_fails(self):
        """Empty list should fail."""
        with pytest.raises(ValidationError):
            validate_list_not_empty([])

    def test_non_list_fails(self):
        """Non-list should fail."""
        with pytest.raises(ValidationError):
            validate_list_not_empty("not a list")


class TestValidateDictHasKeys:
    """Tests for validate_dict_has_keys function."""

    def test_dict_with_keys_passes(self):
        """Dict with required keys should pass."""
        data = {"name": "test", "value": 123}
        assert validate_dict_has_keys(data, ["name", "value"]) == data

    def test_missing_keys_fails(self):
        """Dict missing required keys should fail."""
        data = {"name": "test"}
        with pytest.raises(ValidationError) as exc_info:
            validate_dict_has_keys(data, ["name", "value"])
        assert "value" in str(exc_info.value)

    def test_non_dict_fails(self):
        """Non-dict should fail."""
        with pytest.raises(ValidationError):
            validate_dict_has_keys("not a dict", ["key"])

    def test_extra_keys_allowed(self):
        """Extra keys should be allowed."""
        data = {"name": "test", "extra": "value"}
        assert validate_dict_has_keys(data, ["name"]) == data


class TestValidationError:
    """Tests for ValidationError exception."""

    def test_error_with_field(self):
        """Error with field should include field in message."""
        error = ValidationError("invalid value", field="temperature")
        assert "[temperature]" in str(error)

    def test_error_without_field(self):
        """Error without field should just have message."""
        error = ValidationError("invalid value")
        assert str(error) == "invalid value"

    def test_field_attribute(self):
        """Error should store field attribute."""
        error = ValidationError("invalid", field="test")
        assert error.field == "test"