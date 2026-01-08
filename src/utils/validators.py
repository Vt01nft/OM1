"""
Input validation utilities for OM1.

This module provides helper functions for validating inputs
with clear error messages.

Problem Statement:
    Invalid inputs often cause cryptic errors deep in the code.
    Early validation with clear messages improves debugging.

Solution:
    Provide reusable validation functions that raise descriptive errors.
"""

from typing import Any, Optional, TypeVar

T = TypeVar("T", int, float)


class ValidationError(ValueError):
    """Raised when validation fails."""

    def __init__(self, message: str, field: Optional[str] = None) -> None:
        self.field = field
        full_message = f"[{field}] {message}" if field else message
        super().__init__(full_message)


def validate_positive_number(
    value: T,
    field_name: Optional[str] = None,
    allow_zero: bool = False,
) -> T:
    """
    Validate that a number is positive.

    Args:
        value: Number to validate
        field_name: Name of field for error messages
        allow_zero: Whether zero is allowed

    Returns:
        The validated value

    Raises:
        ValidationError: If validation fails

    Example:
        >>> validate_positive_number(5, "count")
        5
        >>> validate_positive_number(-1, "count")
        ValidationError: [count] Value must be positive, got -1
    """
    if not isinstance(value, (int, float)):
        raise ValidationError(
            f"Expected a number, got {type(value).__name__}",
            field_name,
        )

    if allow_zero:
        if value < 0:
            raise ValidationError(
                f"Value must be non-negative, got {value}",
                field_name,
            )
    else:
        if value <= 0:
            raise ValidationError(
                f"Value must be positive, got {value}",
                field_name,
            )

    return value


def validate_non_empty_string(
    value: Any,
    field_name: Optional[str] = None,
    strip: bool = True,
) -> str:
    """
    Validate that a value is a non-empty string.

    Args:
        value: Value to validate
        field_name: Name of field for error messages
        strip: Whether to strip whitespace before checking

    Returns:
        The validated string

    Raises:
        ValidationError: If validation fails

    Example:
        >>> validate_non_empty_string("hello", "name")
        'hello'
        >>> validate_non_empty_string("", "name")
        ValidationError: [name] String cannot be empty
    """
    if not isinstance(value, str):
        raise ValidationError(
            f"Expected a string, got {type(value).__name__}",
            field_name,
        )

    check_value = value.strip() if strip else value

    if not check_value:
        raise ValidationError(
            "String cannot be empty",
            field_name,
        )

    return value


def validate_in_range(
    value: T,
    min_value: Optional[T] = None,
    max_value: Optional[T] = None,
    field_name: Optional[str] = None,
    inclusive: bool = True,
) -> T:
    """
    Validate that a number is within a range.

    Args:
        value: Number to validate
        min_value: Minimum allowed value (or None for no minimum)
        max_value: Maximum allowed value (or None for no maximum)
        field_name: Name of field for error messages
        inclusive: Whether bounds are inclusive

    Returns:
        The validated value

    Raises:
        ValidationError: If validation fails

    Example:
        >>> validate_in_range(5, 0, 10, "score")
        5
        >>> validate_in_range(15, 0, 10, "score")
        ValidationError: [score] Value must be <= 10, got 15
    """
    if not isinstance(value, (int, float)):
        raise ValidationError(
            f"Expected a number, got {type(value).__name__}",
            field_name,
        )

    if min_value is not None:
        if inclusive and value < min_value:
            raise ValidationError(
                f"Value must be >= {min_value}, got {value}",
                field_name,
            )
        elif not inclusive and value <= min_value:
            raise ValidationError(
                f"Value must be > {min_value}, got {value}",
                field_name,
            )

    if max_value is not None:
        if inclusive and value > max_value:
            raise ValidationError(
                f"Value must be <= {max_value}, got {value}",
                field_name,
            )
        elif not inclusive and value >= max_value:
            raise ValidationError(
                f"Value must be < {max_value}, got {value}",
                field_name,
            )

    return value


def validate_list_not_empty(
    value: Any,
    field_name: Optional[str] = None,
) -> list:
    """
    Validate that a value is a non-empty list.

    Args:
        value: Value to validate
        field_name: Name of field for error messages

    Returns:
        The validated list

    Raises:
        ValidationError: If validation fails
    """
    if not isinstance(value, list):
        raise ValidationError(
            f"Expected a list, got {type(value).__name__}",
            field_name,
        )

    if not value:
        raise ValidationError(
            "List cannot be empty",
            field_name,
        )

    return value


def validate_dict_has_keys(
    value: Any,
    required_keys: list,
    field_name: Optional[str] = None,
) -> dict:
    """
    Validate that a dict contains required keys.

    Args:
        value: Value to validate
        required_keys: List of keys that must be present
        field_name: Name of field for error messages

    Returns:
        The validated dict

    Raises:
        ValidationError: If validation fails
    """
    if not isinstance(value, dict):
        raise ValidationError(
            f"Expected a dict, got {type(value).__name__}",
            field_name,
        )

    missing_keys = [k for k in required_keys if k not in value]

    if missing_keys:
        raise ValidationError(
            f"Missing required keys: {missing_keys}",
            field_name,
        )

    return value