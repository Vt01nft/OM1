"""
Text formatting and sanitization utilities for OM1.

This module provides helper functions for common text operations
used throughout the OM1 system.
"""

import re
from typing import List


def truncate_text(
    text: str,
    max_length: int,
    suffix: str = "...",
) -> str:
    """
    Truncate text to a maximum length with optional suffix.

    Args:
        text: The text to truncate
        max_length: Maximum length of result (including suffix)
        suffix: String to append when truncating

    Returns:
        Truncated text with suffix if needed

    Example:
        >>> truncate_text("Hello World", 8)
        'Hello...'
    """
    if not text:
        return ""
    if max_length <= 0:
        return ""
    if len(text) <= max_length:
        return text
    if max_length <= len(suffix):
        return text[:max_length]
    return text[: max_length - len(suffix)] + suffix


def sanitize_filename(
    filename: str,
    replacement: str = "_",
    max_length: int = 255,
) -> str:
    """
    Sanitize a string to be safe for use as a filename.

    Args:
        filename: The string to sanitize
        replacement: Character to replace invalid chars with
        max_length: Maximum filename length

    Returns:
        Sanitized filename string

    Example:
        >>> sanitize_filename("my:file/name?.txt")
        'my_file_name_.txt'
    """
    if not filename:
        return "unnamed"

    invalid_chars = r'[<>:"/\\|?*\x00-\x1f]'
    sanitized = re.sub(invalid_chars, replacement, filename)
    sanitized = sanitized.strip(" .")

    if replacement:
        sanitized = re.sub(f"{re.escape(replacement)}+", replacement, sanitized)

    if len(sanitized) > max_length:
        sanitized = sanitized[:max_length].rstrip(" .")

    return sanitized if sanitized else "unnamed"


def format_duration(seconds: float, precision: int = 1) -> str:
    """
    Format a duration in seconds to a human-readable string.

    Args:
        seconds: Duration in seconds
        precision: Decimal places for seconds

    Returns:
        Formatted duration string

    Example:
        >>> format_duration(3661.5)
        '1h 1m 1.5s'
    """
    if seconds < 0:
        return f"-{format_duration(-seconds, precision)}"

    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60

    if hours > 0:
        return f"{hours}h {minutes}m {secs:.{precision}f}s"
    elif minutes > 0:
        return f"{minutes}m {secs:.{precision}f}s"
    else:
        return f"{secs:.{precision}f}s"


def capitalize_sentences(text: str) -> str:
    """
    Capitalize the first letter of each sentence in text.

    Args:
        text: Input text with sentences

    Returns:
        Text with capitalized sentences

    Example:
        >>> capitalize_sentences("hello world. how are you?")
        'Hello world. How are you?'
    """
    if not text:
        return ""

    sentence_pattern = r"([.!?]+\s*)"
    parts = re.split(sentence_pattern, text)

    result = []
    capitalize_next = True

    for part in parts:
        if not part:
            continue
        if capitalize_next and part[0].isalpha():
            part = part[0].upper() + part[1:]
        result.append(part)
        capitalize_next = bool(re.match(r"[.!?]+\s*$", part))

    return "".join(result)


def remove_extra_whitespace(text: str) -> str:
    """
    Remove extra whitespace from text.

    Args:
        text: Input text

    Returns:
        Text with normalized whitespace

    Example:
        >>> remove_extra_whitespace("  hello   world  ")
        'hello world'
    """
    if not text:
        return ""
    return " ".join(text.split())


def extract_numbers(text: str) -> List[float]:
    """
    Extract all numbers from a text string.

    Args:
        text: Input text containing numbers

    Returns:
        List of numbers found in text

    Example:
        >>> extract_numbers("I have 3 apples and 2.5 oranges")
        [3.0, 2.5]
    """
    if not text:
        return []
    pattern = r"-?\d+\.?\d*"
    matches = re.findall(pattern, text)
    return [float(m) for m in matches]