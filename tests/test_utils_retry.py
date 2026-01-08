"""
Unit tests for the retry utilities module.

Run with: uv run pytest tests/test_utils_retry.py -v
"""

import pytest
from unittest.mock import MagicMock

from src.utils.retry import (
    RetryExhaustedError,
    calculate_backoff_delay,
    retry_sync,
    with_retry,
    with_network_retry,
)


class TestCalculateBackoffDelay:
    """Tests for backoff delay calculation."""

    def test_first_attempt_returns_base_delay(self):
        """First attempt should return base delay."""
        delay = calculate_backoff_delay(attempt=0, base_delay=1.0, jitter=False)
        assert delay == 1.0

    def test_exponential_growth(self):
        """Delay should grow exponentially."""
        delays = [
            calculate_backoff_delay(attempt=i, base_delay=1.0, jitter=False)
            for i in range(4)
        ]
        assert delays == [1.0, 2.0, 4.0, 8.0]

    def test_max_delay_cap(self):
        """Delay should not exceed max_delay."""
        delay = calculate_backoff_delay(
            attempt=10,
            base_delay=1.0,
            max_delay=30.0,
            jitter=False,
        )
        assert delay == 30.0

    def test_jitter_adds_randomness(self):
        """Jitter should keep delay within expected bounds."""
        for _ in range(10):
            delay = calculate_backoff_delay(
                attempt=1,
                base_delay=1.0,
                max_delay=60.0,
                jitter=True,
            )
            # Base is 2.0, jitter adds up to 25%
            assert 2.0 <= delay <= 2.5


class TestRetrySyncFunction:
    """Tests for synchronous retry function."""

    def test_success_on_first_try(self):
        """Should return immediately on success."""
        mock_func = MagicMock(return_value="success")

        result = retry_sync(mock_func, max_retries=3)

        assert result == "success"
        assert mock_func.call_count == 1

    def test_success_after_retries(self):
        """Should succeed after transient failures."""
        mock_func = MagicMock(
            side_effect=[ConnectionError("fail"), ConnectionError("fail"), "success"]
        )

        result = retry_sync(mock_func, max_retries=3, base_delay=0.01)

        assert result == "success"
        assert mock_func.call_count == 3

    def test_exhausted_retries_raises_error(self):
        """Should raise RetryExhaustedError when all retries fail."""
        mock_func = MagicMock(side_effect=ConnectionError("always fail"))

        with pytest.raises(RetryExhaustedError) as exc_info:
            retry_sync(mock_func, max_retries=2, base_delay=0.01)

        assert exc_info.value.attempts == 3
        assert isinstance(exc_info.value.last_exception, ConnectionError)

    def test_non_retryable_exception_propagates(self):
        """Non-retryable exceptions should propagate immediately."""
        mock_func = MagicMock(side_effect=ValueError("invalid"))

        with pytest.raises(ValueError):
            retry_sync(
                mock_func,
                max_retries=3,
                retry_exceptions=(ConnectionError,),
            )

        assert mock_func.call_count == 1


class TestWithRetryDecorator:
    """Tests for the retry decorator."""

    def test_decorator_on_sync_function(self):
        """Decorator should work on sync functions."""
        call_count = [0]

        @with_retry(max_retries=3, base_delay=0.01)
        def flaky_func():
            call_count[0] += 1
            if call_count[0] < 2:
                raise ConnectionError("fail")
            return "success"

        result = flaky_func()

        assert result == "success"
        assert call_count[0] == 2

    def test_decorator_preserves_metadata(self):
        """Decorator should preserve function name and docstring."""

        @with_retry(max_retries=3)
        def my_function():
            """My docstring."""
            return "result"

        assert my_function.__name__ == "my_function"
        assert my_function.__doc__ == "My docstring."


class TestNetworkRetryDecorator:
    """Tests for network retry convenience decorator."""

    def test_with_network_retry_works(self):
        """with_network_retry should retry on network errors."""
        call_count = [0]

        @with_network_retry
        def network_call():
            call_count[0] += 1
            if call_count[0] < 2:
                raise ConnectionError("fail")
            return "success"

        result = network_call()

        assert result == "success"
        assert call_count[0] == 2


class TestRetryExhaustedError:
    """Tests for RetryExhaustedError exception."""

    def test_error_contains_attempts(self):
        """Error should contain attempt count."""
        error = RetryExhaustedError("failed", attempts=5)
        assert error.attempts == 5

    def test_error_contains_last_exception(self):
        """Error should contain the last exception."""
        original = ConnectionError("original")
        error = RetryExhaustedError("failed", attempts=3, last_exception=original)
        assert error.last_exception is original

    def test_error_message(self):
        """Error should have informative message."""
        error = RetryExhaustedError("Operation failed", attempts=3)
        assert "Operation failed" in str(error)