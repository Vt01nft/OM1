"""
Network retry utilities for OM1.

This module provides retry logic with exponential backoff for network
operations, improving reliability of LLM API calls and other external services.

Problem Statement:
    Network requests to LLM providers (OpenAI, Anthropic, etc.) can fail
    transiently due to rate limits, timeouts, or temporary server issues.
    Without retry logic, these failures crash the agent unnecessarily.

Solution:
    Provide a reusable retry decorator and function with configurable
    exponential backoff, jitter, and exception filtering.

Example:
    >>> @with_network_retry
    ... def call_openai(prompt: str) -> str:
    ...     return client.complete(prompt)
    >>> result = call_openai("Hello")  # Auto-retries on failure
"""

import asyncio
import functools
import logging
import random
import time
from typing import Any, Callable, Optional, Tuple, Type, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")
F = TypeVar("F", bound=Callable[..., Any])

DEFAULT_RETRY_EXCEPTIONS: Tuple[Type[Exception], ...] = (
    ConnectionError,
    TimeoutError,
    OSError,
)


class RetryExhaustedError(Exception):
    """Raised when all retry attempts have been exhausted."""

    def __init__(
        self,
        message: str,
        attempts: int,
        last_exception: Optional[Exception] = None,
    ) -> None:
        super().__init__(message)
        self.attempts = attempts
        self.last_exception = last_exception


def calculate_backoff_delay(
    attempt: int,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    jitter: bool = True,
) -> float:
    """
    Calculate delay for exponential backoff with optional jitter.

    Args:
        attempt: Current attempt number (0-indexed)
        base_delay: Initial delay in seconds
        max_delay: Maximum delay cap in seconds
        exponential_base: Base for exponential calculation
        jitter: Whether to add random jitter

    Returns:
        Calculated delay in seconds
    """
    delay = min(base_delay * (exponential_base**attempt), max_delay)
    if jitter:
        delay += delay * 0.25 * random.random()
    return delay


def retry_sync(
    func: Callable[[], T],
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    retry_exceptions: Tuple[Type[Exception], ...] = DEFAULT_RETRY_EXCEPTIONS,
    on_retry: Optional[Callable[[int, Exception, float], None]] = None,
) -> T:
    """
    Execute a synchronous function with retry logic.

    Args:
        func: Function to execute
        max_retries: Maximum retry attempts
        base_delay: Initial delay between retries
        max_delay: Maximum delay between retries
        retry_exceptions: Exception types to retry on
        on_retry: Callback before each retry

    Returns:
        Result of successful function call

    Raises:
        RetryExhaustedError: If all retries fail
    """
    last_exception: Optional[Exception] = None

    for attempt in range(max_retries + 1):
        try:
            return func()
        except retry_exceptions as e:
            last_exception = e
            if attempt < max_retries:
                delay = calculate_backoff_delay(attempt, base_delay, max_delay)
                logger.warning(
                    "Attempt %d/%d failed: %s. Retrying in %.2fs...",
                    attempt + 1,
                    max_retries + 1,
                    str(e),
                    delay,
                )
                if on_retry:
                    on_retry(attempt, e, delay)
                time.sleep(delay)
            else:
                logger.error("All %d attempts failed.", max_retries + 1)

    raise RetryExhaustedError(
        f"Operation failed after {max_retries + 1} attempts",
        attempts=max_retries + 1,
        last_exception=last_exception,
    )


async def retry_async(
    func: Callable[[], T],
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    retry_exceptions: Tuple[Type[Exception], ...] = DEFAULT_RETRY_EXCEPTIONS,
    on_retry: Optional[Callable[[int, Exception, float], None]] = None,
) -> T:
    """
    Execute an async function with retry logic.

    Args:
        func: Async function to execute
        max_retries: Maximum retry attempts
        base_delay: Initial delay between retries
        max_delay: Maximum delay between retries
        retry_exceptions: Exception types to retry on
        on_retry: Callback before each retry

    Returns:
        Result of successful function call

    Raises:
        RetryExhaustedError: If all retries fail
    """
    last_exception: Optional[Exception] = None

    for attempt in range(max_retries + 1):
        try:
            result = func()
            if asyncio.iscoroutine(result):
                return await result
            return result
        except retry_exceptions as e:
            last_exception = e
            if attempt < max_retries:
                delay = calculate_backoff_delay(attempt, base_delay, max_delay)
                logger.warning(
                    "Attempt %d/%d failed: %s. Retrying in %.2fs...",
                    attempt + 1,
                    max_retries + 1,
                    str(e),
                    delay,
                )
                if on_retry:
                    on_retry(attempt, e, delay)
                await asyncio.sleep(delay)
            else:
                logger.error("All %d attempts failed.", max_retries + 1)

    raise RetryExhaustedError(
        f"Operation failed after {max_retries + 1} attempts",
        attempts=max_retries + 1,
        last_exception=last_exception,
    )


def with_retry(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    retry_exceptions: Tuple[Type[Exception], ...] = DEFAULT_RETRY_EXCEPTIONS,
) -> Callable[[F], F]:
    """
    Decorator to add retry logic to a function.

    Args:
        max_retries: Maximum retry attempts
        base_delay: Initial delay between retries
        max_delay: Maximum delay between retries
        retry_exceptions: Exception types to retry on

    Returns:
        Decorated function with retry logic
    """

    def decorator(func: F) -> F:
        if asyncio.iscoroutinefunction(func):

            @functools.wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                return await retry_async(
                    func=lambda: func(*args, **kwargs),
                    max_retries=max_retries,
                    base_delay=base_delay,
                    max_delay=max_delay,
                    retry_exceptions=retry_exceptions,
                )

            return async_wrapper  # type: ignore
        else:

            @functools.wraps(func)
            def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                return retry_sync(
                    func=lambda: func(*args, **kwargs),
                    max_retries=max_retries,
                    base_delay=base_delay,
                    max_delay=max_delay,
                    retry_exceptions=retry_exceptions,
                )

            return sync_wrapper  # type: ignore

    return decorator


def with_network_retry(func: F) -> F:
    """
    Decorator for network operations with sensible defaults.

    Retries on ConnectionError, TimeoutError, and OSError with
    exponential backoff starting at 1 second, max 60 seconds.
    """
    return with_retry(
        max_retries=3,
        base_delay=1.0,
        max_delay=60.0,
        retry_exceptions=(ConnectionError, TimeoutError, OSError),
    )(func)


def with_api_retry(func: F) -> F:
    """
    Decorator for API calls with rate limit handling.

    Uses longer delays and more retries for API rate limits.
    """
    return with_retry(
        max_retries=5,
        base_delay=2.0,
        max_delay=120.0,
        retry_exceptions=(ConnectionError, TimeoutError, OSError),
    )(func)