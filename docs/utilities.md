# OM1 Utility Modules

This document describes the utility modules available in `src/utils/`.

## Overview

The utilities package provides reusable helper functions for common operations:

- **retry** - Network retry logic with exponential backoff
- **text_helpers** - Text formatting and sanitization
- **validators** - Input validation helpers

## Installation

These utilities are included with OM1. No additional installation required.

## Retry Utilities

Handle transient network failures gracefully with automatic retries.

### Basic Usage
```python
from src.utils import with_network_retry

@with_network_retry
def call_llm_api(prompt: str) -> str:
    """This function will automatically retry on network errors."""
    return api_client.complete(prompt)
```

### Custom Retry Configuration
```python
from src.utils import with_retry

@with_retry(max_retries=5, base_delay=2.0, max_delay=120.0)
def call_external_service():
    return service.request()
```

### Available Functions

| Function | Description |
|----------|-------------|
| `retry_sync()` | Retry a synchronous function |
| `retry_async()` | Retry an async function |
| `with_retry()` | Decorator with custom settings |
| `with_network_retry()` | Decorator with network defaults |

## Text Helpers

Common text processing operations.

### Examples
```python
from src.utils import truncate_text, sanitize_filename, format_duration

# Truncate long text
short = truncate_text("Hello World Example", max_length=10)
# Result: "Hello W..."

# Make safe filenames
safe_name = sanitize_filename("my:file/name?.txt")
# Result: "my_file_name_.txt"

# Format durations
duration = format_duration(3661.5)
# Result: "1h 1m 1.5s"
```

### Available Functions

| Function | Description |
|----------|-------------|
| `truncate_text()` | Truncate text with suffix |
| `sanitize_filename()` | Make strings safe for filenames |
| `format_duration()` | Format seconds as human-readable |
| `capitalize_sentences()` | Capitalize first letter of sentences |
| `remove_extra_whitespace()` | Normalize whitespace |
| `extract_numbers()` | Extract numbers from text |

## Validators

Validate inputs with clear error messages.

### Examples
```python
from src.utils import validate_positive_number, validate_in_range

# Validate positive numbers
count = validate_positive_number(user_input, field_name="count")

# Validate ranges
temperature = validate_in_range(
    value=temp,
    min_value=0.0,
    max_value=2.0,
    field_name="temperature"
)
```

### Available Functions

| Function | Description |
|----------|-------------|
| `validate_positive_number()` | Ensure number is positive |
| `validate_non_empty_string()` | Ensure string is not empty |
| `validate_in_range()` | Ensure number is within range |
| `validate_list_not_empty()` | Ensure list has items |
| `validate_dict_has_keys()` | Ensure dict has required keys |

## Error Handling

All validators raise `ValidationError` with descriptive messages:
```python
from src.utils.validators import ValidationError

try:
    validate_positive_number(-5, field_name="count")
except ValidationError as e:
    print(e)  # "[count] Value must be positive, got -5"
```

## Contributing

When adding new utilities:

1. Add the implementation in `src/utils/`
2. Export from `src/utils/__init__.py`
3. Add tests in `tests/test_utils_*.py`
4. Update this documentation