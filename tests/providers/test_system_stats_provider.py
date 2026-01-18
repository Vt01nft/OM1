"""
Tests for SystemStatsProvider.

This module contains comprehensive tests for the SystemStatsProvider
which monitors system resources for OM1 agents.
"""

import time
from unittest.mock import patch

import pytest


class TestSystemStatsProviderInitialization:
    """Tests for SystemStatsProvider initialization."""

    def test_initialization_default_cache_ttl(self, system_stats_provider_class):
        """Test provider initializes with default cache TTL."""
        provider = system_stats_provider_class()
        assert provider.cache_ttl == 5

    def test_initialization_custom_cache_ttl(self, system_stats_provider_class):
        """Test provider initializes with custom cache TTL."""
        provider = system_stats_provider_class(cache_ttl=10)
        assert provider.cache_ttl == 10

    def test_initialization_cache_is_none(self, system_stats_provider_class):
        """Test provider starts with no cached data."""
        provider = system_stats_provider_class()
        assert provider._cache is None

    def test_initialization_not_running(self, system_stats_provider_class):
        """Test provider starts in not running state."""
        provider = system_stats_provider_class()
        assert provider._running is False


class TestSystemStatsProviderLifecycle:
    """Tests for provider start/stop lifecycle."""

    def test_start_sets_running(self, system_stats_provider_class):
        """Test start() sets running flag to True."""
        provider = system_stats_provider_class()
        provider.start()
        assert provider._running is True

    def test_start_sets_start_time(self, system_stats_provider_class):
        """Test start() updates start time."""
        provider = system_stats_provider_class()
        before = time.time()
        provider.start()
        after = time.time()
        assert before <= provider._start_time <= after

    def test_stop_clears_running(self, system_stats_provider_class):
        """Test stop() sets running flag to False."""
        provider = system_stats_provider_class()
        provider.start()
        provider.stop()
        assert provider._running is False

    def test_stop_clears_cache(self, system_stats_provider_class):
        """Test stop() clears cached data."""
        provider = system_stats_provider_class()
        provider._cache = {"test": "data"}
        provider._cache_time = time.time()
        provider.stop()
        assert provider._cache is None
        assert provider._cache_time == 0


class TestCacheValidation:
    """Tests for cache validation logic."""

    def test_cache_invalid_when_none(self, system_stats_provider_class):
        """Test cache is invalid when None."""
        provider = system_stats_provider_class()
        assert provider._is_cache_valid() is False

    def test_cache_valid_within_ttl(self, system_stats_provider_class):
        """Test cache is valid within TTL period."""
        provider = system_stats_provider_class(cache_ttl=10)
        provider._cache = {"test": "data"}
        provider._cache_time = time.time()
        assert provider._is_cache_valid() is True

    def test_cache_invalid_after_ttl(self, system_stats_provider_class):
        """Test cache is invalid after TTL expires."""
        provider = system_stats_provider_class(cache_ttl=1)
        provider._cache = {"test": "data"}
        provider._cache_time = time.time() - 2  # 2 seconds ago
        assert provider._is_cache_valid() is False


class TestCPUStats:
    """Tests for CPU statistics collection."""

    def test_get_cpu_stats_returns_dict(self, system_stats_provider_class):
        """Test _get_cpu_stats returns a dictionary."""
        provider = system_stats_provider_class()
        stats = provider._get_cpu_stats()
        assert isinstance(stats, dict)

    def test_get_cpu_stats_has_required_keys(self, system_stats_provider_class):
        """Test CPU stats contain all required keys."""
        provider = system_stats_provider_class()
        stats = provider._get_cpu_stats()
        required_keys = [
            "percent",
            "count_physical",
            "count_logical",
            "frequency_mhz",
            "frequency_max_mhz",
        ]
        for key in required_keys:
            assert key in stats

    def test_get_cpu_stats_percent_in_range(self, system_stats_provider_class):
        """Test CPU percent is in valid range."""
        provider = system_stats_provider_class()
        stats = provider._get_cpu_stats()
        assert 0 <= stats["percent"] <= 100

    def test_get_cpu_stats_handles_error(self, system_stats_provider_class):
        """Test CPU stats returns defaults on error."""
        provider = system_stats_provider_class()
        with patch("psutil.cpu_percent", side_effect=Exception("Test error")):
            stats = provider._get_cpu_stats()
            assert stats["percent"] == 0


class TestMemoryStats:
    """Tests for memory statistics collection."""

    def test_get_memory_stats_returns_dict(self, system_stats_provider_class):
        """Test _get_memory_stats returns a dictionary."""
        provider = system_stats_provider_class()
        stats = provider._get_memory_stats()
        assert isinstance(stats, dict)

    def test_get_memory_stats_has_required_keys(self, system_stats_provider_class):
        """Test memory stats contain all required keys."""
        provider = system_stats_provider_class()
        stats = provider._get_memory_stats()
        required_keys = ["total_gb", "available_gb", "used_gb", "percent"]
        for key in required_keys:
            assert key in stats

    def test_get_memory_stats_percent_in_range(self, system_stats_provider_class):
        """Test memory percent is in valid range."""
        provider = system_stats_provider_class()
        stats = provider._get_memory_stats()
        assert 0 <= stats["percent"] <= 100

    def test_get_memory_stats_values_positive(self, system_stats_provider_class):
        """Test memory values are positive."""
        provider = system_stats_provider_class()
        stats = provider._get_memory_stats()
        assert stats["total_gb"] >= 0
        assert stats["available_gb"] >= 0
        assert stats["used_gb"] >= 0

    def test_get_memory_stats_handles_error(self, system_stats_provider_class):
        """Test memory stats returns defaults on error."""
        provider = system_stats_provider_class()
        with patch("psutil.virtual_memory", side_effect=Exception("Test error")):
            stats = provider._get_memory_stats()
            assert stats["percent"] == 0


class TestDiskStats:
    """Tests for disk statistics collection."""

    def test_get_disk_stats_returns_dict(self, system_stats_provider_class):
        """Test _get_disk_stats returns a dictionary."""
        provider = system_stats_provider_class()
        stats = provider._get_disk_stats()
        assert isinstance(stats, dict)

    def test_get_disk_stats_has_required_keys(self, system_stats_provider_class):
        """Test disk stats contain all required keys."""
        provider = system_stats_provider_class()
        stats = provider._get_disk_stats()
        required_keys = ["total_gb", "free_gb", "used_gb", "percent"]
        for key in required_keys:
            assert key in stats

    def test_get_disk_stats_percent_in_range(self, system_stats_provider_class):
        """Test disk percent is in valid range."""
        provider = system_stats_provider_class()
        stats = provider._get_disk_stats()
        assert 0 <= stats["percent"] <= 100

    def test_get_disk_stats_handles_error(self, system_stats_provider_class):
        """Test disk stats returns defaults on error."""
        provider = system_stats_provider_class()
        with patch("psutil.disk_usage", side_effect=Exception("Test error")):
            stats = provider._get_disk_stats()
            assert stats["percent"] == 0


class TestNetworkStats:
    """Tests for network statistics collection."""

    def test_get_network_stats_returns_dict(self, system_stats_provider_class):
        """Test _get_network_stats returns a dictionary."""
        provider = system_stats_provider_class()
        stats = provider._get_network_stats()
        assert isinstance(stats, dict)

    def test_get_network_stats_has_required_keys(self, system_stats_provider_class):
        """Test network stats contain all required keys."""
        provider = system_stats_provider_class()
        stats = provider._get_network_stats()
        required_keys = [
            "bytes_sent_mb",
            "bytes_recv_mb",
            "packets_sent",
            "packets_recv",
        ]
        for key in required_keys:
            assert key in stats

    def test_get_network_stats_values_non_negative(self, system_stats_provider_class):
        """Test network values are non-negative."""
        provider = system_stats_provider_class()
        stats = provider._get_network_stats()
        assert stats["bytes_sent_mb"] >= 0
        assert stats["bytes_recv_mb"] >= 0
        assert stats["packets_sent"] >= 0
        assert stats["packets_recv"] >= 0

    def test_get_network_stats_handles_error(self, system_stats_provider_class):
        """Test network stats returns defaults on error."""
        provider = system_stats_provider_class()
        with patch("psutil.net_io_counters", side_effect=Exception("Test error")):
            stats = provider._get_network_stats()
            assert stats["bytes_sent_mb"] == 0


class TestSystemInfo:
    """Tests for system information collection."""

    def test_get_system_info_returns_dict(self, system_stats_provider_class):
        """Test _get_system_info returns a dictionary."""
        provider = system_stats_provider_class()
        info = provider._get_system_info()
        assert isinstance(info, dict)

    def test_get_system_info_has_required_keys(self, system_stats_provider_class):
        """Test system info contains all required keys."""
        provider = system_stats_provider_class()
        info = provider._get_system_info()
        required_keys = [
            "platform",
            "platform_version",
            "architecture",
            "hostname",
            "python_version",
            "boot_time",
            "uptime_seconds",
            "uptime_hours",
        ]
        for key in required_keys:
            assert key in info

    def test_get_system_info_platform_not_empty(self, system_stats_provider_class):
        """Test platform is not empty."""
        provider = system_stats_provider_class()
        info = provider._get_system_info()
        assert info["platform"] != ""

    def test_get_system_info_handles_error(self, system_stats_provider_class):
        """Test system info returns defaults on error."""
        provider = system_stats_provider_class()
        with patch("psutil.boot_time", side_effect=Exception("Test error")):
            info = provider._get_system_info()
            assert info["platform"] == "Unknown"


class TestGetStats:
    """Tests for comprehensive stats collection."""

    def test_get_stats_returns_dict(self, system_stats_provider_class):
        """Test get_stats returns a dictionary."""
        provider = system_stats_provider_class()
        stats = provider.get_stats()
        assert isinstance(stats, dict)

    def test_get_stats_has_all_sections(self, system_stats_provider_class):
        """Test get_stats includes all stat sections."""
        provider = system_stats_provider_class()
        stats = provider.get_stats()
        required_sections = ["cpu", "memory", "disk", "network", "system", "timestamp"]
        for section in required_sections:
            assert section in stats

    def test_get_stats_uses_cache(self, system_stats_provider_class):
        """Test get_stats uses cached data when valid."""
        provider = system_stats_provider_class(cache_ttl=60)
        first_stats = provider.get_stats()
        first_timestamp = first_stats["timestamp"]

        # Second call should return cached data
        second_stats = provider.get_stats()
        assert second_stats["timestamp"] == first_timestamp

    def test_get_stats_force_refresh_bypasses_cache(self, system_stats_provider_class):
        """Test force_refresh bypasses cache."""
        provider = system_stats_provider_class(cache_ttl=60)
        first_stats = provider.get_stats()

        time.sleep(0.1)
        second_stats = provider.get_stats(force_refresh=True)
        assert second_stats["timestamp"] > first_stats["timestamp"]

    def test_get_stats_returns_cached_on_error(self, system_stats_provider_class):
        """Test get_stats returns cached data on error."""
        provider = system_stats_provider_class()
        # First get valid stats
        valid_stats = provider.get_stats()

        # Force cache to be "stale" but keep the data
        provider._cache_time = 0

        # Mock an error
        with patch.object(provider, "_get_cpu_stats", side_effect=Exception("Error")):
            # Should return cached data
            stats = provider.get_stats()
            assert stats == valid_stats


class TestStatsSummary:
    """Tests for stats summary generation."""

    def test_get_stats_summary_returns_string(self, system_stats_provider_class):
        """Test get_stats_summary returns a string."""
        provider = system_stats_provider_class()
        summary = provider.get_stats_summary()
        assert isinstance(summary, str)

    def test_get_stats_summary_contains_cpu(self, system_stats_provider_class):
        """Test summary mentions CPU."""
        provider = system_stats_provider_class()
        summary = provider.get_stats_summary()
        assert "CPU" in summary

    def test_get_stats_summary_contains_memory(self, system_stats_provider_class):
        """Test summary mentions memory."""
        provider = system_stats_provider_class()
        summary = provider.get_stats_summary()
        assert "memory" in summary

    def test_get_stats_summary_contains_disk(self, system_stats_provider_class):
        """Test summary mentions disk."""
        provider = system_stats_provider_class()
        summary = provider.get_stats_summary()
        assert "disk" in summary

    def test_get_stats_summary_handles_error(self, system_stats_provider_class):
        """Test summary handles errors gracefully."""
        provider = system_stats_provider_class()
        with patch.object(provider, "get_stats", side_effect=Exception("Error")):
            summary = provider.get_stats_summary()
            assert "unable" in summary.lower()


class TestHealthCheck:
    """Tests for system health checking."""

    def test_is_system_healthy_returns_bool(self, system_stats_provider_class):
        """Test is_system_healthy returns boolean."""
        provider = system_stats_provider_class()
        result = provider.is_system_healthy()
        assert isinstance(result, bool)

    def test_is_system_healthy_true_when_all_ok(self, system_stats_provider_class):
        """Test health check passes when all stats below threshold."""
        provider = system_stats_provider_class()
        mock_stats = {
            "cpu": {"percent": 50},
            "memory": {"percent": 50},
            "disk": {"percent": 50},
        }
        with patch.object(provider, "get_stats", return_value=mock_stats):
            assert provider.is_system_healthy() is True

    def test_is_system_healthy_false_high_cpu(self, system_stats_provider_class):
        """Test health check fails on high CPU."""
        provider = system_stats_provider_class()
        mock_stats = {
            "cpu": {"percent": 95},
            "memory": {"percent": 50},
            "disk": {"percent": 50},
        }
        with patch.object(provider, "get_stats", return_value=mock_stats):
            assert provider.is_system_healthy() is False

    def test_is_system_healthy_false_high_memory(self, system_stats_provider_class):
        """Test health check fails on high memory."""
        provider = system_stats_provider_class()
        mock_stats = {
            "cpu": {"percent": 50},
            "memory": {"percent": 95},
            "disk": {"percent": 50},
        }
        with patch.object(provider, "get_stats", return_value=mock_stats):
            assert provider.is_system_healthy() is False

    def test_is_system_healthy_false_high_disk(self, system_stats_provider_class):
        """Test health check fails on high disk."""
        provider = system_stats_provider_class()
        mock_stats = {
            "cpu": {"percent": 50},
            "memory": {"percent": 50},
            "disk": {"percent": 95},
        }
        with patch.object(provider, "get_stats", return_value=mock_stats):
            assert provider.is_system_healthy() is False

    def test_is_system_healthy_returns_true_on_error(self, system_stats_provider_class):
        """Test health check defaults to True on error."""
        provider = system_stats_provider_class()
        with patch.object(provider, "get_stats", side_effect=Exception("Error")):
            assert provider.is_system_healthy() is True


class TestWarnings:
    """Tests for resource warning generation."""

    def test_get_warnings_returns_list(self, system_stats_provider_class):
        """Test get_warnings returns a list."""
        provider = system_stats_provider_class()
        warnings = provider.get_warnings()
        assert isinstance(warnings, list)

    def test_get_warnings_empty_when_healthy(self, system_stats_provider_class):
        """Test no warnings when system is healthy."""
        provider = system_stats_provider_class()
        mock_stats = {
            "cpu": {"percent": 50},
            "memory": {"percent": 50, "available_gb": 8},
            "disk": {"percent": 50, "free_gb": 100},
        }
        with patch.object(provider, "get_stats", return_value=mock_stats):
            warnings = provider.get_warnings()
            assert len(warnings) == 0

    def test_get_warnings_high_cpu(self, system_stats_provider_class):
        """Test warning generated for high CPU."""
        provider = system_stats_provider_class()
        mock_stats = {
            "cpu": {"percent": 95},
            "memory": {"percent": 50, "available_gb": 8},
            "disk": {"percent": 50, "free_gb": 100},
        }
        with patch.object(provider, "get_stats", return_value=mock_stats):
            warnings = provider.get_warnings()
            assert any("CPU" in w for w in warnings)

    def test_get_warnings_high_memory(self, system_stats_provider_class):
        """Test warning generated for high memory."""
        provider = system_stats_provider_class()
        mock_stats = {
            "cpu": {"percent": 50},
            "memory": {"percent": 95, "available_gb": 8},
            "disk": {"percent": 50, "free_gb": 100},
        }
        with patch.object(provider, "get_stats", return_value=mock_stats):
            warnings = provider.get_warnings()
            assert any("memory" in w.lower() for w in warnings)

    def test_get_warnings_low_disk(self, system_stats_provider_class):
        """Test warning generated for low disk space."""
        provider = system_stats_provider_class()
        mock_stats = {
            "cpu": {"percent": 50},
            "memory": {"percent": 50, "available_gb": 8},
            "disk": {"percent": 95, "free_gb": 100},
        }
        with patch.object(provider, "get_stats", return_value=mock_stats):
            warnings = provider.get_warnings()
            assert any("disk" in w.lower() for w in warnings)

    def test_get_warnings_very_low_memory(self, system_stats_provider_class):
        """Test warning generated for very low available memory."""
        provider = system_stats_provider_class()
        mock_stats = {
            "cpu": {"percent": 50},
            "memory": {"percent": 50, "available_gb": 0.5},
            "disk": {"percent": 50, "free_gb": 100},
        }
        with patch.object(provider, "get_stats", return_value=mock_stats):
            warnings = provider.get_warnings()
            assert any("low available memory" in w.lower() for w in warnings)

    def test_get_warnings_very_low_disk_space(self, system_stats_provider_class):
        """Test warning generated for very low disk space."""
        provider = system_stats_provider_class()
        mock_stats = {
            "cpu": {"percent": 50},
            "memory": {"percent": 50, "available_gb": 8},
            "disk": {"percent": 50, "free_gb": 2},
        }
        with patch.object(provider, "get_stats", return_value=mock_stats):
            warnings = provider.get_warnings()
            assert any("low disk space" in w.lower() for w in warnings)

    def test_get_warnings_multiple(self, system_stats_provider_class):
        """Test multiple warnings can be generated."""
        provider = system_stats_provider_class()
        mock_stats = {
            "cpu": {"percent": 95},
            "memory": {"percent": 95, "available_gb": 0.5},
            "disk": {"percent": 95, "free_gb": 2},
        }
        with patch.object(provider, "get_stats", return_value=mock_stats):
            warnings = provider.get_warnings()
            assert len(warnings) >= 3


class TestProperties:
    """Tests for convenience properties."""

    def test_cpu_percent_property(self, system_stats_provider_class):
        """Test cpu_percent property returns float."""
        provider = system_stats_provider_class()
        assert isinstance(provider.cpu_percent, (int, float))
        assert 0 <= provider.cpu_percent <= 100

    def test_memory_percent_property(self, system_stats_provider_class):
        """Test memory_percent property returns float."""
        provider = system_stats_provider_class()
        assert isinstance(provider.memory_percent, (int, float))
        assert 0 <= provider.memory_percent <= 100

    def test_disk_percent_property(self, system_stats_provider_class):
        """Test disk_percent property returns float."""
        provider = system_stats_provider_class()
        assert isinstance(provider.disk_percent, (int, float))
        assert 0 <= provider.disk_percent <= 100

    def test_uptime_hours_property(self, system_stats_provider_class):
        """Test uptime_hours property returns float."""
        provider = system_stats_provider_class()
        assert isinstance(provider.uptime_hours, (int, float))
        assert provider.uptime_hours >= 0


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_zero_cache_ttl(self, system_stats_provider_class):
        """Test provider works with zero cache TTL."""
        provider = system_stats_provider_class(cache_ttl=0)
        stats1 = provider.get_stats()
        stats2 = provider.get_stats()
        # With 0 TTL, cache is always invalid
        assert stats2["timestamp"] >= stats1["timestamp"]

    def test_large_cache_ttl(self, system_stats_provider_class):
        """Test provider works with large cache TTL."""
        provider = system_stats_provider_class(cache_ttl=3600)
        stats = provider.get_stats()
        assert stats is not None

    def test_multiple_get_stats_calls(self, system_stats_provider_class):
        """Test multiple rapid get_stats calls."""
        provider = system_stats_provider_class()
        for _ in range(10):
            stats = provider.get_stats()
            assert stats is not None

    def test_start_stop_start_cycle(self, system_stats_provider_class):
        """Test start/stop/start cycle works correctly."""
        provider = system_stats_provider_class()
        provider.start()
        provider.stop()
        provider.start()
        assert provider._running is True


# Pytest fixture for the provider class
@pytest.fixture
def system_stats_provider_class():
    """
    Fixture that provides the SystemStatsProvider class.

    Returns a fresh class for each test to avoid singleton issues.
    """
    # Import the actual module
    import sys
    import types

    # Create mock singleton
    mock_singleton = types.ModuleType("providers.singleton")

    def singleton(cls):
        return cls

    mock_singleton.singleton = singleton
    sys.modules["providers.singleton"] = mock_singleton

    # Load the module
    with open("src/providers/system_stats_provider.py", "r") as f:
        code = f.read()

    code = code.replace(
        "from .singleton import singleton", "from providers.singleton import singleton"
    )

    # Create a module namespace
    module_ns = {"__name__": "system_stats_provider"}
    exec(code, module_ns)

    return module_ns["SystemStatsProvider"]
