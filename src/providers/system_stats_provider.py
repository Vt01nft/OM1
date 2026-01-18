"""
System Stats Provider for OM1.

This module provides system statistics to OM1 agents,
allowing robots to monitor their own resource usage.
"""

import logging
import platform
import time
from typing import Any, Dict, Optional

import psutil

from .singleton import singleton


@singleton
class SystemStatsProvider:
    """
    Singleton provider for monitoring system statistics.

    This provider allows OM1 agents to access real-time system
    information including CPU, memory, disk, and network stats
    for resource-aware behaviors and self-monitoring.
    """

    DEFAULT_CACHE_TTL = 5  # 5 seconds cache (stats change frequently)

    def __init__(self, cache_ttl: int = DEFAULT_CACHE_TTL):
        """
        Initialize the SystemStatsProvider.

        Parameters
        ----------
        cache_ttl : int, optional
            Cache time-to-live in seconds. Default is 5 seconds.
        """
        self.cache_ttl = cache_ttl

        self._cache: Optional[Dict[str, Any]] = None
        self._cache_time: float = 0
        self._running = False
        self._start_time: float = time.time()

        logging.info("SystemStatsProvider initialized")

    def start(self) -> None:
        """
        Start the SystemStatsProvider.
        """
        self._running = True
        self._start_time = time.time()
        logging.info("SystemStatsProvider started")

    def stop(self) -> None:
        """
        Stop the SystemStatsProvider and clean up resources.
        """
        self._running = False
        self._cache = None
        self._cache_time = 0
        logging.info("SystemStatsProvider stopped")

    def _is_cache_valid(self) -> bool:
        """
        Check if cached stats are still valid.

        Returns
        -------
        bool
            True if cache is valid, False otherwise.
        """
        if self._cache is None:
            return False
        return (time.time() - self._cache_time) < self.cache_ttl

    def _get_cpu_stats(self) -> Dict[str, Any]:
        """
        Get CPU statistics.

        Returns
        -------
        Dict[str, Any]
            CPU statistics dictionary.
        """
        try:
            cpu_freq = psutil.cpu_freq()
            return {
                "percent": psutil.cpu_percent(interval=0.1),
                "count_physical": psutil.cpu_count(logical=False) or 0,
                "count_logical": psutil.cpu_count(logical=True) or 0,
                "frequency_mhz": int(cpu_freq.current) if cpu_freq else 0,
                "frequency_max_mhz": int(cpu_freq.max) if cpu_freq else 0,
            }
        except Exception as e:
            logging.error(f"Error getting CPU stats: {e}")
            return {
                "percent": 0,
                "count_physical": 0,
                "count_logical": 0,
                "frequency_mhz": 0,
                "frequency_max_mhz": 0,
            }

    def _get_memory_stats(self) -> Dict[str, Any]:
        """
        Get memory statistics.

        Returns
        -------
        Dict[str, Any]
            Memory statistics dictionary.
        """
        try:
            mem = psutil.virtual_memory()
            return {
                "total_gb": round(mem.total / (1024**3), 2),
                "available_gb": round(mem.available / (1024**3), 2),
                "used_gb": round(mem.used / (1024**3), 2),
                "percent": mem.percent,
            }
        except Exception as e:
            logging.error(f"Error getting memory stats: {e}")
            return {
                "total_gb": 0,
                "available_gb": 0,
                "used_gb": 0,
                "percent": 0,
            }

    def _get_disk_stats(self) -> Dict[str, Any]:
        """
        Get disk statistics.

        Returns
        -------
        Dict[str, Any]
            Disk statistics dictionary.
        """
        try:
            if platform.system() == "Windows":
                disk = psutil.disk_usage("C:\\")
            else:
                disk = psutil.disk_usage("/")
            return {
                "total_gb": round(disk.total / (1024**3), 2),
                "free_gb": round(disk.free / (1024**3), 2),
                "used_gb": round(disk.used / (1024**3), 2),
                "percent": disk.percent,
            }
        except Exception as e:
            logging.error(f"Error getting disk stats: {e}")
            return {
                "total_gb": 0,
                "free_gb": 0,
                "used_gb": 0,
                "percent": 0,
            }

    def _get_network_stats(self) -> Dict[str, Any]:
        """
        Get network statistics.

        Returns
        -------
        Dict[str, Any]
            Network statistics dictionary.
        """
        try:
            net = psutil.net_io_counters()
            return {
                "bytes_sent_mb": round(net.bytes_sent / (1024**2), 2),
                "bytes_recv_mb": round(net.bytes_recv / (1024**2), 2),
                "packets_sent": net.packets_sent,
                "packets_recv": net.packets_recv,
            }
        except Exception as e:
            logging.error(f"Error getting network stats: {e}")
            return {
                "bytes_sent_mb": 0,
                "bytes_recv_mb": 0,
                "packets_sent": 0,
                "packets_recv": 0,
            }

    def _get_system_info(self) -> Dict[str, Any]:
        """
        Get general system information.

        Returns
        -------
        Dict[str, Any]
            System information dictionary.
        """
        try:
            boot_time = psutil.boot_time()
            uptime_seconds = time.time() - boot_time
            return {
                "platform": platform.system(),
                "platform_version": platform.version(),
                "architecture": platform.machine(),
                "hostname": platform.node(),
                "python_version": platform.python_version(),
                "boot_time": boot_time,
                "uptime_seconds": int(uptime_seconds),
                "uptime_hours": round(uptime_seconds / 3600, 2),
            }
        except Exception as e:
            logging.error(f"Error getting system info: {e}")
            return {
                "platform": "Unknown",
                "platform_version": "Unknown",
                "architecture": "Unknown",
                "hostname": "Unknown",
                "python_version": "Unknown",
                "boot_time": 0,
                "uptime_seconds": 0,
                "uptime_hours": 0,
            }

    def get_stats(self, force_refresh: bool = False) -> Dict[str, Any]:
        """
        Get comprehensive system statistics.

        Parameters
        ----------
        force_refresh : bool, optional
            If True, bypass cache and fetch fresh data. Default is False.

        Returns
        -------
        Dict[str, Any]
            System statistics dictionary containing:
            - cpu: CPU usage and info
            - memory: RAM usage
            - disk: Disk usage
            - network: Network I/O stats
            - system: General system info
            - timestamp: When stats were collected
        """
        if not force_refresh and self._is_cache_valid():
            logging.debug("Returning cached system stats")
            return self._cache

        try:
            stats = {
                "cpu": self._get_cpu_stats(),
                "memory": self._get_memory_stats(),
                "disk": self._get_disk_stats(),
                "network": self._get_network_stats(),
                "system": self._get_system_info(),
                "timestamp": time.time(),
            }

            self._cache = stats
            self._cache_time = time.time()

            logging.debug(
                f"System stats updated: CPU {stats['cpu']['percent']}%, "
                f"Memory {stats['memory']['percent']}%"
            )

            return stats

        except Exception as e:
            logging.error(f"Error collecting system stats: {e}")
            if self._cache:
                return self._cache
            raise

    def get_stats_summary(self) -> str:
        """
        Get a human-readable system stats summary.

        Returns
        -------
        str
            Natural language summary for robot speech.
        """
        try:
            stats = self.get_stats()
            cpu = stats["cpu"]["percent"]
            mem = stats["memory"]["percent"]
            disk = stats["disk"]["percent"]

            return (
                f"System status: CPU usage is {cpu} percent, "
                f"memory usage is {mem} percent, "
                f"and disk usage is {disk} percent."
            )
        except Exception as e:
            logging.error(f"Error generating stats summary: {e}")
            return "I'm unable to get system statistics right now."

    def is_system_healthy(self) -> bool:
        """
        Determine if system resources are at healthy levels.

        Returns
        -------
        bool
            True if all resources below warning thresholds.
        """
        try:
            stats = self.get_stats()
            cpu_ok = stats["cpu"]["percent"] < 90
            mem_ok = stats["memory"]["percent"] < 90
            disk_ok = stats["disk"]["percent"] < 90

            return cpu_ok and mem_ok and disk_ok

        except Exception:
            return True

    def get_warnings(self) -> list:
        """
        Get list of resource warnings.

        Returns
        -------
        list
            List of warning messages for resources above thresholds.
        """
        warnings = []
        try:
            stats = self.get_stats()

            if stats["cpu"]["percent"] >= 90:
                warnings.append(f"High CPU usage: {stats['cpu']['percent']}%")

            if stats["memory"]["percent"] >= 90:
                warnings.append(f"High memory usage: {stats['memory']['percent']}%")

            if stats["disk"]["percent"] >= 90:
                warnings.append(f"Low disk space: {stats['disk']['percent']}% used")

            if stats["memory"]["available_gb"] < 1:
                warnings.append(
                    f"Very low available memory: {stats['memory']['available_gb']}GB"
                )

            if stats["disk"]["free_gb"] < 5:
                warnings.append(
                    f"Very low disk space: {stats['disk']['free_gb']}GB free"
                )

        except Exception as e:
            logging.error(f"Error checking warnings: {e}")

        return warnings

    @property
    def cpu_percent(self) -> float:
        """Get current CPU usage percentage."""
        return self.get_stats().get("cpu", {}).get("percent", 0)

    @property
    def memory_percent(self) -> float:
        """Get current memory usage percentage."""
        return self.get_stats().get("memory", {}).get("percent", 0)

    @property
    def disk_percent(self) -> float:
        """Get current disk usage percentage."""
        return self.get_stats().get("disk", {}).get("percent", 0)

    @property
    def uptime_hours(self) -> float:
        """Get system uptime in hours."""
        return self.get_stats().get("system", {}).get("uptime_hours", 0)
