"""
Enhanced Diagnostics System for OpenMind OM1
"""

import logging
import json
import traceback
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from enum import Enum


class ErrorSeverity(Enum):
    """Severity levels for diagnostics."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class OM1DiagnosticsSystem:
    """Comprehensive diagnostics and error tracking for OM1."""
    
    def __init__(self, log_dir: str = "logs/om1"):
        """Initialize diagnostics system."""
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.error_history: List[Dict[str, Any]] = []
        self.start_time = time.time()
        
        self._setup_logging()
        self.logger = logging.getLogger("OM1Diagnostics")
        self.logger.info("OM1 Diagnostics System initialized")
    
    def _setup_logging(self):
        """Configure logging."""
        from logging.handlers import RotatingFileHandler
        
        formatter = logging.Formatter(
            '[%(asctime)s] %(levelname)s [%(name)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        file_handler = RotatingFileHandler(
            self.log_dir / "om1_diagnostics.log",
            maxBytes=50 * 1024 * 1024,
            backupCount=5
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.DEBUG)
        root_logger.addHandler(file_handler)
        root_logger.addHandler(console_handler)
    
    def log_diagnostic(
        self,
        severity: ErrorSeverity,
        component: str,
        message: str,
        context: Optional[Dict[str, Any]] = None,
        exception: Optional[Exception] = None
    ):
        """Log a diagnostic message with context."""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'severity': severity.value,
            'component': component,
            'message': message,
            'context': context or {},
        }
        
        if exception:
            log_entry['exception'] = {
                'type': type(exception).__name__,
                'message': str(exception),
                'traceback': traceback.format_exc()
            }
        
        self.error_history.append(log_entry)
        
        logger = logging.getLogger(f"OM1.{component}")
        log_func = getattr(logger, severity.value.lower())
        
        log_message = f"{message}"
        if context:
            log_message += f" | Context: {json.dumps(context)}"
        if exception:
            log_message += f" | Exception: {exception}"
        
        log_func(log_message)
    
    def export_diagnostics(self, output_path: Optional[str] = None) -> str:
        """Export diagnostic data to JSON."""
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = str(self.log_dir / f"diagnostics_{timestamp}.json")
        
        diagnostic_data = {
            'export_time': datetime.now().isoformat(),
            'error_history': self.error_history[-100:],
            'uptime_seconds': time.time() - self.start_time
        }
        
        with open(output_path, 'w') as f:
            json.dump(diagnostic_data, f, indent=2)
        
        self.logger.info(f"Diagnostics exported to {output_path}")
        return output_path


# Demo
if __name__ == "__main__":
    print("=== OM1 Enhanced Diagnostics System Demo ===\n")
    
    diagnostics = OM1DiagnosticsSystem()
    
    # Log messages
    print("1. Logging diagnostic messages...")
    diagnostics.log_diagnostic(
        ErrorSeverity.INFO,
        "TestComponent",
        "System initialized successfully"
    )
    
    diagnostics.log_diagnostic(
        ErrorSeverity.WARNING,
        "SensorModule",
        "Camera frame rate dropped",
        context={'fps': 15, 'expected': 30}
    )
    
    # Simulate error
    try:
        raise ValueError("Test error for diagnostics")
    except Exception as e:
        diagnostics.log_diagnostic(
            ErrorSeverity.ERROR,
            "TestComponent",
            "Test error occurred",
            exception=e
        )
    
    # Export
    print("\n2. Exporting diagnostics...")
    export_path = diagnostics.export_diagnostics()
    print(f"  Exported to: {export_path}")
    
    print("\n✅ Diagnostics system demo complete!")