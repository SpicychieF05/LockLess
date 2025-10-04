"""
Logging configuration and utilities for the Lockless system.

This module provides structured logging with security event tracking,
performance monitoring, and privacy-compliant log handling.
"""

import logging
import logging.handlers
import os
import sys
import json
from datetime import datetime
from typing import Dict, Any, Optional
from enum import Enum
from pathlib import Path


class LogLevel(Enum):
    """Log levels for the Lockless system."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class SecurityEventType(Enum):
    """Types of security events to log."""
    AUTHENTICATION_SUCCESS = "auth_success"
    AUTHENTICATION_FAILURE = "auth_failure"
    ENROLLMENT_SUCCESS = "enrollment_success"
    ENROLLMENT_FAILURE = "enrollment_failure"
    LIVENESS_DETECTION_PASS = "liveness_pass"
    LIVENESS_DETECTION_FAIL = "liveness_fail"
    SPOOFING_ATTEMPT = "spoofing_attempt"
    KEY_GENERATION = "key_generation"
    KEY_ACCESS = "key_access"
    TEMPLATE_ACCESS = "template_access"
    UNAUTHORIZED_ACCESS = "unauthorized_access"
    SYSTEM_STARTUP = "system_startup"
    SYSTEM_SHUTDOWN = "system_shutdown"


class PrivacyCompliantFormatter(logging.Formatter):
    """
    Custom formatter that ensures no biometric data is logged.

    Filters out sensitive information while maintaining useful debugging data.
    """

    SENSITIVE_FIELDS = {
        'template', 'embedding', 'biometric_data', 'face_data',
        'password', 'pin', 'key_material', 'private_key',
        'user_image', 'face_image', 'depth_data'
    }

    def format(self, record):
        # Create a copy of the record to avoid modifying the original
        record_copy = logging.makeLogRecord(record.__dict__)

        # Check if the message contains sensitive data
        if hasattr(record_copy, 'args') and record_copy.args:
            filtered_args = []
            for arg in record_copy.args:
                if isinstance(arg, dict):
                    filtered_args.append(self._filter_dict(arg))
                elif isinstance(arg, str) and any(field in arg.lower() for field in self.SENSITIVE_FIELDS):
                    filtered_args.append("[FILTERED]")
                else:
                    filtered_args.append(arg)
            record_copy.args = tuple(filtered_args)

        # Filter the message itself
        message = record_copy.getMessage()
        for field in self.SENSITIVE_FIELDS:
            if field in message.lower():
                # Replace with placeholder while preserving message structure
                message = message.replace(
                    str(record_copy.args), "[FILTERED]") if record_copy.args else message

        record_copy.msg = message
        record_copy.args = ()

        return super().format(record_copy)

    def _filter_dict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Filter sensitive data from dictionary."""
        filtered = {}
        for key, value in data.items():
            if key.lower() in self.SENSITIVE_FIELDS:
                filtered[key] = "[FILTERED]"
            elif isinstance(value, dict):
                filtered[key] = self._filter_dict(value)
            else:
                filtered[key] = value
        return filtered


class SecurityEventLogger:
    """
    Dedicated logger for security events with structured logging.
    """

    def __init__(self, log_dir: Optional[str] = None):
        self.log_dir = log_dir or self._get_default_log_dir()
        self.logger = self._setup_security_logger()

    def _get_default_log_dir(self) -> str:
        """Get default log directory based on platform."""
        if sys.platform == "win32":
            log_dir = os.path.join(os.environ.get(
                "PROGRAMDATA", "C:\\ProgramData"), "Lockless", "logs")
        else:
            log_dir = "/var/log/lockless"

        Path(log_dir).mkdir(parents=True, exist_ok=True)
        return log_dir

    def _setup_security_logger(self) -> logging.Logger:
        """Setup dedicated security event logger."""
        logger = logging.getLogger("lockless.security")
        logger.setLevel(logging.INFO)

        # Prevent duplicate handlers
        if logger.handlers:
            return logger

        # Security log file with rotation
        security_log_file = os.path.join(self.log_dir, "security.log")
        security_handler = logging.handlers.RotatingFileHandler(
            security_log_file,
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=10
        )

        # JSON formatter for structured logging
        security_formatter = logging.Formatter(
            '{"timestamp": "%(asctime)s", "level": "%(levelname)s", '
            '"event": "%(message)s", "module": "%(name)s"}'
        )
        security_handler.setFormatter(security_formatter)
        logger.addHandler(security_handler)

        return logger

    def log_security_event(self, event_type: SecurityEventType,
                           user_id: Optional[str] = None,
                           additional_data: Optional[Dict[str, Any]] = None,
                           success: bool = True):
        """
        Log a security event with structured data.

        Args:
            event_type: Type of security event
            user_id: User identifier (hashed for privacy)
            additional_data: Additional event data (filtered for privacy)
            success: Whether the event was successful
        """
        event_data = {
            "event_type": event_type.value,
            "timestamp": datetime.utcnow().isoformat(),
            "success": success,
            "user_id_hash": self._hash_user_id(user_id) if user_id else None
        }

        if additional_data:
            # Filter sensitive data
            filtered_data = self._filter_sensitive_data(additional_data)
            event_data.update(filtered_data)

        self.logger.info(json.dumps(event_data))

    def _hash_user_id(self, user_id: str) -> str:
        """Hash user ID for privacy-compliant logging."""
        import hashlib
        return hashlib.sha256(user_id.encode('utf-8')).hexdigest()[:16]

    def _filter_sensitive_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Filter sensitive data from additional event data."""
        filtered = {}
        sensitive_keys = PrivacyCompliantFormatter.SENSITIVE_FIELDS

        for key, value in data.items():
            if key.lower() in sensitive_keys:
                filtered[key] = "[FILTERED]"
            elif isinstance(value, dict):
                filtered[key] = self._filter_sensitive_data(value)
            else:
                filtered[key] = value

        return filtered


class PerformanceLogger:
    """
    Logger for performance metrics and monitoring.
    """

    def __init__(self, log_dir: Optional[str] = None):
        self.log_dir = log_dir or self._get_default_log_dir()
        self.logger = self._setup_performance_logger()

    def _get_default_log_dir(self) -> str:
        """Get default log directory."""
        if sys.platform == "win32":
            log_dir = os.path.join(os.environ.get(
                "PROGRAMDATA", "C:\\ProgramData"), "Lockless", "logs")
        else:
            log_dir = "/var/log/lockless"

        Path(log_dir).mkdir(parents=True, exist_ok=True)
        return log_dir

    def _setup_performance_logger(self) -> logging.Logger:
        """Setup performance metrics logger."""
        logger = logging.getLogger("lockless.performance")
        logger.setLevel(logging.INFO)

        if logger.handlers:
            return logger

        # Performance log file
        perf_log_file = os.path.join(self.log_dir, "performance.log")
        perf_handler = logging.handlers.RotatingFileHandler(
            perf_log_file,
            maxBytes=50 * 1024 * 1024,  # 50MB
            backupCount=5
        )

        # CSV-like formatter for easy analysis
        perf_formatter = logging.Formatter(
            '%(asctime)s,%(message)s'
        )
        perf_handler.setFormatter(perf_formatter)
        logger.addHandler(perf_handler)

        return logger

    def log_performance(self, operation: str, duration_ms: float,
                        success: bool = True, additional_metrics: Optional[Dict[str, Any]] = None):
        """
        Log performance metrics.

        Args:
            operation: Name of the operation
            duration_ms: Duration in milliseconds
            success: Whether operation was successful
            additional_metrics: Additional performance metrics
        """
        metrics = {
            "operation": operation,
            "duration_ms": duration_ms,
            "success": success,
            "timestamp": datetime.utcnow().isoformat()
        }

        if additional_metrics:
            metrics.update(additional_metrics)

        # Format as CSV-like string for easy parsing
        metric_parts = [f"{k}={v}" for k, v in metrics.items()]
        self.logger.info(",".join(metric_parts))


# Global logger instances
_security_logger = None
_performance_logger = None


def get_logger(name: str) -> logging.Logger:
    """
    Get a privacy-compliant logger for the specified module.

    Args:
        name: Logger name (usually __name__)

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)

    if not logger.handlers:
        # Setup console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)

        # Use privacy-compliant formatter
        formatter = PrivacyCompliantFormatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # Setup file handler
        log_dir = _get_default_log_dir()
        log_file = os.path.join(log_dir, "lockless.log")

        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=20 * 1024 * 1024,  # 20MB
            backupCount=5
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        logger.setLevel(logging.DEBUG)

    return logger


def get_security_logger() -> SecurityEventLogger:
    """Get the global security event logger."""
    global _security_logger
    if _security_logger is None:
        _security_logger = SecurityEventLogger()
    return _security_logger


def get_performance_logger() -> PerformanceLogger:
    """Get the global performance logger."""
    global _performance_logger
    if _performance_logger is None:
        _performance_logger = PerformanceLogger()
    return _performance_logger


def _get_default_log_dir() -> str:
    """Get default log directory."""
    if sys.platform == "win32":
        log_dir = os.path.join(os.environ.get(
            "PROGRAMDATA", "C:\\ProgramData"), "Lockless", "logs")
    else:
        log_dir = "/var/log/lockless"

    Path(log_dir).mkdir(parents=True, exist_ok=True)
    return log_dir


# Context manager for performance timing
class PerformanceTimer:
    """Context manager for measuring operation performance."""

    def __init__(self, operation_name: str, target_ms: Optional[float] = None):
        self.operation_name = operation_name
        self.target_ms = target_ms
        self.start_time: Optional[float] = None
        self.duration_ms = None
        self.performance_logger = get_performance_logger()

    def __enter__(self):
        import time
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        import time
        end_time = time.perf_counter()
        if self.start_time is not None:
            self.duration_ms = (end_time - self.start_time) * 1000
        else:
            self.duration_ms = 0.0

        success = exc_type is None
        additional_metrics = {}

        if self.target_ms:
            additional_metrics["target_ms"] = self.target_ms
            additional_metrics["within_target"] = self.duration_ms <= self.target_ms

        self.performance_logger.log_performance(
            self.operation_name,
            self.duration_ms,
            success,
            additional_metrics
        )

        # Log warning if exceeds target
        if self.target_ms and self.duration_ms > self.target_ms:
            logger = get_logger(__name__)
            logger.warning(
                f"Operation '{self.operation_name}' exceeded target: "
                f"{self.duration_ms:.2f}ms > {self.target_ms}ms"
            )


# Example usage
if __name__ == "__main__":
    # Test logging
    logger = get_logger(__name__)
    security_logger = get_security_logger()

    logger.info("Testing Lockless logging system")

    # Test security event logging
    security_logger.log_security_event(
        SecurityEventType.AUTHENTICATION_SUCCESS,
        user_id="john_doe",
        additional_data={"method": "face_recognition", "confidence": 0.95}
    )

    # Test performance timing
    with PerformanceTimer("test_operation", target_ms=100):
        import time
        time.sleep(0.05)  # Simulate 50ms operation

    print("Logging test completed successfully!")
