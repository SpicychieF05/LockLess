"""
Custom exceptions for the Lockless biometric authentication system.

This module defines all custom exception classes used throughout the system
for proper error handling and logging.
"""


from typing import Optional


class LocklessError(Exception):
    """Base exception class for all Lockless-specific errors."""

    def __init__(self, message: str, error_code: Optional[str] = None):
        super().__init__(message)
        self.message = message
        self.error_code = error_code


# Security and Encryption Exceptions
class SecurityError(LocklessError):
    """Base class for security-related errors."""
    pass


class EncryptionError(SecurityError):
    """Raised when encryption/decryption operations fail."""
    pass


class KeyDerivationError(SecurityError):
    """Raised when key derivation fails."""
    pass


class KeyManagementError(SecurityError):
    """Raised when key management operations fail."""
    pass


class TPMError(SecurityError):
    """Raised when TPM operations fail."""
    pass


# Biometric Processing Exceptions
class BiometricError(LocklessError):
    """Base class for biometric processing errors."""
    pass


class EnrollmentError(BiometricError):
    """Raised when biometric enrollment fails."""
    pass


class AuthenticationError(BiometricError):
    """Raised when biometric authentication fails."""
    pass


class LivenessDetectionError(BiometricError):
    """Raised when liveness detection fails."""
    pass


class FaceDetectionError(BiometricError):
    """Raised when face detection fails."""
    pass


class FeatureExtractionError(BiometricError):
    """Raised when feature extraction fails."""
    pass


class QualityAssessmentError(BiometricError):
    """Raised when image quality assessment fails."""
    pass


# Camera and Hardware Exceptions
class HardwareError(LocklessError):
    """Base class for hardware-related errors."""
    pass


class CameraError(HardwareError):
    """Raised when camera operations fail."""
    pass


class CameraNotFoundError(CameraError):
    """Raised when no suitable camera is found."""
    pass


class CameraAccessError(CameraError):
    """Raised when camera access is denied."""
    pass


# Configuration and System Exceptions
class ConfigurationError(LocklessError):
    """Raised when configuration is invalid or missing."""
    pass


class ModelLoadError(LocklessError):
    """Raised when AI model loading fails."""
    pass


class DatabaseError(LocklessError):
    """Raised when database operations fail."""
    pass


# User Interface Exceptions
class UIError(LocklessError):
    """Base class for user interface errors."""
    pass


class AccessibilityError(UIError):
    """Raised when accessibility features fail."""
    pass


# API and Integration Exceptions
class APIError(LocklessError):
    """Base class for API-related errors."""
    pass


class ValidationError(APIError):
    """Raised when input validation fails."""
    pass


class AuthorizationError(APIError):
    """Raised when authorization fails."""
    pass


# Platform-Specific Exceptions
class PlatformError(LocklessError):
    """Base class for platform-specific errors."""
    pass


class WindowsError(PlatformError):
    """Raised for Windows-specific errors."""
    pass


class LinuxError(PlatformError):
    """Raised for Linux-specific errors."""
    pass


class AndroidError(PlatformError):
    """Raised for Android-specific errors."""
    pass


# Performance and Timeout Exceptions
class PerformanceError(LocklessError):
    """Raised when performance requirements are not met."""
    pass


class TimeoutError(PerformanceError):
    """Raised when operations exceed time limits."""
    pass


class LatencyError(PerformanceError):
    """Raised when latency requirements are not met."""
    pass


# Template and Storage Exceptions
class TemplateError(LocklessError):
    """Base class for template-related errors."""
    pass


class TemplateCorruptedError(TemplateError):
    """Raised when template data is corrupted."""
    pass


class TemplateNotFoundError(TemplateError):
    """Raised when template is not found."""
    pass


class TemplateVersionError(TemplateError):
    """Raised when template version is incompatible."""
    pass


class StorageError(LocklessError):
    """Base class for storage-related errors."""
    pass


class StorageFullError(StorageError):
    """Raised when storage is full."""
    pass


class StoragePermissionError(StorageError):
    """Raised when storage permissions are insufficient."""
    pass
