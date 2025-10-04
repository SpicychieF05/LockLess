# LockLess API Reference

This document provides comprehensive API documentation for the LockLess biometric authentication system.

## Table of Contents

- [Core API](#core-api)
- [Biometric API](#biometric-api)
- [Security API](#security-api)
- [UI API](#ui-api)
- [REST API](#rest-api)
- [Python SDK](#python-sdk)
- [Configuration API](#configuration-api)
- [Error Handling](#error-handling)
- [Examples](#examples)

## Core API

### ConfigManager

Manages system configuration and settings.

```python
from src.core.config import ConfigManager

# Initialize with default config
config = ConfigManager()

# Initialize with custom config
config = ConfigManager("path/to/config.yaml")

# Get configuration value
value = config.get('camera.device_id', default=0)

# Set configuration value
config.set('camera.device_id', 1)

# Save configuration
config.save()
```

#### Methods

##### `__init__(config_path: Optional[str] = None)`

Initialize configuration manager.

**Parameters:**

- `config_path` (str, optional): Path to configuration file

##### `get(key: str, default: Any = None) -> Any`

Get configuration value by key.

**Parameters:**

- `key` (str): Configuration key (supports dot notation)
- `default` (Any): Default value if key not found

**Returns:**

- Configuration value or default

##### `set(key: str, value: Any) -> None`

Set configuration value.

**Parameters:**

- `key` (str): Configuration key
- `value` (Any): Value to set

### Logging

Structured logging with security event tracking.

```python
from src.core.logging import get_logger, get_security_logger, SecurityEventType

# Get standard logger
logger = get_logger(__name__)

# Get security logger
security_logger = get_security_logger()

# Log security event
security_logger.log_security_event(
    SecurityEventType.TEMPLATE_ACCESS,
    user_id="john_doe",
    additional_data={"action": "read"},
    success=True
)
```

#### SecurityEventType Enum

- `TEMPLATE_ACCESS`: Template access events
- `AUTHENTICATION_ATTEMPT`: Authentication attempts
- `ENROLLMENT_ATTEMPT`: Enrollment attempts
- `SECURITY_VIOLATION`: Security violations
- `SYSTEM_ERROR`: System errors

## Biometric API

### BiometricEnrollment

Handles user enrollment process.

```python
from src.biometric.enrollment import BiometricEnrollment, EnrollmentConfig

# Create enrollment configuration
config = EnrollmentConfig(
    camera_id=0,
    required_samples=5,
    quality_threshold=0.7
)

# Initialize enrollment
enrollment = BiometricEnrollment(config)

# Enroll user
result = enrollment.enroll_user("john_doe", "password123")
```

#### EnrollmentConfig

Configuration for enrollment process.

```python
@dataclass
class EnrollmentConfig:
    camera_id: int = 0
    required_samples: int = 5
    quality_threshold: float = 0.7
    max_enrollment_time: int = 30
    sample_interval: float = 1.0
```

#### Methods

##### `enroll_user(user_id: str, password: str) -> EnrollmentResult`

Enroll a new user with biometric data.

**Parameters:**

- `user_id` (str): Unique user identifier
- `password` (str): Master password for encryption

**Returns:**

- `EnrollmentResult`: Enrollment result with success status and metadata

### AuthenticationEngine

Handles user authentication.

```python
from src.biometric.authentication import AuthenticationEngine, AuthenticationConfig

# Create authentication configuration
config = AuthenticationConfig(
    camera_id=0,
    similarity_threshold=0.7,
    quality_threshold=0.6,
    max_authentication_time=10
)

# Initialize authentication engine
auth_engine = AuthenticationEngine(config)

# Authenticate user
result = auth_engine.authenticate_user("john_doe", "password123")
```

#### AuthenticationConfig

Configuration for authentication process.

```python
@dataclass
class AuthenticationConfig:
    camera_id: int = 0
    similarity_threshold: float = 0.7
    quality_threshold: float = 0.6
    max_authentication_time: int = 10
    max_attempts: int = 3
    lockout_duration: int = 300
```

#### Methods

##### `authenticate_user(user_id: str, password: str) -> AuthenticationResult`

Authenticate a user with biometric data.

**Parameters:**

- `user_id` (str): User identifier
- `password` (str): Master password

**Returns:**

- `AuthenticationResult`: Authentication result with success status and confidence

### FaceDetection

Face detection and feature extraction.

```python
from src.biometric.face_detection import FaceDetector

# Initialize face detector
detector = FaceDetector()

# Detect faces in image
faces = detector.detect_faces(image)

# Extract face features
features = detector.extract_features(face_image)
```

#### Methods

##### `detect_faces(image: np.ndarray) -> List[Face]`

Detect faces in an image.

**Parameters:**

- `image` (np.ndarray): Input image

**Returns:**

- List of detected faces

##### `extract_features(face_image: np.ndarray) -> np.ndarray`

Extract face features for recognition.

**Parameters:**

- `face_image` (np.ndarray): Cropped face image

**Returns:**

- Feature vector (512-dimensional)

### LivenessDetection

Anti-spoofing and liveness detection.

```python
from src.biometric.liveness import LivenessDetector

# Initialize liveness detector
liveness = LivenessDetector()

# Check if face is live
is_live = liveness.check_liveness(face_image, depth_map)
```

#### Methods

##### `check_liveness(face_image: np.ndarray, depth_map: Optional[np.ndarray] = None) -> bool`

Check if detected face is live.

**Parameters:**

- `face_image` (np.ndarray): Face image
- `depth_map` (np.ndarray, optional): Depth information

**Returns:**

- True if face is live, False otherwise

## Security API

### SecureTemplateStorage

Encrypted storage for biometric templates.

```python
from src.security.encryption import SecureTemplateStorage

# Initialize storage
storage = SecureTemplateStorage()

# Store template
storage.store_template("john_doe", encrypted_template, password)

# Retrieve template
template = storage.load_template("john_doe", password)

# Delete template
storage.delete_template("john_doe")

# List users
users = storage.list_users()
```

#### Methods

##### `store_template(user_id: str, template: bytes, password: str) -> bool`

Store encrypted biometric template.

**Parameters:**

- `user_id` (str): User identifier
- `template` (bytes): Encrypted template data
- `password` (str): Encryption password

**Returns:**

- True if successful, False otherwise

##### `load_template(user_id: str, password: str) -> Optional[bytes]`

Load encrypted biometric template.

**Parameters:**

- `user_id` (str): User identifier
- `password` (str): Decryption password

**Returns:**

- Decrypted template or None if not found

### KeyManagement

Secure key derivation and management.

```python
from src.security.key_management import KeyManager

# Initialize key manager
key_manager = KeyManager()

# Derive key from password
key = key_manager.derive_key(password, salt)

# Generate random salt
salt = key_manager.generate_salt()

# Encrypt data
encrypted_data = key_manager.encrypt(data, key)

# Decrypt data
decrypted_data = key_manager.decrypt(encrypted_data, key)
```

## UI API

### MainWindow

Main application window.

```python
from src.ui.main_window import MainWindow

# Create main window
window = MainWindow()

# Show window
window.show()

# Start event loop
window.run()
```

### EnrollmentDialog

User enrollment interface.

```python
from src.ui.enrollment_dialog import EnrollmentDialog

# Create enrollment dialog
dialog = EnrollmentDialog(parent_window)

# Start enrollment
result = dialog.start_enrollment("john_doe")
```

### AuthenticationDialog

User authentication interface.

```python
from src.ui.authentication_dialog import AuthenticationDialog

# Create authentication dialog
dialog = AuthenticationDialog(parent_window)

# Start authentication
result = dialog.start_authentication("john_doe")
```

## REST API

### Authentication Endpoints

#### POST /api/v1/enroll

Enroll a new user.

**Request:**

```json
{
  "user_id": "john_doe",
  "password": "secret123"
}
```

**Response:**

```json
{
  "success": true,
  "user_id": "john_doe",
  "samples_collected": 5,
  "average_quality": 0.85,
  "processing_time": 12.3
}
```

#### POST /api/v1/authenticate

Authenticate a user.

**Request:**

```json
{
  "user_id": "john_doe",
  "password": "secret123"
}
```

**Response:**

```json
{
  "success": true,
  "user_id": "john_doe",
  "confidence": 0.92,
  "quality_score": 0.88,
  "processing_time": 0.35
}
```

#### GET /api/v1/users

List enrolled users.

**Response:**

```json
{
  "users": ["john_doe", "jane_smith"],
  "count": 2
}
```

#### DELETE /api/v1/users/{user_id}

Delete a user.

**Response:**

```json
{
  "success": true,
  "message": "User deleted successfully"
}
```

### System Endpoints

#### GET /api/v1/status

Get system status.

**Response:**

```json
{
  "status": "healthy",
  "version": "1.0.0",
  "uptime": 3600,
  "camera_available": true,
  "users_count": 2
}
```

#### GET /api/v1/diagnostics

Get system diagnostics.

**Response:**

```json
{
  "camera": {
    "available": true,
    "resolution": "640x480",
    "fps": 30
  },
  "performance": {
    "memory_usage": "256MB",
    "cpu_usage": "5%",
    "disk_usage": "50MB"
  },
  "security": {
    "encryption_enabled": true,
    "tpm_available": false
  }
}
```

## Python SDK

### LockLessClient

Main client for interacting with LockLess.

```python
from lockless import LockLessClient

# Initialize client
client = LockLessClient()

# Enroll user
result = client.enroll_user("john_doe", "password123")
print(f"Enrollment: {result.success}")

# Authenticate user
result = client.authenticate_user("john_doe", "password123")
print(f"Authentication: {result.success}")

# List users
users = client.list_users()
print(f"Users: {users}")

# Delete user
success = client.delete_user("john_doe")
print(f"Deletion: {success}")
```

#### Methods

##### `enroll_user(user_id: str, password: str) -> EnrollmentResult`

Enroll a new user.

##### `authenticate_user(user_id: str, password: str) -> AuthenticationResult`

Authenticate a user.

##### `list_users() -> List[str]`

List enrolled users.

##### `delete_user(user_id: str) -> bool`

Delete a user.

##### `get_system_status() -> SystemStatus`

Get system status.

## Configuration API

### Configuration Keys

#### Camera Settings

- `camera.device_id`: Camera device ID (default: 0)
- `camera.resolution.width`: Image width (default: 640)
- `camera.resolution.height`: Image height (default: 480)
- `camera.fps`: Frames per second (default: 30)

#### Authentication Settings

- `authentication.similarity_threshold`: Similarity threshold (default: 0.7)
- `authentication.quality_threshold`: Quality threshold (default: 0.6)
- `authentication.max_attempts`: Maximum attempts (default: 3)
- `authentication.lockout_duration`: Lockout duration in seconds (default: 300)

#### Enrollment Settings

- `enrollment.required_samples`: Required samples (default: 5)
- `enrollment.quality_threshold`: Quality threshold (default: 0.4)
- `enrollment.max_enrollment_time`: Max enrollment time (default: 30)

#### Security Settings

- `security.encryption_enabled`: Enable encryption (default: true)
- `security.tpm_enabled`: Enable TPM (default: false)
- `security.key_derivation_iterations`: Key derivation iterations (default: 100000)

## Error Handling

### Exception Hierarchy

```python
# Base exception
class LockLessError(Exception):
    """Base exception for LockLess errors."""

# Configuration errors
class ConfigurationError(LockLessError):
    """Configuration-related errors."""

# Biometric errors
class BiometricError(LockLessError):
    """Biometric processing errors."""

class EnrollmentError(BiometricError):
    """Enrollment-specific errors."""

class AuthenticationError(BiometricError):
    """Authentication-specific errors."""

# Security errors
class SecurityError(LockLessError):
    """Security-related errors."""

class EncryptionError(SecurityError):
    """Encryption/decryption errors."""

# Camera errors
class CameraError(LockLessError):
    """Camera-related errors."""

# Validation errors
class ValidationError(LockLessError):
    """Input validation errors."""
```

### Error Codes

- `E001`: Configuration file not found
- `E002`: Invalid configuration value
- `E003`: Camera not available
- `E004`: Face not detected
- `E005`: Liveness check failed
- `E006`: Template not found
- `E007`: Authentication failed
- `E008`: Encryption error
- `E009`: Invalid user ID
- `E010`: Password too weak

## Examples

### Complete Enrollment Example

```python
from src.core.config import ConfigManager
from src.biometric.enrollment import BiometricEnrollment, EnrollmentConfig
from src.core.logging import get_logger

logger = get_logger(__name__)

def enroll_user_example():
    """Complete example of user enrollment."""
    try:
        # Load configuration
        config_manager = ConfigManager()

        # Create enrollment configuration
        enrollment_config = EnrollmentConfig(
            camera_id=config_manager.get('camera.device_id', 0),
            required_samples=config_manager.get('enrollment.required_samples', 5),
            quality_threshold=config_manager.get('enrollment.quality_threshold', 0.7)
        )

        # Initialize enrollment
        enrollment = BiometricEnrollment(enrollment_config)

        # Enroll user
        result = enrollment.enroll_user("john_doe", "password123")

        if result.success:
            logger.info(f"User {result.user_id} enrolled successfully")
            logger.info(f"Samples collected: {result.samples_collected}")
            logger.info(f"Average quality: {result.average_quality}")
        else:
            logger.error(f"Enrollment failed: {result.error_message}")

    except Exception as e:
        logger.error(f"Enrollment error: {e}")
```

### Complete Authentication Example

```python
from src.core.config import ConfigManager
from src.biometric.authentication import AuthenticationEngine, AuthenticationConfig
from src.core.logging import get_logger

logger = get_logger(__name__)

def authenticate_user_example():
    """Complete example of user authentication."""
    try:
        # Load configuration
        config_manager = ConfigManager()

        # Create authentication configuration
        auth_config = AuthenticationConfig(
            camera_id=config_manager.get('camera.device_id', 0),
            similarity_threshold=config_manager.get('authentication.similarity_threshold', 0.7),
            quality_threshold=config_manager.get('authentication.quality_threshold', 0.6)
        )

        # Initialize authentication engine
        auth_engine = AuthenticationEngine(auth_config)

        # Authenticate user
        result = auth_engine.authenticate_user("john_doe", "password123")

        if result.success:
            logger.info(f"Authentication successful for {result.user_id}")
            logger.info(f"Confidence: {result.confidence}")
            logger.info(f"Quality score: {result.quality_score}")
        else:
            logger.error(f"Authentication failed: {result.error_message}")

    except Exception as e:
        logger.error(f"Authentication error: {e}")
```

### REST API Client Example

```python
import requests
import json

class LockLessAPIClient:
    """Client for LockLess REST API."""

    def __init__(self, base_url="http://localhost:8080"):
        self.base_url = base_url
        self.session = requests.Session()

    def enroll_user(self, user_id: str, password: str):
        """Enroll a new user."""
        response = self.session.post(
            f"{self.base_url}/api/v1/enroll",
            json={"user_id": user_id, "password": password}
        )
        return response.json()

    def authenticate_user(self, user_id: str, password: str):
        """Authenticate a user."""
        response = self.session.post(
            f"{self.base_url}/api/v1/authenticate",
            json={"user_id": user_id, "password": password}
        )
        return response.json()

    def list_users(self):
        """List enrolled users."""
        response = self.session.get(f"{self.base_url}/api/v1/users")
        return response.json()

    def delete_user(self, user_id: str):
        """Delete a user."""
        response = self.session.delete(f"{self.base_url}/api/v1/users/{user_id}")
        return response.json()

# Usage example
client = LockLessAPIClient()

# Enroll user
result = client.enroll_user("john_doe", "password123")
print(f"Enrollment: {result}")

# Authenticate user
result = client.authenticate_user("john_doe", "password123")
print(f"Authentication: {result}")
```

This API reference provides comprehensive documentation for all LockLess APIs. For more specific examples and advanced usage, refer to the individual module documentation and the test suite.
