"""
Configuration management module for the Lockless system.

This module handles loading and managing configuration from various sources
including YAML files, environment variables, and command-line arguments.
"""

import os
import yaml
from typing import Any, Dict, Optional

from .logging import get_logger
from .exceptions import ConfigurationError

logger = get_logger(__name__)


class ConfigManager:
    """
    Centralized configuration management for the Lockless system.

    Supports hierarchical configuration with the following precedence:
    1. Command-line arguments (highest priority)
    2. Environment variables
    3. Configuration files
    4. Default values (lowest priority)
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration manager.

        Args:
            config_path: Optional path to configuration file
        """
        self.config_data = {}
        self.config_path = config_path

        self._load_default_config()
        self._load_config_file()
        self._load_environment_variables()

        logger.debug("ConfigManager initialized")

    def _load_default_config(self):
        """Load default configuration values."""
        self.config_data = {
            'system': {
                'version': '1.0.0',
                'log_level': 'INFO',
                'data_directory': self._get_default_data_dir()
            },
            'camera': {
                'device_id': 0,
                'resolution': {
                    'width': 640,
                    'height': 480
                },
                'fps': 30
            },
            'authentication': {
                'similarity_threshold': 0.7,
                'quality_threshold': 0.6,
                'max_attempts': 3,
                'lockout_duration': 300,
                'max_time': 10
            },
            'enrollment': {
                'required_samples': 5,
                'quality_threshold': 0.7,
                'max_enrollment_time': 30
            },
            'security': {
                'encryption_enabled': True,
                'tpm_enabled': True,
                'key_derivation_iterations': 100000
            },
            'performance': {
                'max_authentication_time': 5000,
                'enable_gpu_acceleration': False,
                'max_concurrent_users': 10
            }
        }

    def _get_default_data_dir(self) -> str:
        """Get default data directory based on platform."""
        if os.name == 'nt':  # Windows
            return os.path.join(os.environ.get('APPDATA', ''), 'Lockless')
        else:  # Linux/Unix
            return os.path.expanduser('~/.lockless')

    def _load_config_file(self):
        """Load configuration from file."""
        config_paths = []

        # Add specified config path
        if self.config_path:
            config_paths.append(self.config_path)

        # Add default config paths
        config_paths.extend([
            'config/default.yaml',
            'config.yaml',
            os.path.join(self._get_default_data_dir(), 'config.yaml'),
            '/etc/lockless/config.yaml'
        ])

        for config_path in config_paths:
            if os.path.exists(config_path):
                try:
                    with open(config_path, 'r') as f:
                        file_config = yaml.safe_load(f)
                        if file_config:
                            self._merge_config(self.config_data, file_config)
                            logger.debug(
                                f"Loaded configuration from {config_path}")
                            break
                except Exception as e:
                    logger.warning(
                        f"Failed to load config from {config_path}: {e}")

    def _load_environment_variables(self):
        """Load configuration from environment variables."""
        env_mapping = {
            'LOCKLESS_LOG_LEVEL': 'system.log_level',
            'LOCKLESS_CAMERA_ID': 'camera.device_id',
            'LOCKLESS_SIMILARITY_THRESHOLD': 'authentication.similarity_threshold',
            'LOCKLESS_QUALITY_THRESHOLD': 'authentication.quality_threshold',
            'LOCKLESS_DATA_DIR': 'system.data_directory'
        }

        for env_var, config_key in env_mapping.items():
            value = os.environ.get(env_var)
            if value is not None:
                self._set_nested_value(self.config_data, config_key, value)
                logger.debug(
                    f"Set {config_key} from environment variable {env_var}")

    def _merge_config(self, base: Dict[str, Any], update: Dict[str, Any]):
        """Recursively merge configuration dictionaries."""
        for key, value in update.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._merge_config(base[key], value)
            else:
                base[key] = value

    def _set_nested_value(self, data: Dict[str, Any], key_path: str, value: Any):
        """Set a nested configuration value using dot notation."""
        keys = key_path.split('.')
        current = data

        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]

        # Convert string values to appropriate types
        final_key = keys[-1]
        if isinstance(value, str):
            # Try to convert to appropriate type
            if value.lower() in ('true', 'false'):
                value = value.lower() == 'true'
            elif value.isdigit():
                value = int(value)
            elif '.' in value and value.replace('.', '').isdigit():
                value = float(value)

        current[final_key] = value

    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation.

        Args:
            key_path: Configuration key path (e.g., 'camera.device_id')
            default: Default value if key not found

        Returns:
            Configuration value or default
        """
        keys = key_path.split('.')
        current = self.config_data

        try:
            for key in keys:
                current = current[key]
            return current
        except (KeyError, TypeError):
            return default

    def set(self, key_path: str, value: Any):
        """
        Set configuration value using dot notation.

        Args:
            key_path: Configuration key path
            value: Value to set
        """
        self._set_nested_value(self.config_data, key_path, value)

    def get_section(self, section: str) -> Dict[str, Any]:
        """
        Get entire configuration section.

        Args:
            section: Section name

        Returns:
            Configuration section dictionary
        """
        return self.config_data.get(section, {})

    def save(self, file_path: Optional[str] = None):
        """
        Save current configuration to file.

        Args:
            file_path: Optional file path (uses default if not specified)
        """
        if not file_path:
            file_path = self.config_path or 'config.yaml'

        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(file_path), exist_ok=True)

            with open(file_path, 'w') as f:
                yaml.dump(self.config_data, f,
                          default_flow_style=False, indent=2)

            logger.info(f"Configuration saved to {file_path}")

        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")
            raise ConfigurationError(f"Failed to save configuration: {e}")

    def validate(self) -> bool:
        """
        Validate configuration values.

        Returns:
            True if configuration is valid
        """
        try:
            # Validate required sections
            required_sections = ['system', 'camera', 'authentication']
            for section in required_sections:
                if section not in self.config_data:
                    raise ConfigurationError(
                        f"Missing required section: {section}")

            # Validate specific values
            camera_id = self.get('camera.device_id')
            if not isinstance(camera_id, int) or camera_id < 0:
                raise ConfigurationError(
                    f"Invalid camera device_id: {camera_id}")

            similarity_threshold = self.get(
                'authentication.similarity_threshold')
            if not isinstance(similarity_threshold, (int, float)) or not (0 <= similarity_threshold <= 1):
                raise ConfigurationError(
                    f"Invalid similarity_threshold: {similarity_threshold}")

            quality_threshold = self.get('authentication.quality_threshold')
            if not isinstance(quality_threshold, (int, float)) or not (0 <= quality_threshold <= 1):
                raise ConfigurationError(
                    f"Invalid quality_threshold: {quality_threshold}")

            return True

        except ConfigurationError:
            raise
        except Exception as e:
            raise ConfigurationError(f"Configuration validation failed: {e}")

    def __str__(self) -> str:
        """String representation of configuration."""
        return yaml.dump(self.config_data, default_flow_style=False, indent=2)


# Example usage
if __name__ == "__main__":
    # Test configuration manager
    config = ConfigManager()

    print("Default configuration:")
    print(f"Camera device ID: {config.get('camera.device_id')}")
    print(
        f"Similarity threshold: {config.get('authentication.similarity_threshold')}")
    print(f"Data directory: {config.get('system.data_directory')}")

    # Test validation
    try:
        config.validate()
        print("Configuration is valid")
    except ConfigurationError as e:
        print(f"Configuration error: {e}")

    print("Configuration manager test completed!")
