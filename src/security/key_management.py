"""
Key management module for secure key derivation and TPM/Secure Enclave integration.

This module handles the generation, derivation, and secure storage of cryptographic
keys used throughout the Lockless system.
"""

import os
import platform
import secrets
import hashlib
from typing import Optional, Dict
from dataclasses import dataclass
from enum import Enum

from core.exceptions import KeyManagementError, TPMError
from core.logging import get_logger

logger = get_logger(__name__)


class KeyStorageType(Enum):
    """Enumeration of supported key storage types."""
    SOFTWARE = "software"
    TPM = "tpm"
    SECURE_ENCLAVE = "secure_enclave"
    ANDROID_KEYSTORE = "android_keystore"


@dataclass
class KeyMetadata:
    """Metadata for managed keys."""
    key_id: str
    storage_type: KeyStorageType
    created_at: str
    algorithm: str
    key_length: int
    usage: str


class SecureKeyManager:
    """
    Manages cryptographic keys with platform-specific secure storage.

    Provides abstraction over different key storage mechanisms:
    - Windows: TPM 2.0
    - Linux: TPM 2.0 or software keyring
    - Android: Android Keystore
    - Fallback: Software-based encrypted storage
    """

    def __init__(self, storage_type: Optional[KeyStorageType] = None):
        """
        Initialize key manager with platform-appropriate storage.

        Args:
            storage_type: Preferred storage type (auto-detected if None)
        """
        self.storage_type = storage_type or self._detect_optimal_storage()
        self.key_metadata: Dict[str, KeyMetadata] = {}

        logger.info(
            f"KeyManager initialized with storage type: {self.storage_type.value}")

        # Initialize platform-specific components
        self._initialize_platform_storage()

    def _detect_optimal_storage(self) -> KeyStorageType:
        """Detect the best available key storage for the platform."""
        system = platform.system().lower()

        if system == "windows":
            if self._is_tpm_available():
                return KeyStorageType.TPM
            else:
                logger.warning(
                    "TPM not available on Windows, falling back to software storage")
                return KeyStorageType.SOFTWARE

        elif system == "linux":
            if self._is_tpm_available():
                return KeyStorageType.TPM
            else:
                logger.warning(
                    "TPM not available on Linux, falling back to software storage")
                return KeyStorageType.SOFTWARE

        elif system == "android" or "android" in system:
            return KeyStorageType.ANDROID_KEYSTORE

        else:
            logger.warning(
                f"Unknown platform {system}, using software storage")
            return KeyStorageType.SOFTWARE

    def _is_tpm_available(self) -> bool:
        """Check if TPM is available and accessible."""
        try:
            # On Windows, check for TPM via WMI or registry
            if platform.system().lower() == "windows":
                import winreg
                try:
                    key = winreg.OpenKey(
                        winreg.HKEY_LOCAL_MACHINE,
                        r"SYSTEM\CurrentControlSet\Services\TPM"
                    )
                    winreg.CloseKey(key)
                    return True
                except FileNotFoundError:
                    return False

            # On Linux, check for TPM device files
            elif platform.system().lower() == "linux":
                return (os.path.exists("/dev/tpm0") or
                        os.path.exists("/dev/tpmrm0") or
                        os.path.exists("/sys/class/tpm/tpm0"))

            return False

        except Exception as e:
            logger.error(f"Error checking TPM availability: {e}")
            return False

    def _initialize_platform_storage(self):
        """Initialize platform-specific storage components."""
        if self.storage_type == KeyStorageType.TPM:
            self._initialize_tpm()
        elif self.storage_type == KeyStorageType.ANDROID_KEYSTORE:
            self._initialize_android_keystore()
        else:
            self._initialize_software_storage()

    def _initialize_tpm(self):
        """Initialize TPM for key storage."""
        try:
            # In a production implementation, this would use:
            # - Windows: TPM Base Services (TBS) API
            # - Linux: tpm2-tools or TSS2 library

            logger.info("TPM initialization - placeholder implementation")
            # TODO: Implement actual TPM initialization

        except Exception as e:
            logger.error(f"TPM initialization failed: {e}")
            raise TPMError(f"Failed to initialize TPM: {e}")

    def _initialize_android_keystore(self):
        """Initialize Android Keystore."""
        try:
            # Android Keystore initialization would be done via JNI
            # or Python-for-Android integration
            logger.info(
                "Android Keystore initialization - placeholder implementation")
            # TODO: Implement Android Keystore integration

        except Exception as e:
            logger.error(f"Android Keystore initialization failed: {e}")
            raise KeyManagementError(
                f"Failed to initialize Android Keystore: {e}")

    def _initialize_software_storage(self):
        """Initialize software-based key storage."""
        self.software_key_dir = os.path.expanduser("~/.lockless/keys")
        os.makedirs(self.software_key_dir, exist_ok=True)

        # Set restrictive permissions (Unix-like systems)
        if platform.system().lower() != "windows":
            os.chmod(self.software_key_dir, 0o700)

        logger.info("Software key storage initialized")

    def generate_master_key(self, user_id: str, key_length: int = 32) -> str:
        """
        Generate a new master key for a user.

        Args:
            user_id: Unique user identifier
            key_length: Key length in bytes (default: 32 for AES-256)

        Returns:
            Key identifier for future reference

        Raises:
            KeyManagementError: If key generation fails
        """
        try:
            key_id = f"master_key_{user_id}_{secrets.token_hex(8)}"

            # Generate cryptographically secure random key
            key_material = secrets.token_bytes(key_length)

            # Store key based on storage type
            if self.storage_type == KeyStorageType.TPM:
                self._store_key_tpm(key_id, key_material)
            elif self.storage_type == KeyStorageType.ANDROID_KEYSTORE:
                self._store_key_android(key_id, key_material)
            else:
                self._store_key_software(key_id, key_material)

            # Store metadata
            metadata = KeyMetadata(
                key_id=key_id,
                storage_type=self.storage_type,
                created_at=self._get_timestamp(),
                algorithm="AES",
                key_length=key_length * 8,  # Convert to bits
                usage="master_encryption"
            )
            self.key_metadata[key_id] = metadata

            logger.info(f"Master key generated for user {user_id}: {key_id}")
            return key_id

        except Exception as e:
            logger.error(
                f"Master key generation failed for user {user_id}: {e}")
            raise KeyManagementError(f"Failed to generate master key: {e}")

    def derive_user_key(self, master_key_id: str, purpose: str,
                        salt: Optional[bytes] = None) -> bytes:
        """
        Derive a purpose-specific key from master key.

        Args:
            master_key_id: Master key identifier
            purpose: Key purpose (e.g., "template_encryption", "auth_token")
            salt: Optional salt (generated if not provided)

        Returns:
            Derived key material

        Raises:
            KeyManagementError: If key derivation fails
        """
        try:
            # Retrieve master key
            master_key = self._retrieve_key(master_key_id)

            # Generate salt if not provided
            if salt is None:
                salt = secrets.token_bytes(32)

            # Use HKDF for key derivation
            from cryptography.hazmat.primitives.kdf.hkdf import HKDF
            from cryptography.hazmat.primitives import hashes
            from cryptography.hazmat.backends import default_backend

            hkdf = HKDF(
                algorithm=hashes.SHA256(),
                length=32,  # 256 bits
                salt=salt,
                info=purpose.encode('utf-8'),
                backend=default_backend()
            )

            derived_key = hkdf.derive(master_key)

            logger.debug(f"Key derived for purpose: {purpose}")
            return derived_key

        except Exception as e:
            logger.error(f"Key derivation failed: {e}")
            raise KeyManagementError(f"Failed to derive key: {e}")

    def _store_key_tpm(self, key_id: str, key_material: bytes):
        """Store key in TPM."""
        # Placeholder for TPM key storage
        # In production, this would use TPM APIs to:
        # 1. Create a key object in TPM
        # 2. Set appropriate access policies
        # 3. Seal the key to PCR values if needed

        logger.info(f"Storing key in TPM: {key_id}")
        # TODO: Implement actual TPM key storage

    def _store_key_android(self, key_id: str, key_material: bytes):
        """Store key in Android Keystore."""
        # Placeholder for Android Keystore storage
        # In production, this would use Android Keystore APIs

        logger.info(f"Storing key in Android Keystore: {key_id}")
        # TODO: Implement Android Keystore integration

    def _store_key_software(self, key_id: str, key_material: bytes):
        """Store key in software-based encrypted storage."""
        try:
            # For software storage, encrypt the key with a device-specific key
            device_key = self._get_device_key()

            from cryptography.hazmat.primitives.ciphers.aead import AESGCM
            aesgcm = AESGCM(device_key)
            nonce = secrets.token_bytes(12)

            encrypted_key = aesgcm.encrypt(nonce, key_material, None)

            # Store encrypted key and nonce
            key_file = os.path.join(self.software_key_dir, f"{key_id}.key")
            with open(key_file, 'wb') as f:
                f.write(nonce + encrypted_key)

            # Set restrictive permissions
            if platform.system().lower() != "windows":
                os.chmod(key_file, 0o600)

            logger.info(f"Key stored in software storage: {key_id}")

        except Exception as e:
            logger.error(f"Software key storage failed: {e}")
            raise KeyManagementError(
                f"Failed to store key in software storage: {e}")

    def _retrieve_key(self, key_id: str) -> bytes:
        """Retrieve key material by key ID."""
        try:
            if self.storage_type == KeyStorageType.TPM:
                return self._retrieve_key_tpm(key_id)
            elif self.storage_type == KeyStorageType.ANDROID_KEYSTORE:
                return self._retrieve_key_android(key_id)
            else:
                return self._retrieve_key_software(key_id)

        except Exception as e:
            logger.error(f"Key retrieval failed for {key_id}: {e}")
            raise KeyManagementError(f"Failed to retrieve key: {e}")

    def _retrieve_key_tpm(self, key_id: str) -> bytes:
        """Retrieve key from TPM."""
        # Placeholder for TPM key retrieval
        logger.info(f"Retrieving key from TPM: {key_id}")
        # TODO: Implement actual TPM key retrieval
        raise NotImplementedError("TPM key retrieval not yet implemented")

    def _retrieve_key_android(self, key_id: str) -> bytes:
        """Retrieve key from Android Keystore."""
        # Placeholder for Android Keystore retrieval
        logger.info(f"Retrieving key from Android Keystore: {key_id}")
        # TODO: Implement Android Keystore retrieval
        raise NotImplementedError(
            "Android Keystore retrieval not yet implemented")

    def _retrieve_key_software(self, key_id: str) -> bytes:
        """Retrieve key from software storage."""
        try:
            key_file = os.path.join(self.software_key_dir, f"{key_id}.key")

            if not os.path.exists(key_file):
                raise KeyManagementError(f"Key file not found: {key_id}")

            with open(key_file, 'rb') as f:
                data = f.read()

            # Extract nonce and encrypted key
            nonce = data[:12]
            encrypted_key = data[12:]

            # Decrypt with device key
            device_key = self._get_device_key()
            from cryptography.hazmat.primitives.ciphers.aead import AESGCM
            aesgcm = AESGCM(device_key)

            key_material = aesgcm.decrypt(nonce, encrypted_key, None)

            logger.debug(f"Key retrieved from software storage: {key_id}")
            return key_material

        except Exception as e:
            logger.error(f"Software key retrieval failed: {e}")
            raise KeyManagementError(
                f"Failed to retrieve key from software storage: {e}")

    def _get_device_key(self) -> bytes:
        """Generate or retrieve device-specific encryption key."""
        # This is a simplified implementation
        # In production, this should be derived from:
        # - Hardware identifiers (CPU ID, disk serial, etc.)
        # - OS-specific APIs (Windows DPAPI, Linux keyring, etc.)
        # - TPM-derived keys where available

        device_id = platform.node() + platform.machine()
        device_hash = hashlib.sha256(device_id.encode('utf-8')).digest()

        return device_hash[:32]  # Use first 32 bytes for AES-256

    def _get_timestamp(self) -> str:
        """Get current timestamp in ISO format."""
        from datetime import datetime
        return datetime.utcnow().isoformat()

    def delete_key(self, key_id: str) -> bool:
        """
        Securely delete a key.

        Args:
            key_id: Key identifier to delete

        Returns:
            True if deletion successful, False otherwise
        """
        try:
            if self.storage_type == KeyStorageType.SOFTWARE:
                key_file = os.path.join(self.software_key_dir, f"{key_id}.key")
                if os.path.exists(key_file):
                    # Overwrite file with random data before deletion
                    file_size = os.path.getsize(key_file)
                    with open(key_file, 'wb') as f:
                        f.write(secrets.token_bytes(file_size))
                    os.remove(key_file)

            # Remove from metadata
            if key_id in self.key_metadata:
                del self.key_metadata[key_id]

            logger.info(f"Key deleted: {key_id}")
            return True

        except Exception as e:
            logger.error(f"Key deletion failed for {key_id}: {e}")
            return False

    def list_keys(self) -> Dict[str, KeyMetadata]:
        """List all managed keys and their metadata."""
        return self.key_metadata.copy()


# Example usage
if __name__ == "__main__":
    # Initialize key manager
    key_manager = SecureKeyManager()

    # Generate master key for a user
    user_id = "john_doe"
    master_key_id = key_manager.generate_master_key(user_id)
    print(f"Generated master key: {master_key_id}")

    # Derive purpose-specific keys
    template_key = key_manager.derive_user_key(
        master_key_id, "template_encryption")
    auth_key = key_manager.derive_user_key(master_key_id, "auth_token")

    print(f"Template encryption key length: {len(template_key)} bytes")
    print(f"Auth token key length: {len(auth_key)} bytes")

    # List all keys
    keys = key_manager.list_keys()
    print(f"Managed keys: {list(keys.keys())}")

    # Clean up
    key_manager.delete_key(master_key_id)
    print("Test completed successfully!")
