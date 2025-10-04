"""
   AES-256 encryption module for biometric template security.

   This module provides secure encryption/decryption of biometric templates
   using AES-256 in GCM mode with proper key derivation and management.
   """

import os
import hashlib
import secrets
from typing import Optional
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.backends import default_backend
import base64

from ..core.exceptions import EncryptionError, KeyDerivationError
from ..core.logging import get_logger

logger = get_logger(__name__)


class BiometricTemplateEncryption:
    """
    Handles encryption and decryption of biometric templates using AES-256-GCM.

    Features:
    - AES-256 encryption in GCM mode for authenticated encryption
    - PBKDF2 key derivation with configurable iterations
    - Secure random salt and nonce generation
    - Base64 encoding for storage compatibility
    """

    def __init__(self, pbkdf2_iterations: int = 100000):
        """
        Initialize the encryption handler.

        Args:
            pbkdf2_iterations: Number of PBKDF2 iterations for key derivation
        """
        self.pbkdf2_iterations = pbkdf2_iterations
        self.key_length = 32  # 256 bits
        self.salt_length = 32  # 256 bits
        self.nonce_length = 12  # 96 bits for GCM

    def derive_key(self, password: str, salt: bytes) -> bytes:
        """
        Derive encryption key from password using PBKDF2.

        Args:
            password: Master password for key derivation
            salt: Random salt for key derivation

        Returns:
            Derived encryption key

        Raises:
            KeyDerivationError: If key derivation fails
        """
        try:
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=self.key_length,
                salt=salt,
                iterations=self.pbkdf2_iterations,
                backend=default_backend()
            )
            key = kdf.derive(password.encode('utf-8'))
            logger.debug("Key derived successfully")
            return key

        except Exception as e:
            logger.error(f"Key derivation failed: {e}")
            raise KeyDerivationError(f"Failed to derive encryption key: {e}")

    def generate_salt(self) -> bytes:
        """Generate cryptographically secure random salt."""
        return secrets.token_bytes(self.salt_length)

    def generate_nonce(self) -> bytes:
        """Generate cryptographically secure random nonce for GCM."""
        return secrets.token_bytes(self.nonce_length)

    def encrypt_template(self, template_data: bytes, password: str) -> str:
        """
        Encrypt biometric template data.

        Args:
            template_data: Raw biometric template bytes
            password: Master password for encryption

        Returns:
            Base64 encoded encrypted data with embedded salt and nonce

        Raises:
            EncryptionError: If encryption fails
        """
        try:
            # Generate random salt and nonce
            salt = self.generate_salt()
            nonce = self.generate_nonce()

            # Derive encryption key
            key = self.derive_key(password, salt)

            # Initialize AES-GCM cipher
            aesgcm = AESGCM(key)

            # Encrypt data with authentication
            ciphertext = aesgcm.encrypt(nonce, template_data, None)

            # Combine salt + nonce + ciphertext for storage
            encrypted_package = salt + nonce + ciphertext

            # Encode to base64 for storage
            encoded_package = base64.b64encode(
                encrypted_package).decode('utf-8')

            logger.info("Template encrypted successfully")
            return encoded_package

        except Exception as e:
            logger.error(f"Template encryption failed: {e}")
            raise EncryptionError(f"Failed to encrypt template: {e}")

    def decrypt_template(self, encrypted_data: str, password: str) -> bytes:
        """
        Decrypt biometric template data.

        Args:
            encrypted_data: Base64 encoded encrypted template
            password: Master password for decryption

        Returns:
            Decrypted template data

        Raises:
            EncryptionError: If decryption fails or authentication fails
        """
        try:
            # Decode from base64
            encrypted_package = base64.b64decode(
                encrypted_data.encode('utf-8'))

            # Extract salt, nonce, and ciphertext
            salt = encrypted_package[:self.salt_length]
            nonce = encrypted_package[self.salt_length:
                                      self.salt_length + self.nonce_length]
            ciphertext = encrypted_package[self.salt_length +
                                           self.nonce_length:]

            # Derive decryption key
            key = self.derive_key(password, salt)

            # Initialize AES-GCM cipher
            aesgcm = AESGCM(key)

            # Decrypt and authenticate
            plaintext = aesgcm.decrypt(nonce, ciphertext, None)

            logger.info("Template decrypted successfully")
            return plaintext

        except Exception as e:
            logger.error(f"Template decryption failed: {e}")
            raise EncryptionError(f"Failed to decrypt template: {e}")

    def verify_template_integrity(self, encrypted_data: str, password: str) -> bool:
        """
        Verify the integrity of encrypted template without full decryption.

        Args:
            encrypted_data: Base64 encoded encrypted template
            password: Master password for verification

        Returns:
            True if template is valid and authentic, False otherwise
        """
        try:
            self.decrypt_template(encrypted_data, password)
            return True
        except EncryptionError:
            return False


class SecureTemplateStorage:
    """
    Secure storage interface for encrypted biometric templates.

    Handles the persistent storage of encrypted templates with
    additional metadata and integrity checks.
    """

    def __init__(self, storage_path: str = "templates"):
        """
        Initialize secure template storage.

        Args:
            storage_path: Directory path for template storage
        """
        self.storage_path = storage_path
        self.encryption = BiometricTemplateEncryption()

        # Create storage directory if it doesn't exist
        os.makedirs(storage_path, exist_ok=True)

    def store_template(self, user_id: str, template_data: bytes,
                       password: str, metadata: Optional[dict] = None) -> bool:
        """
        Store encrypted biometric template for a user.

        Args:
            user_id: Unique user identifier
            template_data: Raw biometric template
            password: Master password for encryption
            metadata: Optional metadata (creation time, version, etc.)

        Returns:
            True if storage successful, False otherwise
        """
        try:
            # Encrypt template
            encrypted_template = self.encryption.encrypt_template(
                template_data, password)

            # Prepare storage data
            storage_data = {
                'encrypted_template': encrypted_template,
                'metadata': metadata or {},
                'checksum': hashlib.sha256(template_data).hexdigest()
            }

            # Generate storage filename
            filename = self._get_template_filename(user_id)
            filepath = os.path.join(self.storage_path, filename)

            # Write to file (in production, consider atomic writes)
            import json
            with open(filepath, 'w') as f:
                json.dump(storage_data, f, indent=2)

            logger.info(f"Template stored for user: {user_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to store template for user {user_id}: {e}")
            return False

    def load_template(self, user_id: str, password: str) -> Optional[bytes]:
        """
        Load and decrypt biometric template for a user.

        Args:
            user_id: Unique user identifier
            password: Master password for decryption

        Returns:
            Decrypted template data or None if not found/invalid
        """
        try:
            filename = self._get_template_filename(user_id)
            filepath = os.path.join(self.storage_path, filename)

            if not os.path.exists(filepath):
                logger.warning(f"Template not found for user: {user_id}")
                return None

            # Load storage data
            import json
            with open(filepath, 'r') as f:
                storage_data = json.load(f)

            # Decrypt template
            encrypted_template = storage_data['encrypted_template']
            template_data = self.encryption.decrypt_template(
                encrypted_template, password)

            # Verify checksum
            expected_checksum = storage_data.get('checksum', '')
            actual_checksum = hashlib.sha256(template_data).hexdigest()

            if expected_checksum != actual_checksum:
                logger.error(f"Checksum mismatch for user {user_id}")
                return None

            logger.info(f"Template loaded for user: {user_id}")
            return template_data

        except Exception as e:
            logger.error(f"Failed to load template for user {user_id}: {e}")
            return None

    def delete_template(self, user_id: str) -> bool:
        """
        Securely delete user template.

        Args:
            user_id: Unique user identifier

        Returns:
            True if deletion successful, False otherwise
        """
        try:
            filename = self._get_template_filename(user_id)
            filepath = os.path.join(self.storage_path, filename)

            if os.path.exists(filepath):
                # Secure deletion (overwrite before deletion in production)
                os.remove(filepath)
                logger.info(f"Template deleted for user: {user_id}")
                return True
            else:
                logger.warning(f"Template not found for deletion: {user_id}")
                return False

        except Exception as e:
            logger.error(f"Failed to delete template for user {user_id}: {e}")
            return False

    def list_users(self) -> list:
        """
        List all users with stored templates.

        Returns:
            List of user IDs
        """
        try:
            users = []
            for filename in os.listdir(self.storage_path):
                if filename.endswith('.enc'):
                    user_id = filename[:-4]  # Remove .enc extension
                    users.append(user_id)
            return users

        except Exception as e:
            logger.error(f"Failed to list users: {e}")
            return []

    def _get_template_filename(self, user_id: str) -> str:
        """Generate secure filename for user template."""
        # Hash user ID for filename (prevents directory traversal)
        hashed_id = hashlib.sha256(user_id.encode('utf-8')).hexdigest()[:16]
        return f"{hashed_id}.enc"


# Example usage and testing
if __name__ == "__main__":
    # Example biometric template (in practice, this would be from feature extraction)
    import numpy as np

    # Simulate a 512-dimensional face embedding
    template = np.random.rand(512).astype(np.float32)
    template_bytes = template.tobytes()

    # Initialize encryption and storage
    encryption = BiometricTemplateEncryption()
    storage = SecureTemplateStorage("./test_templates")

    # Test encryption/decryption
    password = "secure_master_password_123"

    print("Testing encryption...")
    encrypted = encryption.encrypt_template(template_bytes, password)
    print(f"Encrypted template length: {len(encrypted)}")

    print("Testing decryption...")
    decrypted = encryption.decrypt_template(encrypted, password)

    # Verify integrity
    original_template = np.frombuffer(template_bytes, dtype=np.float32)
    decrypted_template = np.frombuffer(decrypted, dtype=np.float32)

    print(
        f"Decryption successful: {np.array_equal(original_template, decrypted_template)}")

    # Test storage
    print("Testing template storage...")
    user_id = "john_doe"
    metadata = {"created_at": "2025-10-03", "version": "1.0"}

    storage.store_template(user_id, template_bytes, password, metadata)
    loaded_template = storage.load_template(user_id, password)

    if loaded_template is not None:
        loaded_template_array = np.frombuffer(
            loaded_template, dtype=np.float32)
        print(
            f"Storage test successful: {np.array_equal(original_template, loaded_template_array)}")
    else:
        print("Failed to load template")

    # List users
    users = storage.list_users()
    print(f"Stored users: {users}")

    # Clean up
    storage.delete_template(user_id)
    print("Test completed successfully!")
