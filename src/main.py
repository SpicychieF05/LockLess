"""
Main entry point for the Lockless biometric authentication system.

This module provides command-line interface and coordinates all system components
for enrollment, authentication, and system management.
"""

from src.core.logging import get_logger, get_security_logger, SecurityEventType
from src.core.config import ConfigManager
from src.biometric.enrollment import BiometricEnrollment, EnrollmentConfig
from src.biometric.authentication import AuthenticationEngine, AuthenticationConfig
from src.security.encryption import SecureTemplateStorage
import argparse
import sys
import os
import logging
from typing import Optional

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


logger = get_logger(__name__)
security_logger = get_security_logger()


def setup_logging(log_level: str = "INFO"):
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def enroll_user(user_id: str, password: str, config_path: Optional[str] = None) -> bool:
    """
    Enroll a new user with biometric data.

    Args:
        user_id: Unique user identifier
        password: Master password for template encryption
        config_path: Optional path to configuration file

    Returns:
        True if enrollment successful, False otherwise
    """
    try:
        logger.info(f"Starting enrollment for user: {user_id}")

        # Load configuration
        config_manager = ConfigManager(config_path)
        enrollment_config = EnrollmentConfig(
            camera_id=config_manager.get('camera.device_id', 0),
            required_samples=config_manager.get(
                'enrollment.required_samples', 5),
            quality_threshold=config_manager.get(
                'enrollment.quality_threshold', 0.7)
        )

        # Initialize enrollment system
        enrollment = BiometricEnrollment(enrollment_config)

        # Perform enrollment
        result = enrollment.enroll_user(user_id, password)

        if result.success:
            print(f"✓ Enrollment successful for user '{user_id}'")
            print(f"  - Samples collected: {result.samples_collected}")
            print(f"  - Average quality: {result.average_quality:.3f}")
            print(f"  - Processing time: {result.processing_time:.2f}s")
            return True
        else:
            print(
                f"✗ Enrollment failed for user '{user_id}': {result.error_message}")
            return False

    except Exception as e:
        logger.error(f"Enrollment error: {e}")
        print(f"✗ Enrollment failed: {e}")
        return False


def authenticate_user(user_id: Optional[str] = None, password: Optional[str] = None,
                      config_path: Optional[str] = None) -> bool:
    """
    Authenticate a user with biometric data.

    Args:
        user_id: User identifier (None for 1:N authentication)
        password: Template decryption password
        config_path: Optional path to configuration file

    Returns:
        True if authentication successful, False otherwise
    """
    try:
        logger.info(
            f"Starting authentication for user: {user_id or 'any user'}")

        # Load configuration
        config_manager = ConfigManager(config_path)
        auth_config = AuthenticationConfig(
            camera_id=config_manager.get('camera.device_id', 0),
            similarity_threshold=config_manager.get(
                'authentication.similarity_threshold', 0.7),
            quality_threshold=config_manager.get(
                'authentication.quality_threshold', 0.6),
            max_authentication_time=config_manager.get(
                'authentication.max_time', 10)
        )

        # Initialize authentication engine
        auth_engine = AuthenticationEngine(auth_config)

        if user_id and password:
            # 1:1 authentication
            result = auth_engine.authenticate_user(user_id, password)
        else:
            # 1:N authentication (would need enrolled users list)
            print("1:N authentication not implemented in this demo")
            return False

        if result.success:
            print(f"✓ Authentication successful for user '{result.user_id}'")
            print(f"  - Confidence: {result.confidence:.3f}")
            print(f"  - Quality score: {result.quality_score:.3f}")
            print(f"  - Processing time: {result.processing_time:.2f}s")
            return True
        else:
            print(f"✗ Authentication failed: {result.error_message}")
            return False

    except Exception as e:
        logger.error(f"Authentication error: {e}")
        print(f"✗ Authentication failed: {e}")
        return False


def list_users(config_path: Optional[str] = None):
    """List enrolled users."""
    try:
        storage = SecureTemplateStorage()
        users = storage.list_users()

        if users:
            print("Enrolled users:")
            for user_id in users:
                print(f"  - {user_id}")
        else:
            print("No enrolled users found")

    except Exception as e:
        logger.error(f"Error listing users: {e}")
        print(f"✗ Error listing users: {e}")


def delete_user(user_id: str, config_path: Optional[str] = None) -> bool:
    """Delete an enrolled user."""
    try:
        storage = SecureTemplateStorage()
        success = storage.delete_template(user_id)

        if success:
            print(f"✓ User '{user_id}' deleted successfully")
            security_logger.log_security_event(
                SecurityEventType.TEMPLATE_ACCESS,
                user_id=user_id,
                additional_data={"action": "delete"},
                success=True
            )
            return True
        else:
            print(f"✗ Failed to delete user '{user_id}'")
            return False

    except Exception as e:
        logger.error(f"Error deleting user: {e}")
        print(f"✗ Error deleting user: {e}")
        return False


def test_camera(config_path: Optional[str] = None) -> bool:
    """Test camera functionality."""
    try:
        import cv2

        config_manager = ConfigManager(config_path)
        camera_id = config_manager.get('camera.device_id', 0)

        print(f"Testing camera {camera_id}...")

        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            print(f"✗ Cannot open camera {camera_id}")
            return False

        ret, frame = cap.read()
        if not ret:
            print(f"✗ Cannot capture frame from camera {camera_id}")
            cap.release()
            return False

        print(f"✓ Camera {camera_id} working - captured {frame.shape}")
        cap.release()
        return True

    except Exception as e:
        logger.error(f"Camera test error: {e}")
        print(f"✗ Camera test failed: {e}")
        return False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Lockless Biometric Authentication System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --enroll --user john_doe --password secret123
  %(prog)s --authenticate --user john_doe --password secret123
  %(prog)s --list-users
  %(prog)s --test-camera
  %(prog)s --delete-user john_doe
    %(prog)s --gui
        """
    )

    # Main actions
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--enroll', action='store_true',
                       help='Enroll a new user')
    group.add_argument('--authenticate', action='store_true',
                       help='Authenticate a user')
    group.add_argument('--list-users', action='store_true',
                       help='List enrolled users')
    group.add_argument('--delete-user', metavar='USER_ID',
                       help='Delete an enrolled user')
    group.add_argument('--test-camera', action='store_true',
                       help='Test camera functionality')
    group.add_argument('--version', action='store_true',
                       help='Show version information')
    group.add_argument('--gui', action='store_true',
                       help='Launch the graphical interface')

    # Parameters
    parser.add_argument('--user', metavar='USER_ID',
                        help='User identifier')
    parser.add_argument('--password', metavar='PASSWORD',
                        help='Master password for template encryption')
    parser.add_argument('--config', metavar='CONFIG_PATH',
                        help='Path to configuration file')
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                        default='INFO', help='Logging level')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Enable verbose output')

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.log_level)

    if args.verbose:
        print("Lockless Biometric Authentication System")
        print("=" * 40)

    try:
        if args.version:
            print("Lockless v1.0.0")
            print("Privacy-first biometric authentication")
            return 0

        elif args.enroll:
            if not args.user:
                parser.error("--enroll requires --user")
            if not args.password:
                parser.error("--enroll requires --password")

            success = enroll_user(args.user, args.password, args.config)
            return 0 if success else 1

        elif args.authenticate:
            if not args.user:
                parser.error("--authenticate requires --user")
            if not args.password:
                parser.error("--authenticate requires --password")

            success = authenticate_user(args.user, args.password, args.config)
            return 0 if success else 1

        elif args.list_users:
            list_users(args.config)
            return 0

        elif args.delete_user:
            success = delete_user(args.delete_user, args.config)
            return 0 if success else 1

        elif args.test_camera:
            success = test_camera(args.config)
            return 0 if success else 1

        elif args.gui:
            from src.ui.app import run_gui  # Lazy import to avoid PyQt dependency unless needed

            exit_code = run_gui(args.config)
            return exit_code

        else:
            parser.print_help()
            return 1

    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        return 1
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        print(f"✗ Unexpected error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
