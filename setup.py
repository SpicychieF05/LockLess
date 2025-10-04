#!/usr/bin/env python3
"""
Setup script for Lockless Biometric Authentication System.
This script helps set up the development environment and dependencies.
"""

import os
import sys
import subprocess
import platform
from pathlib import Path


def run_command(command, description="", check=True):
    """Run a command and handle errors."""
    print(f"Running: {description or command}")
    try:
        result = subprocess.run(command, shell=True, check=check,
                                capture_output=True, text=True)
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
        if e.stderr:
            print(f"Error output: {e.stderr}")
        return False


def check_python_version():
    """Check if Python version is compatible."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("Error: Python 3.8 or higher is required")
        print(
            f"Current version: {version.major}.{version.minor}.{version.micro}")
        return False
    print(
        f"✓ Python version {version.major}.{version.minor}.{version.micro} is compatible")
    return True


def create_virtual_environment():
    """Create a virtual environment."""
    if os.path.exists(".venv"):
        print("✓ Virtual environment already exists")
        return True

    print("Creating virtual environment...")
    return run_command("python -m venv .venv", "Creating virtual environment")


def activate_venv_command():
    """Get the command to activate virtual environment."""
    if platform.system() == "Windows":
        return ".venv\\Scripts\\activate"
    else:
        return "source .venv/bin/activate"


def install_dependencies():
    """Install Python dependencies."""
    # Determine pip command based on platform
    if platform.system() == "Windows":
        pip_cmd = ".venv\\Scripts\\pip"
        python_cmd = ".venv\\Scripts\\python"
    else:
        pip_cmd = ".venv/bin/pip"
        python_cmd = ".venv/bin/python"

    print("Installing dependencies...")

    # Upgrade pip first
    if not run_command(f"{pip_cmd} install --upgrade pip", "Upgrading pip"):
        return False

    # Install requirements
    if os.path.exists("requirements.txt"):
        return run_command(f"{pip_cmd} install -r requirements.txt",
                           "Installing requirements.txt")
    else:
        # Install basic dependencies manually
        basic_deps = [
            "opencv-python>=4.8.0",
            "numpy>=1.24.0",
            "PyYAML>=6.0",
            "cryptography>=41.0.0",
            "Pillow>=10.0.0",
            "psutil>=5.9.0"
        ]

        for dep in basic_deps:
            if not run_command(f"{pip_cmd} install {dep}", f"Installing {dep}"):
                return False
        return True


def setup_directories():
    """Create necessary directories."""
    directories = [
        "data",
        "logs",
        "models",
        "templates",
        "tests/data"
    ]

    for dir_path in directories:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"✓ Created directory: {dir_path}")

    return True


def check_camera():
    """Check if camera is available."""
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            ret, frame = cap.read()
            cap.release()
            if ret:
                print("✓ Camera is working")
                return True
            else:
                print("⚠ Camera detected but cannot capture frames")
                return False
        else:
            print("⚠ No camera detected or camera is busy")
            return False
    except ImportError:
        print("⚠ OpenCV not installed, cannot test camera")
        return False
    except Exception as e:
        print(f"⚠ Camera test failed: {e}")
        return False


def create_sample_config():
    """Create a sample configuration file."""
    config_content = """# Lockless Configuration
system:
  log_level: "DEBUG"  # Use DEBUG for development
  data_directory: "./data"

camera:
  device_id: 0  # Change if you have multiple cameras

authentication:
  similarity_threshold: 0.6  # Lower for testing (normally 0.7)
  quality_threshold: 0.5     # Lower for testing (normally 0.6)

enrollment:
  required_samples: 3  # Fewer samples for testing (normally 5)
  quality_threshold: 0.5

performance:
  enable_gpu_acceleration: false  # Set to true if you have compatible GPU
"""

    with open("config/development.yaml", "w") as f:
        f.write(config_content)

    print("✓ Created development configuration file")


def main():
    """Main setup function."""
    print("Lockless Biometric Authentication System - Setup")
    print("=" * 50)

    # Check Python version
    if not check_python_version():
        return 1

    # Create virtual environment
    if not create_virtual_environment():
        print("Failed to create virtual environment")
        return 1

    # Install dependencies
    if not install_dependencies():
        print("Failed to install dependencies")
        return 1

    # Setup directories
    if not setup_directories():
        print("Failed to setup directories")
        return 1

    # Create sample config
    create_sample_config()

    print("\n" + "=" * 50)
    print("Setup completed successfully!")
    print("\nNext steps:")
    print(f"1. Activate virtual environment: {activate_venv_command()}")
    print("2. Test camera: python src/main.py --test-camera")
    print("3. Enroll a user: python src/main.py --enroll --user test_user --password test123")
    print("4. Authenticate: python src/main.py --authenticate --user test_user --password test123")

    # Test camera if possible
    print("\nTesting camera...")
    check_camera()

    return 0


if __name__ == "__main__":
    sys.exit(main())
