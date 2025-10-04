#!/usr/bin/env python3
"""
Interactive Getting Started Guide for Lockless
This script provides step-by-step guidance to set up and test the system.
"""

import os
import sys
import time
import subprocess
from pathlib import Path


def print_step(step, title):
    """Print formatted step header."""
    print(f"\n{'='*60}")
    print(f"STEP {step}: {title}")
    print('='*60)


def wait_for_user(message="Press Enter to continue..."):
    """Wait for user input."""
    input(f"\n{message}")


def run_command_interactive(command, description):
    """Run a command with user interaction."""
    print(f"\nAbout to run: {command}")
    print(f"This will: {description}")

    response = input("\nProceed? (y/n): ").lower().strip()
    if response != 'y':
        print("Skipped.")
        return False

    print(f"\nRunning: {command}")
    print("-" * 40)

    try:
        result = subprocess.run(command, shell=True, check=True)
        print("-" * 40)
        print("✓ Command completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print("-" * 40)
        print(f"✗ Command failed with error: {e}")
        return False


def main():
    """Main interactive guide."""
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║                                                              ║
    ║          🔐 LOCKLESS BIOMETRIC AUTHENTICATION SYSTEM         ║
    ║                   Interactive Setup Guide                    ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
    
    This guide will help you set up and test the Lockless system.
    We'll go through each step together to ensure everything works.
    
    What we'll do:
    1. Check your environment
    2. Set up dependencies
    3. Test camera access
    4. Enroll a test user
    5. Test authentication
    """)

    wait_for_user("Ready to start? Press Enter...")

    # Step 1: Environment Check
    print_step(1, "Environment Check")

    # Check Python version
    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")

    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ ERROR: Python 3.8 or higher is required!")
        print("Please install a newer version of Python and try again.")
        return 1
    else:
        print("✓ Python version is compatible!")

    # Check if we're in the right directory
    if not os.path.exists("src/main.py"):
        print("❌ ERROR: Not in the Lockless project directory!")
        print("Please navigate to the Lockless project folder and run this script again.")
        return 1
    else:
        print("✓ Project files found!")

    wait_for_user()

    # Step 2: Dependencies Setup
    print_step(2, "Install Dependencies")

    print("We need to install the required Python packages.")
    print("This includes OpenCV for camera access, PyTorch for AI models, and cryptography for security.")

    if not run_command_interactive("python setup.py", "Install all dependencies and set up the environment"):
        print("\nIf the automatic setup failed, you can try manual installation:")
        print("1. python -m venv venv")
        print("2. venv\\Scripts\\activate  (Windows) or source venv/bin/activate (Linux)")
        print("3. pip install -r requirements.txt")

        manual = input("\nTry manual setup now? (y/n): ").lower().strip()
        if manual == 'y':
            print("\nManual setup steps:")
            run_command_interactive(
                "python -m venv venv", "Create virtual environment")

            # Activation command depends on platform
            if os.name == 'nt':  # Windows
                activate_cmd = "venv\\Scripts\\activate && pip install -r requirements.txt"
            else:  # Linux/Mac
                activate_cmd = "source venv/bin/activate && pip install -r requirements.txt"

            run_command_interactive(
                activate_cmd, "Activate environment and install packages")

    wait_for_user()

    # Step 3: Test Camera
    print_step(3, "Camera Test")

    print("Let's test if your camera is working with the system.")
    print("This will:")
    print("- Detect available cameras")
    print("- Test face detection")
    print("- Verify the biometric pipeline")

    # Determine python command (check if we're in venv)
    if os.path.exists(".venv"):
        if os.name == 'nt':  # Windows
            python_cmd = ".venv\\Scripts\\python"
        else:  # Linux/Mac
            python_cmd = ".venv/bin/python"
    else:
        python_cmd = "python"

    run_command_interactive(
        f"{python_cmd} src/main.py --test-camera", "Test camera functionality")

    wait_for_user()

    # Step 4: User Enrollment
    print_step(4, "Enroll Test User")

    print("Now let's enroll a test user in the system.")
    print("This process will:")
    print("- Capture your face from multiple angles")
    print("- Generate a secure biometric template")
    print("- Encrypt and store the template safely")

    print("\nDuring enrollment:")
    print("- Look directly at the camera")
    print("- Keep your face steady and well-lit")
    print("- Follow the on-screen prompts")

    username = input(
        "\nEnter a username for testing (default: test_user): ").strip()
    if not username:
        username = "test_user"

    password = input("Enter a password (default: test123): ").strip()
    if not password:
        password = "test123"

    enroll_cmd = f'{python_cmd} src/main.py --enroll --user {username} --password {password}'

    if run_command_interactive(enroll_cmd, f"Enroll user '{username}' in the biometric system"):
        print(f"\n✓ Successfully enrolled user: {username}")
    else:
        print(f"\n❌ Enrollment failed for user: {username}")
        print("This might be due to:")
        print("- Camera access issues")
        print("- Poor lighting conditions")
        print("- Missing dependencies")
        return 1

    wait_for_user()

    # Step 5: Authentication Test
    print_step(5, "Test Authentication")

    print(f"Let's test authentication with the enrolled user: {username}")
    print("This will:")
    print("- Capture your face in real-time")
    print("- Perform liveness detection (anti-spoofing)")
    print("- Match against the stored template")
    print("- Report the result and confidence score")

    print("\nDuring authentication:")
    print("- Look at the camera naturally")
    print("- Blink normally to pass liveness detection")
    print("- Keep your face visible and well-lit")

    auth_cmd = f'{python_cmd} src/main.py --authenticate --user {username} --password {password}'

    if run_command_interactive(auth_cmd, f"Authenticate user '{username}'"):
        print(f"\n🎉 Authentication test completed!")
    else:
        print(f"\n❌ Authentication failed")
        print("This might be due to:")
        print("- Different lighting conditions than enrollment")
        print("- Camera angle differences")
        print("- Liveness detection sensitivity")
        print("\nYou can try:")
        print("- Running authentication again")
        print("- Re-enrolling with better lighting")
        print("- Adjusting thresholds in config/default.yaml")

    wait_for_user()

    # Final Steps
    print_step(6, "Next Steps")

    print("🎉 Congratulations! You've successfully set up Lockless!")
    print("\nWhat you can do now:")
    print("1. Enroll more users:")
    print(f"   {python_cmd} src/main.py --enroll --user alice --password secret123")
    print()
    print("2. Test authentication multiple times:")
    print(
        f"   {python_cmd} src/main.py --authenticate --user {username} --password {password}")
    print()
    print("3. Check system performance:")
    print(f"   {python_cmd} src/main.py --benchmark")
    print()
    print("4. List enrolled users:")
    print(f"   {python_cmd} src/main.py --list-users")
    print()
    print("5. Run diagnostics if you encounter issues:")
    print(f"   {python_cmd} src/main.py --diagnostics")
    print()
    print("📖 For more details, check:")
    print("   - QUICKSTART.md - Comprehensive usage guide")
    print("   - docs/ - Full documentation")
    print("   - config/default.yaml - System configuration")

    print("\n" + "="*60)
    print("🔐 Lockless is ready to use! 🔐")
    print("="*60)

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\nSetup interrupted by user.")
        sys.exit(1)
