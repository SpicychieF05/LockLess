# Lockless Biometric Authentication System - Quick Start Guide

This guide will walk you through setting up and running the Lockless biometric authentication system step by step.

## Prerequisites

- **Python 3.8 or higher** (Python 3.9+ recommended)
- **Webcam** (built-in or USB camera)
- **Windows 10/11** (primary support), Linux, or Android development environment

## Step 1: Environment Setup

### Option A: Automated Setup (Recommended)

1. **Run the setup script**:
```bash
python setup.py
```

This will:
- Check Python version compatibility
- Create a virtual environment
- Install all required dependencies
- Set up necessary directories
- Create a development configuration
- Test camera connectivity

### Option B: Manual Setup

1. **Create virtual environment**:
```bash
python -m venv .venv
```

2. **Activate virtual environment**:
```bash
# Windows
.venv\Scripts\activate

# Linux/Mac
source .venv/bin/activate
```

3. **Install dependencies**:
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

4. **Create directories**:
```bash
mkdir data logs models templates tests\data config
```

## Step 2: Verify Installation

1. **Test basic functionality**:
```bash
python src/main.py --version
```

2. **Test camera access**:
```bash
python src/main.py --test-camera
```

Expected output:
```
✓ Camera detected and working
✓ Face detection model loaded
✓ System ready for biometric operations
```

## Step 3: Configure the System

1. **Check configuration**:
```bash
python src/main.py --show-config
```

2. **Edit configuration if needed** (optional):
   - Open `config/default.yaml`
   - Adjust camera settings, thresholds, etc.
   - For development, use lower thresholds for easier testing

## Step 4: Enroll Your First User

1. **Start enrollment process**:
```bash
python src/main.py --enroll --user john_doe --password mypassword123
```

2. **Follow the enrollment prompts**:
   - Position your face in the camera frame
   - Keep your face steady and well-lit
   - The system will capture multiple samples automatically
   - Wait for "Enrollment completed successfully" message

Expected output:
```
Starting enrollment for user: john_doe
Initializing camera...
✓ Camera ready
✓ Face detection active

Please position your face in the frame...
Capturing sample 1/5... ✓
Capturing sample 2/5... ✓
Capturing sample 3/5... ✓
Capturing sample 4/5... ✓
Capturing sample 5/5... ✓

Processing biometric template...
✓ Template generated successfully
✓ Template encrypted and stored
✓ Enrollment completed for user: john_doe
```

## Step 5: Test Authentication

1. **Authenticate the enrolled user**:
```bash
python src/main.py --authenticate --user john_doe --password mypassword123
```

2. **Face the camera**:
   - Position your face as during enrollment
   - Keep steady until authentication completes

Expected output:
```
Starting authentication for user: john_doe
Initializing camera...
✓ Camera ready
✓ Face detection active

Please look at the camera...
✓ Face detected
✓ Liveness check passed
✓ Biometric matching...
✓ Authentication successful!

User: john_doe
Confidence: 95.2%
Time taken: 387ms
```

## Step 6: Explore Additional Features

### List enrolled users:
```bash
python src/main.py --list-users
```

### Run system diagnostics:
```bash
python src/main.py --diagnostics
```

### Performance benchmark:
```bash
python src/main.py --benchmark
```

### Enable debug mode for troubleshooting:
```bash
python src/main.py --debug --authenticate --user john_doe --password mypassword123
```

## Troubleshooting

### Camera Issues

1. **Camera not detected**:
   - Check if other applications are using the camera
   - Try different camera IDs: `--camera-id 1`, `--camera-id 2`
   - Update camera drivers

2. **Poor image quality**:
   - Ensure good lighting (avoid backlighting)
   - Clean camera lens
   - Adjust `quality_threshold` in config for testing

### Performance Issues

1. **Slow authentication (>1 second)**:
   - Enable GPU acceleration in config (if available)
   - Lower image resolution in config
   - Check system resources with `--diagnostics`

### Authentication Failures

1. **Face not detected**:
   - Ensure proper lighting
   - Position face directly facing camera
   - Remove glasses/masks if causing issues

2. **Liveness detection fails**:
   - Blink naturally during authentication
   - Slight head movement helps
   - Ensure you're not using a photo/video

3. **Biometric match fails**:
   - Try multiple authentication attempts
   - Re-enroll if issues persist
   - Check similarity threshold in config

### Error Messages

1. **"Module not found" errors**:
```bash
# Ensure virtual environment is activated
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux

# Reinstall dependencies
pip install -r requirements.txt
```

2. **"Camera permission denied"**:
   - Grant camera permissions to terminal/command prompt
   - Run as administrator (Windows)
   - Check privacy settings

3. **"TPM not available"**:
   - Normal on many systems
   - System will fall back to software encryption
   - No action needed for testing

## Development and Testing

### Run the test suite:
```bash
pytest tests/ -v
```

### Run with different log levels:
```bash
python src/main.py --log-level DEBUG --authenticate --user john_doe --password mypassword123
```

### Test with multiple users:
```bash
# Enroll multiple users
python src/main.py --enroll --user alice --password pass123
python src/main.py --enroll --user bob --password pass456

# Test authentication
python src/main.py --authenticate --user alice --password pass123
python src/main.py --authenticate --user bob --password pass456
```

## Next Steps

1. **Integrate with your application**: Use the API modules in `src/api/`
2. **Customize UI**: Modify the GUI components in `src/ui/`
3. **Deploy**: Follow the deployment guide in `docs/deployment.md`
4. **Scale**: Set up the server components for multi-user environments

## Performance Expectations

- **Authentication time**: < 500ms (typically 200-400ms)
- **Enrollment time**: 10-30 seconds (depends on sample count)
- **Memory usage**: ~200-500MB
- **CPU usage**: Minimal when idle, spike during operations

## Security Notes

- All biometric templates are encrypted at rest
- Passwords are hashed using PBKDF2
- Camera access is only active during operations
- No biometric data is stored in plain text
- System logs exclude sensitive information

## Support

If you encounter issues:
1. Run with `--debug` flag for detailed logs
2. Check the logs in the `logs/` directory
3. Ensure all dependencies are correctly installed
4. Verify camera and system permissions

The system is designed to be robust and user-friendly. Most issues are related to environment setup or camera access permissions.