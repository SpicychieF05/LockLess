# Lockless Deployment Guide

## Overview

This guide provides comprehensive instructions for deploying the Lockless biometric authentication system across Windows, Linux, and Android platforms.

## Prerequisites

### Hardware Requirements

#### Minimum Requirements
- **CPU**: x86_64 or ARM64, 2+ cores, 2.0 GHz
- **RAM**: 4 GB minimum, 8 GB recommended
- **Storage**: 2 GB free space for installation
- **Camera**: RGB camera (webcam), 720p minimum
- **Optional**: IR camera, depth camera for enhanced security

#### Recommended Requirements
- **CPU**: x86_64, 4+ cores, 3.0 GHz or ARM64 equivalent
- **RAM**: 16 GB for optimal performance
- **Storage**: SSD with 10 GB free space
- **Camera**: 1080p RGB + IR camera or depth sensor
- **GPU**: NVIDIA GTX 1060/AMD RX 580 or better for acceleration

### Software Dependencies

#### Windows
- Windows 10 version 1903+ or Windows 11
- Visual C++ Redistributable 2019+
- .NET Framework 4.7.2+
- Windows Hello compatible hardware (optional)

#### Linux
- Ubuntu 20.04+, CentOS 8+, or equivalent
- Python 3.8+
- OpenCV dependencies
- PAM development libraries

#### Android
- Android 8.0 (API level 26)+
- Camera2 API support
- Biometric authentication hardware

## Windows Deployment

### Installation Methods

#### Method 1: MSI Installer (Recommended)

1. **Download Installer**
   ```powershell
   # Download from official release
   Invoke-WebRequest -Uri "https://releases.lockless.app/v1.0.0/lockless-windows-x64.msi" -OutFile "lockless-installer.msi"
   ```

2. **Run Installation**
   ```powershell
   # Silent installation
   msiexec /i lockless-installer.msi /quiet /l*v install.log
   
   # Interactive installation
   msiexec /i lockless-installer.msi
   ```

3. **Verify Installation**
   ```powershell
   # Check service status
   Get-Service -Name "LocklessService"
   
   # Test basic functionality
   & "C:\Program Files\Lockless\lockless.exe" --version
   ```

#### Method 2: Manual Installation

1. **Extract Files**
   ```powershell
   # Create installation directory
   New-Item -ItemType Directory -Path "C:\Program Files\Lockless"
   
   # Extract application files
   Expand-Archive -Path "lockless-windows.zip" -DestinationPath "C:\Program Files\Lockless"
   ```

2. **Install Service**
   ```powershell
   # Register Windows service
   & "C:\Program Files\Lockless\lockless.exe" --install-service
   
   # Start service
   Start-Service -Name "LocklessService"
   ```

3. **Configure Registry**
   ```powershell
   # Import registry settings
   reg import "C:\Program Files\Lockless\config\lockless.reg"
   ```

### Windows Configuration

#### Service Configuration
```xml
<!-- lockless-service.xml -->
<service>
    <id>LocklessService</id>
    <name>Lockless Biometric Authentication</name>
    <description>Provides biometric authentication services</description>
    <executable>C:\Program Files\Lockless\lockless-service.exe</executable>
    <startmode>Automatic</startmode>
    <delayedAutoStart>true</delayedAutoStart>
    <logpath>C:\ProgramData\Lockless\logs</logpath>
    <logmode>rotate</logmode>
</service>
```

#### Registry Configuration
```reg
Windows Registry Editor Version 5.00

[HKEY_LOCAL_MACHINE\SOFTWARE\Lockless]
"InstallPath"="C:\\Program Files\\Lockless"
"DataPath"="C:\\ProgramData\\Lockless"
"Version"="1.0.0"
"EnableLogging"=dword:00000001
"LogLevel"="INFO"

[HKEY_LOCAL_MACHINE\SOFTWARE\Lockless\Security]
"EncryptionEnabled"=dword:00000001
"TPMEnabled"=dword:00000001
"FallbackAuthEnabled"=dword:00000001

[HKEY_LOCAL_MACHINE\SOFTWARE\Lockless\Performance]
"MaxAuthenticationTime"=dword:00001388
"SimilarityThreshold"="0.7"
"QualityThreshold"="0.6"
```

### Windows Hello Integration

1. **Enable Windows Hello**
   ```powershell
   # Check Windows Hello status
   Get-WindowsOptionalFeature -Online -FeatureName "WindowsHelloFace"
   
   # Configure Lockless as credential provider
   reg add "HKLM\SOFTWARE\Microsoft\Windows\CurrentVersion\Authentication\Credential Providers\{LOCKLESS-GUID}" /v "Disabled" /t REG_DWORD /d 0
   ```

2. **Group Policy Configuration**
   ```
   Computer Configuration > Administrative Templates > Windows Components > Windows Hello for Business
   - Use Windows Hello for Business: Enabled
   - Use biometrics: Enabled
   - Configure camera: Allow Lockless Camera Provider
   ```

## Linux Deployment

### Package Installation

#### Ubuntu/Debian
```bash
# Add Lockless repository
curl -fsSL https://packages.lockless.app/gpg | sudo apt-key add -
echo "deb https://packages.lockless.app/ubuntu focal main" | sudo tee /etc/apt/sources.list.d/lockless.list

# Update package list
sudo apt update

# Install Lockless
sudo apt install lockless-biometric

# Install optional components
sudo apt install lockless-pam-module lockless-gui
```

#### CentOS/RHEL/Fedora
```bash
# Add Lockless repository
sudo yum-config-manager --add-repo https://packages.lockless.app/centos/lockless.repo

# Install Lockless
sudo yum install lockless-biometric

# Or using dnf on Fedora
sudo dnf install lockless-biometric
```

### Manual Installation

1. **Extract and Install**
   ```bash
   # Extract package
   tar -xzf lockless-linux-x64.tar.gz
   cd lockless-linux-x64
   
   # Run installation script
   sudo ./install.sh
   
   # Or manual installation
   sudo cp -r bin/* /usr/local/bin/
   sudo cp -r lib/* /usr/local/lib/
   sudo cp -r share/* /usr/local/share/
   ```

2. **Install Dependencies**
   ```bash
   # Ubuntu/Debian
   sudo apt install libopencv-dev libssl-dev libpam-dev python3-dev
   
   # CentOS/RHEL
   sudo yum install opencv-devel openssl-devel pam-devel python3-devel
   ```

### PAM Integration

1. **Configure PAM Module**
   ```bash
   # Copy PAM module
   sudo cp /usr/local/lib/security/pam_lockless.so /lib/security/
   
   # Set permissions
   sudo chmod 755 /lib/security/pam_lockless.so
   ```

2. **Update PAM Configuration**
   ```bash
   # /etc/pam.d/common-auth (Ubuntu/Debian)
   # Add before other auth modules
   auth    sufficient    pam_lockless.so
   auth    [default=1 success=ok]    pam_unix.so nullok_secure
   ```

   ```bash
   # /etc/pam.d/system-auth (CentOS/RHEL)
   auth        sufficient    pam_lockless.so
   auth        include       password-auth-ac
   ```

### Systemd Service

1. **Service Configuration**
   ```ini
   # /etc/systemd/system/lockless.service
   [Unit]
   Description=Lockless Biometric Authentication Service
   After=network.target
   
   [Service]
   Type=forking
   ExecStart=/usr/local/bin/lockless-daemon
   ExecReload=/bin/kill -HUP $MAINPID
   PIDFile=/var/run/lockless.pid
   User=lockless
   Group=lockless
   WorkingDirectory=/var/lib/lockless
   
   # Security settings
   NoNewPrivileges=true
   ProtectSystem=strict
   ProtectHome=true
   ReadWritePaths=/var/lib/lockless /var/log/lockless
   
   [Install]
   WantedBy=multi-user.target
   ```

2. **Enable and Start Service**
   ```bash
   # Reload systemd
   sudo systemctl daemon-reload
   
   # Enable service
   sudo systemctl enable lockless
   
   # Start service
   sudo systemctl start lockless
   
   # Check status
   sudo systemctl status lockless
   ```

### Desktop Integration

1. **Install GUI Components**
   ```bash
   # Install desktop files
   sudo cp share/applications/*.desktop /usr/share/applications/
   sudo cp share/icons/lockless.png /usr/share/icons/
   
   # Update desktop database
   sudo update-desktop-database
   ```

2. **GNOME Integration**
   ```bash
   # Install GNOME Shell extension
   cp -r share/gnome-shell/extensions/lockless@lockless.app ~/.local/share/gnome-shell/extensions/
   
   # Enable extension
   gnome-extensions enable lockless@lockless.app
   ```

## Android Deployment

### APK Installation

1. **Direct Installation**
   ```bash
   # Install via ADB
   adb install lockless-android.apk
   
   # Or download and install from device
   # Settings > Security > Install from Unknown Sources (if needed)
   ```

2. **Google Play Store** (Future)
   ```
   Search for "Lockless Biometric Auth" in Google Play Store
   ```

### Android Integration

1. **Device Administrator Setup**
   ```kotlin
   // Grant device administrator permissions
   val intent = Intent(DevicePolicyManager.ACTION_ADD_DEVICE_ADMIN)
   intent.putExtra(DevicePolicyManager.EXTRA_DEVICE_ADMIN, locklessAdminReceiver)
   startActivityForResult(intent, REQUEST_ENABLE_ADMIN)
   ```

2. **Biometric Manager Configuration**
   ```kotlin
   // Configure biometric authentication
   val biometricManager = BiometricManager.from(this)
   when (biometricManager.canAuthenticate(BIOMETRIC_WEAK)) {
       BiometricManager.BIOMETRIC_SUCCESS ->
           // Configure Lockless as biometric provider
           configureLocklessBiometric()
   }
   ```

### Android Permissions

```xml
<!-- AndroidManifest.xml -->
<uses-permission android:name="android.permission.CAMERA" />
<uses-permission android:name="android.permission.USE_BIOMETRIC" />
<uses-permission android:name="android.permission.USE_FINGERPRINT" />
<uses-permission android:name="android.permission.WRITE_EXTERNAL_STORAGE" />

<!-- Device administrator permission -->
<receiver android:name=".LocklessDeviceAdminReceiver"
          android:permission="android.permission.BIND_DEVICE_ADMIN">
    <meta-data android:name="android.app.device_admin"
               android:resource="@xml/device_admin" />
    <intent-filter>
        <action android:name="android.app.action.DEVICE_ADMIN_ENABLED" />
    </intent-filter>
</receiver>
```

## Configuration Management

### Global Configuration

```yaml
# /etc/lockless/config.yaml
system:
  version: "1.0.0"
  log_level: "INFO"
  data_directory: "/var/lib/lockless"
  
camera:
  device_id: 0
  resolution:
    width: 640
    height: 480
  fps: 30
  
authentication:
  similarity_threshold: 0.7
  quality_threshold: 0.6
  max_attempts: 3
  lockout_duration: 300
  
security:
  encryption_enabled: true
  tpm_enabled: true
  key_derivation_iterations: 100000
  
performance:
  max_authentication_time: 5000  # milliseconds
  enable_gpu_acceleration: true
  max_concurrent_users: 10
```

### User-Specific Configuration

```yaml
# ~/.lockless/user_config.yaml
user:
  id: "john_doe"
  enrolled_at: "2024-01-15T10:30:00Z"
  
preferences:
  auto_lock_timeout: 300
  fallback_auth_enabled: true
  liveness_detection_enabled: true
  
quality_settings:
  min_face_size: 100
  brightness_tolerance: 0.2
  pose_angle_tolerance: 30
```

## Security Hardening

### File Permissions

```bash
# Set secure permissions
sudo chown -R lockless:lockless /var/lib/lockless
sudo chmod 700 /var/lib/lockless
sudo chmod 600 /var/lib/lockless/templates/*
sudo chmod 644 /etc/lockless/config.yaml
```

### Network Security

```bash
# Configure firewall (if API enabled)
sudo ufw allow from 127.0.0.1 to any port 8080
sudo ufw deny 8080

# Or using iptables
sudo iptables -A INPUT -s 127.0.0.1 -p tcp --dport 8080 -j ACCEPT
sudo iptables -A INPUT -p tcp --dport 8080 -j DROP
```

### SELinux Configuration (RHEL/CentOS)

```bash
# Create SELinux policy for Lockless
sudo setsebool -P authlogin_yubikey 1
sudo semanage fcontext -a -t bin_t "/usr/local/bin/lockless"
sudo restorecon -v /usr/local/bin/lockless
```

## Monitoring and Maintenance

### Log Configuration

```yaml
# logging.yaml
version: 1
formatters:
  default:
    format: '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
  security:
    format: '%(asctime)s - SECURITY - %(levelname)s - %(message)s'

handlers:
  file:
    class: logging.handlers.RotatingFileHandler
    filename: /var/log/lockless/lockless.log
    maxBytes: 10485760  # 10MB
    backupCount: 5
    formatter: default
  
  security_file:
    class: logging.handlers.RotatingFileHandler
    filename: /var/log/lockless/security.log
    maxBytes: 10485760
    backupCount: 10
    formatter: security

loggers:
  lockless:
    level: INFO
    handlers: [file]
  lockless.security:
    level: INFO
    handlers: [security_file]
```

### Health Monitoring

```bash
#!/bin/bash
# health_check.sh

# Check service status
if ! systemctl is-active --quiet lockless; then
    echo "ERROR: Lockless service not running"
    exit 1
fi

# Check camera access
if ! /usr/local/bin/lockless --test-camera; then
    echo "WARNING: Camera access issues"
fi

# Check disk space
if [ $(df /var/lib/lockless | tail -1 | awk '{print $5}' | sed 's/%//') -gt 90 ]; then
    echo "WARNING: Low disk space"
fi

echo "OK: System healthy"
```

### Backup and Recovery

```bash
#!/bin/bash
# backup.sh

BACKUP_DIR="/backup/lockless/$(date +%Y%m%d)"
mkdir -p "$BACKUP_DIR"

# Backup configuration
cp -r /etc/lockless "$BACKUP_DIR/"

# Backup user data (encrypted)
cp -r /var/lib/lockless "$BACKUP_DIR/"

# Backup logs
cp -r /var/log/lockless "$BACKUP_DIR/"

# Create archive
tar -czf "$BACKUP_DIR.tar.gz" "$BACKUP_DIR"
rm -rf "$BACKUP_DIR"

echo "Backup completed: $BACKUP_DIR.tar.gz"
```

## Troubleshooting

### Common Issues

1. **Camera Not Detected**
   ```bash
   # List available cameras
   v4l2-ctl --list-devices
   
   # Test camera access
   ffmpeg -f v4l2 -i /dev/video0 -t 5 test.mp4
   ```

2. **Service Won't Start**
   ```bash
   # Check service logs
   sudo journalctl -u lockless -f
   
   # Check configuration
   /usr/local/bin/lockless --check-config
   ```

3. **Authentication Failures**
   ```bash
   # Check authentication logs
   sudo tail -f /var/log/lockless/security.log
   
   # Test enrollment
   /usr/local/bin/lockless --test-enrollment
   ```

### Performance Optimization

1. **GPU Acceleration**
   ```bash
   # Install NVIDIA drivers
   sudo apt install nvidia-driver-470
   
   # Verify GPU access
   nvidia-smi
   
   # Enable GPU in configuration
   echo "performance.enable_gpu_acceleration: true" >> /etc/lockless/config.yaml
   ```

2. **Memory Optimization**
   ```bash
   # Adjust memory limits
   echo "performance.max_memory_mb: 512" >> /etc/lockless/config.yaml
   
   # Enable memory monitoring
   echo "monitoring.memory_alerts: true" >> /etc/lockless/config.yaml
   ```

## Support and Documentation

### Getting Help
- **Documentation**: https://docs.lockless.app
- **Community Forum**: https://forum.lockless.app
- **GitHub Issues**: https://github.com/lockless-auth/lockless/issues
- **Email Support**: support@lockless.app

### Version Updates
```bash
# Check for updates
lockless --check-updates

# Update via package manager
sudo apt update && sudo apt upgrade lockless-biometric

# Manual update
sudo ./update.sh --from-version 1.0.0 --to-version 1.1.0
```

This deployment guide ensures successful installation and configuration of Lockless across all supported platforms while maintaining security and performance standards.