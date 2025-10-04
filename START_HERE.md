# 🔐 Lockless - Quick Start with Virtual Environment

## One-Command Setup

```bash
python setup.py
```

This automatically:
- ✅ Creates `.venv` virtual environment
- ✅ Installs all dependencies
- ✅ Sets up directories and config
- ✅ Tests camera connectivity

## Running Commands

### Option 1: Use the launcher scripts (Recommended)

**Windows:**
```bash
# Test camera
run.bat --test-camera

# Enroll user
run.bat --enroll --user john --password secret123

# Authenticate
run.bat --authenticate --user john --password secret123

# List users
run.bat --list-users
```

**Linux/Mac:**
```bash
# Test camera
./run.sh --test-camera

# Enroll user
./run.sh --enroll --user john --password secret123

# Authenticate
./run.sh --authenticate --user john --password secret123

# List users
./run.sh --list-users
```

### Option 2: Manual virtual environment activation

**Windows:**
```bash
.venv\Scripts\activate
python src\main.py --test-camera
```

**Linux/Mac:**
```bash
source .venv/bin/activate
python src/main.py --test-camera
```

## Interactive Guide

For first-time setup with step-by-step guidance:

```bash
python get_started.py
```

## Quick Test Sequence

1. **Setup** (one time): `python setup.py`
2. **Test camera**: `run.bat --test-camera` (Windows) or `./run.sh --test-camera` (Linux/Mac)
3. **Enroll yourself**: `run.bat --enroll --user your_name --password your_password`
4. **Test authentication**: `run.bat --authenticate --user your_name --password your_password`

## Files Overview

- **`setup.py`** - One-command setup and dependency installation
- **`get_started.py`** - Interactive step-by-step guide
- **`run.bat`** - Windows launcher (auto-activates .venv)
- **`run.sh`** - Linux/Mac launcher (auto-activates .venv)
- **`QUICKSTART.md`** - Detailed documentation
- **`.venv/`** - Virtual environment (created automatically)

The system automatically uses the `.venv` virtual environment so you don't need to worry about activation!