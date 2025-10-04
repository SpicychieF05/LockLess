@echo off
REM Lockless Biometric Authentication System - Windows Launcher
REM This script automatically uses the virtual environment and runs commands

setlocal

REM Check if virtual environment exists
if not exist ".venv" (
    echo Virtual environment not found. Please run setup first:
    echo python setup.py
    pause
    exit /b 1
)

REM Check if arguments provided
if "%~1"=="" (
    echo Lockless Biometric Authentication System
    echo.
    echo Usage: run.bat [command]
    echo.
    echo Examples:
    echo   run.bat --test-camera
    echo   run.bat --enroll --user john --password secret123
    echo   run.bat --authenticate --user john --password secret123
    echo   run.bat --list-users
    echo   run.bat --diagnostics
    echo.
    pause
    exit /b 0
)

REM Set Python path and run the main application with virtual environment python
set PYTHONPATH=.
.venv\Scripts\python.exe src\main.py %*

REM Keep window open if there was an error
if errorlevel 1 (
    echo.
    echo Command failed. Press any key to exit...
    pause >nul
)
    echo.
    echo Command failed. Press any key to exit...
    pause >nul
)