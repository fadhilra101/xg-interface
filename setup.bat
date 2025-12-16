@echo off
setlocal enabledelayedexpansion

echo ========================================
echo xG Prediction Interface - Setup Script
echo ========================================
echo.
echo This script will:
echo - Check and install Python if needed
echo - Check and install/update pip
echo - Install all required packages
echo - Verify installation
echo.

REM Check if running as administrator
net session >nul 2>&1
set "ADMIN_RIGHTS=%errorlevel%"

echo [1/5] Checking Python installation...
python --version >nul 2>&1
if errorlevel 1 (
    echo Python not found in PATH. Checking common locations...
    
    REM Check common Python installation paths
    for %%P in (
        "C:\Python3*\python.exe"
        "C:\Python\python.exe" 
        "%LOCALAPPDATA%\Programs\Python\Python3*\python.exe"
        "%PROGRAMFILES%\Python3*\python.exe"
        "%PROGRAMFILES(X86)%\Python3*\python.exe"
    ) do (
        if exist "%%P" (
            set "PYTHON_PATH=%%P"
            echo Found Python at: !PYTHON_PATH!
            goto :python_found
        )
    )
    
    echo WARNING: Python not found!
    echo.
    echo Please install Python 3.8 or newer from:
    echo https://www.python.org/downloads/windows/
    echo.
    echo Make sure to check "Add Python to PATH" during installation
    echo.
    set /p "choice=Continue anyway? (y/N): "
    if /i not "!choice!"=="y" (
        pause
        exit /b 1
    )
    set "PYTHON_CMD=python"
) else (
    :python_found
    if defined PYTHON_PATH (
        set "PYTHON_CMD=!PYTHON_PATH!"
    ) else (
        set "PYTHON_CMD=python"
    )
    echo Python found!
    !PYTHON_CMD! --version
)

echo.
echo [2/5] Checking and updating pip...
!PYTHON_CMD! -m pip --version >nul 2>&1
if errorlevel 1 (
    echo pip not found. Installing pip...
    !PYTHON_CMD! -m ensurepip --upgrade
    if errorlevel 1 (
        echo Failed to install pip automatically.
        echo Please install pip manually from: https://pip.pypa.io/en/stable/installation/
        pause
        exit /b 1
    )
) else (
    echo pip found! Updating to latest version...
    !PYTHON_CMD! -m pip install --upgrade pip --user
)

echo.
echo [3/5] Checking system architecture and Python version...
!PYTHON_CMD! -c "import sys, platform; print(f'Python {sys.version[:5]} on {platform.machine()}')"

echo.
echo [4/5] Installing requirements...
echo Installing Python packages from requirements.txt...
echo This may take a few minutes, especially for LightGBM...
echo.

REM Try different installation methods
set "INSTALL_SUCCESS=0"

REM Method 1: Standard user install
echo Trying standard user installation...
!PYTHON_CMD! -m pip install --user --no-warn-script-location -r requirements.txt
if not errorlevel 1 (
    set "INSTALL_SUCCESS=1"
    goto :install_complete
)

REM Method 2: With --break-system-packages (for some Python distributions)
echo Trying with --break-system-packages...
!PYTHON_CMD! -m pip install --user --break-system-packages --no-warn-script-location -r requirements.txt
if not errorlevel 1 (
    set "INSTALL_SUCCESS=1"
    goto :install_complete
)

REM Method 3: Global install (if admin rights)
if !ADMIN_RIGHTS! equ 0 (
    echo Trying global installation (Administrator mode)...
    !PYTHON_CMD! -m pip install --no-warn-script-location -r requirements.txt
    if not errorlevel 1 (
        set "INSTALL_SUCCESS=1"
        goto :install_complete
    )
)

REM Method 4: Force reinstall
echo Trying force reinstall...
!PYTHON_CMD! -m pip install --user --force-reinstall --no-warn-script-location -r requirements.txt
if not errorlevel 1 (
    set "INSTALL_SUCCESS=1"
    goto :install_complete
)

:install_complete
if !INSTALL_SUCCESS! equ 0 (
    echo.
    echo ERROR: Failed to install requirements with all methods
    echo.
    echo Troubleshooting suggestions:
    echo 1. Run this script as Administrator
    echo 2. Use virtual environment: setup_venv.bat
    echo 3. Install packages manually: python -m pip install streamlit pandas lightgbm
    echo 4. Check your internet connection
    echo.
    pause
    exit /b 1
)

echo.
echo [5/5] Verifying installation...
echo Checking core packages...

!PYTHON_CMD! -c "import streamlit; print('✓ Streamlit:', streamlit.__version__)" 2>nul
if errorlevel 1 echo ✗ Streamlit installation failed

!PYTHON_CMD! -c "import pandas; print('✓ Pandas:', pandas.__version__)" 2>nul
if errorlevel 1 echo ✗ Pandas installation failed

!PYTHON_CMD! -c "import lightgbm; print('✓ LightGBM:', lightgbm.__version__)" 2>nul
if errorlevel 1 echo ✗ LightGBM installation failed

!PYTHON_CMD! -c "import numpy; print('✓ NumPy:', numpy.__version__)" 2>nul
if errorlevel 1 echo ✗ NumPy installation failed

!PYTHON_CMD! -c "import matplotlib; print('✓ Matplotlib:', matplotlib.__version__)" 2>nul
if errorlevel 1 echo ✗ Matplotlib installation failed

echo.
echo ========================================
echo        SETUP COMPLETED SUCCESSFULLY!
echo ========================================
echo.
echo You can now run the application with:
echo   - run.bat (direct launch)
echo   - run_venv.bat (if using virtual environment)
echo.
echo If you encounter any issues, try using the virtual environment setup:
echo   setup_venv.bat
echo.
pause

echo.
echo ========================================
echo Setup completed successfully!
echo ========================================
echo.
echo Next steps:
echo 1. Place your trained model file (xg_model.joblib) in the root directory
echo 2. Run 'run.bat' to start the application
echo.
echo If you don't have a model file, the app will create a dummy model for demonstration.
echo.
pause
