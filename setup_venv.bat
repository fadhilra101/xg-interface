@echo off
echo ========================================
echo xG Prediction Interface - Virtual Environment Setup
echo ========================================
echo.

echo [1/5] Checking Python installation...
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python from https://python.org
    pause
    exit /b 1
)
echo Python found!

echo.
echo [2/5] Creating virtual environment...
if exist "venv" (
    echo Virtual environment already exists.
) else (
    echo Creating new virtual environment...
    python -m venv venv
    if errorlevel 1 (
        echo ERROR: Failed to create virtual environment
        echo Make sure you have venv module installed
        pause
        exit /b 1
    )
)

echo.
echo [3/5] Activating virtual environment...
call venv\Scripts\activate
if errorlevel 1 (
    echo ERROR: Failed to activate virtual environment
    pause
    exit /b 1
)

echo.
echo [4/5] Upgrading pip...
python -m pip install --upgrade pip

echo.
echo [5/5] Installing requirements...
pip install -r requirements.txt
if errorlevel 1 (
    echo ERROR: Failed to install requirements
    echo Please check your internet connection and try again
    pause
    exit /b 1
)

echo.
echo ========================================
echo Virtual Environment Setup completed!
echo ========================================
echo.
echo Virtual environment created in: venv\
echo.
echo To use the application:
echo 1. Run 'run_venv.bat' (recommended)
echo 2. Or manually activate: venv\Scripts\activate
echo    Then run: streamlit run app.py
echo.
pause
