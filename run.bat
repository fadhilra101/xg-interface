@echo off
echo ========================================
echo xG Prediction Interface
echo ========================================
echo.

echo Checking Python installation...
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please run setup.bat first or install Python
    pause
    exit /b 1
)

echo Checking if requirements are installed...
python -c "import streamlit" 2>nul
if errorlevel 1 (
    echo Requirements not found. Installing with user permissions...
    pip install --user --no-warn-script-location -r requirements.txt
    if errorlevel 1 (
        echo Installation failed. Please run setup.bat or install manually.
        pause
        exit /b 1
    )
)

echo.
echo Checking for model file...
if exist "xg_model.joblib" (
    echo Model file found: xg_model.joblib
) else (
    echo No model file found. App will create a dummy model for demonstration.
    echo See MODEL_PLACEMENT.md for instructions on placing your trained model.
)

echo.
echo Starting xG Prediction Interface...
echo.
echo The application will open in your default web browser.
echo To stop the application, press Ctrl+C in this window.
echo.

REM Try streamlit command first, fallback to python -m streamlit if not found
streamlit run app.py 2>nul
if errorlevel 1 (
    echo Streamlit command not found in PATH. Using python -m streamlit...
    python -m streamlit run app.py
)

pause
