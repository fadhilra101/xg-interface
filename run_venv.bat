@echo off
echo ========================================
echo xG Prediction Interface (Virtual Environment)
echo ========================================
echo.

echo Checking for virtual environment...
if not exist "venv\Scripts\activate.bat" (
    echo Virtual environment not found.
    echo Please run setup_venv.bat first.
    pause
    exit /b 1
)

echo Activating virtual environment...
call venv\Scripts\activate
if errorlevel 1 (
    echo ERROR: Failed to activate virtual environment
    pause
    exit /b 1
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
streamlit run app.py

echo.
echo Deactivating virtual environment...
deactivate

pause
