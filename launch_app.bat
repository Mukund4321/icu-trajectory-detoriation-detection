@echo off
title ICU Trajectory Detection - Web Interface
cls

echo ===============================================
echo   ICU TRAJECTORY DETERIORATION DETECTION
echo   Web Interface Launcher
echo ===============================================
echo.
echo Starting Streamlit web application...
echo.
echo The app will open in your default browser.
echo Press Ctrl+C to stop the server.
echo.
echo ===============================================
echo.

streamlit run app.py

pause
