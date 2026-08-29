@echo off
cd /d "%~dp0"
title SkeleX Settings GUI
echo ====================================================
echo  SkeleX Settings GUI Launcher
echo ====================================================
echo.

:: Try default python first
python main.py
if %errorlevel% equ 0 goto :end

echo.
echo [WARNING] Default python failed or not found.
echo Trying virtual environment python...
echo.

:: Try HeliosProject virtual environment python
"C:\Users\jmike\AppData\Local\HeliosProject\Helios\python\InputSense-CUDA\.henv\Scripts\python.exe" main.py
if %errorlevel% equ 0 goto :end

echo.
echo [ERROR] Failed to launch settings GUI.
pause

:end
