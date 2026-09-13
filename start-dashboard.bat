@echo off
REM Double-click to launch the dashboard.
REM Edit the taker fee below to match your exchange tier.

set TAKER=0.60

cd /d "%~dp0"

python --version >nul 2>&1
if errorlevel 1 (
  echo Python was not found on your PATH.
  echo Install it from python.org and tick "Add python.exe to PATH".
  pause
  exit /b 1
)

echo Starting on http://localhost:8000
start "" http://localhost:8000
python server.py --taker %TAKER%
pause
