@echo off
REM Same thing with synthetic data. No exchange connection needed.
cd /d "%~dp0"
echo Starting with mock data on http://localhost:8000
start "" http://localhost:8000
python server.py --mock
pause
