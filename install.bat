@echo off
echo Setting up venv and installing dependencies (Python 3.13 project env)...
cd /d "%~dp0"
if not exist venv (
    C:\Python313\python.exe -m venv venv
    echo Venv created.
)
call venv\Scripts\activate.bat
pip install --upgrade pip
pip install -r requirements.txt
echo.
echo Setup complete! Run: run_app.bat (uses venv python automatically)
echo Or: .\venv\Scripts\python.exe "record and summarize.py"
pause
