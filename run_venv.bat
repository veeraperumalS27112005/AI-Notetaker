@echo off
echo Activating venv and running AI Learning App...
cd /d "%~dp0"
call venv\Scripts\activate.bat
python "record and summarize.py"
pause

