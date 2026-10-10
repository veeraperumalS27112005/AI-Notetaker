@echo off
echo Running AI Learning App from venv (Python 3.13 + deps)...
cd /d "%~dp0"
venv\Scripts\python.exe "record and summarize.py"
pause

