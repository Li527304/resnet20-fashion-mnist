@echo off
chcp 65001 >nul
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" app.py
) else if exist "F:\深度学习实验\经典视觉算法\_公共资源\环境\vision\python.exe" (
  "F:\深度学习实验\经典视觉算法\_公共资源\环境\vision\python.exe" app.py
) else (
  python app.py
)
pause
