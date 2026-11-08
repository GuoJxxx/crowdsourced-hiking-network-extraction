@echo off
setlocal
cd /d "%~dp0.."
set PYTHONUTF8=1
set PYTHONDONTWRITEBYTECODE=1
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "Chicago\Evaluation\Runtime\workflow\Configure.ps1" -Task Chicago
pause
