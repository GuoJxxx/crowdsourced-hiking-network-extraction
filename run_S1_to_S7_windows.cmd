@echo off
setlocal EnableExtensions DisableDelayedExpansion
title Activity-Aware Pipeline S1-S7 Configurable
set "PS_SCRIPT=%~dp0code\tools\run_pipeline_configurable.ps1"
echo [1/2] Checking launcher...
if not exist "%PS_SCRIPT%" (
  echo ERROR: Missing PowerShell launcher:
  echo "%PS_SCRIPT%"
  pause
  exit /b 1
)
where powershell.exe >nul 2>&1
if errorlevel 1 (
  echo ERROR: powershell.exe was not found.
  pause
  exit /b 1
)
echo [2/2] Opening configuration window. Please check the taskbar if it appears behind this terminal.
powershell.exe -NoLogo -NoProfile -STA -ExecutionPolicy Bypass -File "%PS_SCRIPT%" -Mode S1S7
set "RC=%ERRORLEVEL%"
echo.
if not "%RC%"=="0" echo Pipeline stopped with exit code %RC%.
pause
exit /b %RC%
