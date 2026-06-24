@echo off
REM WorldField global launcher for Windows.

setlocal DISABLEDELAYEDEXPANSION

set "SCRIPT_DIR=%~dp0"
if "%SCRIPT_DIR:~-1%"=="\" set "SCRIPT_DIR=%SCRIPT_DIR:~0,-1%"

set PY=py -3
%PY% --version >nul 2>&1 || set PY=py
%PY% --version >nul 2>&1 || (
    echo Python not found. Install Python 3.10+ from python.org
    pause
    exit /b 1
)

set "PYTHONPATH=%SCRIPT_DIR%;%PYTHONPATH%"

%PY% -c "import worldfield" 2>nul
if errorlevel 1 (
    if exist "%SCRIPT_DIR%\pyproject.toml" (
        echo Installing dependencies first run...
        %PY% -m pip install -e "%SCRIPT_DIR%" --quiet
        if errorlevel 1 (
            echo pip install failed.
            pause
            exit /b 1
        )
    ) else (
        echo worldfield package not found.
        pause
        exit /b 1
    )
)

%PY% -m worldfield --check-deps %*
