@echo off
setlocal
cd /d "%~dp0"

echo ==============================================================
echo   Foundations  -  Article 3: Mathematics of the Inference Stack
echo   s08 tokenization . s09 the model as a function
echo   s10 decoding     . s11 KV cache . s12 latency and cost
echo ==============================================================
echo.

where python >nul 2>&1
if errorlevel 1 (
  echo ERROR: python was not found on PATH.
  echo        Install Python 3.10+ or open a shell where "python" resolves.
  goto :end
)

if not exist "logs" mkdir "logs"

echo [1/5]  Dependency check
python -c "import numpy" 2>nul
if errorlevel 1 (
  echo        numpy missing - installing from requirements.txt
  python -m pip install -r requirements.txt --quiet
) else (
  echo        numpy present
)
echo.

echo [2/5]  Compile check  (catches syntax errors before anything runs)
python -m compileall -q foundations run_all.py
if errorlevel 1 (
  echo        FAILED - syntax error shown above. Nothing was executed.
  goto :end
)
echo        ok
echo.

echo [3/5]  Register check  (no em-dash, en-dash or unicode minus anywhere)
python check_register.py
if errorlevel 1 goto :end
echo.

echo [4/5]  Unit tests
python -m pytest tests -q
echo.

echo [5/5]  Running sections and writing logs
echo.
python run_all.py --article 3
set RC=%errorlevel%

echo.
echo ==============================================================
if "%RC%"=="0" (
  echo   RESULT: every claim passed.
) else (
  echo   RESULT: one or more claims FAILED.
  echo           The failing claims are listed above, with the measured
  echo           value that decided each one. That list is the useful
  echo           output - send it back and it says what to change.
)
echo.
echo   Section transcripts : %cd%\logs\s08.log .. s12.log
echo   Machine-readable    : %cd%\logs\claims.json
echo ==============================================================
echo.

:end
pause
endlocal
