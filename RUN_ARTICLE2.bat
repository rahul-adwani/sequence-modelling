@echo off
setlocal
cd /d "%~dp0"

echo ==============================================================
echo   Foundations  -  Article 2: Mathematics of Sequence Modelling
echo   s01 output/loss . s02 RNN . s03 gradients . s04 LSTM
echo   s05 bottleneck  . s06 attention . s07 transformer/GPT
echo ==============================================================
echo.

where python >nul 2>&1
if errorlevel 1 (
  echo ERROR: python was not found on PATH.
  echo        Install Python 3.10+ or open a shell where "python" resolves.
  goto :end
)

if not exist "logs" mkdir "logs"

echo [1/4]  Dependency check
python -c "import numpy" 2>nul
if errorlevel 1 (
  echo        numpy missing - installing from requirements.txt
  python -m pip install -r requirements.txt --quiet
) else (
  echo        numpy present
)
echo.

echo [2/4]  Compile check  (catches syntax errors before anything runs)
python -m compileall -q foundations run_all.py
if errorlevel 1 (
  echo        FAILED - syntax error shown above. Nothing was executed.
  goto :end
)
echo        ok
echo.

echo [3/4]  Unit tests
python -m pytest tests -q
echo.

echo [4/4]  Running sections and writing logs
echo.
python run_all.py --article 2
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
echo   Section transcripts : %cd%\logs\s0*.log
echo   Machine-readable    : %cd%\logs\claims.json
echo ==============================================================
echo.

:end
pause
endlocal
