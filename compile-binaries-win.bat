@echo off
setlocal EnableExtensions
cd /d "%~dp0"

where python >nul 2>&1
if errorlevel 1 (
  echo ERROR: python not found on PATH.
  echo Install Python 3.13 ^(or 3.11^) and check "Add python.exe to PATH".
  exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
  echo Creating .venv...
  python -m venv .venv
  if errorlevel 1 exit /b 1
)

echo Installing project dependencies into .venv...
".venv\Scripts\python.exe" -m pip install -U pip
if errorlevel 1 exit /b 1
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 exit /b 1

echo Installing PyInstaller into .venv (if needed^)...
".venv\Scripts\python.exe" -m pip install -q -U pyinstaller
if errorlevel 1 exit /b 1

echo Building PyStereo (batch GUI + CLI^)...
".venv\Scripts\python.exe" -m PyInstaller packaging\pystereo_batch.spec
if errorlevel 1 exit /b 1

echo Building PyStereoWeb (Flask server^)...
".venv\Scripts\python.exe" -m PyInstaller packaging\pystereo_web.spec
if errorlevel 1 exit /b 1

".venv\Scripts\python.exe" -c "from pystereo_core._version import __version__; print(__version__)" > _ver_tmp.txt
set /p VERSION=<_ver_tmp.txt
del _ver_tmp.txt

echo Packaging archives...
cd dist
powershell -NoProfile -Command "Compress-Archive -Force -Path PyStereo -DestinationPath 'PyStereo-%VERSION%-win.zip'; Compress-Archive -Force -Path PyStereoWeb -DestinationPath 'PyStereoWeb-%VERSION%-win.zip'"
cd ..

echo.
echo ========================================================================
echo Build finished OK -- version %VERSION%.
echo.
echo Batch tool (GUI/CLI^):
echo   %cd%\dist\PyStereo\PyStereo.exe
echo   Folder: %cd%\dist\PyStereo\
echo.
echo Web UI (Flask server -- open http://127.0.0.1:8766 after starting^):
echo   %cd%\dist\PyStereoWeb\PyStereoWeb.exe
echo   Folder: %cd%\dist\PyStereoWeb\
echo.
echo Archives (upload these to the GitHub release^):
echo   %cd%\dist\PyStereo-%VERSION%-win.zip
echo   %cd%\dist\PyStereoWeb-%VERSION%-win.zip
echo ========================================================================
if /i "%~1"=="nopause" (
  endlocal
  exit /b 0
)
echo.
pause
endlocal
exit /b 0
