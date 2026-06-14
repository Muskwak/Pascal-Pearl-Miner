@echo off
REM Build the closed-source Windows binary (dist\p40-miner\p40-miner.exe).
REM Produces the torch-free ~60 MB bundle. Run from p40-pearl-gemm\ :
REM   packaging\build_windows.bat
setlocal
cd /d "%~dp0\.."

echo [1/3] Building the torch-free CUDA library (p40cuda.dll)...
if not exist p40cuda.dll call packaging\build_capi.bat || goto :err

echo [2/3] Installing PyInstaller if needed...
python -c "import PyInstaller" 2>nul || python -m pip install pyinstaller || goto :err

echo [3/3] Freezing the torch-free miner...
pyinstaller packaging\p40-miner-lite.spec --noconfirm --distpath dist --workpath build_pyi || goto :err

echo.
echo Done (~60 MB). Share the whole folder:  dist\p40-miner\
echo Run with:  dist\p40-miner\p40-miner.exe --wallet prl1YOURWALLET --worker p40
goto :eof

:err
echo BUILD FAILED
exit /b 1
