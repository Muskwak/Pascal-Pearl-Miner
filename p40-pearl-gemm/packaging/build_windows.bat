@echo off
REM Build the closed-source Windows binary (dist\p40-miner\p40-miner.exe).
REM Run from the p40-pearl-gemm directory:  packaging\build_windows.bat
setlocal
cd /d "%~dp0\.."

echo [1/3] Ensuring the CUDA extension is built (sm_61)...
python -c "import glob,sys; sys.exit(0 if glob.glob('build/lib.*/p40_pearl_gemm_cuda.*') else 1)"
if %ERRORLEVEL% NEQ 0 (
    echo   building extension...
    python setup.py build_ext --inplace || goto :err
)

echo [2/3] Installing PyInstaller if needed...
python -c "import PyInstaller" 2>nul || python -m pip install pyinstaller || goto :err

echo [3/3] Freezing the miner...
pyinstaller packaging\p40-miner.spec --noconfirm --distpath dist --workpath build_pyi || goto :err

echo.
echo Done. Share the whole folder:  dist\p40-miner\
echo Run with:  dist\p40-miner\p40-miner.exe --wallet prl1YOURWALLET --worker p40
goto :eof

:err
echo BUILD FAILED
exit /b 1
