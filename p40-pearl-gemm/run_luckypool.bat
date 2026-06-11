@echo off
REM P40 Pearl Miner — luckypool.io
REM Usage: run_luckypool.bat --wallet YOUR_WALLET_ADDRESS [--worker NAME] [--pool HOST:PORT]

setlocal
set "DIR=%~dp0"
cd /d "%DIR%"

REM Ensure CUDA DLLs and PyTorch libs are findable
set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8"
if exist "%CUDA_PATH%\bin" set "PATH=%CUDA_PATH%\bin;%PATH%"

REM Find conda's torch lib directory for DLLs
if exist "%CONDA_PREFIX%\Lib\site-packages\torch\lib" (
    set "PATH=%CONDA_PREFIX%\Lib\site-packages\torch\lib;%PATH%"
)

python -m p40_pearl_gemm.luckypool_miner %*
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo If the miner fails, make sure you have:
    echo   1. CUDA 12.8 installed at C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8
    echo   2. The pearl_mining package installed (py-pearl-mining)
    echo   3. A Pascal GPU (Tesla P40, GTX 1070, etc.)
    pause
)
