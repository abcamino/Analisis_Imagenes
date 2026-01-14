@echo off
echo ============================================
echo  Setup Training Environment for Aneurysm Detection
echo ============================================
echo.

REM Check if Python 3.11 is available
where py >nul 2>nul
if %errorlevel% neq 0 (
    echo ERROR: Python launcher 'py' not found.
    echo Please install Python from https://www.python.org/downloads/
    pause
    exit /b 1
)

echo Checking for Python 3.11...
py -3.11 --version >nul 2>nul
if %errorlevel% neq 0 (
    echo.
    echo Python 3.11 not found!
    echo.
    echo Please install Python 3.11 from:
    echo   https://www.python.org/downloads/release/python-3119/
    echo.
    echo After installing, run this script again.
    pause
    exit /b 1
)

echo Python 3.11 found!
py -3.11 --version
echo.

REM Check if virtual environment exists
if exist "training_env" (
    echo Virtual environment 'training_env' already exists.
    set /p RECREATE="Recreate it? (y/N): "
    if /i "%RECREATE%"=="y" (
        echo Removing old environment...
        rmdir /s /q training_env
    ) else (
        echo Using existing environment.
        goto :activate
    )
)

echo Creating virtual environment...
py -3.11 -m venv training_env

:activate
echo.
echo Activating virtual environment...
call training_env\Scripts\activate.bat

echo.
echo Installing PyTorch (CPU version)...
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

echo.
echo Installing additional dependencies...
pip install timm onnx onnxruntime
pip install opencv-python-headless numpy matplotlib
pip install albumentations scikit-learn tqdm pyyaml

echo.
echo ============================================
echo  Setup Complete!
echo ============================================
echo.
echo To use the training environment:
echo   1. Activate: training_env\Scripts\activate
echo   2. Train: python train_model.py --data_dir ..\data\processed --epochs 50
echo   3. Export: python export_onnx.py --checkpoint models\best_model.pth
echo.
echo To export a pretrained model for testing:
echo   python export_onnx.py --pretrained --output ..\models\onnx\mobilenetv3_aneurysm.onnx
echo.
pause
