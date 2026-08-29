@echo off
echo ================================================
echo  SkeleX – Dependency Installer
echo ================================================
echo.

echo [1/6] Checking Python...
python --version
if errorlevel 1 (
    echo ERROR: Python not found. Install Python 3.12 from https://python.org
    pause & exit /b 1
)

echo.
echo [2/6] Core packages...
pip install opencv-contrib-python pillow numpy

echo.
echo [3/6] PyTorch with CUDA 12.8...
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

echo.
echo [4/6] Ultralytics (YOLOv8)...
pip install -U ultralytics

echo.
echo [5/6] TensorRT...
pip install tensorrt
pip install "nvidia-modelopt[onnx]>=0.44"

echo.
echo [6/6] Windows utils...
pip install pywin32 pyserial

echo.
echo ================================================
echo  All done!
echo  Run:  python main.py       (settings GUI)
echo  Then load SkeleX_GCV.py in Gtuner IV CV tab.
echo ================================================
pause
