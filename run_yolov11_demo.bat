@echo off
echo ====================================
echo   YOLO-World + YOLOv11 Demo 啟動器
echo ====================================
echo.

cd /d "C:\Users\Ryan\OneDrive - 中山醫學大學\桌面\YOLO-World-MHT"

echo 設定環境變數...
set PYTHONPATH=C:\Users\Ryan\OneDrive - 中山醫學大學\桌面\YOLO-World-MHT

echo.
echo 選擇要啟動的demo:
echo 1. YOLOv8 版本 (已驗證可用) - Port 8081
echo 2. YOLOv11 簡化版 - Port 8083  
echo 3. 測試YOLOv11組件
echo.

set /p choice="請輸入選擇 (1-3): "

if "%choice%"=="1" (
    echo 啟動YOLOv8版本...
    conda run -n yolo_wd python simple_gradio_demo.py
) else if "%choice%"=="2" (
    echo 啟動YOLOv11簡化版...
    conda run -n yolo_wd python yolov11_simple_demo.py
) else if "%choice%"=="3" (
    echo 測試YOLOv11組件...
    conda run -n yolo_wd python -c "import yolo_world.models.layers.yolov11_blocks as yv11; import torch; conv = yv11.YOLOv11Conv(3, 64, 3, 2); x = torch.randn(1, 3, 640, 640); out = conv(x); print(f'YOLOv11組件測試成功: {x.shape} -> {out.shape}')"
) else (
    echo 無效的選擇
)

echo.
pause