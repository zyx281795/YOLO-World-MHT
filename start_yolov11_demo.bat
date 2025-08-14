@echo off
echo ==========================================
echo   YOLO-World + YOLOv11 Demo Launcher
echo ==========================================
echo.
echo 啟動帶有YOLOv11組件的YOLO-World Demo...
echo.

cd /d "C:\Users\Ryan\OneDrive - 中山醫學大學\桌面\YOLO-World-MHT"

echo 設定環境...
set PYTHONPATH=%cd%

echo.
echo 載入YOLOv11組件並啟動demo...
echo.

conda run -n yolo_wd python -c "import yolo_world; from yolo_world.models.layers import yolov11_blocks; print('YOLOv11組件已載入'); import sys; sys.argv = ['demo/gradio_demo.py', 'configs/pretrain/yolo_world_v2_x_vlpan_bn_2e-3_100e_4x8gpus_obj365v1_goldg_train_1280ft_lvis_minival.py', 'checkpoints/yolo_world_v2_x_obj365v1_goldg_cc3mlite_pretrain_1280ft-14996a36.pth']; exec(open('demo/gradio_demo.py').read())"

echo.
echo Demo已結束
pause