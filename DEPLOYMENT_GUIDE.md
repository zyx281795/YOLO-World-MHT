# YOLO-World YOLOv11 Integration - Cross-Platform Deployment Guide

## 概述

本指南提供YOLO-World與YOLOv11整合的完整部署說明，包含所有必要的檔案修改和配置步驟，適用於跨平台部署。

## 📁 檔案結構

### 新增檔案

```
yolo_world/
├── models/
│   ├── layers/
│   │   └── yolov11_blocks.py          # YOLOv11核心組件
│   └── backbones/
│       ├── yolov11_backbone.py        # YOLOv11骨幹網絡
│       └── yolov11_mm_backbone.py     # YOLOv11多模態骨幹
├── configs/
│   └── yolov11_integration/
│       └── yolo_world_v2_l_yolov11_backbone.py  # 配置檔案
└── test_yolov11_integration.py        # 整合測試腳本
```

### 修改檔案

```
yolo_world/models/
├── layers/__init__.py                 # 註冊YOLOv11組件
└── backbones/__init__.py              # 註冊YOLOv11骨幹網絡
```

## 🚀 部署步驟

### 第一步：環境準備

1. **安裝依賴**
```bash
# 在您的conda環境中執行
pip install ultralytics  # YOLOv11支援
```

2. **確認環境**
```bash
# 檢查YOLO-World基本環境
python -c "import torch; import mmdet; import mmyolo; print('Environment OK')"
```

### 第二步：複製檔案

將以下檔案複製到目標平台的相應位置：

1. **核心組件檔案**
   - `yolo_world/models/layers/yolov11_blocks.py`
   - `yolo_world/models/backbones/yolov11_backbone.py`
   - `yolo_world/models/backbones/yolov11_mm_backbone.py`

2. **配置檔案**
   - `configs/yolov11_integration/yolo_world_v2_l_yolov11_backbone.py`

3. **測試檔案**
   - `test_yolov11_integration.py`

### 第三步：更新註冊檔案

#### 更新 `yolo_world/models/layers/__init__.py`

```python
# 在原有導入後添加
from .yolov11_blocks import (
    YOLOv11Conv,
    YOLOv11Bottleneck,
    C3k,
    C3k2,
    PSABlock,
    MultiHeadAttention,
    C2PSA,
    SPPF
)

# 在__all__列表中添加
__all__.extend([
    'YOLOv11Conv',
    'YOLOv11Bottleneck', 
    'C3k',
    'C3k2',
    'PSABlock',
    'MultiHeadAttention',
    'C2PSA',
    'SPPF'
])
```

#### 更新 `yolo_world/models/backbones/__init__.py`

```python
# 在原有導入後添加
from .yolov11_backbone import (
    YOLOv11Backbone,
    YOLOv11CSPBackbone,
    YOLOv11MultiScaleBackbone)

from .yolov11_mm_backbone import (
    YOLOv11MultiModalBackbone,
    AttentionFusion,
    AdaptiveFusion)

# 在__all__列表中添加
__all__.extend([
    'YOLOv11Backbone',
    'YOLOv11CSPBackbone', 
    'YOLOv11MultiScaleBackbone',
    'YOLOv11MultiModalBackbone',
    'AttentionFusion',
    'AdaptiveFusion'
])
```

### 第四步：驗證安裝

運行測試腳本驗證整合是否成功：

```bash
python test_yolov11_integration.py
```

預期輸出應包含：
- ✓ YOLOv11 blocks working correctly
- ✓ YOLOv11 Backbone working correctly  
- ✓ Multi-Modal Backbone working correctly
- ✓ Configuration validation passed
- ✓ Model building successful

## 🛠️ 配置選項

### YOLOv11 架構選擇

```python
# 在配置檔案中修改
backbone=dict(
    type='YOLOv11MultiModalBackbone',
    image_model=dict(
        type='YOLOv11Backbone',
        arch='YOLOv11l',  # 可選: YOLOv11n, YOLOv11s, YOLOv11l
        depth_multiple=1.0,  # 深度倍數
        width_multiple=1.0,  # 寬度倍數
    ),
    # ... 其他配置
)
```

### 特徵融合方式

```python
# 選擇融合方式
feature_fusion_cfg=dict(
    type='attention',  # 選項: 'attention', 'adaptive', 'simple'
    text_channels=512,
)
```

### 輸出層選擇

```python
# 配置輸出層
image_model=dict(
    out_indices=(4, 6, 10),  # P3, P4, P5對應層
    # 或者使用更多輸出: (2, 4, 6, 8, 10)
)
```

## 🔧 訓練配置

### 基本訓練

```bash
# 使用新配置進行訓練
python tools/train.py configs/yolov11_integration/yolo_world_v2_l_yolov11_backbone.py
```

### 從預訓練權重開始

```python
# 在配置檔案中設置
load_from = 'path/to/yolo_world_pretrained.pth'
```

### 學習率調整

```python
# 針對YOLOv11優化的學習率
base_lr = 1e-3  # 較低的學習率以確保穩定性
optim_wrapper = dict(
    optimizer=dict(
        type='AdamW',
        lr=base_lr,
        weight_decay=0.025,  # 降低權重衰減
    ),
    paramwise_cfg=dict(
        custom_keys={
            'backbone.text_model': dict(lr_mult=0.01),  # 文本模型較低學習率
            'backbone.image_model': dict(lr_mult=1.0),  # YOLOv11骨幹全學習率
        }
    )
)
```

## 🐛 故障排除

### 常見問題

1. **導入錯誤**
   ```
   ModuleNotFoundError: No module named 'yolo_world.models.layers.yolov11_blocks'
   ```
   **解決方案**: 確認檔案路徑正確，並更新了`__init__.py`

2. **CUDA記憶體不足**
   ```
   RuntimeError: CUDA out of memory
   ```
   **解決方案**: 
   - 降低batch size
   - 使用較小的模型 (YOLOv11s instead of YOLOv11l)
   - 啟用gradient checkpointing

3. **配置檔案錯誤**
   ```
   KeyError: 'YOLOv11MultiModalBackbone'
   ```
   **解決方案**: 確認所有組件都已正確註冊

### 性能優化

1. **記憶體優化**
```python
# 在配置中添加
train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=80,
    val_interval=5,
    # 啟用混合精度訓練
    fp16=dict(loss_scale=512.0)
)
```

2. **推理加速**
```python
# 模型融合以提升推理速度
model.fuse()
```

## 📊 預期改進

相較於原始YOLOv8-based YOLO-World，YOLOv11整合版本預期提供：

- **精度提升**: C2PSA注意力機制改善小物體檢測
- **效率提升**: C3k2塊降低計算複雜度
- **穩定性**: 改進的特徵提取和融合
- **泛化性**: 更好的跨領域表現

## 🔄 版本兼容性

| 組件 | 最低版本 | 推薦版本 |
|------|---------|---------|
| PyTorch | 1.8.0 | 2.0+ |
| MMCV | 2.0.0 | 2.1+ |
| MMDetection | 3.0.0 | 3.2+ |
| MMYOLO | 0.6.0 | 最新 |
| Ultralytics | 8.3.0 | 8.3.178+ |

## 📝 重要注意事項

1. **備份原始配置**: 在進行修改前備份原始的YOLO-World配置
2. **逐步測試**: 建議先在小型數據集上測試整合效果
3. **監控記憶體**: YOLOv11 + 文本處理可能增加記憶體使用
4. **調整超參數**: 根據具體任務調整學習率和訓練策略

## 🚀 快速開始範例

```bash
# 1. 複製所有必要檔案到目標位置
# 2. 更新__init__.py檔案
# 3. 運行測試
python test_yolov11_integration.py

# 4. 開始訓練
python tools/train.py configs/yolov11_integration/yolo_world_v2_l_yolov11_backbone.py

# 5. 進行推理
python demo/gradio_demo.py configs/yolov11_integration/yolo_world_v2_l_yolov11_backbone.py path/to/trained/model.pth
```

---

**完成！** 您現在擁有一個完整的YOLO-World + YOLOv11整合系統，可以在任何支援的平台上部署。