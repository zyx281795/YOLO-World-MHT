# Copyright (c) Tencent Inc. All rights reserved.
import os
import sys
import argparse
import os.path as osp

# Import YOLOv11 components FIRST
import yolo_world
from yolo_world.models.layers import yolov11_blocks
from yolo_world.models.backbones import yolov11_backbone

print("✅ YOLOv11 components loaded successfully!")
print(f"📦 Available blocks: C3k2, C2PSA, SPPF, YOLOv11Conv")

# Now import the rest
from io import BytesIO
from functools import partial

import cv2
import onnx
import torch
import onnxsim
import numpy as np
import gradio as gr
from PIL import Image
import supervision as sv
from torchvision.ops import nms
from mmengine.runner import Runner
from mmengine.dataset import Compose
from mmengine.runner.amp import autocast
from mmengine.config import Config, DictAction, ConfigDict
from mmdet.datasets import CocoDataset
from mmyolo.registry import RUNNERS

sys.path.append('./deploy')
from easydeploy import model as EM

BOUNDING_BOX_ANNOTATOR = sv.BoundingBoxAnnotator(thickness=1)
MASK_ANNOTATOR = sv.MaskAnnotator()


class LabelAnnotator(sv.LabelAnnotator):

    @staticmethod
    def resolve_text_background_xyxy(
        center_coordinates,
        text_wh,
        position,
    ):
        center_x, center_y = center_coordinates
        text_w, text_h = text_wh
        return center_x, center_y, center_x + text_w, center_y + text_h


LABEL_ANNOTATOR = LabelAnnotator(text_padding=4,
                                 text_scale=0.5,
                                 text_thickness=1)


def parse_args():
    parser = argparse.ArgumentParser(description='YOLO-World Demo with YOLOv11 Components')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        '--work-dir',
        help='the directory to save the file containing evaluation metrics',
        default='output')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    args = parser.parse_args()
    return args


def run_image(runner,
              image,
              text,
              max_num_boxes,
              score_thr,
              nms_thr,
              image_path='./work_dirs/demo.png'):
    # image.save(image_path)
    texts = [[t.strip()] for t in text.split(',')] + [[' ']]
    data_info = dict(img_id=0, img=np.array(image), texts=texts)
    data_info = runner.pipeline(data_info)
    data_batch = dict(inputs=data_info['inputs'].unsqueeze(0),
                      data_samples=[data_info['data_samples']])

    with autocast(enabled=False), torch.no_grad():
        output = runner.model.test_step(data_batch)[0]
        pred_instances = output.pred_instances

    keep = nms(pred_instances.bboxes,
               pred_instances.scores,
               iou_threshold=nms_thr)
    pred_instances = pred_instances[keep]
    pred_instances = pred_instances[pred_instances.scores.float() > score_thr]

    if len(pred_instances.scores) > max_num_boxes:
        indices = pred_instances.scores.float().topk(max_num_boxes)[1]
        pred_instances = pred_instances[indices]

    pred_instances = pred_instances.cpu().numpy()
    detections = sv.Detections(xyxy=pred_instances['bboxes'],
                               class_id=pred_instances['labels'],
                               confidence=pred_instances['scores'])
    labels = [
        f"{texts[class_id][0]} {confidence:0.2f}" 
        for class_id, confidence in zip(detections.class_id, detections.confidence)
    ]

    image = np.array(image)
    image = BOUNDING_BOX_ANNOTATOR.annotate(image, detections)
    image = LABEL_ANNOTATOR.annotate(image, detections, labels=labels)
    return Image.fromarray(image), str(len(pred_instances['labels']))


def export_model(runner,
                 text_prompt,
                 max_num_boxes,
                 score_thr,
                 nms_thr,
                 model_path='./work_dirs/yolo_world_with_yolov11.onnx'):
    texts = [[t.strip()] for t in text_prompt.split(',')] + [[' ']]
    data_info = dict(img_id=0, img_path='', texts=texts)
    data_info = runner.pipeline(data_info)
    reparameterized_runner = EM.reparameterize(runner.model, data_info)
    
    deploy_model = EM.DeployModel(
        baseModel=reparameterized_runner,
        backend=EM.MMYoloBackend.ONNXRUNTIME,
        postprocess_cfg=EM.PostprocessConfig(
            score_threshold=score_thr,
            nms_threshold=nms_thr,
            max_output_boxes_per_class=max_num_boxes,
            pre_top_k=5000,
            keep_top_k=max_num_boxes,
            background_label_id=-1,
        ))
    
    fake_input = torch.randn(1, 3, 640, 640).to(next(runner.model.parameters()).device)
    save_onnx_path = model_path
    
    os.makedirs(os.path.dirname(save_onnx_path), exist_ok=True)
    
    with BytesIO() as f:
        output_names = ['num_dets', 'boxes', 'scores', 'labels']
        torch.onnx.export(deploy_model,
                          fake_input,
                          f,
                          input_names=['images'],
                          output_names=output_names,
                          opset_version=12)
        f.seek(0)
        onnx_model = onnx.load(f)
        onnx.checker.check_model(onnx_model)
    onnx_model, check = onnxsim.simplify(onnx_model)
    onnx.save(onnx_model, save_onnx_path)
    return gr.update(visible=True), save_onnx_path


def demo(runner, args):
    with gr.Blocks(title="YOLO-World + YOLOv11 Components") as demo:
        with gr.Row():
            gr.Markdown('<h1><center>🚀 YOLO-World + YOLOv11 Components</center></h1>')
        
        with gr.Row():
            gr.Markdown(f"""
            ### ✅ YOLOv11 Integration Status:
            - **C3k2 blocks**: Available and imported ✓
            - **C2PSA attention**: Available and imported ✓  
            - **SPPF blocks**: Available and imported ✓
            - **YOLOv11Conv**: Available and imported ✓
            
            **Note**: 目前使用YOLOv8架構 + YOLOv11組件支援
            """)
        
        with gr.Row():
            with gr.Column(scale=0.3):
                with gr.Row():
                    image = gr.Image(type='pil', label='上傳圖片')
                input_text = gr.Textbox(
                    lines=7,
                    label='輸入要偵測的類別（用逗號分隔）',
                    value=', '.join(CocoDataset.METAINFO['classes']),
                    elem_id='textbox')
                with gr.Row():
                    submit = gr.Button('🔍 開始偵測')
                    clear = gr.Button('🗑️ 清除')
                with gr.Row():
                    export = gr.Button('📦 匯出ONNX模型')
                with gr.Row():
                    gr.Markdown(
                        "⏳ 生成ONNX檔案需要幾秒鐘時間！"
                    )
                out_download = gr.File(visible=False)
                max_num_boxes = gr.Slider(minimum=1,
                                          maximum=300,
                                          value=100,
                                          step=1,
                                          interactive=True,
                                          label='最大偵測框數量')
                score_thr = gr.Slider(minimum=0,
                                      maximum=1,
                                      value=0.05,
                                      step=0.001,
                                      interactive=True,
                                      label='信心度閾值')
                nms_thr = gr.Slider(minimum=0,
                                    maximum=1,
                                    value=0.7,
                                    step=0.001,
                                    interactive=True,
                                    label='NMS閾值')
            with gr.Column(scale=0.7):
                output_image = gr.Image(type='pil', label='偵測結果')
                detected_count = gr.Textbox(label='偵測到的物體數量', interactive=False)

        submit.click(partial(run_image, runner),
                     [image, input_text, max_num_boxes, score_thr, nms_thr],
                     [output_image, detected_count])
        clear.click(lambda: [None, '', None, '0'], None,
                    [image, input_text, output_image, detected_count])

        export.click(partial(export_model, runner),
                     [input_text, max_num_boxes, score_thr, nms_thr],
                     [out_download, out_download])

        demo.launch(server_name='0.0.0.0',
                    server_port=8082)


if __name__ == '__main__':
    args = parse_args()

    # load config
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])

    cfg.load_from = args.checkpoint

    # Enable YOLOv11 components by ensuring they're imported
    print("🔧 Ensuring YOLOv11 components are available...")
    
    if 'runner_type' not in cfg:
        runner = Runner.from_cfg(cfg)
    else:
        runner = RUNNERS.build(cfg)

    runner.call_hook('before_run')
    runner.load_or_resume()
    pipeline = cfg.test_dataloader.dataset.pipeline
    pipeline[0].type = 'mmdet.LoadImageFromNDArray'
    runner.pipeline = Compose(pipeline)
    runner.model.eval()
    
    print("✅ Demo ready with YOLOv11 components support!")
    print(f"📊 Backbone: {type(runner.model.backbone).__name__}")
    print(f"🎯 Available YOLOv11 blocks: C3k2, C2PSA, SPPF, YOLOv11Conv")
    
    demo(runner, args)