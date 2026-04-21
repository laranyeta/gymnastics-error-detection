import os
import cv2
import json
import torch
import numpy as np
import pandas as pd

from utils.data import sapiens2mediapipe
from utils.vision import normalize_pose_tensor, interpolation_smoothing, draw_skeleton
from sklearn.impute import KNNImputer
from scipy.signal import savgol_filter
from mmpretrain.models.backbones.vision_transformer import VisionTransformer
from mmpose.apis import inference_topdown, init_model

VisionTransformer.arch_zoo.update({
    'sapiens_0.3b': {'embed_dims': 1024, 'num_layers': 24, 'num_heads': 16, 'feedforward_channels': 4096},
    'sapiens_0.6b': {'embed_dims': 1280, 'num_layers': 32, 'num_heads': 16, 'feedforward_channels': 5120},
    'sapiens_1b':   {'embed_dims': 1536, 'num_layers': 40, 'num_heads': 24, 'feedforward_channels': 6144},
    'sapiens_2b':   {'embed_dims': 1920, 'num_layers': 48, 'num_heads': 32, 'feedforward_channels': 7680},
})

os.chdir("/workspace/sapiens/pose")
POSE_CONFIG = "/workspace/sapiens/pose/configs/sapiens_pose/coco_wholebody/sapiens_2b-210e_coco_wholebody-1024x768.py" 
POSE_CHECKPOINT = "/workspace/sapiens/pose/checkpoints/2b/sapiens_2b_coco_wholebody_best_coco_wholebody_AP_745.pth"

category = "pike"
INPUT_VIDEO = f"/workspace/sapiens/dataset/videos/{category}/001.mov"
OUTPUT_FOLDER = f"/workspace/sapiens/outputs/{category}"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

filename = os.path.splitext(os.path.basename(INPUT_VIDEO))[0]
if torch.backends.mps.is_available(): #macos compatibility
    device = 'mps'
else: #windows compatibility
    device = 'cuda'

print("[LOADING] Initializing Sapiens-2B model for Human Pose Estimation extraction")
model = init_model(POSE_CONFIG, POSE_CHECKPOINT, device=device, override_ckpt_meta=True)
cap = cv2.VideoCapture(INPUT_VIDEO)
frames = []
raw_coords_list = []

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

bbox = np.array([[0, 0, width, height]])
padding = 100 
frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
        
    frame_count += 1
    print(f"[DEBUG] {frame_count} frames have been read")
    
    frames.append(frame)
    pose_results = inference_topdown(model, frame, bbox)[0]
    
    if len(pose_results.pred_instances.keypoints) > 0:
        kpts = pose_results.pred_instances.keypoints[0]
        
        valid_x = [p[0] for p in kpts if p[0] > 0]
        valid_y = [p[1] for p in kpts if p[1] > 0]
        
        if valid_x and valid_y:
            x_min = max(0, min(valid_x) - padding)
            y_min = max(0, min(valid_y) - padding)
            x_max = min(width, max(valid_x) + padding)
            y_max = min(height, max(valid_y) + padding)
            bbox = np.array([[x_min, y_min, x_max, y_max]])
        else:
            bbox = np.array([[0, 0, width, height]])
    else:
        kpts = []
        bbox = np.array([[0, 0, width, height]]) 
        
    coords = sapiens2mediapipe(kpts)
    raw_coords_list.append(coords)
    
cap.release()
fixed_draw, rnn_tensor = interpolation_smoothing(raw_coords_list)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(f"{OUTPUT_FOLDER}/{filename}.avi", fourcc, fps, (width, height))

for i, frame in enumerate(frames):
    frame_with_skeleton = draw_skeleton(frame.copy(), fixed_draw[i])
    out.write(frame_with_skeleton)
    
out.release()
with open(f"{OUTPUT_FOLDER}/{filename}.json", 'w') as f:
    json.dump(rnn_tensor, f)

print(f"[SUCCESS] Output video has been saved in directory {OUTPUT_FOLDER}")