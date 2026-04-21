import os
import cv2
import json
import torch
import numpy as np

from utils.data import sapiens2mediapipe, calculate_joint_angle
from utils.vision import interpolation_smoothing, draw_skeleton
from mmpretrain.models.backbones.vision_transformer import VisionTransformer
from mmpose.apis import inference_topdown, init_model

VisionTransformer.arch_zoo.update({'sapiens_2b':   {'embed_dims': 1920, 'num_layers': 48, 'num_heads': 32, 'feedforward_channels': 7680}})
os.chdir("backend/hpe/pose") #directory main branch
POSE_CONFIG = "backend/hpe/pose/configs/sapiens_pose/coco_wholebody/sapiens_2b-210e_coco_wholebody-1024x768.py" 
POSE_CHECKPOINT = "backend/hpe/pose/checkpoints/2b/sapiens_2b_coco_wholebody_best_coco_wholebody_AP_745.pth"

category = "pike"
INPUT_VIDEO = f"backend/hpe/dataset/videos/{category}/001.mov"
OUTPUT_FOLDER = f"backend/hpe/dataset/outputs/{category}"
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
raw_coords = []

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
        
    frame_count += 1 #only for debug purposes
    print(f"[DEBUG] {frame_count} frames have been read")
    
    frames.append(frame)
    pose = inference_topdown(model, frame, bbox)[0]
    
    if len(pose.pred_instances.keypoints) > 0:
        keypoints = pose.pred_instances.keypoints[0]
        
        valid_x = [p[0] for p in keypoints if p[0] > 0]
        valid_y = [p[1] for p in keypoints if p[1] > 0]
        
        if valid_x and valid_y:
            x_min = max(0, min(valid_x) - padding)
            y_min = max(0, min(valid_y) - padding)
            x_max = min(width, max(valid_x) + padding)
            y_max = min(height, max(valid_y) + padding)
            bbox = np.array([[x_min, y_min, x_max, y_max]])
        else:
            bbox = np.array([[0, 0, width, height]])
    else:
        keypoints = []
        bbox = np.array([[0, 0, width, height]]) 
        
    coords = sapiens2mediapipe(keypoints)
    raw_coords.append(coords)
    
cap.release()
smoothed_coords, dataframe = interpolation_smoothing(raw_coords)
rnn_tensor = [calculate_joint_angle(frame) for frame in dataframe] 
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(f"{OUTPUT_FOLDER}/{filename}.avi", fourcc, fps, (width, height))

for i, frame in enumerate(frames):
    pose_estimation = draw_skeleton(frame.copy(), smoothed_coords[i])
    out.write(pose_estimation)
    
out.release()
with open(f"{OUTPUT_FOLDER}/{filename}.json", 'w') as f:
    json.dump(rnn_tensor, f)

print(f"[SUCCESS] Output video has been saved in directory {OUTPUT_FOLDER}")