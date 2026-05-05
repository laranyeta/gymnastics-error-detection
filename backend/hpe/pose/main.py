import os
import cv2
import json
import torch
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__)) # tfg/backend/hpe/pose
project_root = os.path.abspath(os.path.join(current_dir, "..", "..", "..")) #tfg/
os.chdir(current_dir)

#importing the following libraries once in directory /pose
from utils.data import coco2mediapipe, calculate_joint_angle
from utils.vision import interpolation_smoothing, draw_skeleton
from mmpretrain.models.backbones.vision_transformer import VisionTransformer
from mmpose.apis import inference_topdown, init_model

VisionTransformer.arch_zoo.update({'sapiens_2b': {'embed_dims': 1920, 'num_layers': 48, 'num_heads': 32, 'feedforward_channels': 7680}})

POSE_CONFIG = "configs/sapiens_pose/coco_wholebody/sapiens_2b-210e_coco_wholebody-1024x768.py" 
POSE_CHECKPOINT = "checkpoints/sapiens_2b_coco_wholebody_best_coco_wholebody_AP_745.pth"

INPUT_VIDEO = os.path.join(project_root, "backend", "rnn", "test.mov")
OUTPUT_FOLDER = os.path.join(project_root, "backend", "rnn")
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

filename = os.path.splitext(os.path.basename(INPUT_VIDEO))[0]

if torch.backends.mps.is_available(): #macos compatibility
    device = 'mps'
else: #windows compatibility
    device = 'cuda'

_original_load = torch.load #support for latest pytorch updates (v2.6)
def patched_load(*args, **kwargs): 
    kwargs['weights_only'] = False
    return _original_load(*args, **kwargs)
torch.load = patched_load

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
        
    frame_count += 1 
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
        
    coords = coco2mediapipe(keypoints)
    raw_coords.append(coords)
    
cap.release()
smoothed_coords, dataframe = interpolation_smoothing(raw_coords)
rnn_tensor = [calculate_joint_angle(frame) for frame in dataframe] 
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(os.path.join(OUTPUT_FOLDER, f"{filename}.avi"), fourcc, fps, (width, height))

for i, frame in enumerate(frames):
    pose_estimation = draw_skeleton(frame.copy(), smoothed_coords[i])
    out.write(pose_estimation)
    
out.release()
with open(os.path.join(OUTPUT_FOLDER, f"{filename}.json"), 'w') as f:
    json.dump(rnn_tensor, f)

print(f"[SUCCESS] Output video has been saved in directory {OUTPUT_FOLDER}")