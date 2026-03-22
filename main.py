import os
import json
import mediapipe as mp

from utils.data import read_video, save_video
from utils.pipeline_mediapipe import extract_pose, draw_skeleton
from utils.optimization import interpolation_smoothing

category = "straddle"
input_path = f"dataset/{category}/006.mov"
output_path = f"outputs/{category}"
os.makedirs(output_path, exist_ok=True)

valid_extensions = (".mp4", ".mov", ".avi")
mp_pose = mp.solutions.pose
pose_model = mp_pose.Pose(
    static_image_mode=False, 
    model_complexity=2,
    min_detection_confidence=0.8,
    min_tracking_confidence=0.8
)
connections = mp_pose.POSE_CONNECTIONS

filename = os.path.splitext(os.path.basename(input_path))[0]

if input_path.lower().endswith(valid_extensions):
    frames = read_video(input_path)
    print(f"[DEBUG] {len(frames)} frames have been read")

    raw_coords_list = []
    print("[LOADING] Extracting pose...")
    for frame in frames:
        coords = extract_pose(frame, pose_model)
        raw_coords_list.append(coords)

    print("[LOADING] Processing interpolation and Pose Tensor...")
    fixed_draw, rnn_tensor = interpolation_smoothing(raw_coords_list)

    result_frames = []
    for i, frame in enumerate(frames):
        frame_with_skeleton = draw_skeleton(frame, fixed_draw[i], connections)
        result_frames.append(frame_with_skeleton)

    os.makedirs(output_path, exist_ok=True)
    save_video(result_frames, f"{output_path}/{filename}.avi")

    print("[LOADING] Creating coordinate JSON for later RNN extraction...")
    with open(f"{output_path}/{filename}.json", 'w') as f:
        json.dump(rnn_tensor, f)

    print(f"[SUCCESS] Output video and coordinate data have been saved in {output_path} directory.")
else:
    print(f"[ERROR] Not a valid extension for the input video.")