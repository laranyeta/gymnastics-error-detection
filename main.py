### testing
import os
from utils.data import read_video, save_video
from utils.pipeline_mediapipe import extract_pose, draw_skeleton

input_path = "dataset/video_test/aerial_doublebackflip.mp4"
filename = input_path[19:-4]
output_path = "outputs"

frames = read_video(input_path) #every frame of the video
print(f"[DEBUG] {len(frames)} frames have been read.")

result = []
for i, frame in enumerate(frames):
    results, connections = extract_pose(frame)
    pose_estimation = draw_skeleton(frame, results, connections)
    result.append(pose_estimation)

os.makedirs(output_path, exist_ok=True)
save_video(result, f"{output_path}/{filename}_out.avi")