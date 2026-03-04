#### TESTING SAMPLE (WIP)
## first samples and test will be shown through this code via balance beam videos
import os
from utils.data import read_video, save_video, filter_annotations
from utils.visualization import extract_pose, draw_skeleton

output_path = "outputs"
filter_annotations("dataset/original/finegym_v1.1.json")
frames = read_video("dataset/video_test/flicflac_backflip.mp4") #every frame of the video
print(f"[DEBUG] {len(frames)} frames have been read.")

result = []
for i, frame in enumerate(frames):
    results, connections = extract_pose(frame)
    pose_estimation = draw_skeleton(frame, results, connections)
    result.append(pose_estimation)

os.makedirs(output_path, exist_ok=True)
save_video(result, output_path)