#### TESTING SAMPLE (WIP)
## first samples and test will be shown through this code via balance beam videos
import os
from utils.data import read_video, save_video, filter_annotations
from utils.visualization import extract_pose, draw_skeleton

input_path = "dataset/video_test/aerial_doublebackflip.mp4"
filename = input_path[19:-4]
output_path = "outputs"
filter_annotations("dataset/original/finegym_v1.1.json")
frames = read_video(input_path) #every frame of the video
print(f"[DEBUG] {len(frames)} frames have been read.")

result = []
for i, frame in enumerate(frames):
    results, connections = extract_pose(frame)
    pose_estimation = draw_skeleton(frame, results, connections)
    result.append(pose_estimation)

os.makedirs(output_path, exist_ok=True)
save_video(result, f"{output_path}/{filename}_out.avi")