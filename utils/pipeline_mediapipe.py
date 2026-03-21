import cv2
import mediapipe as mp
import numpy as np

def extract_pose(frame, pose_model):
    h, w, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose_model.process(frame_rgb)

    coords = {}
    for i in range(33):
        coords[f"x_{i}"] = np.nan
        coords[f"y_{i}"] = np.nan

    if results.pose_landmarks:
        for idx, landmark in enumerate(results.pose_landmarks.landmark):
                coords[f"x_{idx}"] = w * landmark.x
                coords[f"y_{idx}"] = h * landmark.y
    
    return coords

def draw_skeleton(frame, fixed_coords, connections):
    h,w = frame.shape[:2]
    visible_landmarks = [11,12,13,14,15,16,23,24,25,26,27,28,31,32]
    y_pelvis = fixed_coords.get("y_24", h)
    thr = h*0.5

    if y_pelvis < thr:
        for connection in connections:
            start_idx, end_idx = connection #start -> [0], end -> [1]

            if start_idx in visible_landmarks and end_idx in visible_landmarks:
                x1 = fixed_coords.get(f"x_{start_idx}",0)
                y1 = fixed_coords.get(f"y_{start_idx}",0)
                x2 = fixed_coords.get(f"x_{end_idx}",0)
                y2 = fixed_coords.get(f"y_{end_idx}",0)

                if not (np.isnan(x1) or np.isnan(y1) or np.isnan(x2) or np.isnan(y2)):
                    point1 = (int(x1), int(y1))
                    point2 = (int(x2), int(y2))

                    cv2.line(frame, point1, point2, (0, 255, 0), 2)
                    cv2.circle(frame, point1, 4, (0, 0, 255), -1)
                    cv2.circle(frame, point2, 4, (0, 0, 255), -1)
    return frame