#### VISUALIZATION.PY (WIP) ###############################
## involves visual processes related to pose estimation
## extraction from human skeleton
###########################################################
import cv2
import mediapipe as mp

def extract_pose(frame):
    pose = mp.solutions.pose
    connections = pose.POSE_CONNECTIONS
    pose_estimation = pose.Pose(static_image_mode=True,
                        model_complexity=2,
                        enable_segmentation=False,
                        min_detection_confidence=0.5,
                        min_tracking_confidence=0.3)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) #convert to rgb to process
    results = pose_estimation.process(frame)
    return results, connections

def draw_skeleton(frame, results, connections):
    #bone mapping
    body_parts = {
        "shoulders": [11,12],
        "arms": [13,14,15,16],
        "pelvis": [23,24],
        "legs": [25,26,27,28],
        "toe": [31,32]
    }
    skeleton = mp.solutions.drawing_utils
    style = mp.solutions.drawing_styles

    if results.pose_landmarks:
        visible_landmarks = []
        for points in body_parts.values():
            visible_landmarks.extend(points)
        for i in range(1,33): #iterate for every landmark
            if i not in visible_landmarks:
                results.pose_landmarks.landmark[i].visibility = 0.0
        skeleton.draw_landmarks(
            frame,
            results.pose_landmarks,
            connections,
            landmark_drawing_spec=style.get_default_pose_landmarks_style())
    return frame