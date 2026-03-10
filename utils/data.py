#### DATA.PY (WIP) ##########################
## involves data processing and treatment
#############################################
import numpy as np
import cv2

def read_video(path):
    frames = []
    video = cv2.VideoCapture(path)
    while True:
        ret, frame = video.read()
        if not ret:
            break
        h, w, _ = frame.shape
        frame = cv2.resize(frame, (int(w*2), int(h*2)), cv2.INTER_LINEAR)
        frames.append(frame)
    video.release()
    cv2.destroyAllWindows()
    return frames

def save_video(frames, path):
    h,w,_ = frames[0].shape
    codec = cv2.VideoWriter_fourcc(*"XVID")
    output = cv2.VideoWriter(path, codec, fps=30.0, frameSize=(w,h))
    for frame in frames:
        output.write(frame)
    output.release()

def process_nans(frame, results):
    h, w, _ = frame.shape
    coords = {}
    #nans conversion for later processing
    if results.pose_landmarks:
        for idx, landmark in enumerate(results.pose_landmarks.landmark):
            if landmark.visibility > 0.5:
                coords[f"x_{idx}"] = w*landmark.x
                coords[f"y_{idx}"] = h*landmark.y
            else:
                coords[f"x_{idx}"] = np.nan
                coords[f"y_{idx}"] = np.nan
    #TO DO: processing remaining!!
    return results

def get_point(idx, results, width, height): #returns point given a landmark
    return [int(width*results.pose_landmarks.landmark[idx].x), 
            int(height*results.pose_landmarks.landmark[idx].y)]

def create_bones(results,w,h):
    bones = {}
    if not results.pose_landmarks:
        return bones #None
    
    bodyparts_idx = {"shoulder": 11,"elbow": 13,"wrist": 15,"hip": 23,"knee": 25,"ankle": 27,"tiptoe": 31}
    points = {}
    for part, idx in bodyparts_idx.items():
        points[f"{part}_left"] = get_point(idx, results, width=w, height=h)
        points[f"{part}_right"] = get_point(idx+1, results, width=w, height=h) #mediapipe structure pattern

    bones = {
        "torso": [points["shoulder_right"], points["shoulder_left"]],
        "shoulder_left": [points["shoulder_left"], points["elbow_left"]],
        "shoulder_right": [points["shoulder_right"], points["elbow_right"]],
        "elbow_left": [points["elbow_left"], points["wrist_left"]],
        "elbow_right": [points["elbow_right"], points["wrist_right"]],
        "pelvis": [points["hip_right"], points["hip_left"]],
        "hips_left": [points["hip_left"], points["knee_left"]],
        "hips_right": [points["hip_right"], points["knee_right"]],
        "knee_left": [points["knee_left"], points["ankle_left"]],
        "knee_right": [points["knee_right"], points["ankle_right"]],
        "toe_left": [points["ankle_left"], points["tiptoe_left"]],
        "toe_right": [points["ankle_right"], points["tiptoe_right"]]
    }
    return bones

def create_joints(bones):
    joints = {}
    if bones:
        joints = {
            "left": {
                "shoulder": [bones["torso"], bones["shoulder_left"]],
                "elbow": [bones["shoulder_left"], bones["elbow_left"]],
                "hips": [bones["pelvis"], bones["hips_left"]],
                "knee": [bones["hips_left"], bones["knee_left"]],
                "toe": [bones["knee_left"], bones["toe_left"]]
            },
            "right": {
                "shoulder": [bones["torso"], bones["shoulder_right"]],
                "elbow": [bones["shoulder_right"], bones["elbow_right"]],
                "hips": [bones["pelvis"], bones["hips_right"]],
                "knee": [bones["hips_right"], bones["knee_right"]],
                "toe": [bones["knee_right"], bones["toe_right"]]
            }
        }
    return joints