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
                coords[f"x_{idx}"] = int(w*landmark.x)
                coords[f"y_{idx}"] = int(h*landmark.y)
            else:
                coords[f"x_{idx}"] = np.nan
                coords[f"y_{idx}"] = np.nan
    else:
        for idx in range(33): #every landmark is nan
            coords[f"x_{idx}"] = np.nan
            coords[f"x_{idx}"] = np.nan
    return coords

def get_point(coords, idx): #returns point given a landmark
    return [coords[f"x_{idx}"], coords[f"y_{idx}"]]

def create_bones(coords):
    bones = {}
    if np.isnan(coords[f"x_0"]): #assuming if first coord is nan everything is nan
        return bones #None
    
    bodyparts_idx = {"shoulder": 11,"elbow": 13,"wrist": 15,"hip": 23,"knee": 25,"ankle": 27,"tiptoe": 31}
    points = {}
    for part, idx in bodyparts_idx.items():
        points[f"{part}_left"] = get_point(coords, idx)
        points[f"{part}_right"] = get_point(coords, idx+1) #mediapipe structure pattern (right=left+1)

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