import numpy as np

COCO_TO_MEDIAPIPE = { #translating keypoints to mediapipe
    0: 0,  
    5: 11, 
    6: 12, 
    7: 13, 
    8: 14, 
    9: 15,  
    10: 16,
    11: 23, 
    12: 24,
    13: 25, 
    14: 26, 
    15: 27, 
    16: 28,
    19: 29,
    22: 30,
    17: 31,
    20: 32
}

def coco2mediapipe(keypoints): #translates sapiens to mediapipe keypoints
    dataframe = {}
    for i in range(33):
        dataframe[f"x_{i}"] = np.nan
        dataframe[f"y_{i}"] = np.nan

    for coco_idx, mp_idx in  COCO_TO_MEDIAPIPE.items():
        if coco_idx < len(keypoints):
            dataframe[f"x_{mp_idx}"] = float(keypoints[coco_idx][0])
            dataframe[f"y_{mp_idx}"] = float(keypoints[coco_idx][1])
    return dataframe


def calculate_angle(a, b, c): #a,b,c -> points (x,y)
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    if np.any(np.isnan(a)) or np.any(np.isnan(b)) or np.any(np.isnan(c)):
        return np.nan
        
    ba = a - b
    bc = c - b
    
    cosangle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.degrees(np.arccos(np.clip(cosangle, -1.0, 1.0)))
    
    return angle

def calculate_joint_angle(dataframe): #groups angle by joint distribution
    position = dataframe.get('position', {}) #sub-dict -> position

    kp_shoulder_L = (position.get(['x_11'], np.nan), position.get(['y_11'], np.nan)) #kp -> keypoint
    kp_hip_L = (position.get(['x_23'], np.nan), position.get(['y_23'], np.nan))
    kp_knee_L = (position.get(['x_25'], np.nan), position.get(['y_25'], np.nan))
    kp_ankle_L = (position.get(['x_27'], np.nan), position.get(['y_27'], np.nan))
    kp_toe_L = (position.get(['x_31'], np.nan), position.get(['y_31'], np.nan))
    
    kp_shoulder_R = (position.get(['x_12'], np.nan), position.get(['y_12'], np.nan))
    kp_hip_R = (position.get(['x_24'], np.nan), position.get(['y_24'], np.nan))
    kp_knee_R = (position.get(['x_26'], np.nan), position.get(['y_26'], np.nan))
    kp_ankle_R = (position.get(['x_28'], np.nan), position.get(['y_28'], np.nan))
    kp_toe_R = (position.get(['x_32'], np.nan), position.get(['y_32'], np.nan))

    #adding joint to dataframe
    dataframe['angles'] = {
        'joint_hip_L': calculate_angle(kp_shoulder_L, kp_hip_L, kp_knee_L),
        'joint_hip_R': calculate_angle(kp_shoulder_R, kp_hip_R, kp_knee_R),
        'joint_knee_L': calculate_angle(kp_hip_L, kp_knee_L, kp_ankle_L),
        'joint_knee_R': calculate_angle(kp_hip_R, kp_knee_R, kp_ankle_R),
        'joint_ankle_L': calculate_angle(kp_knee_L, kp_ankle_L, kp_toe_L),
        'joint_ankle_R': calculate_angle(kp_knee_R, kp_ankle_R, kp_toe_R),
        'opening_L': calculate_angle(kp_knee_L, kp_hip_L, kp_hip_R),
        'opening_R': calculate_angle(kp_knee_R, kp_hip_R, kp_hip_L)
    }
    return dataframe