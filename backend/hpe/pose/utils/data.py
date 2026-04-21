import os
import cv2
import json
import torch
import numpy as np
import pandas as pd

from sklearn.impute import KNNImputer
from scipy.signal import savgol_filter
from mmpretrain.models.backbones.vision_transformer import VisionTransformer
from mmpose.apis import inference_topdown, init_model

SAPIENS_TO_MEDIAPIPE = {
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

def sapiens2mediapipe(sapiens_kpts): #translates sapiens to mediapipe keypoints
    dataframe = {}
    for i in range(33):
        dataframe[f"x_{i}"] = np.nan
        dataframe[f"y_{i}"] = np.nan

    for sap_idx, mp_idx in SAPIENS_TO_MEDIAPIPE.items():
        if sap_idx < len(sapiens_kpts):
            dataframe[f"x_{mp_idx}"] = float(sapiens_kpts[sap_idx][0])
            dataframe[f"y_{mp_idx}"] = float(sapiens_kpts[sap_idx][1])
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
    kp_shoulder_L = (dataframe['x_11'], dataframe['y_11']) #kp -> keypoint
    kp_hip_L = (dataframe['x_23'], dataframe['y_23'])
    kp_knee_L = (dataframe['x_25'], dataframe['y_25'])
    kp_ankle_L = (dataframe['x_27'], dataframe['y_27'])
    kp_toe_L = (dataframe['x_31'], dataframe['y_31'])
    
    kp_shoulder_R = (dataframe['x_12'], dataframe['y_12'])
    kp_hip_R = (dataframe['x_24'], dataframe['y_24'])
    kp_knee_R = (dataframe['x_26'], dataframe['y_26'])
    kp_ankle_R = (dataframe['x_28'], dataframe['y_28'])
    kp_toe_R = (dataframe['x_32'], dataframe['y_32'])

    #adding joint to dataframe
    dataframe['joint_hip_L'] = calculate_angle(kp_shoulder_L, kp_hip_L, kp_knee_L)
    dataframe['joint_hip_R'] = calculate_angle(kp_shoulder_R, kp_hip_R, kp_knee_R)
    
    dataframe['joint_knee_L'] = calculate_angle(kp_hip_L, kp_knee_L, kp_ankle_L)
    dataframe['joint_knee_R'] = calculate_angle(kp_hip_R, kp_knee_R, kp_ankle_R)
    
    dataframe['joint_ankle_L'] = calculate_angle(kp_knee_L, kp_ankle_L, kp_toe_L)
    dataframe['joint_ankle_R'] = calculate_angle(kp_knee_R, kp_ankle_R, kp_toe_R)
    
    dataframe['opening_L'] = calculate_angle(kp_knee_L, kp_hip_L, kp_hip_R)
    dataframe['opening_R'] = calculate_angle(kp_knee_R, kp_hip_R, kp_hip_L)
    return dataframe