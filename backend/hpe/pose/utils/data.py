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
    frame_data = {}
    for i in range(33):
        frame_data[f"x_{i}"] = np.nan
        frame_data[f"y_{i}"] = np.nan

    for sap_idx, mp_idx in SAPIENS_TO_MEDIAPIPE.items():
        if sap_idx < len(sapiens_kpts):
            frame_data[f"x_{mp_idx}"] = float(sapiens_kpts[sap_idx][0])
            frame_data[f"y_{mp_idx}"] = float(sapiens_kpts[sap_idx][1])
    return frame_data