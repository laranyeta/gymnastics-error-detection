import cv2
import numpy as np
import pandas as pd

from sklearn.impute import KNNImputer
from scipy.signal import savgol_filter
from mmpretrain.models.backbones.vision_transformer import VisionTransformer
from mmpose.apis import inference_topdown, init_model


def normalize_pose_tensor(df):
    neck_x, neck_y = (df['x_11'] + df['x_12'])/2, (df['y_11'] + df['y_12'])/2
    pelvis_x, pelvis_y = (df['x_23'] + df['x_24'])/2, (df['y_23'] + df['y_24'])/2
    
    mid_torso_x = (neck_x + pelvis_x) / 2
    mid_torso_y = (neck_y + pelvis_y) / 2
    
    d = np.sqrt((neck_x - pelvis_x)**2 + (neck_y - pelvis_y)**2).replace(0, 1)

    df_norm = df.copy()
    for i in range(33):
        df_norm[f'x_{i}'] = (df[f'x_{i}'] - mid_torso_x) / d
        df_norm[f'y_{i}'] = (df[f'y_{i}'] - mid_torso_y) / d
        
    return df_norm

def interpolation_smoothing(frames):
    data = pd.DataFrame(frames)
    data = data.interpolate(method="linear", limit=10, limit_direction="both")
    
    body_parts = {
        "torso_head": [f"{ax}_{i}" for i in [0, 11, 12, 23, 24] for ax in ['x','y']],
        "r_arm": [f"{ax}_{i}" for i in [12, 14, 16] for ax in ['x','y']],
        "l_arm": [f"{ax}_{i}" for i in [11, 13, 15] for ax in ['x','y']],
        "r_leg": [f"{ax}_{i}" for i in [24, 26, 28, 30, 32] for ax in ['x','y']],
        "l_leg": [f"{ax}_{i}" for i in [23, 25, 27, 29, 31] for ax in ['x','y']]
    }

    imputer = KNNImputer(n_neighbors=2, weights="distance") 
    for part, indices in body_parts.items():
        cols = [c for c in indices if c in data.columns]
        if cols and data[cols].isnull().values.any():
            data[cols] = imputer.fit_transform(data[cols])

    window = 5
    if len(data) > window:
        for part, indices in body_parts.items():
            cols = [c for c in indices if c in data.columns]
            for col in cols:
                data[col] = savgol_filter(data[col], window_length=window, polyorder=2)

    fixed_draw = data.to_dict("records")
    pos_norm = normalize_pose_tensor(data)
    pos_norm = pos_norm.fillna(0)
    
    velocity = pos_norm.diff().fillna(0)
    acceleration = velocity.diff().fillna(0)
    
    rnn_data = []
    for i in range(len(pos_norm)):
        frame_data = {
            "position": pos_norm.iloc[i].to_dict(),
            "velocity": velocity.iloc[i].to_dict(),
            "acceleration": acceleration.iloc[i].to_dict()
        }
        rnn_data.append(frame_data)
            
    return fixed_draw, rnn_data

def draw_skeleton(frame, fixed_coords):
    h, w = frame.shape[:2]
    CONNECTIONS = [
        (11, 12), (11, 13), (13, 15), (12, 14), (14, 16), #shoulders and arms
        (11, 23), (12, 24), (23, 24), # torso and hips           
        (23, 25), (25, 27), (24, 26), (26, 28), #legs
        (27, 31), (28, 32), #ankle
    ]

    for start_idx, end_idx in CONNECTIONS:
        x1 = fixed_coords.get(f"x_{start_idx}")
        y1 = fixed_coords.get(f"y_{start_idx}")
        x2 = fixed_coords.get(f"x_{end_idx}")
        y2 = fixed_coords.get(f"y_{end_idx}")

        if pd.notna(x1) and pd.notna(y1) and pd.notna(x2) and pd.notna(y2):
            point1 = (int(x1), int(y1))
            point2 = (int(x2), int(y2))

            cv2.line(frame, point1, point2, (255, 255, 255), 3)
            cv2.circle(frame, point1, 7, (255, 255, 255), -1)
            cv2.circle(frame, point2, 7, (255, 255, 255), -1)
    return frame