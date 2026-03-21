import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from scipy.signal import savgol_filter

def normalize_pose_tensor(df): #solution to point freezing
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
        "torso_head": [f"{ax}_{i}" for i in range(13) for ax in ['x','y']] + ["x_23","y_23","x_24","y_24"],
        "r_arm": [f"{ax}_{i}" for i in [12,14,16,18,20,22] for ax in ['x','y']],
        "l_arm": [f"{ax}_{i}" for i in [11,13,15,17,19,21] for ax in ['x','y']],
        "r_leg": [f"{ax}_{i}" for i in [24,26,28,30,32] for ax in ['x','y']],
        "l_leg": [f"{ax}_{i}" for i in [23,25,27,29,31] for ax in ['x','y']]
    }

    imputer = KNNImputer(n_neighbors=2, weights="distance") 
    for part, indices in body_parts.items():
        cols = [c for c in indices if c in data.columns]
        if cols and data[cols].isnull().values.any():
            data[cols] = imputer.fit_transform(data[cols])

    window = 5
    if len(data) > window:
        for col in data.columns:
            data[col] = savgol_filter(data[col], window_length=window, polyorder=2)

    fixed_draw = data.to_dict("records")
    pos_norm = normalize_pose_tensor(data)
    
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