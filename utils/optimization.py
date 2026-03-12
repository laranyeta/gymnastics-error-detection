#### OPTIMIZATION.PY (WIP) ######################################
## involves improvements and optimizations for logical pipeline
#################################################################
import pandas as pd
from scipy.signal import savgol_filter

def interpolation_smoothing(frames):
    '''data = pd.DataFrame(frames)
    #data = data.interpolate(method="quadratic", limit_direction="both")
    data = data.interpolate(method="linear", limit_direction="both") #incase

    window = 11
    for col in data.columns:
        if len(data) > window:
            data[col] = savgol_filter(data[col], window_length=window, polyorder=3)

    print("[DEBUG COORDS] Hombro Izquierdo (Frame 0):", data.iloc[0]['x_11'], data.iloc[0]['y_11'])
    print("[DEBUG COORDS] Hombro Izquierdo (Frame 40):", data.iloc[40]['x_11'], data.iloc[40]['y_11'])
    result = data.to_dict("records")'''
    return frames