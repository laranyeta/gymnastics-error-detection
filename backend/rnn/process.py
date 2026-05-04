import os
import json
import numpy as np
from scipy.interpolate import interp1d

FRAMES = 30 #cut up to 30 frames per video
INPUT_DIR = "backend/hpe/dataset"
OUTPUT_DIR = "backend/rnn/data"

CATEGORIES = ["pike", "split", "straddle", "tuck"]

KEYPOINT_SWAPPING = [ #left to right swapping for data augmentation
    (1,2), (3,4), (5,6), (7,8), (9,10), (11,12), (13,14), (15,16), 
    (17,18), (19,20), (21,22), (23,24), (25,26), (27,28), (29,30), (31,32)
]

ANGLE_SWAPPING = [
    ("joint_hip_L", "joint_hip_R"),
    ("joint_knee_L", "joint_knee_R"),
    ("joint_ankle_L", "joint_ankle_R"),
    ("opening_L", "opening_R")
]

# crops data to 30 frames per sequence
def interpolate_sequence(sequence, target_length=FRAMES): #json -> matrix -> new_json
    original_length = len(sequence)
    if original_length == 0:
        return []
    elif original_length == target_length: #is already 30 frames
        return sequence

    frame = sequence[0]
    flat_keys = []
    for key in ["position", "velocity", "acceleration", "angles"]:
        if key in frame: #frame -> structure 
            for subkey in frame[key].keys(): #position -> x_1; angle -> joint_hip_L
                flat_keys.append((key, subkey))

    data_matrix = np.zeros((original_length, len(flat_keys))) #to apply interpolation
    for i, frame in enumerate(sequence):
        for j, (key, subkey) in enumerate(flat_keys):
            data_matrix[i, j] = frame.get(key, {}).get(subkey, 0.0)

    old_time = np.linspace(0, 1, original_length)
    new_time = np.linspace(0, 1, target_length)

    interpolator = interp1d(old_time, data_matrix, axis=0, kind='linear') #axis = 0 -> temporal
    new_matrix = interpolator(new_time)

    new_sequence = []
    for i in range(target_length):
        new_frame = {"position": {}, "velocity": {}, "acceleration": {}, "angles": {}}
        for j, (key, subkey) in enumerate(flat_keys):
            new_frame[key][subkey] = float(new_matrix[i, j])
        new_sequence.append(new_frame)

    return new_sequence #new json format

# data augmentation
def mirror_sequence(sequence):
    mirror = []
    for frame in sequence:
        new_frame = {"position": {}, "velocity": {}, "acceleration": {}, "angles": {}}

        for key in ["position", "velocity", "acceleration"]:
            # CORRECCIÓ 1: Canviat 'mk' per 'key'
            for subkey, value in frame[key].items():
                new_subkey = subkey
                # CORRECCIÓ 2: Hem de fer el split al 'subkey' (ex: "x_11"), no al 'key' (ex: "position")
                parts = subkey.split('_') 
                if len(parts) == 2 and parts[1].isdigit():
                    idx = int(parts[1])
                    for idxL, idxR in KEYPOINT_SWAPPING:
                        if idx == idxL:
                            # CORRECCIÓ 3: Guardar el canvi a 'new_subkey'
                            new_subkey = f"{parts[0]}_{idxR}" 
                            break
                        elif idx == idxR:
                            new_subkey = f"{parts[0]}_{idxL}"
                            break
                
                if new_subkey.startswith('x'): #only inverts x
                    new_frame[key][new_subkey] = -value
                else:
                    new_frame[key][new_subkey] = value #y remains the same (horizontal mirror)

        for key, value in frame.get("angles", {}).items(): #key -> joint
            new_key = key
            for angleL, angleR in ANGLE_SWAPPING:
                if key == angleL:
                    new_key = angleR
                    break
                elif key == angleR:
                    new_key = angleL
                    break
            new_frame["angles"][new_key] = value
        mirror.append(new_frame)
    return mirror

def process_dataset():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("[LOADING] Processing dataset...")

    for category in CATEGORIES:
        input_path = os.path.join(INPUT_DIR, category) 
        output_path = os.path.join(OUTPUT_DIR, category)
        os.makedirs(output_path, exist_ok=True)
        
        files = [f for f in os.listdir(input_path) if f.endswith(".json")]
        files.sort()
        
        total_originals = len(files)
        
        for i, file in enumerate(files):
            with open(os.path.join(input_path, file), "r") as f:
                raw_sequence = json.load(f)
            sequence = interpolate_sequence(raw_sequence)
            mirrored = mirror_sequence(sequence) 

            with open(os.path.join(output_path, file), "w") as f:
                json.dump(sequence, f)
            mirror_index = total_originals + i + 1 #mirrored json start from last standard json (+1 as they start from 001)
            mirror_filename = f"{mirror_index:03d}.json" 
            
            with open(os.path.join(output_path, mirror_filename), "w") as f:
                json.dump(mirrored, f)
                
    print(f"[SUCCESS] The dataset has been processed")

if __name__ == "__main__":
    process_dataset()
