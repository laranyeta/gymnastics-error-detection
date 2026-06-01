import json
import cv2
import math
import numpy as np

from backend.hpe.pose.utils.vision import CONNECTIONS
from backend.scoring.evaluator import AcrobaticEvaluator
from backend.scoring.rules import COLOR_MAP
from backend.rnn.predict import load_prediction_model, predict

def denormalize_point(x_val, y_val, view_range=3.0, canvas_size=640): #denormalizes point from normalized coordinates by torso placement
    px = int((float(x_val)+view_range)/(view_range*2)*canvas_size)
    py = int((float(y_val)+view_range)/(view_range*2)*canvas_size)
    return px, py

def get_pt(idx, pos, VIEW_RANGE=3.0, canvas_size=640):
    x_val = float(pos.get(f"x_{idx}", 0))
    y_val = float(pos.get(f"y_{idx}", 0))
    return denormalize_point(x_val, y_val, VIEW_RANGE, canvas_size)

def get_ankle_displacement(frame_A, frame_B, view_range=3.0, canvas_size=640): #euclidean distance of ankle between two frames (less distance->on ground, more distance->on air)
    pos_A = frame_A.get("position", {})
    pos_B = frame_B.get("position", {})
    
    #frame A
    x1_norm = pos_A.get("x_27", 0) #keypoint 27 -> left ankle (default)
    y1_norm = pos_A.get("y_27", 0)
    px1, py1 = denormalize_point(x1_norm, y1_norm, view_range, canvas_size)
    
    #frame B
    x2_norm = pos_B.get("x_27", 0)
    y2_norm = pos_B.get("y_27", 0)
    px2, py2 = denormalize_point(x2_norm, y2_norm, view_range, canvas_size)
    
    return math.sqrt(math.pow(px2 - px1, 2) + math.pow(py2 - py1, 2)) #euclidean distance

def calculate_split_angle(pos): #calculates the real angle by making an imaginary keypoint for extra vectorial angle calculation
    hip_L_x, hip_L_y = pos.get("x_23", 0), pos.get("y_23", 0)
    hip_R_x, hip_R_y = pos.get("x_24", 0), pos.get("y_24", 0)

    knee_L_x, knee_L_y = pos.get("x_25", 0), pos.get("y_25", 0)
    knee_R_x, knee_R_y = pos.get("x_26", 0), pos.get("y_26", 0)

    hip_middle_x = (hip_L_x + hip_R_x) / 2
    hip_middle_y = (hip_L_y + hip_R_y) / 2

    vector_L = (knee_L_x - hip_middle_x, knee_L_y - hip_middle_y)
    vector_R = (knee_R_x - hip_middle_x, knee_R_y - hip_middle_y)

    dot_prod = (vector_L[0]*vector_R[0]) + (vector_L[1]*vector_R[1]) #u*v (dot product)
    magnitude_L = math.sqrt(pow(vector_L[0], 2) + pow(vector_L[1], 2)) #|u|
    magnitude_R = math.sqrt(pow(vector_R[0], 2) + pow(vector_R[1], 2)) #|v|
    
    if magnitude_L*magnitude_R == 0:
        print("[ERROR] Division by 0 not calculable.")
    cos_theta = dot_prod/(magnitude_L*magnitude_R) #u*v/|u||v|
    cos_theta = max(min(cos_theta, 1.0), -1.0) #limited to range -1 to 1

    return (math.degrees(math.acos(cos_theta))) #radians -> degrees

def find_acrobatic_peak(sequence, pred): #finds peak based on predicted acrobatic (split -> max opening, tuck-> min hip-knee angle, pike -> min hip angle/max knee angle)
    if pred in ["split", "straddle"]:
        idx, peak_frame = max(enumerate(sequence), key=lambda x: calculate_split_angle(x[1].get("position", {})))
    elif pred == "tuck":
        #filter for realistic frames (joints over 20 degrees)
        valid_frames = []
        for i, frame in enumerate(sequence):
            angles = frame.get("angles", {})
            if angles.get("joint_hip_L", 0) > 20 and angles.get("joint_knee_L", 0) > 20 and angles.get("joint_knee_R", 0) > 20 and angles.get("joint_ankle_L", 0) > 20 and angles.get("joint_ankle_R", 0) > 20:
                valid_frames.append((i, frame)) #tuple list
        idx, peak_frame = min(
            valid_frames, 
            key=lambda x: (
                x[1].get("angles", {}).get("joint_hip_L", 45) + 
                x[1].get("angles", {}).get("joint_knee_L", 45) + 
                x[1].get("angles", {}).get("joint_knee_R", 45)
            )
        )
    elif pred == "pike":
        valid_frames = []
        for i, frame in enumerate(sequence):
            angles = frame.get("angles", {})
            if angles.get("joint_hip_L", 0) > 20 and angles.get("joint_knee_L", 0) > 20 and angles.get("joint_knee_R", 0) > 20 and angles.get("joint_ankle_L", 0) > 20 and angles.get("joint_ankle_R", 0) > 20:
                valid_frames.append((i, frame))
        idx, peak_frame = min(
            valid_frames,
            key=lambda x: (
                x[1].get("angles", {}).get("joint_hip_L", 45) - 
                x[1].get("angles", {}).get("joint_knee_L", 180) - 
                x[1].get("angles", {}).get("joint_knee_R", 180)
            )
        )
    else:
        idx = len(sequence)//2
        peak_frame = sequence[idx] 
    return idx, peak_frame

def find_acrobatic_window(sequence, peak_idx, thr=3.0): #finds start and end frames of acrobatic (ankle displacement)
    start_idx = peak_idx
    end_idx = peak_idx

    for i in range(peak_idx, 0, -1): #searching backwards for the start of the acrobatic
        d = get_ankle_displacement(sequence[i], sequence[i-1])
        if d < thr: 
            start_idx = i
            break
            
    for i in range(peak_idx, len(sequence)-1): #searching forwards for the end of the acrobatic
        d = get_ankle_displacement(sequence[i], sequence[i+1])
        if d < thr: 
            end_idx = i
            break
    return start_idx, end_idx

def evaluate_performance(sequence, pred): #evaluates only one acrobatic
    evaluator = AcrobaticEvaluator()
    peak_idx, _ = find_acrobatic_peak(sequence, pred)
    angles = sequence[peak_idx].get("angles", {})
    pos = sequence[peak_idx].get("position", {})
    
    penalty = 0.0
    if pred == "tuck":
        hip_angle = (angles.get("joint_hip_L", 180) + angles.get("joint_hip_R", 180)) / 2
        knee_L = angles.get("joint_knee_L", 180)
        knee_R = angles.get("joint_knee_R", 180)
        ankle_L = angles.get("joint_ankle_L", 180)
        ankle_R = angles.get("joint_ankle_R", 180)
        penalty, breakdown = evaluator.eval_tuck(hip_angle, knee_L, knee_R, ankle_L, ankle_R)

    elif pred == "pike":
        hip_angle = (angles.get("joint_hip_L", 180) + angles.get("joint_hip_R", 180)) / 2
        knee_L = angles.get("joint_knee_L", 180)
        knee_R = angles.get("joint_knee_R", 180)
        ankle_L = angles.get("joint_ankle_L", 180)
        ankle_R = angles.get("joint_ankle_R", 180)
        penalty, breakdown = evaluator.eval_pike(hip_angle, knee_L, knee_R, ankle_L, ankle_R)
        
    elif pred == "split":
        opening_angle = calculate_split_angle(pos)
        knee_L = angles.get("joint_knee_L", 180)
        knee_R = angles.get("joint_knee_R", 180)
        ankle_L = angles.get("joint_ankle_L", 180)
        ankle_R = angles.get("joint_ankle_R", 180)
        penalty, breakdown = evaluator.eval_split(opening_angle, knee_L, knee_R, ankle_L, ankle_R)
        
    elif pred == "straddle":
        opening_angle = calculate_split_angle(pos) #vectorial
        knee_L = angles.get("joint_knee_L", 180)
        knee_R = angles.get("joint_knee_R", 180)
        ankle_L = angles.get("joint_ankle_L", 180)
        ankle_R = angles.get("joint_ankle_R", 180)
        penalty, breakdown = evaluator.eval_straddle(opening_angle, knee_L, knee_R, ankle_L, ankle_R)

    return penalty, breakdown, peak_idx

def evaluate_routine(json_path, window=40, step=20): #evaluates more than one acrobatic (sliding window)
    with open(json_path, "r") as f:
        sequence = json.load(f) #full sequence

    model, device = load_prediction_model() 
    total_frames = len(sequence)
    results = []
    last_pred = None

    start = 0
    while start <= total_frames-window:
        end = start+window
        window_seq = sequence[start:end]
        pred, conf = predict(model, device, window_seq) #predicts
    
        if pred != "none" and conf > 50 and pred != last_pred:
            penalty, breakdown, local_peak = evaluate_performance(window_seq, pred)
            global_peak = start+local_peak

            #print(f"\nAcrobatic: {pred.upper()} (Peak Global: Frame {global_peak})")
            start_acro, end_acro = find_acrobatic_window(sequence, global_peak, thr=3.0)
            results.append({
                "acrobatic": pred,
                "global_peak": global_peak,
                "start_frame": start_acro,
                "end_frame": end_acro, 
                "penalty": penalty,
                "breakdown": breakdown,
                "confidence": conf,
                "angles": sequence[global_peak].get("angles", {}), #to draw the skeleton
                "position": sequence[global_peak].get("position", {})
            })
            start = end_acro
            last_pred = pred 
        else:
            if pred == "none":
                last_pred = None 
            start += step
    return results

def color_joint_deduction(COLOR_MAP, breakdown): #assigns color depending on penalty type (minor->green, medium->yellow, severe->red)
    colors = {}
    for reason in breakdown:
        if "MINOR" in reason: 
            c = COLOR_MAP["MINOR"]
        elif "MEDIUM" in reason: 
            c = COLOR_MAP["MEDIUM"]
        elif "SEVERE" in reason: 
            c = COLOR_MAP["SEVERE"]
        else: 
            continue

        if "Opening" in reason: #Opening (16.8º) below 135º - SEVERE (-0.5) -> example
            colors["opening_L"], colors["opening_R"] = c, c
        if "torso" in reason: #Bent torso (88.8º) above 65º - MINOR (-0.1) -> example
            colors["torso_L"], colors["torso_R"], colors["pelvis"] = c, c, c
        if "knee" in reason: #Bent knee (68.0º) above 65º - MINOR (-0.1) -> example
            colors["upperleg_L"], colors["lowerleg_L"] = c, c
            colors["upperleg_R"], colors["lowerleg_R"] = c, c
        if "ankle" in reason: #Bent ankle (41.3º) above 160º - MINOR (-0.1) -> example
            colors["toe_L"], colors["toe_R"] = c, c      
    return colors

def generate_skeleton_canvas(pos, breakdown, is_false_positive=False, canvas_size=640): #skeleton canvas with colored deductions for visual aid
    canvas = np.zeros((canvas_size, canvas_size, 3), dtype=np.uint8)
    VIEW_RANGE = 3.0
    for start_idx, end_idx in CONNECTIONS:
        pt1 = get_pt(start_idx, pos, VIEW_RANGE, canvas_size)
        pt2 = get_pt(end_idx, pos, VIEW_RANGE, canvas_size)
        cv2.line(canvas, pt1, pt2, (255, 255, 255), 2)
        cv2.circle(canvas, pt1, 4, (255, 255, 255), -1)
        cv2.circle(canvas, pt2, 4, (255, 255, 255), -1)

    if is_false_positive: #blank canvas for false positives (no deductions)
        return canvas

    x_11, x_12 = get_pt(11, pos, VIEW_RANGE, canvas_size), get_pt(12, pos, VIEW_RANGE, canvas_size)
    x_23, x_24 = get_pt(23, pos, VIEW_RANGE, canvas_size), get_pt(24, pos, VIEW_RANGE, canvas_size)
    x_25, x_26 = get_pt(25, pos, VIEW_RANGE, canvas_size), get_pt(26, pos, VIEW_RANGE, canvas_size) 
    x_27, x_28 = get_pt(27, pos, VIEW_RANGE, canvas_size), get_pt(28, pos, VIEW_RANGE, canvas_size) 
    x_31, x_32 = get_pt(31, pos, VIEW_RANGE, canvas_size), get_pt(32, pos, VIEW_RANGE, canvas_size) 
    x_i = (int((x_23[0]+x_24[0])/2), int((x_23[1]+x_24[1])/2)) #virtual pelvis keypoint 

    colors = color_joint_deduction(COLOR_MAP, breakdown)
    for level in ["MINOR", "MEDIUM", "SEVERE"]:
        target_color = COLOR_MAP[level]

        if "torso_L" in colors and colors["torso_L"] == target_color:
            cv2.line(canvas, x_12, x_24, target_color, 3)
            cv2.line(canvas, x_24, x_26, target_color, 3)
            cv2.circle(canvas, x_24, 5, target_color, -1)

        if "opening_L" in colors and colors["opening_L"] == target_color:
            cv2.line(canvas, x_i, x_25, target_color, 3) 
            cv2.line(canvas, x_i, x_26, target_color, 3) 
            cv2.circle(canvas, x_i, 5, target_color, -1)

        if "upperleg_L" in colors and colors["upperleg_L"] == target_color:
            cv2.line(canvas, x_23, x_25, target_color, 3)
            cv2.line(canvas, x_24, x_26, target_color, 3)
            cv2.circle(canvas, x_25, 5, target_color, -1)
            cv2.circle(canvas, x_26, 5, target_color, -1)

        if "lowerleg_L" in colors and colors["lowerleg_L"] == target_color:
            cv2.line(canvas, x_25, x_27, target_color, 3)
            cv2.line(canvas, x_26, x_28, target_color, 3)

        if "toe_L" in colors and colors["toe_L"] == target_color:
            cv2.line(canvas, x_27, x_31, target_color, 3)
            cv2.line(canvas, x_28, x_32, target_color, 3)
            cv2.circle(canvas, x_27, 5, target_color, -1)
            cv2.circle(canvas, x_28, 5, target_color, -1)
    return canvas