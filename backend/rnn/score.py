import json
import cv2
import math
import numpy as np

from backend.hpe.pose.utils.vision import CONNECTIONS
from backend.scoring.evaluator import AcrobaticEvaluator
from backend.scoring.rules import COLOR_MAP
from backend.rnn.predict import load_prediction_model, predict_window 

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

def find_acrobatic_peak(sequence, pred):
    if pred in ["split", "straddle"]:
        idx, peak_frame = max(enumerate(sequence), key=lambda x: calculate_split_angle(x[1].get("position", {})))
    elif pred == "tuck":
        idx, peak_frame = min(
            enumerate(sequence), 
            key=lambda x: (
                x[1].get("angles", {}).get("joint_hip_L", 45) + 
                x[1].get("angles", {}).get("joint_knee_L", 45) + 
                x[1].get("angles", {}).get("joint_knee_R", 45)
            )
        )
    elif pred == "pike":
        idx, peak_frame = min(
            enumerate(sequence), 
            key=lambda x: (
                x[1].get("angles", {}).get("joint_hip_L", 180) - 
                x[1].get("angles", {}).get("joint_knee_L", 180) - 
                x[1].get("angles", {}).get("joint_knee_R", 180)
            )
        )

    else:
        idx = len(sequence)//2
        peak_frame = sequence[idx] 
        
    return idx, peak_frame

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

def evaluate_routine(json_path, window=40, step=15): #evaluates more than one acrobatic (sliding window)
    with open(json_path, "r") as f:
        sequence = json.load(f) #full sequence

    model, device = load_prediction_model() 
    total_frames = len(sequence)
    results = []
    last_pred = None #last prediction of the model

    for start in range(0, total_frames - window+1, step):
        end = start + window
        window_seq = sequence[start:end]
        pred, conf = predict_window(model, device, window_seq)
    
        if pred != "none" and conf > 50 and pred != last_pred:
            penalty, breakdown, local_peak = evaluate_performance(window_seq, pred)
            global_peak = start + local_peak
            results.append({
                "acrobatic": pred,
                "global_peak": global_peak,
                "penalty": penalty,
                "breakdown": breakdown,
                "confidence": conf,
                "angles": sequence[global_peak].get("angles", {}), #to draw the skeleton
                "position": sequence[global_peak].get("position", {})
            })
            last_pred = pred
        elif pred == "none":
            last_pred = None 
            
    return results

def color_joint_deduction(COLOR_MAP, breakdown):
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

def visualize_peak_frame(video_path, peak_frame, pos, breakdown, pred, conf):
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, peak_frame)
    ret, original_frame = cap.read()
    cap.release()
    
    canvas_size = 800
    skeleton_canvas = np.zeros((canvas_size, canvas_size, 3), dtype=np.uint8)

    VIEW_RANGE = 3.0
    def get_pt(idx):
        x_val = float(pos.get(f"x_{idx}", 0))
        y_val = float(pos.get(f"y_{idx}", 0))
        px = int((x_val + VIEW_RANGE) / (VIEW_RANGE * 2) * canvas_size)
        py = int((y_val + VIEW_RANGE) / (VIEW_RANGE * 2) * canvas_size)
        return (px, py)
    
    for start_idx, end_idx in CONNECTIONS:
        pt1 = get_pt(start_idx)
        pt2 = get_pt(end_idx)
        cv2.line(skeleton_canvas, pt1, pt2, (255, 255, 255), 2)
        cv2.circle(skeleton_canvas, pt1, 4, (255, 255, 255), -1)
        cv2.circle(skeleton_canvas, pt2, 4, (255, 255, 255), -1)

    x_11, x_12 = get_pt(11), get_pt(12) 
    x_23, x_24 = get_pt(23), get_pt(24) 
    x_25, x_26 = get_pt(25), get_pt(26) 
    x_27, x_28 = get_pt(27), get_pt(28) 
    x_31, x_32 = get_pt(31), get_pt(32) 
    x_i = (int((x_23[0] + x_24[0]) / 2), int((x_23[1] + x_24[1]) / 2)) 

    colors = color_joint_deduction(COLOR_MAP, breakdown)

    for level in ["MINOR", "MEDIUM", "SEVERE"]:
        target_color = COLOR_MAP[level]

        if "torso_L" in colors and colors["torso_L"] == target_color:
            cv2.line(skeleton_canvas, x_11, x_23, target_color, 6)
            cv2.line(skeleton_canvas, x_12, x_24, target_color, 6)
            cv2.line(skeleton_canvas, x_23, x_24, target_color, 6)

        if "opening_L" in colors and colors["opening_L"] == target_color:
            cv2.line(skeleton_canvas, x_i, x_25, target_color, 6) 
            cv2.line(skeleton_canvas, x_i, x_26, target_color, 6) 
            cv2.circle(skeleton_canvas, x_i, 8, target_color, -1)

        if "upperleg_L" in colors and colors["upperleg_L"] == target_color:
            cv2.line(skeleton_canvas, x_23, x_25, target_color, 6)
            cv2.line(skeleton_canvas, x_24, x_26, target_color, 6)
            cv2.circle(skeleton_canvas, x_25, 8, target_color, -1)
            cv2.circle(skeleton_canvas, x_26, 8, target_color, -1)

        if "lowerleg_L" in colors and colors["lowerleg_L"] == target_color:
            cv2.line(skeleton_canvas, x_25, x_27, target_color, 6)
            cv2.line(skeleton_canvas, x_26, x_28, target_color, 6)

        if "toe_L" in colors and colors["toe_L"] == target_color:
            cv2.line(skeleton_canvas, x_27, x_31, target_color, 6)
            cv2.line(skeleton_canvas, x_28, x_32, target_color, 6)
            cv2.circle(skeleton_canvas, x_27, 8, target_color, -1)
            cv2.circle(skeleton_canvas, x_28, 8, target_color, -1)

    text_pred = f"Acrobatic: {pred.upper()}"
    text_conf = f"Confidence: {conf:.2f}%"
    
    cv2.putText(skeleton_canvas, text_pred, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(skeleton_canvas, text_conf, (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 1, cv2.LINE_AA)

    if ret:
        orig_h, orig_w = original_frame.shape[:2]
        scale = canvas_size/orig_h
        resized_orig = cv2.resize(original_frame, (int(orig_w*scale), canvas_size))
        
        combined_view = np.hstack((resized_orig, skeleton_canvas))
        cv2.imshow(f"AI Judge UI - Peak Frame: {peak_frame}", combined_view) 
    else:
        cv2.imshow(f"Virtual Skeleton - Peak Frame: {peak_frame}", skeleton_canvas) 
    
    cv2.waitKey(0) 
    cv2.destroyAllWindows()

#testing purposes
if __name__ == "__main__":
    json_path = "backend/rnn/test/test08.json"
    video_path = "backend/rnn/test/test08_skeleton.avi"
    d_score = 5.0 
    total_penalty = 0.0

    print("\n--- GYMNASTICS EVALUATION REPORT ---")
    results = evaluate_routine(json_path, window=40, step=20)

    if not results:
        print("No acrobatics detected.")
        exit()
    else:
        for item in results:
            print(f"\nDetected Acrobatic: {item['acrobatic'].upper()} ({item['confidence']:.2f}%)")
            print(f"Peak frame: {item['global_peak']}")
            
            if len(item['breakdown']) == 0:
                print("Perfect execution. No deductions applied.")
            else:
                print("\n-DEDUCTIONS-")
                for reason in item['breakdown']:
                        print(f"{reason}")

            total_penalty += item['penalty']
            visualize_peak_frame(video_path, item['global_peak'], item['position'], item['breakdown'], item['acrobatic'], item['confidence'])

    evaluator = AcrobaticEvaluator()
    final_score = evaluator.calculate_final_score(d_score, total_penalty)

    print("\n-FINAL SCORE-")
    print(f"Total deductions applied: -{total_penalty:.1f}")
    print(f"Final Score (D + E): {final_score:.1f}")