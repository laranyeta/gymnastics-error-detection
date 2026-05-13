import json
import cv2
import math

from backend.scoring.evaluator import AcrobaticEvaluator
from backend.rnn.predict import predict

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
        idx = len(sequence) // 2
        peak_frame = sequence[idx] 
        
    return idx, peak_frame

def evaluate_performance(json_path, pred):
    with open(json_path, 'r') as f:
        sequence = json.load(f)
        
    evaluator = AcrobaticEvaluator()
    peak_idx, _ = find_acrobatic_peak(sequence, pred)
    angles = sequence[peak_idx].get("angles", {})
    pos = sequence[peak_idx].get("position", {})
    
    penalty = 0.0
    if pred == "tuck":
        hip_angle = (angles.get("joint_hip_L", 180) + angles.get("joint_hip_R", 180)) / 2
        knee_L = angles.get("joint_knee_L", 45)
        knee_R = angles.get("joint_knee_R", 45)
        ankle_L = angles.get("joint_ankle_L", 180)
        ankle_R = angles.get("joint_ankle_R", 180)
        penalty, breakdown = evaluator.eval_tuck(hip_angle, knee_L, knee_R, ankle_L, ankle_R)
        print(f"Torso angle: {hip_angle:.2f}")
        print(f"Knee L angle: {knee_L:.2f}")
        print(f"Knee R angle: {knee_R:.2f}")
        print(f"Ankle L angle: {ankle_L:.2f}")
        print(f"Ankle R angle: {ankle_R:.2f}")

    elif pred == "pike":
        hip_angle = (angles.get("joint_hip_L", 180) + angles.get("joint_hip_R", 180)) / 2
        knee_L = angles.get("joint_knee_L", 180)
        knee_R = angles.get("joint_knee_R", 180)
        ankle_L = angles.get("joint_ankle_L", 180)
        ankle_R = angles.get("joint_ankle_R", 180)
        penalty, breakdown = evaluator.eval_pike(hip_angle, knee_L, knee_R, ankle_L, ankle_R)
        print(f"Torso angle: {hip_angle:.2f}")
        print(f"Knee L angle: {knee_L:.2f}")
        print(f"Knee R angle: {knee_R:.2f}")
        print(f"Ankle L angle: {ankle_L:.2f}")
        print(f"Ankle R angle: {ankle_R:.2f}")
        
    elif pred == "split":
        opening_angle = calculate_split_angle(pos)
        knee_L = angles.get("joint_knee_L", 180)
        knee_R = angles.get("joint_knee_R", 180)
        ankle_L = angles.get("joint_ankle_L", 180)
        ankle_R = angles.get("joint_ankle_R", 180)
        print(f"Opening angle: {opening_angle:.2f}º")
        print(f"Knee L angle: {knee_L:.2f}")
        print(f"Knee R angle: {knee_R:.2f}")
        print(f"Ankle L angle: {ankle_L:.2f}")
        print(f"Ankle R angle: {ankle_R:.2f}")
        penalty, breakdown = evaluator.eval_split(opening_angle, knee_L, knee_R, ankle_L, ankle_R)
        
    elif pred == "straddle":
        opening_angle = angles.get("opening_L", 180) #not sure if needed in straddle
        knee_L = angles.get("joint_knee_L", 180)
        knee_R = angles.get("joint_knee_R", 180)
        ankle_L = angles.get("joint_ankle_L", 180)
        ankle_R = angles.get("joint_ankle_R", 180)
        print(f"Opening angle: {opening_angle:.2f}º")
        print(f"Knee L angle: {knee_L:.2f}")
        print(f"Knee R angle: {knee_R:.2f}")
        print(f"Ankle L angle: {ankle_L:.2f}")
        print(f"Ankle R angle: {ankle_R:.2f}")
        penalty, breakdown = evaluator.eval_straddle(opening_angle, knee_L, knee_R, ankle_L, ankle_R)

    return penalty, breakdown, peak_idx

def visualize_peak_frame(video_path, peak_frame):
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, peak_frame)
    
    ret, frame = cap.read()
    if ret:
        cv2.imshow(f"Peak Frame - Index: {peak_frame}", frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("\n[ERROR] Video has not been read.")
        
    cap.release()

#testing purposes
if __name__ == "__main__":
    json_path = "backend/rnn/test/test06.json"
    d_score = 5.0 #hardcoded, d-score is not automatized

    pred = predict(json_path, debug=False)
    
    print("\n--- GYMNASTICS EVALUATION REPORT ---")
    penalty, breakdown, peak_idx = evaluate_performance(json_path, pred)
    
    evaluator = AcrobaticEvaluator()
    final_score = evaluator.calculate_final_score(d_score, penalty)
    visualize_peak_frame("backend/rnn/test/test06_skeleton.avi", peak_idx)
    print(f"Detected Acrobatic: {pred.upper()}")
    
    print(f"\n--- DEDUCTION BREAKDOWN ---")
    if len(breakdown) == 0:
        print("Perfect execution. No deductions applied.")
    else:
        for reason in breakdown:
            print(f"{reason}")
            
    print(f"\nTotal Execution Deductions (E-Score): -{penalty:.1f}")
    print(f"Final Score (D + E): {final_score:.1f}")