import json
import cv2

from backend.scoring.evaluator import AcrobaticEvaluator
from backend.rnn.predict import predict

'''def find_acrobatic_peak(sequence, pred):
    if pred in ["split", "straddle"]:
        idx, peak_frame = max(enumerate(sequence), key=lambda x: x[1]["angles"].get("opening_L", 0))
    elif pred in ["tuck", "pike"]:
        idx, peak_frame = min(enumerate(sequence), key=lambda x: x[1]["angles"].get("joint_hip_L", 180)) 
    else:
        idx = len(sequence) // 2
        peak_frame = sequence[idx] 
        
    return idx, peak_frame'''

def evaluate_performance(json_path, pred):
    with open(json_path, 'r') as f:
        sequence = json.load(f)
        
    evaluator = AcrobaticEvaluator()
    
    #peak_idx, peak_frame = find_acrobatic_peak(sequence, pred)
    angles = sequence[30].get("angles", {})
    pos = sequence[30].get("position", {})
    
    penalty = 0.0
    if pred == "tuck":
        hip_angle = (angles.get("joint_hip_L", 180) + angles.get("joint_hip_R", 180)) / 2
        knee_angle = (angles.get("joint_knee_L", 180) + angles.get("joint_knee_R", 180)) / 2
        toe_distance = 0.0 
        shoulder_width = 1.0 
        penalty = evaluator.eval_tuck(hip_angle, knee_angle, toe_distance, shoulder_width)
        
    elif pred == "pike":
        hip_angle = (angles.get("joint_hip_L", 180) + angles.get("joint_hip_R", 180)) / 2
        knee_angle = (angles.get("joint_knee_L", 180) + angles.get("joint_knee_R", 180)) / 2
        toe_distance = 0.0
        shoulder_width = 1.0
        toes_flexed = False
        penalty = evaluator.eval_pike(hip_angle, knee_angle, toe_distance, shoulder_width, toes_flexed)
        
    elif pred == "split":
        opening_angle = angles.get("opening_R", 180)
        knee_angle = min(angles.get("joint_knee_L", 180), angles.get("joint_knee_R", 180)) 
        toes_flexed = False
        print(f"\n[DEBUG] Read values: {30}:")
        print(f"Opening angle: {opening_angle:.2f}º")
        print(f"Most flexed angles: {knee_angle:.2f}º")
        penalty = evaluator.eval_split(opening_angle, knee_angle, toes_flexed)
        
    elif pred == "straddle":
        opening_angle = angles.get("opening_L", 180)
        knee_angle = min(angles.get("joint_knee_L", 180), angles.get("joint_knee_R", 180))
        print(f"\n[DEBUG] Read values: {30}:")
        print(f"Opening angle: {opening_angle:.2f}º")
        print(f"Most flexed angle: {knee_angle:.2f}º")
        penalty = evaluator.eval_straddle(opening_angle, knee_angle)

    return penalty

def visualize_peak_frame(video_path, frame_idx, output_img="peak_frame.jpg"):
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    
    ret, frame = cap.read()
    if ret:
        cv2.imshow(f"Peak Frame - Index: {frame_idx}", frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("\n[ERROR] Video has not been read.")
        
    cap.release()

#testing purposes
if __name__ == "__main__":
    json_path = "backend/rnn/test/test01.json"
    d_score = 5.0 #hardcoded, d-score is not automatized

    pred = predict(json_path, debug=False)
    
    print("\n--- GYMNASTICS EVALUATION ---")
    
    penalty = evaluate_performance(json_path, pred)
    evaluator = AcrobaticEvaluator()
    final_score = evaluator.calculate_final_score(d_score, penalty)
    
    visualize_peak_frame("backend/rnn/test/test01_skeleton.avi", 30)
    print(f"Detected Acrobatic: {pred.upper()}")
    print(f"Evaluated at Frame: 30")
    print(f"Execution Deductions (E-Score): -{penalty:.1f}")
    print(f"Final Score (D + E): {final_score:.1f}")