import torch
import json
import cv2
import os

from model import RNNAcrobaticClassificator, LABEL_MAPPING
from process import process_json

def predict(json_path, skeleton_path, output_path):
    if torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cuda"

    best_path = "backend/rnn/checkpoints/best.pth"
    
    model = RNNAcrobaticClassificator(input_size=206, hidden_size=128, n_classes=4, n_layers=2)
    model.load_state_dict(torch.load(best_path, map_location=device))
    model.to(device)
    model.eval()
    
    input_tensor = process_json(json_path).to(device)
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)[0] * 100 #output to %
        confidence, predicted_idx = torch.max(probabilities, 0) #max probability is best result
        
    class2text = {v: k for k, v in LABEL_MAPPING.items()} #0:tuck, 1:pike...
    final_prediction = class2text[predicted_idx.item()]
    
    print(f"Detected acrobatic : {final_prediction.upper()}")
    print(f"Confidence: {confidence.item():.2f}%")
    print(f"-"*15)
    print("Probability report")
    for exercise, idx in LABEL_MAPPING.items():
        print(f"{exercise}: {probabilities[idx].item():.2f}%")

    #classification information shown in video
    cap = cv2.VideoCapture(skeleton_path)
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    text = f"Acrobatic: {final_prediction.upper()} | Conf: {confidence.item():.2f}%"
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break     
        cv2.putText(frame, text, (90, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)
        out.write(frame)
        
    cap.release()
    out.release()
    print(f"[SUCCESS] Predicted video has been saved in rnn/test/outputs directory")
    
#testing purposes
if __name__ == "__main__":
    json_path = "backend/rnn/test/test02.json" #raw spatial data
    skeleton_path = "backend/rnn/test/test02_skeleton.avi" #raw spatial data into video frames (visual skeleton)

    os.makedirs("backend/rnn/test/outputs", exist_ok=True)
    output_path = "backend/rnn/test/outputs/test02_out.mov"
    predict(json_path, skeleton_path, output_path)