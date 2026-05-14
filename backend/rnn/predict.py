import torch
import cv2
import os
import json

from backend.rnn.model import RNNAcrobaticClassificator, LABEL_MAPPING
from backend.rnn.process import process_sequence

def load_prediction_model():
    if torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cuda"

    best_path = "backend/rnn/checkpoints/best.pth"
    model = RNNAcrobaticClassificator(input_size=206, hidden_size=128, n_classes=4, n_layers=2)

    if os.path.exists(best_path):
        model.load_state_dict(torch.load(best_path, map_location=device))
    else:
        print(f"[ERROR] No model has been found in directory {best_path}")
        
    model.to(device)
    model.eval()
    
    return model, device

def predict_window(model, device, window_sequence):
    input_tensor = process_sequence(window_sequence).to(device)
    
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)[0] * 100
        confidence, predicted_idx = torch.max(probabilities, 0)
        
    class2text = {v: k for k, v in LABEL_MAPPING.items()}
    pred = class2text[predicted_idx.item()]
    
    return pred, confidence.item()

def predict(json_path, skeleton_path=None, output_path=None, debug=False):
    model, device = load_prediction_model()

    with open(json_path, 'r') as f:
        sequence = json.load(f)
        
    pred, confidence = predict_window(model, device, sequence)

    if debug:
        print(f"\n--- DEBUG PREDICTION ---")
        print(f"Detected: {pred.upper()} | Confidence: {confidence:.2f}%")
        
        if skeleton_path and output_path:
            save_predicted_video(skeleton_path, output_path, pred, confidence)
            
    return pred

def save_predicted_video(input_path, output_path, pred, conf):
    cap = cv2.VideoCapture(input_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    text = f"Acrobatic: {pred.upper()} | Conf: {conf:.2f}%"
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break     
        cv2.putText(frame, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
        out.write(frame)
        
    cap.release()
    out.release()
    print(f"[SUCCESS] Output video has been saved in directory {output_path}")

#testing purposes
if __name__ == "__main__":
    json_routine_path = "backend/rnn/test/test06.json"
    rnn_model, device = load_prediction_model()

    with open(json_routine_path, 'r') as f:
        full_routine = json.load(f)
    
    window_size = 40
    step = 20

    for i in range(0, len(full_routine) - window_size, step):
        window = full_routine[i : i + window_size]
        pred, conf = predict_window(rnn_model, device, window)
        
        if conf > 90.0:
            print(f"Frames {i}-{i+window_size}: {pred.upper()} ({conf:.1f}%)")