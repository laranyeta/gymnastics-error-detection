import torch
import json

from model import RNNAcrobaticClassificator, LABEL_MAPPING

def process_single_json(json_path):
    with open(json_path, 'r') as f:
        sequence = json.load(f)
    
    frames = []
    for frame in sequence:
        x = list(frame.get("position", {}).values())
        v = list(frame.get("velocity", {}).values())
        a = list(frame.get("acceleration", {}).values())
        ang = list(frame.get("angles", {}).values())
        frames.append(x + v + a + ang)
    
    tensor = torch.tensor(frames, dtype=torch.float32)
    return tensor.unsqueeze(0) 

def predict_video(json_path):
    if torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cuda"

    best_path = "backend/rnn/checkpoints/best.pth"
    
    model = RNNAcrobaticClassificator(input_size=206, hidden_size=128, n_classes=4, n_layers=2)
    model.load_state_dict(torch.load(best_path, map_location=device))
    model.to(device)
    model.eval()
    
    input_tensor = process_single_json(json_path).to(device)
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

if __name__ == "__main__":
    test_path = "backend/rnn/test.json" 
    predict_video(test_path)