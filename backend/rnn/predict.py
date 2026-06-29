import torch
import os
import sys

from backend.rnn.model import RNNAcrobaticClassificator, LABEL_MAPPING
from backend.rnn.process import process_sequence

def resource_path(relative_path): #to create pyinstaller executable file
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)

def load_prediction_model():
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    best_path = resource_path("backend/rnn/checkpoints/best.pth")
    model = RNNAcrobaticClassificator(input_size=206, hidden_size=128, n_classes=4, n_layers=2)

    if os.path.exists(best_path):
        model.load_state_dict(torch.load(best_path, map_location=device))
    else:
        print(f"[ERROR] No model has been found in directory {best_path}")
        
    model.to(device)
    model.eval()
    
    return model, device

def predict(model, device, window_sequence):
    input_tensor = process_sequence(window_sequence).to(device)
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)[0] * 100
        confidence, predicted_idx = torch.max(probabilities, 0)
        
    class2text = {v: k for k, v in LABEL_MAPPING.items()}
    pred = class2text[predicted_idx.item()]
    return pred, confidence.item()