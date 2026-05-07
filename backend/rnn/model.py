import torch 
import torch.nn as nn
import os
import json

from torch.utils.data import Dataset

LABEL_MAPPING = {"tuck": 0, "pike": 1, "split": 2, "straddle": 3}

class RNNAcrobaticClassificator(nn.Module):
    def __init__(self, input_size, hidden_size, n_classes, n_layers=2): #default value for n_layers
        super(RNNAcrobaticClassificator, self).__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=n_layers, batch_first=True, dropout=0.5) #batch, seq_length, input_size
        self.fc = nn.Linear(hidden_size, n_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :]) #last frame
        return out

class GymnasticsDataset(Dataset):
    def __init__(self, data_path):
        self.samples = []
        self.labels = []
        
        for label, idx in LABEL_MAPPING.items():
            dir = os.path.join(data_path, label)
            if not os.path.exists(dir):
                continue

            for file in os.listdir(dir):
                if file.endswith(".json"):
                    with open(os.path.join(dir, file), "r") as f:
                        sequence = json.load(f)
                        tensor = self._json_to_tensor(sequence)
                        self.samples.append(tensor)
                        self.labels.append(idx) 

    def _json_to_tensor(self, sequence):
        frames = []
        for frame in sequence:
            x = list(frame.get("position", {}).values())
            v = list(frame.get("velocity", {}).values())
            a = list(frame.get("acceleration", {}).values())
            ang = list(frame.get("angles", {}).values())
            frames.append(x + v + a + ang)
            
        return torch.tensor(frames, dtype=torch.float32)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx], self.labels[idx]