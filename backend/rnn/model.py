import torch 
import torch.nn as nn
import torch.optim as optim
import os
import json

from torch.utils.data import random_split
from torch.utils.data import Dataset, DataLoader

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

def train():
    if torch.backends.mps.is_available():
        device = 'mps' #macos compatibility
    elif torch.cuda.is_available():
        device = 'cuda' #windows compatibility
    else:
        device = 'cpu'
        
    print(f"[SYSTEM] Running on device: {device}")
    
    input_size = 206 # 33 keypoints*2 (x,y) * 3 (x,v,a) + 8 ang
    hidden_size = 128
    n_classes = len(LABEL_MAPPING) #4
    n_layers = 2
    epochs = 50
    batch_size = 4

    dataset = GymnasticsDataset("backend/rnn/data")
    train_size = int(0.8 * len(dataset)) #80% train
    val_size = len(dataset) - train_size #20% validation

    torch.manual_seed(42) #same split for each execution
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    model = RNNAcrobaticClassificator(input_size, hidden_size, n_classes, n_layers).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):
        model.train() #training phase
        total_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        model.eval() #evaluation phase
        val_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for X_val, y_val in val_loader:
                X_val, y_val = X_val.to(device), y_val.to(device)
                
                outputs = model(X_val)
                loss = criterion(outputs, y_val)
                val_loss += loss.item()
                
                _, y_pred = torch.max(outputs.data, 1)
                total += y_val.size(0)
                correct += (y_pred == y_val).sum().item()

        train_loss_avg = total_loss/len(train_loader)
        val_loss_avg = val_loss/len(val_loader)
        accuracy = 100 * correct/total
        print(f"Epoch [{epoch+1}/{epochs}] | Train Loss: {train_loss_avg:.4f} | Val Loss: {val_loss_avg:.4f} | Val Acc: {accuracy:.2f}%")

    os.makedirs("backend/rnn/checkpoints", exist_ok=True)
    torch.save(model.state_dict(), "backend/rnn/checkpoints/best.pth")
    print("[SUCCESS] Model file has been saved in directory /checkpoints")

if __name__ == "__main__":
    train()