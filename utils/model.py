import torch 
import torch.nn as nn
import torch.optim as optim
import os
import json

from torch.utils.data import Dataset, DataLoader

LABEL_MAPPING = {"tuck": 0, "pike": 1, "stretch": 2, "split": 3, "straddle": 4}
class RNNAcrobaticClassificator(nn.Module):
    def __init__(self, input_size, hidden_size, n_classes, n_layers):
        super(RNNAcrobaticClassificator, self).__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True, dropout=0.2) #allows memory
        self.fc = nn.Linear(hidden_size, n_classes)

    def forward(self,x):
        out, _ = self.lstm(x)
        out = self.fc(out[:,-1,:])
        return out

class GymnasticsDataset(Dataset):
    def __init__(self, data_path):
        self.samples = []
        self.labels = []
        
        for label, _ in LABEL_MAPPING.items():
            dir = os.path.join(data_path, label)
            if not os.path.exists(dir):
                continue

            for file in os.listdir(dir):
                if file.endswith(".json"):
                    with open(os.path.join(dir, file), "r") as f:
                        sequence = json.load(f)
                        tensor = self._json_to_tensor(sequence)
                        self.samples.append(tensor)
                        self.labels.append(label)

    def _json_to_tensor(self, sequence):
        frames = []
        for frame in sequence:
            x = list(frame["position"].values())
            v = list(frame["velocity"].values())
            a= list(frame["acceleration"].values())
            frames.append(x+v+a)
        return torch.tensor(frames, dtype=torch.float32)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self,idx):
        return self.samples[idx], self.labels[idx]

def train():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu") #running on macbook air m5
    input_size = 198
    hidden_size = 128
    n_classes = 5
    epochs = 50
    batch_size = 8

    dataset = GymnasticsDataset("data")
    loader = DataLoader(dataset, batch_size, shuffle=True)
    model = RNNAcrobaticClassificator(input_size, hidden_size, n_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch+1)%5 == 0: #info shows every 5 epochs
            print(f"Epoch [{epoch+1}/{epochs}], Loss:  {total_loss/len(loader):.4f}")

    torch.save(model.state_dict(), "checkpoint.pth")
