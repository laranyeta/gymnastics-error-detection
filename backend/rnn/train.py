import torch
import torch.nn as nn
import os

from backend.rnn.model import LABEL_MAPPING, GymnasticsDataset, RNNAcrobaticClassificator
from torch.utils.data import random_split
from torch.utils.data import DataLoader

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
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

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

#testing purposes
if __name__ == "__main__":
    train()