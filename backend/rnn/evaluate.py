import torch
import os
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix, classification_report
from torch.utils.data import random_split, DataLoader
from model import RNNAcrobaticClassificator, GymnasticsDataset, LABEL_MAPPING

def evaluate():
    if torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cuda'
    
    data_path = "backend/rnn/data"
    best_path = "backend/rnn/checkpoints/best.pth"

    dataset = GymnasticsDataset(data_path)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    

    torch.manual_seed(42) 
    _, val_dataset = random_split(dataset, [train_size, val_size])
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    
    model = RNNAcrobaticClassificator(input_size=206, hidden_size=128, n_classes=4, n_layers=2)
    model.load_state_dict(torch.load(best_path, map_location=device))
    model.to(device)
    model.eval() #dropout turned off

    y_true = []
    y_pred = []
    
    with torch.no_grad():
        for X_val, y_val in val_loader:
            X_val, y_val = X_val.to(device), y_val.to(device)
            outputs = model(X_val)
            _, predicted = torch.max(outputs.data, 1)
            
            y_true.extend(y_val.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
            
    categories = list(LABEL_MAPPING.keys())
    report = classification_report(y_true, y_pred, target_names=categories)
    print(report)
    
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=categories, yticklabels=categories)
    
    plt.title("Confusion Matrix - Acrobatic Classificator")
    plt.ylabel("True")
    plt.xlabel("Predict")
    
    plt.savefig("metrics/model_confusion_matrix.png", bbox_inches='tight', dpi=300)
    print(f"\n[SUCCESS] Confusion matrix has been saved in /metrics directory")

if __name__ == "__main__":
    evaluate()