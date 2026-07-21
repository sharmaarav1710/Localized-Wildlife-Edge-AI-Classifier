import torch
import torch.nn as nn
from torchvision import models
from dataset import get_wildlife_dataloaders

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, _, _ = get_wildlife_dataloaders("data/")

    # Use ResNet18
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    
    # Freeze the feature extractor completely
    for param in model.parameters():
        param.requires_grad = False
    
    # Custom Head: Adding two layers with Dropout to force generalization
    model.fc = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(model.fc.in_features, 512),
        nn.ReLU(),
        nn.Dropout(p=0.5),
        nn.Linear(512, 2)
    )
    
    model = model.to(device)
    
    # Use AdamW (better weight decay handling than Adam)
    optimizer = torch.optim.AdamW(model.fc.parameters(), lr=0.001, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(50): # Longer training
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    import os
    os.makedirs("models", exist_ok=True)
    
    # Save the model weights
    torch.save(model.state_dict(), "models/wildlife_model.pth")
    print("Model saved successfully to models/wildlife_model.pth")
    
    # Exporting...
if __name__ == "__main__":
    train()