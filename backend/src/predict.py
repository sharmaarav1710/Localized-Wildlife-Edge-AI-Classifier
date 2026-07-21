import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# 1. Define image transform
# Note: For standard evaluation/inference, we use direct resizing.
# If using TTA, we apply transformations per pass.
eval_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

tta_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def run_edge_inference(img_path, num_passes=5):
    # Resolve relative path to models/wildlife_model.pth correctly
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(base_dir, "models", "wildlife_model.pth")

    # Initialize ResNet-18 architecture
    my_model = models.resnet18()
    my_model.fc = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(my_model.fc.in_features, 512),
        nn.ReLU(),
        nn.Dropout(p=0.5),
        nn.Linear(512, 2)
    )
    
    # Load model weights on CPU
    my_model.load_state_dict(torch.load(model_path, map_location="cpu"))
    my_model.eval()
    
    # Process Image
    img = Image.open(img_path).convert('RGB')
    logits_list = []
    
    with torch.no_grad():
        for i in range(num_passes):
            # Pass 0 uses deterministic transform, subsequent passes use TTA
            tf = eval_transform if i == 0 else tta_transform
            input_tensor = tf(img).unsqueeze(0)
            logits = my_model(input_tensor)
            logits_list.append(logits)
            
    # Calculate average logits and probabilities
    avg_logits = torch.stack(logits_list).mean(dim=0)
    probabilities = torch.softmax(avg_logits, dim=1)
    
    pred_idx = torch.argmax(probabilities, dim=1).item()
    confidence = probabilities[0][pred_idx].item()
    
    classes = ["bullfrog", "cane_toad"]
    return classes[pred_idx], confidence