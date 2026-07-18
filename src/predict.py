import torch
from torchvision import models
import torch.nn as nn
from torchvision import transforms
from PIL import Image

# 1. Define the transformation
tta_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def run_edge_inference(img_path, num_passes=5):
    # Load model inside the function scope
    # Ensure this architecture matches your train.py exactly
# 1. Initialize the architecture (Must match train.py exactly)
    my_model = models.resnet18()
    
    # This matches the 'Sequential' head used in training
    my_model.fc = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(my_model.fc.in_features, 512),
        nn.ReLU(),
        nn.Dropout(p=0.5),
        nn.Linear(512, 2)
    )
    
    # 2. Load the weights
    my_model.load_state_dict(torch.load("models/wildlife_model.pth", map_location="cpu"))
    my_model.eval()
    
    # Inference
    img = Image.open(img_path).convert('RGB')
    logits_list = []
    
    with torch.no_grad():
        for _ in range(num_passes):
            input_tensor = tta_transform(img).unsqueeze(0)
            logits = my_model(input_tensor)
            logits_list.append(logits)
            
    avg_logits = torch.stack(logits_list).mean(dim=0)
    pred_idx = torch.argmax(avg_logits, dim=1).item()
    
    classes = ["bullfrog", "cane_toad"]
    return classes[pred_idx], None