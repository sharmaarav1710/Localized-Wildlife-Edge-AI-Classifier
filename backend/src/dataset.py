import os
import torch
from PIL import Image
from torch.utils.data import Dataset, random_split
from torchvision import transforms

class WildlifeDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.class_to_idx = {'bullfrog': 0, 'cane_toad': 1}
        self.classes = ['bullfrog', 'cane_toad']
        self.samples = []
        self._build_dataset_index()

    def _build_dataset_index(self):
        valid_extensions = ('.jpg', '.jpeg', '.png', '.webp')
        for class_name in self.classes:
            class_dir = os.path.join(self.root_dir, class_name)
            class_idx = self.class_to_idx[class_name]
            
            print(f"DEBUG: Mapping folder '{class_name}' to index {class_idx}")
            
            for root, _, files in os.walk(class_dir):
                for f in files:
                    if f.lower().endswith(valid_extensions):
                        file_path = os.path.join(root, f)
                        self.samples.append((file_path, class_idx))
                        
        if len(self.samples) == 0:
            raise RuntimeError(f"Found 0 images in {self.root_dir}. Check your folder structure!")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        if idx % 10 == 0: 
            print(f"DEBUG: Loading {os.path.basename(img_path)} as label {label}")
            
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"⚠️ Warning: Failed to load {img_path}. Error: {e}")
            image = Image.new("RGB", (224, 224), (128, 128, 128))

        if self.transform:
            image = self.transform(image)
        return image, label

class TransformedSubset(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform
        
    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y
        
    def __len__(self):
        return len(self.subset)

def get_train_transforms():
    return transforms.Compose([
        # 1. Random Resized Crop is mandatory: Force model to focus on the frog
        transforms.RandomResizedCrop(224, scale=(0.5, 1.0)),
        # 2. Add Grayscale: Remove color bias (if bullfrogs are green and toads are brown, 
        # color is a shortcut. Grayscale forces focus on shape/texture)
        transforms.RandomGrayscale(p=0.3),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(30),
        # 3. Aggressive Color Jitter
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
def get_val_transforms():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def get_wildlife_dataloaders(root_dir, batch_size=16, val_split=0.2, seed=42):
    full_dataset = WildlifeDataset(root_dir=root_dir, transform=None)
    
    total_len = len(full_dataset)
    val_len = int(total_len * val_split)
    train_len = total_len - val_len
    
    generator = torch.Generator().manual_seed(seed)
    train_subset, val_subset = random_split(full_dataset, [train_len, val_len], generator=generator)
    
    # Debug: Map indices to original dataset labels
    labels = [full_dataset[i][1] for i in train_subset.indices]
    print(f"DEBUG: Training labels distribution: {labels}")
    
    train_dataset = TransformedSubset(train_subset, transform=get_train_transforms())
    val_dataset = TransformedSubset(val_subset, transform=get_val_transforms())
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    return train_loader, val_loader, full_dataset.classes