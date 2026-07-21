from src.dataset import get_wildlife_dataloaders
import matplotlib.pyplot as plt

# Load your data
train_loader, val_loader, classes = get_wildlife_dataloaders('data/')

# Get one batch
images, labels = next(iter(train_loader))
print(f"Batch shape: {images.shape}") # Should be [16, 3, 224, 224]

# Optional: Visualize one image to confirm augmentation (e.g., rotation/flip)
plt.imshow(images[0].permute(1, 2, 0))
plt.show()