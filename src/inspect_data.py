import os
import matplotlib.pyplot as plt
from PIL import Image

def inspect_images(data_dir="data/"):
    classes = ['bullfrog', 'cane_toad']
    
    for cls in classes:
        cls_path = os.path.join(data_dir, cls)
        images = [f for f in os.listdir(cls_path) if f.endswith(('.jpg', '.png', '.jpeg'))]
        
        print(f"\n--- Inspecting {cls}: {len(images)} images ---")
        
        # Display up to 20 images in a grid
        fig, axes = plt.subplots(4, 5, figsize=(15, 12))
        axes = axes.flatten()
        
        for i, img_name in enumerate(images[:20]):
            img = Image.open(os.path.join(cls_path, img_name))
            axes[i].imshow(img)
            axes[i].set_title(img_name, fontsize=8)
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    inspect_images()