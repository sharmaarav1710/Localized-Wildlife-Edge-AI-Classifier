import os
from predict import run_edge_inference

TEST_DIR = "test_samples"

def evaluate():
    correct = 0
    total = 0
    
    # Track performance per species
    mismatches = []
    
    for class_name in ["bullfrog", "cane_toad"]:
        class_dir = os.path.join(TEST_DIR, class_name)
        
        # Ensure directory exists before iterating
        if not os.path.exists(class_dir):
            continue
            
        for img_name in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_name)

            # Get prediction
            pred_species, _ = run_edge_inference(img_path)

            if pred_species == class_name:
                correct += 1
            else:
                # Track the mismatch
                mismatches.append(f"MISMATCH: File {img_name} in '{class_name}' was predicted as '{pred_species}'")
                
            total += 1

    # Print results
    print("\n--- Evaluation Details ---")
    for m in mismatches:
        print(m)
        
    print(f"\nFinal Test Accuracy: {(correct/total)*100:.2f}% ({correct}/{total})")

if __name__ == "__main__":
    evaluate()