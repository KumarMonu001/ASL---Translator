import os
import sys
import torch
import torch.nn as nn
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report

# ==========================================
# ADD PROJECT ROOT TO PATH (DYNAMIC FIX)
# ==========================================
# Script path: AI_Intern_ASL/pipeline/training/evaluate.py
current_dir = os.path.dirname(os.path.abspath(__file__))
# Move up TWO levels to reach: AI_Intern_ASL/
PROJECT_ROOT = os.path.abspath(os.path.join(current_dir, "..", ".."))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ==========================================
# IMPORTS (Anchored to Project Root)
# ==========================================
# Since 'models' is inside 'pipeline', we use that nesting
from pipeline.models.cnn_model import ASLCNN
from pipeline.transforms.transform import RobustTransform

# ==========================================
# DEVICE & CONFIG
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n--- Evaluation Setup ---")
print(f"Project Root: {PROJECT_ROOT}")
print(f"Using Device: {device}")

# Corrected Paths based on your system audit
MODEL_PATH = os.path.join(PROJECT_ROOT, "pipeline", "saved_models", "robust_model.pth")
TEST_DIR = os.path.join(PROJECT_ROOT, "datasets", "asl_alphabet_test", "asl_alphabet_test")

IMAGE_SIZE = 64
BATCH_SIZE = 32
NUM_CLASSES = 29

# ==========================================
# TRANSFORMS
# ==========================================
clean_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

robust_test_transform = transforms.Compose([
    RobustTransform(),
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# ==========================================
# DATASETS & LOADERS
# ==========================================
try:
    clean_dataset = datasets.ImageFolder(root=TEST_DIR, transform=clean_transform)
    robust_dataset = datasets.ImageFolder(root=TEST_DIR, transform=robust_test_transform)

    clean_loader = DataLoader(clean_dataset, batch_size=BATCH_SIZE, shuffle=False)
    robust_loader = DataLoader(robust_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    class_names = list(clean_dataset.class_to_idx.keys())
    print(f"✅ Found {len(class_names)} classes in Test Directory.")
except Exception as e:
    print(f"❌ Dataset Error: {e}")
    sys.exit(1)

# ==========================================
# LOAD MODEL
# ==========================================
model = ASLCNN(num_classes=NUM_CLASSES)

if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model = model.to(device)
    model.eval()
    print(f"✅ Model loaded successfully from: {MODEL_PATH}")
else:
    print(f"❌ Critical Error: Model file NOT found at {MODEL_PATH}")
    sys.exit(1)

# ==========================================
# EVALUATION FUNCTION
# ==========================================
def evaluate(loader, mode_name):
    correct, total = 0, 0
    all_labels, all_predictions = [], []

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    accuracy = (100 * correct / total)
    print(f"{mode_name} Accuracy: {accuracy:.2f}%")
    return accuracy, all_labels, all_predictions

# ==========================================
# EXECUTION
# ==========================================
if __name__ == '__main__':
    print("\n--- Starting Evaluation ---")
    
    # 1. Evaluate Clean Data
    clean_acc, _, _ = evaluate(clean_loader, "Clean Dataset")

    # 2. Evaluate Robust Data
    robust_acc, robust_labels, robust_preds = evaluate(robust_loader, "Robust Dataset")

    # 3. Final Comparison Output
    print("\n" + "="*30)
    print("ROBUSTNESS BENCHMARK")
    print("="*30)
    print(f"Clean (Baseline): {clean_acc:.2f}%")
    print(f"Robust (Distorted): {robust_acc:.2f}%")
    print(f"Accuracy Gap: {clean_acc - robust_acc:.2f}%")
    print("="*30)

    # 4. Generate Confusion Matrix
    cm = confusion_matrix(robust_labels, robust_preds)
    plt.figure(figsize=(15, 12))
    sns.heatmap(cm, annot=False, cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix: Robust Test Set")
    plt.show()