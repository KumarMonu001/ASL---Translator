import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from PIL import Image

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from torchvision.utils import make_grid

# ==========================================
# ADD PROJECT ROOT TO PATH
# ==========================================
sys.path.append(
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            ".."
        )
    )
)

# ==========================================
# IMPORTS
# ==========================================
from pipeline.transforms.transform import RobustTransform
from pipeline.models.cnn_model import ASLCNN

# ==========================================
# DEVICE & CONFIG
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# Point to the main directory containing the 29 class folders
TRAIN_DIR = os.path.join(PROJECT_ROOT, "datasets", "asl_alphabet_train", "asl_alphabet_train")

IMAGE_SIZE = 64
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 20
NUM_CLASSES = 29

# ==========================================
# TRANSFORMS
# ==========================================
train_transform = transforms.Compose([
    RobustTransform(),
    transforms.ToPILImage() if not isinstance(RobustTransform(), Image.Image) else transforms.Lambda(lambda x: x), 
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomRotation(10),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Use the same normalization for validation
val_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# ==========================================
# DATASETS & LOADERS (THE 90/10 SPLIT)
# ==========================================
if __name__ == '__main__':
    # 1. Load the full 87,000 image dataset
    # Note: We apply train_transform here, but we will override val_data later
    full_dataset = datasets.ImageFolder(root=TRAIN_DIR, transform=train_transform)

    # 2. Split into 90% Train (78,300) and 10% Val (8,700)
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_data, val_data = random_split(
        full_dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    # 3. Optimization: Ensure Validation data uses the clean val_transform (no augmentations)
    val_data.dataset.transform = val_transform

    # 4. Final Loaders
    train_loader = DataLoader(
        train_data, batch_size=BATCH_SIZE, shuffle=True, 
        num_workers=4, pin_memory=True
    )

    val_loader = DataLoader(
        val_data, batch_size=BATCH_SIZE, shuffle=False, 
        num_workers=4, pin_memory=True
    )

    print(f"\n--- Dataset Audit ---")
    print(f"Total Images: {len(full_dataset)}")
    print(f"Training: {len(train_data)} | Validation: {len(val_data)}")
    print(f"Using Device: {device}\n")

# ==========================================
# MODEL, LOSS, OPTIMIZER, SCHEDULER
# ==========================================
model = ASLCNN(num_classes=NUM_CLASSES).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

# ==========================================
# TRAINING FUNCTIONS
# ==========================================
def imshow(img):
    img = img / 2 + 0.5
    plt.imshow(img.numpy().transpose((1, 2, 0)))
    plt.axis("off")

def train_one_epoch(loader):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    return running_loss / len(loader), 100 * correct / total

def evaluate(loader):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return running_loss / len(loader), 100 * correct / total

# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == '__main__':
    # Visualization check
    try:
        sample_batch, _ = next(iter(train_loader))
        plt.figure(figsize=(10, 5))
        imshow(make_grid(sample_batch[:8]))
        plt.title("Robust Training Samples")
        plt.show()
    except Exception as e:
        print(f"Visualization skipped: {e}")

    best_acc = 0.0
    save_path = os.path.join(PROJECT_ROOT, "saved_models", "robust_model.pth")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    for epoch in range(EPOCHS):
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch [{epoch+1}/{EPOCHS}] | LR: {current_lr:.6f}")

        train_loss, train_acc = train_one_epoch(train_loader)
        val_loss, val_acc = evaluate(val_loader)

        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), save_path)
            print(f">>> New Best Model Saved ({best_acc:.2f}%)")
        
        scheduler.step()

    print("\nRobust Training Complete!")