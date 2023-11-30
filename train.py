import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
from model.uNet import UNet
from utils.metrics import iou
from utils.customDataset import CustomDataset
from utils.imageMasker import train_annotation_paths, val_annotation_paths, test_annotation_paths


# Set device for training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load paths for training, validation, and test sets
with open('train_images.json', 'r') as f:
    train_image_paths = json.load(f)
with open('val_images.json', 'r') as f:
    val_image_paths = json.load(f)
with open('test_images.json', 'r') as f:
    test_image_paths = json.load(f)

# Data loaders
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(degrees=30),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
])

def apply_transform(img):
    # Convert PIL Image to tensor if needed
    if isinstance(img, Image.Image):
        img = transforms.ToTensor()(img)
    
    # Check if img is a tensor, if yes, apply data augmentation
    if isinstance(img, torch.Tensor):
        img = transform(img)
    
    return img

# Create datasets and dataloaders using the paths
train_dataset = CustomDataset(
    image_paths=train_image_paths,
    annotation_paths=train_annotation_paths,  # Assuming train_annotation_paths is defined
    mask_dir='monusegdata/trainMask',  # Adjust the path accordingly
    transform=apply_transform
)

val_dataset = CustomDataset(
    image_paths=val_image_paths,
    annotation_paths=val_annotation_paths,  # Assuming val_annotation_paths is defined
    mask_dir='monusegdata/valMask',  # Adjust the path accordingly
)

test_dataset = CustomDataset(
    image_paths=test_image_paths,
    annotation_paths=test_annotation_paths,  # Assuming test_annotation_paths is defined
    mask_dir='monusegdata/testMask',  # Adjust the path accordingly
)

# Data loaders
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

# Model, Loss function, Optimizer
model = UNet(n_channels=3, n_classes=1).to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training and Validation Loops
num_epochs = 3
train_losses, val_losses = [], []
train_ious, val_ious = [], []

for epoch in range(num_epochs):
    # Training Loop
    model.train()
    total_train_loss = 0.0
    total_train_iou = 0.0

    for images, masks in train_loader:
        images, masks = images.to(device), masks.to(device)

        # Apply data augmentation
        if transform:
            images = torch.stack([transform(img) for img in images])

        outputs = model(images)
        loss = criterion(outputs, masks)
        total_train_loss += loss.item()

        preds = torch.sigmoid(outputs) > 0.5
        masks_binary = masks > 0.5
        total_train_iou += iou(preds, masks_binary).item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    train_losses.append(total_train_loss / len(train_loader))
    train_ious.append(total_train_iou / len(train_loader))

    # Validation Loop
    model.eval()
    total_val_loss, total_val_iou = 0.0, 0.0

    with torch.no_grad():
        for val_images, val_masks in val_loader:
            val_images, val_masks = val_images.to(device), val_masks.to(device)
            val_outputs = model(val_images)
            val_loss = criterion(val_outputs, val_masks)
            total_val_loss += val_loss.item()

            # Apply data augmentation if needed
            if transform:
                val_images_transformed = torch.stack([transform(img) for img in val_images])
                val_preds = torch.sigmoid(model(val_images_transformed)) > 0.5
            else:
                val_preds = torch.sigmoid(val_outputs) > 0.5

            total_val_iou += iou(val_preds, val_masks).item()

    val_losses.append(total_val_loss / len(val_loader))
    val_ious.append(total_val_iou / len(val_loader))

    print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_losses[-1]:.4f}, Train IoU: {train_ious[-1]:.4f}, '
          f'Val Loss: {val_losses[-1]:.4f}, Val IoU: {val_ious[-1]:.4f}')

# Save the trained model
torch.save(model.state_dict(), 'trained_unet_model.pth')

# Plot the training and validation loss and IoU
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss')
plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss')
plt.title('Loss over Epochs')
plt.xlabel('Epochs')
plt.xticks(range(1, num_epochs + 1))
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(range(1, num_epochs + 1), train_ious, label='Train IoU')
plt.plot(range(1, num_epochs + 1), val_ious, label='Validation IoU')
plt.title('IoU over Epochs')
plt.xlabel('Epochs')
plt.xticks(range(1, num_epochs + 1))
plt.ylabel('IoU')
plt.legend()

plt.tight_layout()
plt.show()
plt.savefig('Loss_and_IoU_over_Epochs.png')
