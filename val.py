import torch
import torch.nn as nn
import json
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from model.uNet import UNet
from utils.metrics import iou
from utils.customDataset import CustomDataset
from utils.imageMasker import val_annotation_paths


# Set device for evaluation
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load paths for validation set
with open('val_images.json', 'r') as f:
    val_image_paths = json.load(f)

# Create validation dataset and dataloader
val_dataset = CustomDataset(
    image_paths=val_image_paths,
    annotation_paths=val_annotation_paths,
    mask_dir='monusegdata/valMask',
)

val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

# Initialize the model
model = UNet(n_channels=3, n_classes=1).to(device)

# Load the trained weights
model.load_state_dict(torch.load('trained_unet_model.pth'))
model.eval()

# Loss function
criterion = nn.BCEWithLogitsLoss()

# Validation loop
total_val_loss, total_val_iou = 0.0, 0.0

with torch.no_grad():
    for val_images, val_masks in val_loader:
        val_images, val_masks = val_images.to(device), val_masks.to(device)
        val_outputs = model(val_images)
        val_loss = criterion(val_outputs, val_masks)
        total_val_loss += val_loss.item()

        val_preds = torch.sigmoid(val_outputs) > 0.5
        val_preds = val_preds.float()

        masks_binary = val_masks > 0.5
        total_val_iou += iou(val_preds, masks_binary).item()

# Calculate average validation loss
average_val_loss = total_val_loss / len(val_loader)

# Calculate average IoU across all validation batches
average_val_iou = total_val_iou / len(val_loader)

# Print or log the results
print(f'Validation Loss: {average_val_loss:.4f}, Validation IoU: {average_val_iou:.4f}')

# Visualise predictions
plt.figure(figsize=(12, 6))
for i in range(min(3, len(val_images))):  # Visualise up to 3 images
    plt.subplot(3, 3, i + 1)  # Change the number of rows and columns to 3
    plt.imshow(val_images[i].cpu().permute(1, 2, 0))
    plt.title('Original Image')

    plt.subplot(3, 3, i + 4)  # Change the number of rows and columns to 3
    plt.imshow(val_preds[i].cpu().squeeze(), cmap='gray')
    plt.title('Predicted Mask')

    plt.subplot(3, 3, i + 7)  # Change the number of rows and columns to 3
    plt.imshow(val_masks[i].cpu().squeeze(), cmap='gray')
    plt.title('True Mask')

plt.tight_layout()
plt.show()
