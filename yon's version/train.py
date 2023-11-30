import os
import pickle
from torch.utils.data import Dataset, DataLoader, random_split
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torchvision import transforms

script_dir = os.path.dirname(os.path.realpath(__file__))
preprocessed_folder = os.path.join(script_dir, "preprocessed_data")

class CustomDataset(Dataset):
    def __init__(self, images, annotations, transform=None):
        self.images = images
        self.annotations = annotations
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = self.images[index]
        annotation = self.annotations[index]

        # Apply transformations if specified
        if self.transform:
            image = self.transform(image)

        return image, annotation

# Load preprocessed datasets
val_dataset_path = os.path.join(preprocessed_folder, "val_preprocessed_data.pkl")
train_dataset_path = os.path.join(preprocessed_folder, "train_preprocessed_data.pkl")

with open(val_dataset_path, 'rb') as file:
    val_dataset = pickle.load(file)

with open(train_dataset_path, 'rb') as file:
    train_dataset = pickle.load(file)

# Create data loaders
batch_size = 4
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Assuming input size matches the pre-trained ResNet model
input_size = (3, 256, 256)

# Define the model
model = models.segmentation.fcn_resnet50(pretrained=True, progress=True)

# Freeze pre-trained layers
for param in model.parameters():
    param.requires_grad = False

# Modify the output layer to match your task
# For example, if you have two classes (foreground and background):
model.classifier[4] = nn.Conv2d(512, 2, kernel_size=(1, 1), stride=(1, 1))

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

# Move the model to CUDA if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Training loop
num_epochs = 10

for epoch in range(num_epochs):
    model.train()

    for inputs, annotations in train_loader:
        inputs, annotations = inputs.to(device), annotations.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)['out']
        loss = criterion(outputs, annotations)
        loss.backward()
        optimizer.step()

    # Validation loop (if you have a validation set)
    model.eval()

    with torch.no_grad():
        for inputs, annotations in val_loader:
            inputs, annotations = inputs.to(device), annotations.to(device)

            outputs = model(inputs)['out']
            val_loss = criterion(outputs, annotations)

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}, Val Loss: {val_loss.item()}")

# Save the trained model
torch.save(model.state_dict(), "resnet_segmentation_model.pth")
