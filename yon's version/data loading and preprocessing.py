import os
import pickle
import numpy as np
from PIL import Image
import xml.etree.ElementTree as ET
from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.data import random_split

script_dir = os.path.dirname(os.path.realpath(__file__))

def load_data(data_folder, data_type="training"):
    image_folder = os.path.join(data_folder, "Tissue Images")
    annotation_folder = os.path.join(data_folder, "Annotations")

    image_files = [f for f in os.listdir(image_folder) if f.endswith(".tif")]
    annotation_files = [f for f in os.listdir(annotation_folder) if f.endswith(".xml")]

    # Make sure the lists are sorted for proper synchronization
    image_files.sort()
    annotation_files.sort()

    # Initialize lists to store image paths and corresponding annotation paths
    image_paths = [os.path.join(image_folder, filename) for filename in image_files]
    annotation_paths = [os.path.join(annotation_folder, filename) for filename in annotation_files]

    # Ensure that the number of images and annotations match
    assert len(image_paths) == len(annotation_paths), "Mismatch between images and annotations"

    # Load images and annotations
    images = [Image.open(image_path) for image_path in image_paths]
    annotations = [load_annotations(annotation_path) for annotation_path in annotation_paths]

    return images, annotations

def load_annotations(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    annotations = []

    for annotation_elem in root.findall('.//Annotation'):
        annotation = {'Id': annotation_elem.get('Id'),
                      'Type': annotation_elem.get('Type'),
                      'Vertices': []}

        for vertex_elem in annotation_elem.findall('.//Vertex'):
            vertex = {'X': float(vertex_elem.get('X')),
                      'Y': float(vertex_elem.get('Y'))}
            annotation['Vertices'].append(vertex)

        annotations.append(annotation)

    return annotations

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

"""
def preprocess_and_save_data(data_folder, save_folder, target_size=(256, 256), data_type="training"):
    # Load and preprocess data
    images, annotations = load_data(data_folder)
    resized_images = [image.resize(target_size) for image in images]

    # Create dataset
    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    dataset = CustomDataset(resized_images, annotations, transform=data_transform)
    os.makedirs(save_folder, exist_ok=True)

    # Save preprocessed data
    save_path = os.path.join(save_folder, f"{data_type}_preprocessed_data.pkl")
    with open(save_path, 'wb') as file:
        pickle.dump(dataset, file)

    print(f"Preprocessed data saved at: {save_path}")
"""
def preprocess_and_save_data(testdata_folder, traindata_folder, save_folder):
    target_size=(256, 256)
    split_ratio=(0.8, 0.1, 0.1)
    # Load and preprocess data
    test_images, test_annotations = load_data(testdata_folder)
    testresized_images = [image.resize(target_size) for image in test_images]
    training_images, training_annotations = load_data(traindata_folder)
    trainresized_images = [image.resize(target_size) for image in training_images]
    # Create dataset
    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    testdataset = CustomDataset(testresized_images, test_annotations, transform=data_transform)
    train_dataset = CustomDataset(trainresized_images, training_annotations, transform=data_transform)
    os.makedirs(save_folder, exist_ok=True)

    # Split the dataset
    total_size = len(testdataset)
    train_size = int(split_ratio[0] * total_size)
    val_size = int(split_ratio[1] * total_size)
    test_size = total_size - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(testdataset, [train_size, val_size, test_size])

    # Save preprocessed data
    save_path_train = os.path.join(save_folder, "train_preprocessed_data.pkl")
    with open(save_path_train, 'wb') as file:
        pickle.dump(train_dataset, file)

    save_path_val = os.path.join(save_folder, "val_preprocessed_data.pkl")
    with open(save_path_val, 'wb') as file:
        pickle.dump(val_dataset, file)

    save_path_test = os.path.join(save_folder, "test_preprocessed_data.pkl")
    with open(save_path_test, 'wb') as file:
        pickle.dump(test_dataset, file)

    print(f"Preprocessed data saved at: {save_path_train}, {save_path_val}, {save_path_test}")

'''
# Example usage for loading training data
training_data_folder = os.path.join("MoNuSegTrainData", "MoNuSeg 2018 Training Data")
training_images, training_annotations = load_data(training_data_folder, data_type="training")

# Example usage for loading test data
test_data_folder = "MoNuSegTestData"
test_images, test_annotations = load_data(test_data_folder, data_type="test")

target_size = (256, 256)

data_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Resize images, create a dataset
train_resized_images = [image.resize(target_size) for image in training_images]
test_resized_images = [image.resize(target_size) for image in test_images]

train_dataset = CustomDataset(train_resized_images, training_annotations, transform=data_transform)
test_dataset  = CustomDataset(test_resized_images, test_annotations, transform=data_transform)
'''

training_data_folder = os.path.join(script_dir, "MoNuSegTrainData", "MoNuSeg 2018 Training Data")
training_images, training_annotations = load_data(training_data_folder, data_type="training")

test_data_folder = os.path.join(script_dir, "MoNuSegTestData")
test_images, test_annotations = load_data(test_data_folder, data_type="test")

# Preprocess and save the data
save_folder = os.path.join(script_dir, "preprocessed_data")
preprocess_and_save_data(test_data_folder, training_data_folder, save_folder)
