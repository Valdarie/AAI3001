import os
from PIL import Image
import xml.etree.ElementTree as ET
import cv2
import numpy as np
from torchvision import transforms
from sklearn.model_selection import train_test_split
import json

class ImageMasker:
    def __init__(self, image_paths, annotation_paths, mask_dir):
        self.image_paths = image_paths
        self.annotation_paths = annotation_paths
        self.mask_dir = mask_dir
        self.transform = transforms.ToTensor()

    def create_masks(self):
        # Ensure the mask directory exists
        if not os.path.exists(self.mask_dir):
            os.makedirs(self.mask_dir)

        for img_path, annotation_path in zip(self.image_paths, self.annotation_paths):
            mask_name = os.path.basename(img_path).replace('.tif', '.png')
            mask_path = os.path.join(self.mask_dir, mask_name)

            if not os.path.exists(mask_path):
                self._create_mask(annotation_path, Image.open(img_path).size, mask_path)

    def _create_mask(self, xml_path, img_shape, mask_path):
        mask = self._xml_to_mask(xml_path, img_shape)
        mask_image = Image.fromarray(mask)
        mask_image.save(mask_path)

    def _xml_to_mask(self, xml_file, img_shape):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        mask = np.zeros(img_shape[:2], dtype=np.uint8)  # Assuming img_shape is in (H, W) format

        for region in root.iter('Region'):
            polygon = []
            for vertex in region.iter('Vertex'):
                x = int(float(vertex.get('X')))
                y = int(float(vertex.get('Y')))
                polygon.append((x, y))

            np_polygon = np.array([polygon], dtype=np.int32)
            cv2.fillPoly(mask, np_polygon, 255)  # Fill polygon with 255

        return mask

class DatasetSplitter:
    def __init__(self, train_paths, val_paths, test_paths):
        self.train_set = set(train_paths)
        self.val_set = set(val_paths)
        self.test_set = set(test_paths)

    def _check_disjoint(self, *sets):
        """
        Checks if the provided sets are disjoint.

        Args:
        *sets: Variable number of sets to check for disjointedness.

        Returns:
        bool: True if all sets are disjoint, False otherwise.
        """
        for i, set_i in enumerate(sets):
            for j, set_j in enumerate(sets):
                if i != j and not set_i.isdisjoint(set_j):
                    return False  # Sets are not disjoint
        return True  # All sets are disjoint

    def verify_disjointedness(self):
        """
        Verifies disjointedness among the dataset splits.

        Returns:
        bool: True if train, validation, and test sets are disjoint, False otherwise.
        """
        return self._check_disjoint(self.train_set, self.val_set, self.test_set)

# DATA DIRECTORIES
base_dir = 'monusegdata'
train_annotation_dir = f'{base_dir}/trainData/annotations'
train_image_dir = f'{base_dir}/trainData/tissueimages'
test_data_dir = f'{base_dir}/testData/'

train_image_paths = [os.path.join(train_image_dir, f) for f in os.listdir(train_image_dir) if f.endswith('.tif')]
train_annotation_paths = [os.path.join(train_annotation_dir, f.replace('.tif', '.xml')) for f in os.listdir(train_image_dir) if f.endswith('.tif')]

test_image_paths = [os.path.join(test_data_dir, f) for f in os.listdir(test_data_dir) if f.endswith('.tif')]
test_annotation_paths = [os.path.join(test_data_dir, f.replace('.tif', '.xml')) for f in os.listdir(test_data_dir) if f.endswith('.tif')]

# DATA SPLITTING
train_image_paths, val_image_paths, train_annotation_paths, val_annotation_paths = train_test_split(
    train_image_paths, train_annotation_paths, test_size=0.2, random_state=42)

# VERIFY DISJOINTEDNESS
splitter = DatasetSplitter(train_image_paths, val_image_paths, test_image_paths)
if splitter.verify_disjointedness():
    print("The train, validation, and test sets are disjoint.")
else:
    print("There is an overlap between the sets. They are not disjoint.")

# IMAGE MASKING
train_mask_dir = f'{base_dir}/trainMask'
val_mask_dir = f'{base_dir}/valMask'
test_mask_dir = f'{base_dir}/testMask'

train_masker = ImageMasker(train_image_paths, train_annotation_paths, train_mask_dir)
train_masker.create_masks()

val_masker = ImageMasker(val_image_paths, val_annotation_paths, val_mask_dir)
val_masker.create_masks()

test_masker = ImageMasker(test_image_paths, test_annotation_paths, test_mask_dir)
test_masker.create_masks()

def save_paths_to_json(file_path, paths):
    with open(file_path, 'w') as f:
        json.dump(paths, f)

save_paths_to_json('train_images.json', train_image_paths)
save_paths_to_json('val_images.json', val_image_paths)
save_paths_to_json('test_images.json', test_image_paths)

