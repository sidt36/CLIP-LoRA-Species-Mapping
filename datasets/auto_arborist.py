import os
import math
import random
from collections import defaultdict

import torch
import torchvision
import json
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, Dataset
from PIL import Image


class CustomDataset(Dataset):
    def __init__(self, image_dir, metadata_path, transform=None):
        self.image_dir = image_dir
        self.transform = transform

        # Load metadata
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)

        # Extract image paths and labels
        self.image_paths = [os.path.join(self.image_dir, item['tree_id'] + '_streetlevel.png') for item in self.metadata]
        self.labels = [item['genus'] for item in self.metadata]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        # Load image
        image = Image.open(img_path).convert('RGB')

        # Apply transformations
        if self.transform:
            image = self.transform(image)

        return image, label

# Path to the JSON file

PROCESSED_DATA_DIR = r'C:\Users\sidt3\Documents\research_tree\SpeciesMapping\data\auto_arborist_processed'
unique_meta_data_path = os.path.join(PROCESSED_DATA_DIR, 'unique_genus.json')

# Read the JSON file
with open(unique_meta_data_path, 'r') as f:
    meta_data_unique = json.load(f)

auto_arborist_genus_classes = meta_data_unique
auto_arborist_templates = ["a photo of a {}."]

class AutoArborist():
    dataset_dir = 'auto_arborist_processed'

    def __init__(self, root, num_shots, preprocess, train_preprocess=None, test_preprocess=None):
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, 'images')

        if train_preprocess is None:
            train_preprocess = transforms.Compose([
                transforms.RandomResizedCrop(size=224, scale=(0.08, 1), interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
            ])
        
        if test_preprocess is None:
            test_preprocess = preprocess

        # Initialize the custom datasets
        train_metadata_path = os.path.join(self.dataset_dir, 'train.json')
        test_metadata_path = os.path.join(self.dataset_dir, 'test.json')

        self.train_x = CustomDataset(self.image_dir, train_metadata_path, transform=train_preprocess)
        self.val = CustomDataset(self.image_dir, train_metadata_path, transform=preprocess)
        self.test = CustomDataset(self.image_dir, test_metadata_path, transform=test_preprocess)

        num_shots_val = min(4, num_shots)
        
        self.template = auto_arborist_templates
        self.classnames = auto_arborist_genus_classes

        # Multi-shot logic
        split_by_label_dict = defaultdict(list)
        for i in range(len(self.train_x.image_paths)):
            split_by_label_dict[self.train_x.labels[i]].append(self.train_x.image_paths[i])
        
        imgs = []
        targets = []
        imgs_val = []
        targets_val = []
        for label, items in split_by_label_dict.items():
            samples = random.sample(items, min(len(items), num_shots + num_shots_val))
            imgs.extend(samples[:num_shots])
            imgs_val.extend(samples[num_shots:num_shots + num_shots_val])
            targets.extend([label] * min(len(items), num_shots))
            targets_val.extend([label] * min(len(items), num_shots_val))
            
        self.train_x.image_paths = imgs
        self.train_x.labels = targets
        
        self.val.image_paths = imgs_val
        self.val.labels = targets_val

    def get_dataloaders(self, batch_size, shuffle=True):
        train_loader = DataLoader(self.train_x, batch_size=batch_size, shuffle=shuffle)
        val_loader = DataLoader(self.val, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(self.test, batch_size=batch_size, shuffle=False)
        return train_loader, val_loader, test_loader