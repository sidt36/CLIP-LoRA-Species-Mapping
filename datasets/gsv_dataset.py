"""
Tree Species Dataset for Bio-CLIP Training
Following the ImageNet dataloader pattern for compatibility
"""

import os
import json
import random
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import pandas as pd
import numpy as np
from PIL import Image


# Tree species templates following imagenet_templates pattern
tree_species_templates = [
    "a specimen of a {} tree.",
    "a specimen of a {} tree species.",
    "a photo of a {} tree."
]

# Alternative: simpler template
# tree_species_templates = ["a photo of a {}."]


class TreeSpeciesDataset():
    """
    Tree Species Dataset for Bio-CLIP training
    Follows the ImageNet dataloader structure for compatibility
    """
    
    dataset_dir = 'tree_species'
    
    def __init__(self, root, num_shots, preprocess, train_preprocess=None, test_preprocess=None):
        """
        Args:
            root: Root directory containing the dataset
            num_shots: Number of examples per class for few-shot learning
            preprocess: Default preprocessing (used for val)
            train_preprocess: Training-specific preprocessing
            test_preprocess: Test-specific preprocessing
        """
        
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, 'images')
        
        # Load the integrated dataset
        self.metadata_path = os.path.join(self.dataset_dir, 'species_classification_vectors', 'metadata', 'trees_with_vectors.csv')
        self.vector_info_path = os.path.join(self.dataset_dir, 'species_classification_vectors', 'metadata', 'species_vector_mapping.json')
        
        # Load metadata
        self.df = pd.read_csv(self.metadata_path)
        with open(self.vector_info_path, 'r') as f:
            self.vector_info = json.load(f)
        
        # Get class names from vector info
        self.classnames = self.vector_info['species_order']  # Already ordered list
        self.class_to_idx = {name: idx for idx, name in enumerate(self.classnames)}
        
        # Default transforms if not provided
        if train_preprocess is None:
            train_preprocess = transforms.Compose([
                transforms.CenterCrop((400, 300)),  # Center crop for GSV images
                transforms.RandomResizedCrop(size=224, scale=(0.08, 1), interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
            ])
        
        if test_preprocess is None:
            test_preprocess = preprocess
        
        # Load data splits
        train_df = pd.read_csv(os.path.join(self.dataset_dir, 'species_classification_vectors', 'data_splits', 'train.csv'))
        val_df = pd.read_csv(os.path.join(self.dataset_dir, 'species_classification_vectors', 'data_splits', 'val.csv'))
        test_df = pd.read_csv(os.path.join(self.dataset_dir, 'species_classification_vectors', 'data_splits', 'test.csv'))
        
        # Filter for labeled data only
        train_df = train_df[train_df['has_species_label'] == True]
        val_df = val_df[val_df['has_species_label'] == True]
        test_df = test_df[test_df['has_species_label'] == True]
        
        # Create ImageFolder-compatible structure
        self.train_x = self._create_dataset(train_df, train_preprocess)
        self.val = self._create_dataset(val_df, preprocess)
        self.test = self._create_dataset(test_df, test_preprocess)
        
        # Set templates
        self.template = tree_species_templates
        
        # Apply few-shot sampling if needed
        if num_shots > 0:
            self._apply_few_shot_sampling(num_shots)
    
    def _create_dataset(self, df, transform):
        """Create a dataset structure compatible with ImageFolder"""
        dataset = SimpleDataset(transform=transform)
        
        # Build imgs and targets lists
        imgs = []
        targets = []
        
        for _, row in df.iterrows():
            img_path = os.path.join(self.dataset_dir, 'images', row['image_filename'])
            
            # Use species_name to get class index
            species_name = row['species_name']
            if species_name in self.class_to_idx:
                class_idx = self.class_to_idx[species_name]
                imgs.append((img_path, class_idx))
                targets.append(class_idx)
        
        dataset.imgs = imgs
        dataset.targets = targets
        dataset.samples = imgs  # For compatibility
        
        return dataset
    
    def _apply_few_shot_sampling(self, num_shots):
        """Apply few-shot sampling following ImageNet pattern"""
        num_shots_val = min(4, num_shots)
        
        # Group training data by label
        split_by_label_dict = defaultdict(list)
        for i in range(len(self.train_x.imgs)):
            split_by_label_dict[self.train_x.targets[i]].append(self.train_x.imgs[i])
        
        # Sample for train and val
        imgs = []
        targets = []
        imgs_val = []
        targets_val = []
        
        for label, items in split_by_label_dict.items():
            if len(items) >= num_shots + num_shots_val:
                samples = random.sample(items, num_shots + num_shots_val)
                imgs.extend(samples[0:num_shots])
                imgs_val.extend(samples[num_shots:num_shots+num_shots_val])
                targets.extend([label] * num_shots)
                targets_val.extend([label] * num_shots_val)
            elif len(items) >= num_shots:
                # If not enough for val split, use all for training
                samples = random.sample(items, num_shots)
                imgs.extend(samples)
                targets.extend([label] * num_shots)
        
        # Update train dataset
        self.train_x.imgs = imgs
        self.train_x.targets = targets
        self.train_x.samples = imgs
        
        # Update val dataset if we have samples
        if imgs_val:
            self.val.imgs = imgs_val
            self.val.targets = targets_val
            self.val.samples = imgs_val


class SimpleDataset(Dataset):
    """Simple dataset that mimics ImageFolder behavior"""
    
    def __init__(self, transform=None):
        self.transform = transform
        self.imgs = []
        self.targets = []
        self.samples = []
    
    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, idx):
        path, target = self.imgs[idx]
        
        # Load image
        img = Image.open(path).convert('RGB')
        
        # Apply transform
        if self.transform is not None:
            img = self.transform(img)
        
        return img, target
