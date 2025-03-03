import os
import math
import random
import json
from collections import defaultdict
from pathlib import Path
from typing import Optional, Tuple, List, Dict

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define global variables to maintain compatibility with original code
auto_arborist_templates = ["a photo of a {}."]

class DataSample:
    def __init__(self, img, label, impath):
        self.img = img
        self.label = label
        self.impath = impath

class CustomDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None, view_type='streetlevel'):
        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform
        self.view_type = view_type
        
        # Load global genus list
        with open(self.root_dir / 'unique_genus.json', 'r') as f:
            self.genus_list = json.load(f)
            
        global auto_arborist_genus_classes
        auto_arborist_genus_classes = self.genus_list
            
        self.genus_to_idx = {genus: idx for idx, genus in enumerate(self.genus_list)}
        
        # Get all city directories
        self.cities = [d for d in self.root_dir.iterdir() if d.is_dir() and not d.name.endswith('.json')]
        
        # Initialize image paths and labels for compatibility with original code
        self.samples = []
        self.image_paths = []
        self.labels = []
        
        # Load data based on split
        if split == 'test':
            self._load_test_data()
        else:
            self._load_train_data()

    def _load_train_data(self):
        """Load training data from all cities"""
        for city_dir in self.cities:
            try:
                json_path = city_dir / 'train.json'
                if not json_path.exists():
                    continue
                    
                with open(json_path, 'r') as f:
                    city_data = json.load(f)
                
                for item in city_data:
                    base_path = city_dir / 'images' / 'train'
                    street_path = str(base_path / f"{item['tree_id']}_streetlevel.png")
                    aerial_path = str(base_path / f"{item['tree_id']}_aerial.png")
                    label = self.genus_to_idx[item['genus']]
                    
                    sample = {
                        'tree_id': item['tree_id'],
                        'genus': item['genus'],
                        'streetlevel_path': street_path,
                        'aerial_path': aerial_path,
                        'label': label
                    }
                    
                    # For compatibility with original code
                    self.samples.append(sample)
                    if self.view_type == 'streetlevel':
                        self.image_paths.append(street_path)
                    elif self.view_type == 'aerial':
                        self.image_paths.append(aerial_path)
                    else:
                        self.image_paths.append(street_path)  # Default to streetlevel for both
                    
                    self.labels.append(label)
            except Exception as e:
                print(f"Error loading train data from {city_dir}: {e}")
                continue

    def _load_test_data(self):
        """Load test data from all cities"""
        for city_dir in self.cities:
            try:
                json_path = city_dir / 'test.json'
                if not json_path.exists():
                    continue
                    
                with open(json_path, 'r') as f:
                    city_data = json.load(f)
                
                for item in city_data:
                    base_path = city_dir / 'images' / 'test'
                    street_path = str(base_path / f"{item['tree_id']}_streetlevel.png")
                    aerial_path = str(base_path / f"{item['tree_id']}_aerial.png")
                    label = self.genus_to_idx[item['genus']]
                    
                    sample = {
                        'tree_id': item['tree_id'],
                        'genus': item['genus'],
                        'streetlevel_path': street_path,
                        'aerial_path': aerial_path,
                        'label': label
                    }
                    
                    # For compatibility with original code
                    self.samples.append(sample)
                    if self.view_type == 'streetlevel':
                        self.image_paths.append(street_path)
                    elif self.view_type == 'aerial':
                        self.image_paths.append(aerial_path)
                    else:
                        self.image_paths.append(street_path)  # Default to streetlevel for both
                    
                    self.labels.append(label)
            except Exception as e:
                print(f"Error loading test data from {city_dir}: {e}")
                continue

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        max_retries = 3
        
        for retry in range(max_retries):
            try:
                sample = self.samples[idx]
                
                if self.view_type == 'streetlevel':
                    image_path = sample['streetlevel_path']
                elif self.view_type == 'aerial':
                    image_path = sample['aerial_path']
                else:
                    # Handle both views if needed
                    street_img = Image.open(sample['streetlevel_path']).convert('RGB')
                    aerial_img = Image.open(sample['aerial_path']).convert('RGB')
                    
                    if self.transform:
                        street_img = self.transform(street_img)
                        aerial_img = self.transform(aerial_img)
                    
                    img = torch.cat([street_img, aerial_img], dim=0)
                    return DataSample(img=img, label=sample['label'], impath=sample['streetlevel_path'])
                
                # For single view (streetlevel or aerial)
                image = Image.open(image_path).convert('RGB')
                if self.transform:
                    image = self.transform(image)
                
                return DataSample(img=image, label=sample['label'], impath=image_path)
                
            except (FileNotFoundError, IOError) as e:
                print(f"Warning: Image not found (attempt {retry + 1}/{max_retries}): {str(e)}")
                idx = (idx + 1) % len(self)
                continue
            except Exception as e:
                print(f"Unexpected error loading image: {str(e)}")
                idx = (idx + 1) % len(self)
                continue
        
        # Default return for failed loads
        channels = 6 if self.view_type == 'both' else 3
        default_image = torch.zeros((channels, 224, 224))
        return DataSample(img=default_image, label=0, impath="failed_load")

class AutoArborist():
    dataset_dir = 'auto_arborist_processed'  # Kept for compatibility
    
    def __init__(self, root, num_shots, preprocess, train_preprocess=None, test_preprocess=None, view_type='streetlevel'):
        # Handle both string and Path objects for root
        self.root = Path(root)
        self.view_type = view_type
        
        # For compatibility with the original code structure
        self.dataset_dir = self.root
        self.image_dir = self.root
        self.train_dir = self.root
        self.test_dir = self.root
        
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
        self.train_x = CustomDataset(root, split='train', transform=train_preprocess, view_type=view_type)
        self.val = CustomDataset(root, split='train', transform=preprocess, view_type=view_type)  # Will be modified for validation
        self.test = CustomDataset(root, split='test', transform=test_preprocess, view_type=view_type)

        # Use global variables for compatibility
        self.template = auto_arborist_templates
        self.classnames = auto_arborist_genus_classes

        # Create few-shot train/val split
        self._create_few_shot_split(num_shots)

    def _create_few_shot_split(self, num_shots):
        """Create few-shot training and validation sets"""
        num_shots_val = min(4, num_shots)
        
        # Group samples by class - use labels directly for compatibility with original code
        split_by_label_dict = defaultdict(list)
        for i in range(len(self.train_x.labels)):
            split_by_label_dict[self.train_x.labels[i]].append(i)
        
        # Create few-shot training and validation sets
        imgs = []
        targets = []
        imgs_val = []
        targets_val = []
        
        for label, indices in split_by_label_dict.items():
            if len(indices) > 0:
                # Shuffle indices for this class
                random.shuffle(indices)
                # Select samples for training
                selected_train = indices[:min(len(indices), num_shots)]
                # Select samples for validation (different from training)
                selected_val = indices[num_shots:min(len(indices), num_shots + num_shots_val)]
                
                # Collect image paths for training
                for idx in selected_train:
                    imgs.append(self.train_x.image_paths[idx])
                    targets.append(self.train_x.labels[idx])
                
                # Collect image paths for validation
                for idx in selected_val:
                    imgs_val.append(self.train_x.image_paths[idx])
                    targets_val.append(self.train_x.labels[idx])
        
        # Update image paths and labels for compatibility with original code
        self.train_x.image_paths = imgs
        self.train_x.labels = targets
        
        self.val.image_paths = imgs_val
        self.val.labels = targets_val
        
        # Update samples list for our new implementation
        self.train_x.samples = [
            {'label': self.train_x.labels[i], 'streetlevel_path': self.train_x.image_paths[i]} 
            for i in range(len(self.train_x.labels))
        ]
        
        self.val.samples = [
            {'label': self.val.labels[i], 'streetlevel_path': self.val.image_paths[i]} 
            for i in range(len(self.val.labels))
        ]
        
        # Convert target labels to tensor for compatibility
        self.targets = torch.tensor(targets, dtype=torch.long).to(device)

def get_auto_arborist_dataset(
    root_dir: str,
    num_shots: int = 16,
    view_type: str = 'streetlevel',
    seed: int = 42
):
    """
    Create Auto Arborist dataset with few-shot sampling
    
    Args:
        root_dir: Root directory containing the Auto Arborist data
        num_shots: Number of shots (samples per class)
        view_type: Type of views to use ('streetlevel', 'aerial', or 'both')
        seed: Random seed for reproducibility
        
    Returns:
        AutoArborist dataset object
    """
    random.seed(seed)
    
    # Define transforms
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    train_preprocess = transforms.Compose([
        transforms.RandomResizedCrop(size=224, scale=(0.08, 1), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create dataset
    dataset = AutoArborist(
        root=root_dir,
        num_shots=num_shots,
        preprocess=preprocess,
        train_preprocess=train_preprocess,
        view_type=view_type
    )
    
    return dataset

# Example usage:
if __name__ == "__main__":
    root_dir = "/app/DATASET/auto_arborist_processed"
    
    # Create dataset with few-shot sampling
    dataset = get_auto_arborist_dataset(
        root_dir=root_dir,
        num_shots=16,
        view_type='streetlevel'
    )
    
    # Print dataset information
    print(f"Number of classes: {len(dataset.classnames)}")
    print(f"Training samples: {len(dataset.train_x)}")
    print(f"Validation samples: {len(dataset.val)}")
    print(f"Test samples: {len(dataset.test)}")
    
    # Create dataloaders if needed
    train_loader = DataLoader(dataset.train_x, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(dataset.val, batch_size=32, shuffle=False, num_workers=4)
    test_loader = DataLoader(dataset.test, batch_size=32, shuffle=False, num_workers=4)