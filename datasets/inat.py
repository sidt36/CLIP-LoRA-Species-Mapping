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
inaturalist_templates = ["a photo of a {}."]
inaturalist_species_classes = []  # Will be populated later

class DataSample:
    def __init__(self, img, label, impath):
        self.img = img
        self.label = label
        self.impath = impath

class INaturalistDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None):
        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform
        
        # Load image data
        with open(self.root_dir / 'image_data.json', 'r') as f:
            self.image_data = json.load(f)
            
        # Load species counts if available
        try:
            with open(self.root_dir / 'species_counts.json', 'r') as f:
                self.species_counts = json.load(f)['species_counts']
        except (FileNotFoundError, KeyError):
            self.species_counts = {}
        
        # Create species list and mapping
        self.species_list = sorted(list(set(item['species'] for item in self.image_data)))
        self.species_to_idx = {species: idx for idx, species in enumerate(self.species_list)}
        
        # Update global variables
        global inaturalist_species_classes
        inaturalist_species_classes = self.species_list
        
        # Initialize containers
        self.all_samples = []
        self.train_samples = []
        self.val_samples = []
        self.test_samples = []
        
        # Load all data and create splits
        self._load_all_data()
        self._create_splits()
        
        # Set current samples based on split
        if split == 'train':
            self.samples = self.train_samples
        elif split == 'val':
            self.samples = self.val_samples
        else:  # test
            self.samples = self.test_samples
            
        # Initialize image paths and labels for compatibility with original code
        self.image_paths = []
        self.labels = []
        
        for sample in self.samples:
            self.image_paths.append(sample['image_path'])
            self.labels.append(sample['label'])

    def _load_all_data(self):
        """Load all valid images."""
        print(f"Loading data from directory: {self.root_dir}")
        
        valid_count = 0
        invalid_count = 0
        
        for item in self.image_data:
            image_path = self.root_dir / item['image_filename'].split('/')[-1]
            if image_path.exists():
                species = item['species']
                self.all_samples.append({
                    'image_path': str(image_path),
                    'species': species,
                    'label': self.species_to_idx[species],
                    'location': item.get('location', None)
                })
                valid_count += 1
            else:
                invalid_count += 1
                
        print(f"Found {valid_count} valid images and {invalid_count} invalid/missing images")

    def _create_splits(self, train_ratio=0.7, min_eval_samples=10, seed=42):
        """
        Create train/val/test splits ensuring minimum samples per class.
        Default is 70-15-15 split with minimum 10 samples per class in val and test.
        """
        # Set random seed
        random.seed(seed)
        
        # Group samples by species
        species_samples = defaultdict(list)
        for sample in self.all_samples:
            species_samples[sample['species']].append(sample)
        
        # For valid species, create splits
        for species, samples in species_samples.items():
            num_samples = len(samples)
            if num_samples >= 2 * min_eval_samples:
                random.shuffle(samples)
                
                # Calculate split sizes
                num_test = max(min_eval_samples, int(num_samples * 0.15))
                num_val = max(min_eval_samples, int(num_samples * 0.15))
                num_train = num_samples - (num_test + num_val)
                
                # Split samples
                self.test_samples.extend(samples[:num_test])
                self.val_samples.extend(samples[num_test:num_test + num_val])
                self.train_samples.extend(samples[num_test + num_val:])
                
                print(f"Species: {species}")
                print(f"  Total samples: {num_samples}")
                print(f"  Train: {num_train}, Val: {num_val}, Test: {num_test}")
                print(f"  Original count from API: {self.species_counts.get(species, 0)}")
            else:
                print(f"Warning: Species {species} has only {num_samples} samples, " 
                      f"which is less than the required {2*min_eval_samples}. Skipping.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        max_retries = 3
        
        for retry in range(max_retries):
            try:
                sample = self.samples[idx]
                image_path = sample['image_path']
                
                # Load and convert image
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
        default_image = torch.zeros((3, 224, 224))
        return DataSample(img=default_image, label=0, impath="failed_load")

class INaturalist():
    """
    Main class for iNaturalist dataset, following a similar interface to AutoArborist
    """
    def __init__(self, root, num_shots, preprocess, train_preprocess=None, test_preprocess=None):
        # Handle both string and Path objects for root
        self.root = Path(root)
        
        # For compatibility with the original code structure
        self.dataset_dir = self.root
        self.image_dir = self.root
        self.train_dir = self.root
        self.test_dir = self.root
        
        if train_preprocess is None:
            train_preprocess = transforms.Compose([
                transforms.RandomResizedCrop(size=224, scale=(0.08, 1), interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
            ])
        
        if test_preprocess is None:
            test_preprocess = preprocess

        # Initialize the custom datasets
        self.train_x = INaturalistDataset(root, split='train', transform=train_preprocess)
        self.val = INaturalistDataset(root, split='val', transform=preprocess)
        self.test = INaturalistDataset(root, split='test', transform=test_preprocess)

        # Use global variables for compatibility
        self.template = inaturalist_templates
        self.classnames = inaturalist_species_classes

        # Create few-shot train split if specified
        if num_shots > 0:
            self._create_few_shot_split(num_shots)

    def _create_few_shot_split(self, num_shots):
        """Create few-shot training set by selecting a subset of training data"""
        # Group samples by class
        split_by_label_dict = defaultdict(list)
        for i in range(len(self.train_x.labels)):
            split_by_label_dict[self.train_x.labels[i]].append(i)
        
        # Create few-shot training set
        imgs = []
        targets = []
        
        for label, indices in split_by_label_dict.items():
            if len(indices) > 0:
                # Shuffle indices for this class
                random.shuffle(indices)
                # Select samples for training
                selected = indices[:min(len(indices), num_shots)]
                
                # Collect image paths and labels
                for idx in selected:
                    imgs.append(self.train_x.image_paths[idx])
                    targets.append(self.train_x.labels[idx])
        
        # Update image paths and labels
        self.train_x.image_paths = imgs
        self.train_x.labels = targets
        
        # Update samples list
        self.train_x.samples = [
            {
                'image_path': self.train_x.image_paths[i],
                'label': self.train_x.labels[i],
                'species': self.train_x.species_list[self.train_x.labels[i]]
            } 
            for i in range(len(self.train_x.labels))
        ]
        
        # Convert target labels to tensor for compatibility
        self.targets = torch.tensor(targets, dtype=torch.long).to(device)

def get_inaturalist_dataset(
    root_dir: str,
    num_shots: int = 16,
    seed: int = 42
):
    """
    Create iNaturalist dataset
    
    Args:
        root_dir: Root directory containing the iNaturalist data
        num_shots: Number of shots (samples per class) for few-shot learning
        seed: Random seed for reproducibility
        
    Returns:
        INaturalist dataset object
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
    dataset = INaturalist(
        root=root_dir,
        num_shots=num_shots,
        preprocess=preprocess,
        train_preprocess=train_preprocess
    )
    
    return dataset

# Example usage:
if __name__ == "__main__":
    root_dir = "/app/DATASET/iNaturalist_edible_indian_trees"
    
    # Create dataset with few-shot sampling
    dataset = get_inaturalist_dataset(
        root_dir=root_dir,
        num_shots=16
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