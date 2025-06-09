"""
SkySat Tree Species Dataset for Bio-CLIP Training
Following the ImageNet dataloader pattern for compatibility
Loads SkySat satellite imagery crops instead of GSV images
"""

import os
import json
import random
from collections import defaultdict
from pathlib import Path
import logging

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import pandas as pd
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# SkySat tree species templates - adapted for satellite imagery context
skysat_tree_species_templates = [
    "a satellite view of a {} tree.",
    "an aerial view of a {} tree species.",
    "a satellite image of a {} tree."
]


class SkySatTreeSpeciesDataset():
    """
    SkySat Tree Species Dataset for Bio-CLIP training
    Follows the ImageNet dataloader structure for compatibility
    """
    
    dataset_dir = 'tree_species'
    
    def __init__(self, root, num_shots, preprocess, train_preprocess=None, test_preprocess=None, 
                 enlargement_factor=2.0, crop_format='jpg'):
        """
        Args:
            root: Root directory containing the dataset
            num_shots: Number of examples per class for few-shot learning
            preprocess: Default preprocessing (used for val)
            train_preprocess: Training-specific preprocessing
            test_preprocess: Test-specific preprocessing
            enlargement_factor: The enlargement factor used when extracting crops
            crop_format: Format of the saved crops ('jpg' or 'png')
        """
        
        self.enlargement_factor = enlargement_factor
        self.crop_format = crop_format
        
        # Find the dataset directory - handle different possible structures
        possible_paths = [
            os.path.join(root, self.dataset_dir),
            os.path.join(root, 'species_classification_vectors'),
            root  # In case root is already the dataset directory
        ]
        
        self.dataset_dir = None
        for path in possible_paths:
            if os.path.exists(path):
                # Check if this path contains the expected structure
                if os.path.exists(os.path.join(path, 'species_classification_vectors')):
                    self.dataset_dir = path
                    break
                elif os.path.exists(os.path.join(path, 'metadata', 'trees_with_vectors.csv')):
                    self.dataset_dir = os.path.dirname(path)
                    break
        
        if self.dataset_dir is None:
            raise ValueError(f"Could not find dataset directory in {root}")
        
        logger.info(f"Using dataset directory: {self.dataset_dir}")
        
        # Find SkySat crops directory
        possible_skysat_dirs = [
            os.path.join(self.dataset_dir, 'skysat_crops'),
            os.path.join(os.path.dirname(self.dataset_dir), 'skysat_crops'),
            os.path.join(self.dataset_dir, '..', 'skysat_crops')
        ]
        
        self.skysat_dir = None
        for skysat_dir in possible_skysat_dirs:
            if os.path.exists(skysat_dir):
                self.skysat_dir = os.path.abspath(skysat_dir)
                break
        
        if self.skysat_dir is None:
            raise ValueError(f"Could not find skysat_crops directory")
        
        logger.info(f"Using SkySat directory: {self.skysat_dir}")
        
        # Load the integrated dataset
        metadata_base = os.path.join(self.dataset_dir, 'species_classification_vectors')
        if not os.path.exists(metadata_base):
            metadata_base = self.dataset_dir
        
        self.metadata_path = os.path.join(metadata_base, 'metadata', 'trees_with_vectors.csv')
        self.vector_info_path = os.path.join(metadata_base, 'metadata', 'species_vector_mapping.json')
        
        if not os.path.exists(self.metadata_path):
            raise FileNotFoundError(f"Metadata file not found: {self.metadata_path}")
        
        if not os.path.exists(self.vector_info_path):
            raise FileNotFoundError(f"Vector info file not found: {self.vector_info_path}")
        
        # Load metadata
        self.df = pd.read_csv(self.metadata_path)
        with open(self.vector_info_path, 'r') as f:
            self.vector_info = json.load(f)
        
        # Get class names from vector info
        self.classnames = self.vector_info['species_order']  # Already ordered list
        self.class_to_idx = {name: idx for idx, name in enumerate(self.classnames)}
        
        # Default transforms for SkySat imagery (no center crop needed)
        if train_preprocess is None:
            train_preprocess = transforms.Compose([
                transforms.RandomResizedCrop(size=224, scale=(0.8, 1.0), interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),  # Added for satellite imagery
                transforms.RandomRotation(degrees=45),  # Added for rotation invariance
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
            ])
        
        if test_preprocess is None:
            test_preprocess = preprocess
        
        # Load data splits
        splits_dir = os.path.join(metadata_base, 'data_splits')
        train_df = pd.read_csv(os.path.join(splits_dir, 'train.csv'))
        val_df = pd.read_csv(os.path.join(splits_dir, 'val.csv'))
        test_df = pd.read_csv(os.path.join(splits_dir, 'test.csv'))
        
        # Filter for labeled data AND SkySat availability
        train_df = train_df[(train_df['has_species_label'] == True) & (train_df.get('has_skysat', True) == True)].copy()
        val_df = val_df[(val_df['has_species_label'] == True) & (val_df.get('has_skysat', True) == True)].copy()
        test_df = test_df[(test_df['has_species_label'] == True) & (test_df.get('has_skysat', True) == True)].copy()
        
        logger.info(f"SkySat Dataset sizes - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
        
        # Create ImageFolder-compatible structure
        self.train_x = self._create_dataset(train_df, train_preprocess)
        self.val = self._create_dataset(val_df, preprocess)
        self.test = self._create_dataset(test_df, test_preprocess)
        
        # Set templates
        self.template = skysat_tree_species_templates
        
        # Apply few-shot sampling if needed
        if num_shots > 0:
            self._apply_few_shot_sampling(num_shots)
    
    def _get_skysat_filename(self, gsv_filename):
        """Convert GSV filename to SkySat filename"""
        base_name = os.path.splitext(gsv_filename)[0]
        return f"{base_name}_skysat.{self.crop_format}"
    
    def _create_dataset(self, df, transform):
        """Create a dataset structure compatible with ImageFolder"""
        dataset = SimpleDataset(transform=transform)
        
        # Build imgs and targets lists
        imgs = []
        targets = []
        data = []  # List of Datum objects
        missing_files = 0
        
        for _, row in df.iterrows():
            # Get SkySat filename from GSV filename
            skysat_filename = self._get_skysat_filename(row['image_filename'])
            
            # Try multiple possible paths for the SkySat image
            possible_paths = [
                os.path.join(self.skysat_dir, skysat_filename),
                os.path.join(self.dataset_dir, 'skysat_crops', skysat_filename),
            ]
            
            # Also check if there's a specific skysat path in the metadata
            if 'skysat_path' in row and pd.notna(row['skysat_path']):
                possible_paths.insert(0, row['skysat_path'])
            
            img_path = None
            for path in possible_paths:
                if path and os.path.exists(path):
                    img_path = path
                    break
            
            if img_path is None:
                missing_files += 1
                logger.debug(f"SkySat image not found: {skysat_filename}")
                continue
            
            # Use species_name to get class index
            species_name = row['species_name']
            if species_name in self.class_to_idx:
                class_idx = self.class_to_idx[species_name]
                imgs.append((img_path, class_idx))
                targets.append(class_idx)
                
                # Create Datum object for Bio-CLIP compatibility
                datum = Datum(
                    impath=img_path,
                    label=class_idx,
                    classname=species_name
                )
                data.append(datum)
            else:
                logger.warning(f"Species '{species_name}' not found in class index")
        
        if missing_files > 0:
            logger.warning(f"Could not find {missing_files} SkySat image files")
        
        dataset.imgs = imgs
        dataset.targets = targets
        dataset.samples = imgs  # For compatibility
        dataset.data = data  # Bio-CLIP compatibility
        
        logger.info(f"Created SkySat dataset with {len(imgs)} samples")
        
        return dataset
    
    def _apply_few_shot_sampling(self, num_shots):
        """Apply few-shot sampling following ImageNet pattern"""
        num_shots_val = min(4, num_shots)
        
        # Group training data by label
        split_by_label_dict = defaultdict(list)
        data_by_label_dict = defaultdict(list)
        
        for i in range(len(self.train_x.imgs)):
            label = self.train_x.targets[i]
            split_by_label_dict[label].append(self.train_x.imgs[i])
            data_by_label_dict[label].append(self.train_x.data[i])
        
        # Log class distribution
        class_counts = {label: len(items) for label, items in split_by_label_dict.items()}
        logger.info(f"SkySat class distribution before sampling: {class_counts}")
        
        # Sample for train and val
        imgs = []
        targets = []
        data = []
        imgs_val = []
        targets_val = []
        data_val = []
        
        for label, items in split_by_label_dict.items():
            datum_items = data_by_label_dict[label]
            
            if len(items) >= num_shots + num_shots_val:
                # Random sample indices
                indices = list(range(len(items)))
                sampled_indices = random.sample(indices, num_shots + num_shots_val)
                
                # Split into train and val
                train_indices = sampled_indices[0:num_shots]
                val_indices = sampled_indices[num_shots:num_shots+num_shots_val]
                
                # Add to train
                for idx in train_indices:
                    imgs.append(items[idx])
                    targets.append(label)
                    data.append(datum_items[idx])
                
                # Add to val
                for idx in val_indices:
                    imgs_val.append(items[idx])
                    targets_val.append(label)
                    data_val.append(datum_items[idx])
                    
            elif len(items) >= num_shots:
                # If not enough for val split, use all for training
                indices = list(range(len(items)))
                sampled_indices = random.sample(indices, num_shots)
                
                for idx in sampled_indices:
                    imgs.append(items[idx])
                    targets.append(label)
                    data.append(datum_items[idx])
            else:
                # Use all available samples if less than num_shots
                for idx in range(len(items)):
                    imgs.append(items[idx])
                    targets.append(label)
                    data.append(datum_items[idx])
                logger.warning(f"Class {label} has only {len(items)} SkySat samples, less than requested {num_shots}")
        
        # Update train dataset
        self.train_x.imgs = imgs
        self.train_x.targets = targets
        self.train_x.samples = imgs
        self.train_x.data = data
        
        logger.info(f"SkySat few-shot sampling: {len(imgs)} training samples")
        
        # Update val dataset if we have samples
        if imgs_val:
            self.val.imgs = imgs_val
            self.val.targets = targets_val
            self.val.samples = imgs_val
            self.val.data = data_val
            logger.info(f"SkySat few-shot sampling: {len(imgs_val)} validation samples")


class Datum:
    """Data instance for Bio-CLIP compatibility"""
    def __init__(self, impath, label, classname):
        self.impath = impath
        self.label = label
        self.classname = classname


class SimpleDataset(Dataset):
    """Simple dataset that mimics ImageFolder behavior with Bio-CLIP compatibility"""
    
    def __init__(self, transform=None):
        self.transform = transform
        self.imgs = []
        self.targets = []
        self.samples = []
        self.data = []  # List of Datum objects for compatibility
    
    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, idx):
        datum = self.data[idx]
        path = datum.impath
        target = datum.label

        try:
            img = Image.open(path).convert('RGB')
            if self.transform is not None:
                img = self.transform(img)
        except Exception as e:
            logger.error(f"Error loading SkySat image {path}: {e}")
            if self.transform is not None:
                # Create a dummy image with typical SkySat crop size
                dummy_img = Image.new('RGB', (224, 224), color='black')
                img = self.transform(dummy_img)
            else:
                img = torch.zeros(3, 224, 224)

        # Always return a Datum object
        return datum


# Utility function to create the dataset
def get_skysat_tree_species_dataset(root, num_shots=16, preprocess=None):
    """
    Utility function to create SkySat tree species dataset
    
    Args:
        root: Root directory containing the dataset
        num_shots: Number of examples per class for few-shot learning
        preprocess: Preprocessing pipeline
    
    Returns:
        SkySatTreeSpeciesDataset instance
    """
    if preprocess is None:
        # Default CLIP preprocessing for SkySat imagery
        preprocess = transforms.Compose([
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), 
                               std=(0.26862954, 0.26130258, 0.27577711))
        ])
    
    return SkySatTreeSpeciesDataset(
        root=root,
        num_shots=num_shots,
        preprocess=preprocess
    )


if __name__ == "__main__":
    # Test the dataloader
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, required=True, help='Dataset root directory')
    parser.add_argument('--num_shots', type=int, default=16, help='Number of shots per class')
    args = parser.parse_args()
    
    # Create dataset
    dataset = get_skysat_tree_species_dataset(args.root, args.num_shots)
    
    # Print dataset info
    print(f"Number of classes: {len(dataset.classnames)}")
    print(f"Classes: {dataset.classnames[:10]}...")  # First 10 classes
    print(f"Train samples: {len(dataset.train_x)}")
    print(f"Val samples: {len(dataset.val)}")
    print(f"Test samples: {len(dataset.test)}")
    
    # Test loading a sample
    if len(dataset.train_x) > 0:
        sample = dataset.train_x[0]
        if hasattr(sample, 'impath'):
            print(f"Sample image path: {sample.impath}")
            print(f"Sample label: {sample.label} ({sample.classname})")
        else:
            print(f"Sample loaded successfully")