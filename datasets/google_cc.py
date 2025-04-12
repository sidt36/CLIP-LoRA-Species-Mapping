import os
import json
from typing import Optional, Tuple, List, Dict
from pathlib import Path
import random
from collections import defaultdict

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import gc

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define global variables to maintain compatibility with Auto Arborist
google_cc_templates = ["a photo of a {}."]
google_cc_species_classes = []  # Will be populated later

class DataSample:
    def __init__(self, img, label, impath):
        self.img = img
        self.label = label
        self.impath = impath

class GoogleCCTreeDataset(Dataset):
    """
    Dataset class for the Google Creative Commons tree dataset.
    """
    def __init__(
        self,
        root_dir: str,
        split: str = 'train',
        transform: Optional[transforms.Compose] = None,
        train_ratio: float = 0.7,
        min_eval_samples_per_class: int = 1,
        seed: int = 42
    ):
        """
        Args:
            root_dir: Root directory containing the classification_results.json and images
            split: One of 'train', 'val', or 'test'
            transform: Optional transform to be applied to images
            train_ratio: Ratio of data to use for training (0.7 for 70-15-15 split)
            min_eval_samples_per_class: Minimum number of samples per class in val/test sets
            seed: Random seed for reproducibility
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform
        self.train_ratio = train_ratio
        self.min_eval_samples = min_eval_samples_per_class
        
        # Load classification results
        with open(self.root_dir / 'classification_results.json', 'r') as f:
            self.data = json.load(f)
        
        # Extract filtered data
        self.filtered_data = self.data['filtered_data']
        
        # Create species list and mapping
        self.species_list = sorted(list(self.filtered_data.keys()))
        self.species_to_idx = {species: idx for idx, species in enumerate(self.species_list)}
        
        # Update global variables
        global google_cc_species_classes
        google_cc_species_classes = self.species_list
        
        # Load all data
        self.all_samples = []
        self._load_all_data()
        
        # Set random seed
        random.seed(seed)
        
        # Create splits
        self.train_samples, self.val_samples, self.test_samples = self._create_splits()
        
        # Assign appropriate split
        if split == 'train':
            self.samples = self.train_samples
        elif split == 'val':
            self.samples = self.val_samples
        else:  # test
            self.samples = self.test_samples
            
        # Initialize image paths and labels for compatibility with Auto Arborist
        self.image_paths = []
        self.labels = []
        
        for sample in self.samples:
            self.image_paths.append(str(sample['image_path']))
            self.labels.append(self.species_to_idx[sample['species']])

        # Pre-check image paths to remove invalid images
        self._verify_images()

    def _verify_images(self):
        """Verify all images exist and are valid before training starts"""
        valid_samples = []
        valid_paths = []
        valid_labels = []
        
        for i, sample in enumerate(self.samples):
            image_path = sample['image_path']
            if os.path.exists(image_path):
                try:
                    # Just try opening the image to verify it's valid
                    with Image.open(image_path) as img:
                        pass
                    valid_samples.append(sample)
                    valid_paths.append(self.image_paths[i])
                    valid_labels.append(self.labels[i])
                except Exception as e:
                    print(f"Removing invalid image: {image_path}, Error: {str(e)}")
            else:
                print(f"Removing non-existent image: {image_path}")
        
        self.samples = valid_samples
        self.image_paths = valid_paths
        self.labels = valid_labels
        print(f"Verified {len(self.samples)} valid images for {self.split} split")

    def _load_all_data(self):
        """Load all filtered images data."""
        for species, images in self.filtered_data.items():
            for img_data in images:
                image_path = self.root_dir / 'filtered_trees' / img_data['image_filename'].split('/')[-1]
                self.all_samples.append({
                    'image_path': image_path,
                    'species': species,
                    'confidence': img_data['confidence'],
                    'source_url': img_data['source_url'],
                    'license': img_data['license'],
                    'attribution': img_data['attribution']
                })

    def _create_splits(self) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """
        Create train/val/test splits ensuring minimum samples per class.
        70-15-15 split with minimum samples per class in val and test.
        """
        # Group samples by species
        species_samples = defaultdict(list)
        for sample in self.all_samples:
            species_samples[sample['species']].append(sample)

        # For compatibility with Auto Arborist
        self.genus_list = sorted(self.species_list)
        self.genus_to_idx = {genus: idx for idx, genus in enumerate(self.genus_list)}
        
        train_samples = []
        val_samples = []
        test_samples = []
        
        # For each species
        for species, samples in species_samples.items():
            num_samples = len(samples)
            random.shuffle(samples)
            
            # Ensure minimum samples for val and test
            min_required = 2 * self.min_eval_samples  # For both val and test
            if num_samples < min_required:
                print(f"Warning: Species {species} has only {num_samples} samples, "
                      f"which is less than minimum required {min_required}")
                # Split evenly between val and test, no train samples
                mid = num_samples // 2
                val_samples.extend(samples[:mid])
                test_samples.extend(samples[mid:])
                continue
            
            # Calculate split sizes
            num_test = max(self.min_eval_samples, int(num_samples * 1))
            num_val = max(self.min_eval_samples, int(num_samples * 0))
            
            # Adjust train size to maintain minimum eval samples
            num_train = num_samples - (num_test + num_val)
            
            # Split samples
            test_samples.extend(samples[:num_test])
            val_samples.extend(samples[num_test:num_test + num_val])
            train_samples.extend(samples[num_test + num_val:])
            
            # Print split info for this species
            print(f"Species: {species}")
            print(f"  Total samples: {num_samples}")
            print(f"  Train: {len(samples[num_test + num_val:])}")
            print(f"  Val: {len(samples[num_test:num_test + num_val])}")
            print(f"  Test: {len(samples[:num_test])}")
            
        return train_samples, val_samples, test_samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int):
        max_retries = 3
        retried_indices = set()
        original_idx = idx
        
        for retry in range(max_retries):
            if len(retried_indices) >= min(len(self), 10):  # Prevent too many retries
                break
                
            try:
                sample = self.samples[idx]
                image_path = sample['image_path']
                
                # Check if file exists
                if not os.path.exists(image_path):
                    raise FileNotFoundError(f"Image not found: {image_path}")
                    
                # Load and convert image
                image = Image.open(image_path).convert('RGB')
                if self.transform:
                    image = self.transform(image)
                
                # Get label
                label = self.species_to_idx[sample['species']]
                
                # Create DataSample object but return tuple as expected by DataLoader
                data_sample = DataSample(img=image, label=label, impath=str(image_path))
                
                # Return the sample in format expected by evaluation function
                # This matches what DataLoader unpacks as (images, target)
                return data_sample
                
            except (FileNotFoundError, IOError) as e:
                print(f"Warning: Image not found (attempt {retry + 1}/{max_retries}): {str(e)}")
                retried_indices.add(idx)
                # Try a different random index instead of sequential next
                new_indices = [i for i in range(len(self)) if i not in retried_indices]
                if not new_indices:
                    break
                idx = random.choice(new_indices)
                continue
            except Exception as e:
                print(f"Unexpected error loading image {image_path}: {str(e)}")
                retried_indices.add(idx)
                # Try a different random index
                new_indices = [i for i in range(len(self)) if i not in retried_indices]
                if not new_indices:
                    break
                idx = random.choice(new_indices)
                continue
        
        # Default return for failed loads - using a small tensor to minimize memory usage
        print(f"Failed to load image after {max_retries} attempts, using default zero tensor")
        default_image = torch.zeros((3, 224, 224), dtype=torch.float32)
        default_label = 0
        return default_image, default_label

class GoogleCCArborist():
    """
    Wrapper class for Google CC tree dataset that follows Auto Arborist interface.
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
        self.train_x = GoogleCCTreeDataset(root, split='train', transform=train_preprocess)
        self.val = GoogleCCTreeDataset(root, split='val', transform=preprocess)
        self.test = GoogleCCTreeDataset(root, split='test', transform=test_preprocess)

        # Use global variables for compatibility with Auto Arborist
        self.template = google_cc_templates
        self.classnames = google_cc_species_classes

        # Create few-shot train split if specified
        # Skip few-shot sampling if num_shots is 0 or negative (use full dataset)
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
        new_samples = []
        for i, (img_path, label) in enumerate(zip(imgs, targets)):
            species = self.train_x.species_list[label]
            # Find original sample with this path to get all metadata
            for sample in self.train_x.all_samples:
                if str(sample['image_path']) == img_path:
                    new_samples.append(sample)
                    break
        
        self.train_x.samples = new_samples
        
        # Convert target labels to tensor for compatibility
        self.targets = torch.tensor(targets, dtype=torch.long).to(device)

def get_google_cc_arborist_dataset(
    root_dir: str,
    num_shots: int = 16,
    use_full_dataset: bool = False,
    seed: int = 42
):
    """
    Create Google CC dataset with Auto Arborist-compatible interface
    
    Args:
        root_dir: Root directory containing the Google CC data
        num_shots: Number of shots (samples per class) for few-shot learning
        use_full_dataset: If True, use full dataset instead of few-shot subset
        seed: Random seed for reproducibility
        
    Returns:
        GoogleCCArborist dataset object
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
    dataset = GoogleCCArborist(
        root=root_dir,
        num_shots=0 if use_full_dataset else num_shots,  # Use 0 to skip few-shot sampling
        preprocess=preprocess,
        train_preprocess=train_preprocess
    )
    
    return dataset

# Example usage:
if __name__ == "__main__":
    root_dir = "/app/DATASET/google_creative_commons_edible_indian_trees"
    
    # Create dataset with few-shot sampling
    dataset = get_google_cc_arborist_dataset(
        root_dir=root_dir,
        num_shots=16,
        # use_full_dataset=True  # Uncomment to use full dataset
    )
    
    # Print dataset information
    print(f"Number of classes: {len(dataset.classnames)}")
    print(f"Training samples: {len(dataset.train_x)}")
    print(f"Validation samples: {len(dataset.val)}")
    print(f"Test samples: {len(dataset.test)}")
    
    # Memory cleanup before creating dataloaders
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Create dataloaders with reduced number of workers for validation and test
    train_loader = DataLoader(dataset.train_x, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(dataset.val, batch_size=16, shuffle=False, num_workers=1)  # Reduced workers and batch size
    test_loader = DataLoader(dataset.test, batch_size=16, shuffle=False, num_workers=1)  # Reduced workers and batch size

