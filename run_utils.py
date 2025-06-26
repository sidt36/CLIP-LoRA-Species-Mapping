import random
import argparse  
import numpy as np 
import torch
import copy
from collections import defaultdict

from lora import run_lora

    

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_arguments():

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=1, type=int)
    # Dataset arguments
    parser.add_argument('--root_path', type=str, default='')
    parser.add_argument('--dataset', type=str, default='gsv')
    parser.add_argument('--shots', default=16, type=int)
    # Model arguments
    #Important: The backbone should be the same as the one used to train the LoRA modules
    parser.add_argument('--backbone', default='ViT-B/16', type=str)
    # Training arguments
    parser.add_argument('--lr', default=2e-4, type=float)
    parser.add_argument('--n_iters', default=500, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--subsample_ratio', default=None, type=float, help='fraction of training data to use (e.g., 0.1 for 10%)')
    # LoRA arguments
    parser.add_argument('--position', type=str, default='all', choices=['bottom', 'mid', 'up', 'half-up', 'half-bottom', 'all', 'top3'], help='where to put the LoRA modules')
    parser.add_argument('--encoder', type=str, choices=['text', 'vision', 'both'], default='both')
    parser.add_argument('--params', metavar='N', type=str, nargs='+', default=['q', 'k', 'v'], help='list of attention matrices where putting a LoRA') 
    parser.add_argument('--r', default=2, type=int, help='the rank of the low-rank matrices')
    parser.add_argument('--alpha', default=1, type=int, help='scaling (see LoRA paper)')
    parser.add_argument('--dropout_rate', default=0.25, type=float, help='dropout rate applied before the LoRA module')
    # Where the LoRA modules will be saved
    parser.add_argument('--save_path', default=None, help='path to save the lora modules after training, not saved if None')
    #Where the LoRA modules will be loaded from for evaluation
    parser.add_argument('--filename', default='lora_weights', help='file name to save the lora weights (.pt extension will be added)')
    #Only evaluate the LoRA modules
    parser.add_argument('--eval_only', default=False, action='store_true', help='only evaluate the LoRA modules (save_path should not be None)')
    args = parser.parse_args()

    return args
    

def subsample_training_data(dataset, subsample_ratio, seed=42):
    """
    Subsample training data while maintaining class distribution.
    Ensures that smaller subsets are nested within larger ones (e.g., 10% ⊆ 20% ⊆ 30%).
    
    Args:
        dataset: Dataset object with train_x attribute
        subsample_ratio: Float between 0 and 1, fraction of data to keep
        seed: Random seed for reproducible subsampling
        
    Returns:
        Subsampled dataset
    """
    if subsample_ratio is None or subsample_ratio >= 1.0:
        return dataset
    
    # Set seed for reproducible subsampling
    np.random.seed(seed)
    random.seed(seed)
    
    # Group samples by class
    class_to_indices = defaultdict(list)
    for idx, sample in enumerate(dataset.train_x):
        # The sample is a Datum object with a label property
        label = sample.label
        class_to_indices[label].append(idx)
    
    # Subsample each class maintaining the same ratio
    subsampled_indices = []
    for class_label, indices in class_to_indices.items():
        # Sort indices to ensure consistent ordering for nesting
        indices.sort()
        
        # Calculate number of samples to keep for this class
        num_samples = len(indices)
        num_to_keep = max(1, int(num_samples * subsample_ratio))  # Keep at least 1 sample per class
        
        # Take the first num_to_keep samples (ensures nesting property)
        selected_indices = indices[:num_to_keep]
        subsampled_indices.extend(selected_indices)
    
    # Create new training data list
    original_train_x = dataset.train_x
    subsampled_train_x = [original_train_x[i] for i in sorted(subsampled_indices)]
    
    # Create a new dataset object with subsampled training data
    # We need to create a copy and modify its train_x
    import copy
    subsampled_dataset = copy.deepcopy(dataset)
    subsampled_dataset._train_x = subsampled_train_x
    
    # Update the number of classes and class mappings in case some classes were removed
    subsampled_dataset._num_classes = subsampled_dataset.get_num_classes(subsampled_train_x)
    subsampled_dataset._lab2cname, subsampled_dataset._classnames = subsampled_dataset.get_lab2cname(subsampled_train_x)
    
    print(f"Subsampled training data: {len(original_train_x)} -> {len(subsampled_train_x)} samples ({subsample_ratio:.1%})")
    
    # Print class distribution info
    class_counts = defaultdict(int)
    for sample in subsampled_train_x:
        class_counts[sample.label] += 1
    
    print(f"Classes in subsampled data: {len(class_counts)}")
    print(f"Samples per class range: {min(class_counts.values())} - {max(class_counts.values())}")
    
    return subsampled_dataset


# Sample run function with eval on

#