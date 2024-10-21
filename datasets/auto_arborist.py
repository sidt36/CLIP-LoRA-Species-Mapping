import os
import json
import random
from PIL import Image
from torch.utils.data import Dataset
from .utils import DatasetBase  
from copy import deepcopy


template = ['a photo of a {}.']

class SplitDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        image = Image.open(sample['streetlevel_path'])
        im = deepcopy(image)
        image.close()

        label = sample['genus']

        return sample, im, label

class AutoArboristStreetviewDataset(DatasetBase):
    dataset_dir = 'AutoArborist'

    def __init__(self, root):
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.data = []

        # Load data from the directory
        self.load_data(self.dataset_dir)

        self.template = template

        # Split data into train, val, test
        train_data, val_data, test_data = self.split_data(self.data)

        self.train = SplitDataset(train_data)
        self.val = SplitDataset(val_data)
        self.test = SplitDataset(test_data)

        super().__init__(train_x=self.train, val=self.val, test=self.test)

    def load_data(self, directory):
        for subdir, _, files in os.walk(directory):
            for file in files:
                if file.endswith('.json'):
                    json_path = os.path.join(subdir, file)
                    with open(json_path, 'r') as f:
                        json_data = json.load(f)
                        self.data.extend(json_data)  

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        raise NotImplementedError("We don't use this aT ALLL")

    def split_data(self, data):
        random.shuffle(data)  
        total = len(data)

        train_end = int(0.8 * total)
        val_end = train_end + int(0.1 * total)

        train_data = data[:train_end]
        val_data = data[train_end:val_end]
        test_data = data[val_end:]

        return train_data, val_data, test_data
