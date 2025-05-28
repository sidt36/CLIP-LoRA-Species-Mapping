from .oxford_pets import OxfordPets
from .eurosat import EuroSAT
from .ucf101 import UCF101
from .sun397 import SUN397
from .caltech101 import Caltech101
from .dtd import DescribableTextures
from .fgvc import FGVCAircraft
from .food101 import Food101
from .oxford_flowers import OxfordFlowers
from .stanford_cars import StanfordCars
from .imagenet import ImageNet
from .auto_arborist import AutoArborist
from .inat import INaturalist
from .google_cc import GoogleCCArborist
from .gsv_dataset import TreeSpeciesDataset  # Placeholder for GSV dataset, not implemented
dataset_list = {
                "oxford_pets": OxfordPets,
                "eurosat": EuroSAT,
                "ucf101": UCF101,
                "sun397": SUN397,
                "caltech101": Caltech101,
                "dtd": DescribableTextures,
                "fgvc": FGVCAircraft,
                "food101": Food101,
                "oxford_flowers": OxfordFlowers,
                "stanford_cars": StanfordCars,
                "imagenet": ImageNet,
                "auto_arborist":AutoArborist,
                "inat" : INaturalist,
                "google_cc": GoogleCCArborist,
                "gsv": TreeSpeciesDataset  # Placeholder for GSV dataset, not implemented
                }


def build_dataset(dataset, root_path, shots, preprocess):
    if dataset == 'imagenet' or dataset == 'auto_arborist' or dataset == 'google_cc' or dataset == 'inat': 
        return dataset_list[dataset](root_path, shots, preprocess)
    else:
        return dataset_list[dataset](root_path, shots)