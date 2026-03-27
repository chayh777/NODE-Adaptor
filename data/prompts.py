import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import List, Tuple, Dict

CIFAR100_CLASSES = [
    'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle', 'bottle',
    'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 'can', 'castle', 'caterpillar', 'cattle',
    'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur',
    'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 'house', 'kangaroo', 'keyboard',
    'lamp', 'lawn_mower', 'leopard', 'lion', 'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain',
    'mouse', 'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pan', 'pear', 'pickup_truck',
    'pine_tree', 'plain', 'plate', 'poppy', 'porcupine', 'possum', 'rabbit', 'raccoon', 'ray', 'road',
    'rocket', 'rose', 'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake',
    'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table', 'tank', 'telephone', 'television', 'tiger',
    'toaster', 'tractor', 'train', 'trout', 'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf',
    'woman', 'worm'
]

NUM_PROMPTS = 20

CIFAR100_PROMPTS = [
    "a photo of a {}.",
    "a picture of a {}.",
    "an image of a {}.",
    "a photo of a {} in the scene.",
    "a photograph of a {}.",
    "this is a photo of a {}.",
    "a photo showing a {}.",
    "a {} in the photograph.",
    "a picture displaying a {}.",
    "a photo of the {}.",
    "a photo of a {} object.",
    "a {} captured in photo.",
    "a photo containing a {}.",
    "an image depicting a {}.",
    "a photo featuring a {}.",
    "a photo of a {} for classification.",
    "a {} visible in the image.",
    "a photo with a {} in it.",
    "a photo that shows a {}.",
    "a {} in a photo."
]


def get_cifar100_dataset(root: str = './data', train: bool = True, transform=None):
    """Download and return CIFAR-100 dataset."""
    if transform is None:
        transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    dataset = torchvision.datasets.CIFAR100(
        root=root,
        train=train,
        download=True,
        transform=transform
    )
    return dataset


def create_prompt_templates(class_name: str) -> List[str]:
    """Create M=20 prompt templates for a given class name."""
    prompts = [prompt.format(class_name) for prompt in CIFAR100_PROMPTS]
    return prompts


class FewShotDataset(Dataset):
    """Few-shot dataset for N-way K-shot learning."""
    
    def __init__(self, dataset: Dataset, num_ways: int, num_shots: int, num_query: int = 15):
        self.dataset = dataset
        self.num_ways = num_ways
        self.num_shots = num_shots
        self.num_query = num_query
        
        self.class_to_indices = self._build_class_index()
        self.classes = list(self.class_to_indices.keys())
    
    def _build_class_index(self) -> Dict[int, List[int]]:
        class_to_indices = {}
        for idx, label in enumerate(self.dataset.targets):
            if label not in class_to_indices:
                class_to_indices[label] = []
            class_to_indices[label].append(idx)
        return class_to_indices
    
    def __len__(self):
        return self.num_ways * (self.num_shots + self.num_query)
    
    def __getitem__(self, idx):
        class_idx = idx // (self.num_shots + self.num_query)
        sample_idx = idx % (self.num_shots + self.num_query)
        
        class_label = self.classes[class_idx]
        available_indices = self.class_to_indices[class_label]
        
        if sample_idx < self.num_shots:
            img_idx = available_indices[sample_idx]
        else:
            query_idx = sample_idx - self.num_shots
            img_idx = available_indices[self.num_shots + query_idx]
        
        image, label = self.dataset[img_idx]
        return image, label, class_idx
