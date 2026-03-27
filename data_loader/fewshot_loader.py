import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import List, Dict
import torchvision


class FewShotSampler:
    """Sampler for few-shot learning tasks."""
    
    def __init__(
        self, 
        dataset: Dataset, 
        num_ways: int, 
        num_shots: int,
        num_query: int = 15,
        num_tasks: int = 1000,
        replace: bool = False
    ):
        self.dataset = dataset
        self.num_ways = num_ways
        self.num_shots = num_shots
        self.num_query = num_query
        self.num_tasks = num_tasks
        self.replace = replace
        
        self.class_to_indices = self._build_class_index()
        self.classes = list(self.class_to_indices.keys())
    
    def _build_class_index(self) -> Dict[int, List[int]]:
        targets = getattr(self.dataset, 'targets', None)
        if targets is None:
            targets = [self.dataset[i][1] for i in range(len(self.dataset))]
        
        class_to_indices = {}
        for idx, label in enumerate(targets):
            if label not in class_to_indices:
                class_to_indices[label] = []
            class_to_indices[label].append(idx)
        return class_to_indices
    
    def __iter__(self):
        for _ in range(self.num_tasks):
            yield self._sample_task()
    
    def _sample_task(self) -> Dict:
        selected_classes = np.random.choice(
            self.classes, self.num_ways, replace=self.replace
        )
        
        support_images = []
        support_labels = []
        query_images = []
        query_labels = []
        
        for class_idx, class_label in enumerate(selected_classes):
            available_indices = self.class_to_indices[class_label]
            num_available = len(available_indices)
            num_needed = self.num_shots + self.num_query
            
            if num_available < num_needed:
                sampled_indices = np.random.choice(
                    available_indices, 
                    num_available, 
                    replace=False
                )
                support_idx = sampled_indices[:min(self.num_shots, len(sampled_indices))]
                query_idx = sampled_indices[min(self.num_shots, len(sampled_indices)):]
            else:
                sampled_indices = np.random.choice(
                    available_indices, 
                    num_needed, 
                    replace=False
                )
                support_idx = sampled_indices[:self.num_shots]
                query_idx = sampled_indices[self.num_shots:]
            
            class_support_images = []
            for img_idx in support_idx:
                image, _ = self.dataset[img_idx]
                class_support_images.append(image)
                support_labels.append(class_idx)
            
            if len(class_support_images) > 0:
                support_images.append(torch.stack(class_support_images))
            
            for img_idx in query_idx:
                image, _ = self.dataset[img_idx]
                query_images.append(image)
                query_labels.append(class_idx)
        
        support_images_tensor = torch.cat(support_images, dim=0) if support_images else torch.zeros(0, 3, 32, 32)
        support_images_tensor = support_images_tensor.view(self.num_ways, self.num_shots, *support_images_tensor.shape[1:])
        
        return {
            'support_images': support_images_tensor,
            'support_labels': torch.tensor(support_labels, dtype=torch.long),
            'query_images': torch.stack(query_images),
            'query_labels': torch.tensor(query_labels, dtype=torch.long),
            'selected_classes': selected_classes.tolist(),
            'num_ways': self.num_ways,
            'num_shots': self.num_shots
        }
    
    def __len__(self):
        return self.num_tasks


class FewShotDataset(Dataset):
    """Dataset that returns few-shot tasks."""
    
    def __init__(
        self, 
        dataset: Dataset, 
        num_ways: int, 
        num_shots: int,
        num_query: int = 15,
        num_tasks: int = 100,
        fix_seed: bool = False
    ):
        self.dataset = dataset
        self.num_ways = num_ways
        self.num_shots = num_shots
        self.num_query = num_query
        self.num_tasks = num_tasks
        self.fix_seed = fix_seed
        
        self.sampler = FewShotSampler(
            dataset, num_ways, num_shots, num_query, num_tasks
        )
        
        self.tasks = list(self.sampler)
    
    def __len__(self):
        return len(self.tasks)
    
    def __getitem__(self, idx):
        return self.tasks[idx]


def get_clip_transforms(image_size: int = 224) -> torchvision.transforms.Compose:
    """Get CLIP-compatible image transforms."""
    return torchvision.transforms.Compose([
        torchvision.transforms.Resize(image_size),
        torchvision.transforms.CenterCrop(image_size),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711]
        )
    ])


def create_fewshot_dataloader(
    dataset: Dataset,
    num_ways: int,
    num_shots: int,
    num_query: int = 15,
    num_tasks: int = 100,
    batch_size: int = 1,
    shuffle: bool = False,
    num_workers: int = 0
) -> DataLoader:
    """Create a DataLoader for few-shot learning."""
    fewshot_dataset = FewShotDataset(
        dataset=dataset,
        num_ways=num_ways,
        num_shots=num_shots,
        num_query=num_query,
        num_tasks=num_tasks
    )
    
    return DataLoader(
        fewshot_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fewshot_tasks
    )


def collate_fewshot_tasks(batch):
    """Collate function for few-shot tasks."""
    return batch[0]
