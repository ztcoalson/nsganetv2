import torch
import torchvision.datasets as dset

from abc import ABC, abstractmethod

DATASET_CLASSES = {
    'cifar100': dset.CIFAR100,
    'mnist': dset.MNIST,
    'fashion_mnist': dset.FashionMNIST,
    'svhn': lambda root, split, **kwargs: dset.SVHN(root=root, split=split, **kwargs),
    'cifar10': dset.CIFAR10
}

# ------------------------------------------------------------------------------
#   Dataset wrapper for managing the attacks in our paper
# ------------------------------------------------------------------------------
class PoisonWrapper(ABC):
    """
    Abstract class for wrapping clean datasets with poison loaders.
    """
    def __init__(self, dataset, train_kwargs, transform):
        self.clean_dataset = DATASET_CLASSES[dataset](**train_kwargs)
        self.transform = transform

    @abstractmethod
    def __getitem__(self, idx):
        pass

    def get_num_poisons(self):
        return len(self.indices)
    
    def __len__(self):
        return len(self.clean_dataset)
    

class DirtyLabelPoisonedDataset(PoisonWrapper):
    """
    Dataset used for dirty label attacks.
    (e.g., Random and Confidence-Based Label Flipping)
    """
    def __init__(self, dataset, poisons_path, transform, train_kwargs):
        super().__init__(dataset, train_kwargs, transform)

        if poisons_path is None:
            raise ValueError("poisons_path is None")

        try:
            poisons = torch.load(poisons_path)
            self.indices = poisons["indices"]
            self.poisoned_labels = poisons["poisoned_labels"]
        except:
            raise ValueError(
                "Failed to load poisons from path: {}\nEnsure format is correct." 
                " For dirty label poisons, poisons dictionary must contain 'indices' and 'poisoned_labels' keys.".format(poisons_path)
            )
    
    def __getitem__(self, idx):
        image, label = self.clean_dataset[idx]

        # flip label if idx is in poisons
        if idx in self.indices:
            label = self.poisoned_labels[self.indices.index(idx)]

        # apply transforms
        return self.transform(image), label


class CleanLabelPoisonedDataset(PoisonWrapper):
    """
    Dataset used for clean label attacks.
    (e.g., Gaussian Noise and Gradient Canceling)
    """
    def __init__(self, dataset, poisons_path, transform, train_kwargs):
        super().__init__(dataset, train_kwargs, transform)

        if poisons_path is None:
            raise ValueError("poisons_path is None")

        try:
            poisons = torch.load(poisons_path)
            self.indices = poisons["indices"]
            self.poisoned_images = poisons["poisoned_images"]
        except:
            raise ValueError(
                "Failed to load poisons from path: {}\nEnsure format is correct." 
                " For clean label poisons, poisons dictionary must contain 'indices' and 'poisoned_images' keys.".format(poisons_path)
            )
    
    def __getitem__(self, idx):
        image, label = self.clean_dataset[idx]

        # replace image with poisoned image if idx is in poisons
        if idx in self.indices:
            image = self.poisoned_images[self.indices.index(idx)]
        
        # apply transforms
        return self.transform(image), label