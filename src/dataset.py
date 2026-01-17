import os
import sys
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import random_split, Subset

class Dataset:
    """
    Handles the preparation, transformation, and loading of datasets.
    """
    def __init__(self, config, project_root):
        self.config = config
        self.project_root = project_root
        self.data_path = os.path.join(project_root, 'data')
        
        # Extract configuration
        self.dataset_config = config.get('dataset', {})
        self.dataset_name = self.dataset_config.get('name', 'CIFAR10')
        self.validation_split = self.dataset_config.get('validation_split', 0.1)
        self.subset_fraction = self.dataset_config.get('subset_fraction', 1.0)
        
        # Attributes to store Dataset objects
        self.train_dataset = None
        self.val_dataset = None

    def get_transforms(self):
        """Defines transforms based on the dataset."""
        if self.dataset_name == 'MNIST':
            # MNIST is grayscale, but models like AlexNet often expect 3 channels.
            return transforms.Compose([
                transforms.Resize((227, 227)),
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ])
        elif self.dataset_name == 'CIFAR10':
            return transforms.Compose([
                transforms.Resize((227, 227)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        else:
            raise ValueError(f"Dataset '{self.dataset_name}' not recognized.")

    def _load_raw_dataset(self, transform):
        """Downloads and loads the dataset object."""
        if self.dataset_name == 'MNIST':
            print("Using MNIST dataset.")
            return torchvision.datasets.MNIST(root=self.data_path, train=True,
                                              download=True, transform=transform)
        elif self.dataset_name == 'CIFAR10':
            print("Using CIFAR10 dataset.")
            return torchvision.datasets.CIFAR10(root=self.data_path, train=True,
                                                download=True, transform=transform)
        else:
            print(f"Error: Dataset '{self.dataset_name}' not recognized. Exiting.")
            sys.exit(1)

    def prepare_datasets(self):
        """
        Prepare data: Download, split Train/Val, and create Dataset objects (Subset).
        """
        if self.train_dataset is not None:
            return self.train_dataset, self.val_dataset

        print("Preparing datasets...")
        transform = self.get_transforms()
        full_dataset = self._load_raw_dataset(transform)

        # Calculate split sizes
        total_size = len(full_dataset)
        val_size = int(np.floor(self.validation_split * total_size))
        train_size = total_size - val_size

        # Split dataset using random_split (standard PyTorch)
        generator = torch.Generator().manual_seed(42) # Ensure reproducibility
        self.train_dataset, self.val_dataset = random_split(
            full_dataset, [train_size, val_size], generator=generator
        )

        # Handle subset creation (if config requests running on less data)
        if self.subset_fraction < 1.0:
            self.train_dataset = self._create_subset(self.train_dataset, self.subset_fraction)
            self.val_dataset = self._create_subset(self.val_dataset, self.subset_fraction)
            print(f"Using a subset of the data: {self.subset_fraction*100:.1f}%")

        print(f"Datasets prepared: {len(self.train_dataset)} training, {len(self.val_dataset)} validation.")
        return self.train_dataset, self.val_dataset

    def _create_subset(self, dataset, fraction):
        """Helper function to create a subset of the dataset."""
        subset_size = int(len(dataset) * fraction)
        indices = list(range(subset_size))
        return Subset(dataset, indices)