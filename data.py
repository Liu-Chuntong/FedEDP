"""Dataset loading with a common, deterministic public input order."""

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset, TensorDataset
from torchvision import datasets, transforms

from data_partition import dirichlet_partition, split_public_private


def seed_worker(worker_id):
    import random

    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_loader(dataset, config, shuffle=False, batch_size=None):
    return DataLoader(
        dataset,
        batch_size=batch_size or config.batch_size,
        shuffle=shuffle,
        num_workers=config.num_workers,
        pin_memory=str(config.device).startswith("cuda"),
        worker_init_fn=seed_worker,
    )


def load_datasets(config):
    if config.smoke_test:
        generator = torch.Generator().manual_seed(config.seed)
        shape = (config.input_channels, config.image_size, config.image_size)
        train = TensorDataset(
            torch.randn(config.smoke_train_size, *shape, generator=generator),
            torch.arange(config.smoke_train_size) % config.num_classes,
        )
        test = TensorDataset(
            torch.randn(config.smoke_test_size, *shape, generator=generator),
            torch.arange(config.smoke_test_size) % config.num_classes,
        )
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(config.normalize_mean, config.normalize_std),
        ])
        kwargs = {
            "root": config.data_dir,
            "download": config.download,
            "transform": transform,
        }
        if config.dataset == "fashionmnist":
            train = datasets.FashionMNIST(train=True, **kwargs)
            test = datasets.FashionMNIST(train=False, **kwargs)
        elif config.dataset == "emnist":
            train = datasets.EMNIST(split="byclass", train=True, **kwargs)
            test = datasets.EMNIST(split="byclass", train=False, **kwargs)
            if config.emnist_train_size:
                if config.emnist_train_size > len(train):
                    raise ValueError("The EMNIST training subset exceeds the dataset size.")
                indices = np.random.RandomState(config.seed).choice(
                    len(train), config.emnist_train_size, replace=False
                )
                train = Subset(train, indices.tolist())
        elif config.dataset == "svhn":
            train = datasets.SVHN(split="train", **kwargs)
            test = datasets.SVHN(split="test", **kwargs)
        elif config.dataset == "cifar10":
            train = datasets.CIFAR10(train=True, **kwargs)
            test = datasets.CIFAR10(train=False, **kwargs)
        else:
            raise ValueError(f"Unknown dataset: {config.dataset}")

    private, public = split_public_private(train, config.public_ratio, config.seed)
    clients = dirichlet_partition(
        private, config.num_clients, config.num_classes,
        config.dirichlet_alpha, config.seed, config.partition_max_attempts,
    )
    return clients, public, test


class DistillationDataset(Dataset):
    """Keep teacher targets aligned while allowing shuffled training batches."""

    def __init__(self, public_dataset, soft_labels, hard_labels):
        if len(public_dataset) != len(soft_labels) or len(soft_labels) != len(hard_labels):
            raise ValueError("Public inputs and teacher targets must have identical lengths.")
        self.public_dataset = public_dataset
        self.soft_labels = soft_labels
        self.hard_labels = hard_labels

    def __len__(self):
        return len(self.public_dataset)

    def __getitem__(self, index):
        image, _ = self.public_dataset[index]
        return image, self.soft_labels[index], self.hard_labels[index]
