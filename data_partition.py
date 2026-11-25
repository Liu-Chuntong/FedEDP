"""Seeded public/private splits and the legacy Dirichlet allocation."""

import numpy as np
import torch
from torch.utils.data import ConcatDataset, Subset, TensorDataset


def get_targets(dataset):
    """Resolve labels without loading images, including nested subsets."""
    if isinstance(dataset, Subset):
        return get_targets(dataset.dataset)[np.asarray(dataset.indices, dtype=np.int64)]
    if isinstance(dataset, ConcatDataset):
        return np.concatenate([get_targets(part) for part in dataset.datasets])
    if isinstance(dataset, TensorDataset):
        return dataset.tensors[1].cpu().numpy().astype(np.int64)
    for attribute in ("targets", "labels"):
        if hasattr(dataset, attribute):
            return np.asarray(getattr(dataset, attribute), dtype=np.int64)
    return np.asarray([int(label) for _, label in dataset], dtype=np.int64)


def split_public_private(dataset, public_ratio, seed):
    public_size = int(len(dataset) * public_ratio)
    if not 0 < public_size < len(dataset):
        raise ValueError("The public and private datasets must both be nonempty.")
    indices = torch.randperm(
        len(dataset), generator=torch.Generator().manual_seed(seed)
    ).tolist()
    return Subset(dataset, indices[public_size:]), Subset(dataset, indices[:public_size])


def dirichlet_partition(dataset, num_clients, num_classes, alpha, seed, max_attempts):
    """Draw client-wise class preferences, then allocate each class across clients.

    This intentionally preserves the legacy sampling axis. It is not the
    alternative procedure that draws one Dirichlet vector per class.
    """
    if alpha <= 0 or num_clients < 1 or len(dataset) < num_clients:
        raise ValueError("Invalid alpha, client count, or insufficient private samples.")
    targets = get_targets(dataset)
    if targets.size == 0 or targets.min() < 0 or targets.max() >= num_classes:
        raise ValueError("Dataset labels are outside the configured class range.")
    rng = np.random.RandomState(seed)
    class_indices = [np.flatnonzero(targets == c) for c in range(num_classes)]
    for _ in range(max_attempts):
        preferences = rng.dirichlet(np.full(num_classes, alpha), size=num_clients)
        allocations = [[] for _ in range(num_clients)]
        invalid_draw = False
        for class_id, indices in enumerate(class_indices):
            if len(indices) == 0:
                continue
            weights = preferences[:, class_id]
            if not np.isfinite(weights.sum()) or weights.sum() == 0:
                invalid_draw = True
                break
            counts = rng.multinomial(len(indices), weights / weights.sum())
            offset = 0
            for client_id, count in enumerate(counts):
                allocations[client_id].extend(indices[offset:offset + count].tolist())
                offset += count
        if not invalid_draw and all(allocations):
            return [Subset(dataset, indices) for indices in allocations]
    raise RuntimeError(
        f"No nonempty client partition after {max_attempts} attempts. "
        "Increase alpha, the dataset size, or the retry limit."
    )
