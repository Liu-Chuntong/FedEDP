

import math

import torch


def aggregate_soft_labels(client_soft_labels, client_entropies, lambda_value, temperature):
    """Apply sample-wise client weights using one global lambda."""
    if not client_soft_labels or len(client_soft_labels) != len(client_entropies):
        raise ValueError("Each participating client must supply predictions and entropy.")
    if not math.isfinite(lambda_value) or lambda_value < 0 or temperature <= 0:
        raise ValueError("Invalid global lambda or temperature.")
    probabilities = torch.stack(client_soft_labels)
    entropies = torch.stack(client_entropies)
    if probabilities.ndim != 3 or entropies.shape != probabilities.shape[:2]:
        raise ValueError("Expected predictions [clients, samples, classes] and aligned entropy.")
    if not torch.isfinite(probabilities).all() or not torch.isfinite(entropies).all():
        raise ValueError("Non-finite client predictions or entropies.")
    weights = torch.softmax(-lambda_value * entropies, dim=0)
    aggregated = (weights.unsqueeze(-1) * probabilities).sum(dim=0)
    # Preserve the second server-side softmax in the legacy implementation.
    global_probabilities = torch.softmax(aggregated / temperature, dim=1)
    return global_probabilities, global_probabilities.argmax(dim=1)


def tensor_bytes(tensor):
    return tensor.numel() * tensor.element_size()


def communication_bytes(predictions, entropies, global_soft, global_hard, recipients):
    upload = sum(tensor_bytes(values) for values in predictions + entropies)
    download = recipients * (tensor_bytes(global_soft) + tensor_bytes(global_hard))
    return upload, download
