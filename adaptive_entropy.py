"""Global entropy-gap estimation and upper-endpoint lambda selection."""

from dataclasses import dataclass
import math

import numpy as np
import torch


@dataclass
class EntropyWeightResult:
    lambda_value: float
    lambda_lower: float
    lambda_upper: float
    raw_lambda_lower: float = math.nan
    raw_lambda_upper: float = math.nan
    representative_gap: float = math.nan
    gap_mean: float = math.nan
    gap_median: float = math.nan
    gap_tie_rate: float = math.nan
    gap_coverage: float = math.nan
    lower_rank: int = 0
    upper_rank: int = 0
    low_group_size: int = 0
    high_group_size: int = 0
    used_fallback: bool = False


def calculate_adaptive_lambda(client_entropies, config):
    """Choose one lambda for the whole round from the saturation endpoint."""
    fallback = float(np.clip(
        config.entropy_weight_scale,
        config.adaptive_lambda_min_clip,
        config.adaptive_lambda_max_clip,
    ))
    result = EntropyWeightResult(fallback, fallback, fallback, used_fallback=True)
    if len(client_entropies) < 2:
        return result

    entropies = torch.stack([
        values.detach().float().cpu().reshape(-1) for values in client_entropies
    ])
    num_clients, num_samples = entropies.shape
    lower_rank = max(1, math.ceil(config.adaptive_client_low_quantile * num_clients))
    upper_rank = min(num_clients, math.ceil(
        config.adaptive_client_high_quantile * num_clients
    ))
    if lower_rank >= upper_rank:
        raise ValueError("The participating clients cannot form distinct entropy ranks.")
    result.lower_rank = lower_rank
    result.upper_rank = upper_rank
    result.low_group_size = lower_rank
    result.high_group_size = num_clients - upper_rank + 1
    saturation_ratio = result.high_group_size / (
        result.low_group_size * config.adaptive_high_entropy_tolerance
    )
    if config.adaptive_lambda_min_ratio > saturation_ratio:
        raise ValueError("Empty lambda interval: R_min must not exceed n_U / (n_L * xi).")
    if num_samples == 0:
        return result

    ordered = entropies.sort(dim=0).values
    gaps = (ordered[upper_rank - 1] - ordered[lower_rank - 1]).numpy()
    gap = float(np.quantile(gaps, config.adaptive_sample_gap_quantile))
    result.representative_gap = gap
    result.gap_mean = float(np.mean(gaps))
    result.gap_median = float(np.median(gaps))
    result.gap_tie_rate = float(np.mean(gaps <= config.adaptive_entropy_gap_floor))
    result.gap_coverage = float(np.mean(gaps >= gap))
    if not math.isfinite(gap) or gap <= config.adaptive_entropy_gap_floor:
        return result

    result.raw_lambda_lower = math.log(config.adaptive_lambda_min_ratio) / gap
    result.raw_lambda_upper = math.log(saturation_ratio) / gap
    result.lambda_lower = float(np.clip(
        result.raw_lambda_lower,
        config.adaptive_lambda_min_clip,
        config.adaptive_lambda_max_clip,
    ))
    result.lambda_upper = float(np.clip(
        result.raw_lambda_upper,
        config.adaptive_lambda_min_clip,
        config.adaptive_lambda_max_clip,
    ))
    result.lambda_value = result.lambda_upper
    result.used_fallback = False
    return result
