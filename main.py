
import argparse
import copy
from dataclasses import asdict, dataclass, replace
from datetime import datetime
import json
import math
import os
from pathlib import Path
import random

import numpy as np
import torch

from adaptive_entropy import EntropyWeightResult, calculate_adaptive_lambda
from client import FederatedClient, current_learning_rate, legacy_noise_std
from data import load_datasets, make_loader
from data_partition import get_targets
from models import ClientModel
from results import ResultWriter, write_seed_summary
from server import aggregate_soft_labels, communication_bytes


# Change these settings here, or use the command-line overrides shown in README.md.
DATASET = "fashionmnist"
SEED = 42
RECOMMENDED_SEEDS = (42, 43, 44, 45, 46)
NUM_CLIENTS = 10
CLIENT_FRACTION = 1.0
NUM_ROUNDS = 50
LOCAL_EPOCHS = 3
BATCH_SIZE = 64
PUBLIC_BATCH_SIZE = 64
TEST_BATCH_SIZE = 64
NUM_WORKERS = 0
CPU_THREADS = 0
DEVICE = "auto"
DATA_DIR = str(Path(__file__).resolve().parent / "data")
RESULTS_DIR = str(Path(__file__).resolve().parent / "results")
DOWNLOAD = True
SAVE_MODELS = False
PARTITION_MAX_ATTEMPTS = 100
EMNIST_TRAIN_SIZE = 150000

TEMPERATURE = 2.0
HARD_LABEL_WEIGHT = 0.8
EPSILON = 1.5
DELTA = 1e-5
LIPSCHITZ_CONSTANT = 1.0

ADAPTIVE_LAMBDA_ENABLED = True
ADAPTIVE_LAMBDA_MIN_RATIO = 2.0
ADAPTIVE_HIGH_ENTROPY_TOLERANCE = 0.10
ADAPTIVE_CLIENT_LOW_QUANTILE = 0.25
ADAPTIVE_CLIENT_HIGH_QUANTILE = 0.75
ADAPTIVE_SAMPLE_GAP_QUANTILE = 0.10
ADAPTIVE_ENTROPY_GAP_FLOOR = 1e-6
ADAPTIVE_LAMBDA_MIN_CLIP = 0.0
ADAPTIVE_LAMBDA_MAX_CLIP = 20.0

MOMENTUM = 0.9
WEIGHT_DECAY = 0.0
ADAM_BETAS = (0.9, 0.999)
ADAM_EPSILON = 1e-8
DROPOUT_INITIAL = 0.2
DROPOUT_MIN = 0.1
DROPOUT_MAX = 0.5
DROPOUT_WINDOW = 3
DROPOUT_STEP = 0.05
DROPOUT_LOSS_TOLERANCE = 0.05
SMOKE_TRAIN_SIZE = 160
SMOKE_TEST_SIZE = 16

# Dataset-specific values follow the legacy scripts, with unified Adam,
# learning-rate schedule, noise calibration, and global adaptive lambda.
# entropy_weight_scale is used only for fixed-lambda runs or unresolved gaps.
DATASET_DEFAULTS = {
    "fashionmnist": dict(
        num_classes=10, input_channels=1, image_size=28, public_ratio=0.10,
        dirichlet_alpha=0.1, optimizer="adam", initial_lr=0.0001,
        min_lr=1e-5, lr_decay_rate=0.95, delta_theta=0.1,
        entropy_weight_scale=0.0, dynamic_dropout=True,
        conv_dropout=0.4, classifier_dropout=0.5,
        distillation_loss="cross_entropy", shared_initialization=False,
        normalize_mean=(0.2860,), normalize_std=(0.3530,),
    ),
    "emnist": dict(
        num_classes=62, input_channels=1, image_size=28, public_ratio=0.05,
        dirichlet_alpha=0.1, optimizer="adam", initial_lr=0.0001,
        min_lr=1e-5, lr_decay_rate=0.95, delta_theta=0.1,
        entropy_weight_scale=0.0, dynamic_dropout=True,
        conv_dropout=0.4, classifier_dropout=0.5,
        distillation_loss="cross_entropy", shared_initialization=False,
        normalize_mean=(0.1751,), normalize_std=(0.3332,),
    ),
    "svhn": dict(
        num_classes=10, input_channels=3, image_size=32, public_ratio=0.10,
        dirichlet_alpha=0.03, optimizer="adam", initial_lr=0.0001,
        min_lr=1e-5, lr_decay_rate=0.95, delta_theta=0.1,
        entropy_weight_scale=2.5, dynamic_dropout=True,
        conv_dropout=0.4, classifier_dropout=0.5,
        distillation_loss="cross_entropy", shared_initialization=False,
        normalize_mean=(0.4377, 0.4438, 0.4728),
        normalize_std=(0.1980, 0.2010, 0.1970),
    ),
    "cifar10": dict(
        num_classes=10, input_channels=3, image_size=32, public_ratio=0.10,
        dirichlet_alpha=1.0, optimizer="adam", initial_lr=0.0001,
        min_lr=1e-5, lr_decay_rate=0.95, delta_theta=0.1,
        entropy_weight_scale=4.5, dynamic_dropout=False,
        conv_dropout=0.25, classifier_dropout=0.3,
        distillation_loss="kl", shared_initialization=True,
        normalize_mean=(0.4914, 0.4822, 0.4465),
        normalize_std=(0.2470, 0.2435, 0.2616),
    ),
}


@dataclass(frozen=True)
class ExperimentConfig:
    dataset: str
    num_classes: int
    input_channels: int
    image_size: int
    public_ratio: float
    dirichlet_alpha: float
    optimizer: str
    initial_lr: float
    min_lr: float
    lr_decay_rate: float
    delta_theta: float
    entropy_weight_scale: float
    dynamic_dropout: bool
    conv_dropout: float
    classifier_dropout: float
    distillation_loss: str
    shared_initialization: bool
    normalize_mean: tuple
    normalize_std: tuple
    seed: int = SEED
    num_clients: int = NUM_CLIENTS
    client_fraction: float = CLIENT_FRACTION
    num_rounds: int = NUM_ROUNDS
    local_epochs: int = LOCAL_EPOCHS
    batch_size: int = BATCH_SIZE
    public_batch_size: int = PUBLIC_BATCH_SIZE
    test_batch_size: int = TEST_BATCH_SIZE
    num_workers: int = NUM_WORKERS
    cpu_threads: int = CPU_THREADS
    device: str = DEVICE
    data_dir: str = DATA_DIR
    results_dir: str = RESULTS_DIR
    download: bool = DOWNLOAD
    save_models: bool = SAVE_MODELS
    partition_max_attempts: int = PARTITION_MAX_ATTEMPTS
    emnist_train_size: int = EMNIST_TRAIN_SIZE
    temperature: float = TEMPERATURE
    hard_label_weight: float = HARD_LABEL_WEIGHT
    epsilon: float = EPSILON
    delta: float = DELTA
    lipschitz_constant: float = LIPSCHITZ_CONSTANT
    adaptive_lambda_enabled: bool = ADAPTIVE_LAMBDA_ENABLED
    adaptive_lambda_min_ratio: float = ADAPTIVE_LAMBDA_MIN_RATIO
    adaptive_high_entropy_tolerance: float = ADAPTIVE_HIGH_ENTROPY_TOLERANCE
    adaptive_client_low_quantile: float = ADAPTIVE_CLIENT_LOW_QUANTILE
    adaptive_client_high_quantile: float = ADAPTIVE_CLIENT_HIGH_QUANTILE
    adaptive_sample_gap_quantile: float = ADAPTIVE_SAMPLE_GAP_QUANTILE
    adaptive_entropy_gap_floor: float = ADAPTIVE_ENTROPY_GAP_FLOOR
    adaptive_lambda_min_clip: float = ADAPTIVE_LAMBDA_MIN_CLIP
    adaptive_lambda_max_clip: float = ADAPTIVE_LAMBDA_MAX_CLIP
    momentum: float = MOMENTUM
    weight_decay: float = WEIGHT_DECAY
    adam_betas: tuple = ADAM_BETAS
    adam_epsilon: float = ADAM_EPSILON
    dropout_initial: float = DROPOUT_INITIAL
    dropout_min: float = DROPOUT_MIN
    dropout_max: float = DROPOUT_MAX
    dropout_window: int = DROPOUT_WINDOW
    dropout_step: float = DROPOUT_STEP
    dropout_loss_tolerance: float = DROPOUT_LOSS_TOLERANCE
    smoke_test: bool = False
    smoke_train_size: int = SMOKE_TRAIN_SIZE
    smoke_test_size: int = SMOKE_TEST_SIZE


def build_config(dataset=DATASET, **overrides):
    return replace(ExperimentConfig(dataset=dataset, **DATASET_DEFAULTS[dataset]), **overrides)


def validate_config(config):
    for name, value in asdict(config).items():
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f"{name} must be finite.")
    positive = (
        "num_clients", "num_rounds", "local_epochs", "batch_size",
        "public_batch_size", "test_batch_size", "partition_max_attempts",
        "temperature", "epsilon", "lipschitz_constant", "delta_theta",
        "dirichlet_alpha", "initial_lr", "min_lr", "lr_decay_rate",
        "adaptive_entropy_gap_floor", "dropout_window", "adam_epsilon",
    )
    for name in positive:
        if getattr(config, name) <= 0:
            raise ValueError(f"{name} must be positive.")
    if not 0 < config.delta < 1 or not 0 < config.public_ratio < 1:
        raise ValueError("delta and public_ratio must lie strictly between zero and one.")
    if not 0 < config.client_fraction <= 1 or not 0 <= config.hard_label_weight <= 1:
        raise ValueError("Invalid client fraction or hard-label loss weight.")
    if config.num_workers < 0 or config.cpu_threads < 0 or config.emnist_train_size < 0:
        raise ValueError("Worker counts, thread counts, and subset sizes must be nonnegative.")
    if not 0 <= config.seed < 2 ** 32:
        raise ValueError("seed must lie in [0, 2**32).")
    if config.optimizer not in ("sgd", "adam"):
        raise ValueError("optimizer must be sgd or adam.")
    if config.distillation_loss not in ("kl", "cross_entropy"):
        raise ValueError("Unknown distillation loss.")
    if config.entropy_weight_scale < 0:
        raise ValueError("The fixed or fallback entropy weight must be nonnegative.")
    if not 0 < config.adaptive_client_low_quantile < config.adaptive_client_high_quantile <= 1:
        raise ValueError("Entropy quantiles must satisfy 0 < q_low < q_high <= 1.")
    if not 0 <= config.adaptive_sample_gap_quantile <= 1:
        raise ValueError("The sample gap quantile must lie in [0, 1].")
    if config.adaptive_lambda_min_ratio <= 1:
        raise ValueError("The minimum pairwise weight ratio must exceed one.")
    if not 0 < config.adaptive_high_entropy_tolerance < 1:
        raise ValueError("The high-entropy tolerance must lie in (0, 1).")
    if not 0 <= config.adaptive_lambda_min_clip <= config.adaptive_lambda_max_clip:
        raise ValueError("Invalid lambda clipping interval.")
    if config.adaptive_lambda_enabled:
        participants = max(1, int(config.num_clients * config.client_fraction))
        calculate_adaptive_lambda([torch.zeros(1) for _ in range(participants)], config)


def set_random_seed(seed):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def run_experiment(config):
    validate_config(config)
    device = config.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable.")
    config = replace(config, device=device)
    set_random_seed(config.seed)
    if config.cpu_threads:
        torch.set_num_threads(config.cpu_threads)
    client_datasets, public_dataset, test_dataset = load_datasets(config)
    public_loader = make_loader(
        public_dataset, config, batch_size=config.public_batch_size
    )
    test_loader = make_loader(test_dataset, config, batch_size=config.test_batch_size)
    public_targets = torch.as_tensor(get_targets(public_dataset))
    shared_state = (
        copy.deepcopy(ClientModel(config).state_dict())
        if config.shared_initialization else None
    )
    clients = []
    for client_id, dataset in enumerate(client_datasets):
        model = ClientModel(config)
        if shared_state is not None:
            model.load_state_dict(shared_state)
        clients.append(FederatedClient(client_id, model, dataset, config))

    writer = ResultWriter(config, client_datasets, len(public_dataset), len(test_dataset))
    selection_rng = np.random.RandomState(config.seed)
    participants = max(1, int(config.num_clients * config.client_fraction))
    total_upload, total_download = 0, 0
    sigma = legacy_noise_std(config)
    print(
        f"Dataset={config.dataset}, seed={config.seed}, alpha={config.dirichlet_alpha}, "
        f"device={device}, sigma={sigma:.8f}, public_samples={len(public_dataset)}"
    )
    print(f"Results: {writer.directory}")
    for round_number in range(1, config.num_rounds + 1):
        selected = selection_rng.choice(config.num_clients, participants, replace=False)
        predictions, entropies, client_rows = [], [], {}
        for client_id in selected:
            client = clients[client_id]
            private_loss = client.train_private(round_number)
            probabilities, entropy = client.predict_public(public_loader)
            predictions.append(probabilities)
            entropies.append(entropy)
            pre_accuracy, _ = client.evaluate(test_loader)
            client_rows[int(client_id)] = dict(
                round=round_number, client_id=int(client_id),
                private_loss=private_loss, pre_accuracy_pct=pre_accuracy, sigma=sigma,
            )
        if config.adaptive_lambda_enabled:
            weight = calculate_adaptive_lambda(entropies, config)
        else:
            value = config.entropy_weight_scale
            weight = EntropyWeightResult(value, value, value)
        global_soft, global_hard = aggregate_soft_labels(
            predictions, entropies, weight.lambda_value, config.temperature
        )
        upload, download = communication_bytes(
            predictions, entropies, global_soft, global_hard, participants
        )
        total_upload += upload
        total_download += download
        for client_id in selected:
            client_rows[int(client_id)]["distillation_loss"] = clients[client_id].distill(
                public_dataset, global_soft, global_hard, round_number
            )
        # Evaluate all clients so the reported mean has a fixed population.
        accuracies, losses = [], []
        for client_id, client in enumerate(clients):
            accuracy, loss = client.evaluate(test_loader)
            accuracies.append(accuracy)
            losses.append(loss)
            row = client_rows.get(client_id, dict(
                round=round_number, client_id=client_id, sigma=sigma
            ))
            row.update(
                participated=client_id in client_rows,
                final_accuracy_pct=accuracy, test_loss=loss,
                private_samples=len(client.private_dataset),
            )
            writer.client_rows.append(row)
        writer.adaptive_rows.append(dict(round=round_number, **asdict(weight)))
        writer.round_rows.append(dict(
            round=round_number,
            mean_accuracy_pct=float(np.mean(accuracies)),
            mean_test_loss=float(np.mean(losses)),
            public_prediction_accuracy_pct=float(
                (global_hard == public_targets).float().mean() * 100
            ),
            lambda_value=weight.lambda_value, sigma=sigma,
            learning_rate=current_learning_rate(config, round_number),
            participants=participants, upload_bytes=upload, download_bytes=download,
            total_bytes=upload + download,
            cumulative_upload_bytes=total_upload,
            cumulative_download_bytes=total_download,
            cumulative_total_bytes=total_upload + total_download,
            upload_mib=upload / 2 ** 20, download_mib=download / 2 ** 20,
            cumulative_total_mib=(total_upload + total_download) / 2 ** 20,
        ))
        writer.save_progress()
        print(
            f"Round {round_number:02d}/{config.num_rounds}: "
            f"accuracy={np.mean(accuracies):.2f}%, lambda={weight.lambda_value:.4f}, "
            f"gap={weight.representative_gap:.6f}, "
            f"communication={(total_upload + total_download) / 2**20:.4f} MiB"
        )
    writer.finish(clients)
    return writer.directory, writer.round_rows[-1]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=DATASET_DEFAULTS, default=DATASET)
    seeds = parser.add_mutually_exclusive_group()
    seeds.add_argument("--seed", type=int)
    seeds.add_argument(
        "--seeds", type=int, nargs="+",
        help="Suggested seeds: " + ", ".join(map(str, RECOMMENDED_SEEDS)),
    )
    overrides = (
        ("alpha", "dirichlet_alpha", float), ("lr", "initial_lr", float),
        ("min-lr", "min_lr", float), ("lr-decay", "lr_decay_rate", float),
        ("clients", "num_clients", int), ("client-fraction", "client_fraction", float),
        ("rounds", "num_rounds", int), ("local-epochs", "local_epochs", int),
        ("batch-size", "batch_size", int), ("num-workers", "num_workers", int),
        ("cpu-threads", "cpu_threads", int), ("temperature", "temperature", float),
        ("epsilon", "epsilon", float), ("delta", "delta", float),
        ("delta-theta", "delta_theta", float),
        ("fixed-lambda", "entropy_weight_scale", float),
        ("emnist-train-size", "emnist_train_size", int),
        ("device", "device", str), ("data-dir", "data_dir", str),
        ("results-dir", "results_dir", str),
    )
    for option, destination, value_type in overrides:
        parser.add_argument(f"--{option}", dest=destination, type=value_type)
    parser.add_argument("--optimizer", choices=("sgd", "adam"))
    parser.add_argument("--adaptive-lambda", dest="adaptive_lambda_enabled",
                        action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--download", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--save-models", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--smoke-test", action="store_true",
                        help="Run one short synthetic round without downloading datasets.")
    parser.add_argument("--print-config", action="store_true",
                        help="Show resolved hyperparameters without loading data or training.")
    return parser.parse_args()


def main():
    args = parse_args()
    excluded = {"dataset", "seed", "seeds", "print_config"}
    overrides = {
        name: value for name, value in vars(args).items()
        if name not in excluded and value is not None
    }
    if args.smoke_test:
        overrides.update(num_clients=2, num_rounds=1, local_epochs=1,
                         batch_size=8, public_batch_size=8, test_batch_size=8,
                         cpu_threads=2, num_workers=0)
    seeds = args.seeds if args.seeds is not None else [
        args.seed if args.seed is not None else SEED
    ]
    if len(seeds) != len(set(seeds)):
        raise ValueError("Repeated seeds would not provide independent runs.")
    summaries = []
    for seed in seeds:
        config = build_config(args.dataset, seed=seed, **overrides)
        validate_config(config)
        if args.print_config:
            print(json.dumps(asdict(config), indent=2))
            continue
        directory, last_round = run_experiment(config)
        summaries.append(dict(
            seed=seed, smoke_test=config.smoke_test,
            run_directory=str(directory), **last_round
        ))
    if len(summaries) > 1:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        tag = "smoke_" if config.smoke_test else ""
        prefix = Path(config.results_dir) / f"{tag}{config.dataset}_seeds_{timestamp}"
        write_seed_summary(prefix, summaries)
        print(f"Seed summary: {prefix}.csv")


if __name__ == "__main__":
    main()
