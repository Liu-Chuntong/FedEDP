# FedEDP

Federated distillation with adaptive entropy weighting and Gaussian
perturbation of soft labels.

# Project layout

```text
FedEDP/
  main.py                 Experiment entry point and all hyperparameter defaults
  adaptive_entropy.py     Global entropy gap and upper-endpoint lambda selection
  client.py               Local training, probability noise, and distillation
  server.py               Entropy-weighted aggregation and payload accounting
  data_partition.py       Public/private and Dirichlet client partitions
  data.py                 Dataset loading and aligned public input batches
  models.py               Dataset-specific CNN architectures
  results.py              CSV, Excel, configuration, and figure export
  tests/test_core.py      Numerical checks and synthetic integration tests
  requirements.txt
  .gitignore
```



# Installation

Use Python 3.10 or later. From this directory:

```bash
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
# Linux or macOS: source .venv/bin/activate
python -m pip install -r requirements.txt
```

Install a matching PyTorch/torchvision pair for your CPU or CUDA environment.
Dataset files are downloaded on the first real run. `--no-download` uses only
previously downloaded files. The default data and results directories are
relative to this project's location, regardless of the working directory.

# Run experiments

```bash
python main.py --dataset fashionmnist --alpha 0.01 --seed 42
python main.py --dataset emnist --alpha 0.01 --seed 42
python main.py --dataset svhn --alpha 0.01 --seed 42
python main.py --dataset cifar10 --alpha 0.01 --seed 42
```

Use **42, 43, 44, 45, and 46** as the five experimental random seeds:

```bash
python main.py --dataset fashionmnist --alpha 0.01 --seeds 42 43 44 45 46
```

Repeat for the other datasets and the desired Dirichlet settings, such as
`--alpha 0.03` and `--alpha 0.1`. Seeds govern initialization, data selection,
partitioning, participation, batch shuffling, dropout, and Gaussian draws.
The multi-seed command runs sequentially and reports the mean and sample
standard deviation of the final accuracy.


```bash
python main.py --dataset emnist --print-config
python main.py --dataset fashionmnist --no-adaptive-lambda --fixed-lambda 3.0
python main.py --help
```

`--print-config` performs no downloads or training. Use `--device cpu` or
`--device cuda:0` to select a device. The default is CUDA when available,
otherwise CPU.

# Hyperparameters

Edit the constants and `DATASET_DEFAULTS` in **main.py**. Shared defaults are
10 clients, full participation, 50 communication rounds, batch size 64,
temperature 2, and 3 local epochs for both private training and distillation.
The epoch count stays constant throughout training.

| Dataset | Public ratio | Optimizer | Initial LR | Minimum LR | LR decay |
| --- | --- | --- | --- | --- | --- |
| Fashion-MNIST | 0.10 | Adam | 0.0001 | 0.00001 | 0.95 |
| EMNIST ByClass | 0.05 | Adam | 0.0001 | 0.00001 | 0.95 |
| SVHN | 0.10 | Adam | 0.0001 | 0.00001 | 0.95 |
| CIFAR-10 | 0.10 | Adam | 0.0001 | 0.00001 | 0.95 |



# Adaptive entropy weighting

1. Sort the participating clients' clean prediction entropies for each public
   sample. The lower and upper ranks are `ceil(q_L * N)` and `ceil(q_U * N)`.
2. Subtract the lower-rank entropy from the upper-rank entropy for each sample.
3. Use the configured quantile of these sample gaps as the global gap `Delta`.
4. Compute `lambda_lower = log(R_min) / Delta` and
   `lambda_upper = log(n_U / (n_L * xi)) / Delta`.
5. Clip the endpoints to the configured numerical range and select the upper
   endpoint. Use the fallback for unresolved gaps or a single participant.

| Parameter in main.py | Default |
| --- | --- |
| `ADAPTIVE_LAMBDA_MIN_RATIO` | 2.0 |
| `ADAPTIVE_HIGH_ENTROPY_TOLERANCE` | 0.10 |
| `ADAPTIVE_CLIENT_LOW_QUANTILE` | 0.25 |
| `ADAPTIVE_CLIENT_HIGH_QUANTILE` | 0.75 |
| `ADAPTIVE_SAMPLE_GAP_QUANTILE` | 0.10 |
| `ADAPTIVE_ENTROPY_GAP_FLOOR` | 0.000001 |
| `ADAPTIVE_LAMBDA_MIN_CLIP` | 0.0 |
| `ADAPTIVE_LAMBDA_MAX_CLIP` | 20.0 |

## Tests

Run the offline numerical and synthetic training tests:

```bash
python -m unittest discover -s tests -v
```

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE).
