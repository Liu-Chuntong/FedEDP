"""Per-round CSV logs, Excel results, configuration snapshots, and figures."""

from dataclasses import asdict
from datetime import datetime
import json
from pathlib import Path
import platform

import numpy as np
import pandas as pd
import torch
import torchvision

from data_partition import get_targets


class ResultWriter:
    def __init__(self, config, client_datasets, public_size, test_size):
        self.config = config
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        tag = "smoke_" if config.smoke_test else ""
        name = (
            f"{tag}{config.dataset}_alpha{config.dirichlet_alpha:g}_"
            f"seed{config.seed}_{timestamp}"
        )
        self.directory = Path(config.results_dir).resolve() / name
        self.directory.mkdir(parents=True, exist_ok=False)
        (self.directory / "config.json").write_text(
            json.dumps(asdict(config), indent=2), encoding="utf-8"
        )
        environment = dict(
            python=platform.python_version(), torch=torch.__version__,
            torchvision=torchvision.__version__, numpy=np.__version__,
            device=config.device, public_samples=public_size, test_samples=test_size,
            private_samples=sum(map(len, client_datasets)),
            public_data_in_round_cost=False,
        )
        (self.directory / "environment.json").write_text(
            json.dumps(environment, indent=2), encoding="utf-8"
        )
        self.distribution = pd.DataFrame([
            dict(client_id=index, sample_count=len(dataset), **{
                f"class_{class_id}": int(count)
                for class_id, count in enumerate(np.bincount(
                    get_targets(dataset), minlength=config.num_classes
                ))
            })
            for index, dataset in enumerate(client_datasets)
        ])
        self.distribution.to_csv(self.directory / "client_distribution.csv", index=False)
        self.round_rows = []
        self.client_rows = []
        self.adaptive_rows = []

    def save_progress(self):
        for filename, rows in (
            ("rounds.csv", self.round_rows),
            ("clients.csv", self.client_rows),
            ("adaptive_entropy.csv", self.adaptive_rows),
        ):
            pd.DataFrame(rows).to_csv(self.directory / filename, index=False)

    def finish(self, clients):
        rounds = pd.DataFrame(self.round_rows)
        final_accuracy = float(rounds.iloc[-1]["mean_accuracy_pct"])
        best_index = rounds["mean_accuracy_pct"].idxmax()
        summary = dict(
            final_accuracy_pct=final_accuracy,
            best_accuracy_pct=float(rounds.loc[best_index, "mean_accuracy_pct"]),
            best_round=int(rounds.loc[best_index, "round"]),
            completed_rounds=int(rounds.iloc[-1]["round"]),
            cumulative_total_mib=float(rounds.iloc[-1]["cumulative_total_mib"]),
            smoke_test=self.config.smoke_test,
        )
        (self.directory / "summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
        with pd.ExcelWriter(self.directory / "results.xlsx", engine="openpyxl") as writer:
            rounds.to_excel(writer, sheet_name="Rounds", index=False)
            pd.DataFrame(self.client_rows).to_excel(writer, sheet_name="Clients", index=False)
            pd.DataFrame(self.adaptive_rows).to_excel(
                writer, sheet_name="Adaptive entropy", index=False
            )
            self.distribution.to_excel(writer, sheet_name="Client distribution", index=False)
            pd.DataFrame([summary]).to_excel(writer, sheet_name="Summary", index=False)
            pd.DataFrame([
                {"parameter": key, "value": str(value)}
                for key, value in asdict(self.config).items()
            ]).to_excel(writer, sheet_name="Configuration", index=False)
        if self.config.save_models:
            model_directory = self.directory / "models"
            model_directory.mkdir()
            for client in clients:
                torch.save(
                    {key: value.detach().cpu() for key, value in client.model.state_dict().items()},
                    model_directory / f"client_{client.client_id:03d}.pt",
                )
        self._plot(rounds)

    def _plot(self, rounds):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        adaptive = pd.DataFrame(self.adaptive_rows)
        figure, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
        steps = rounds["round"]
        axes[0, 0].plot(steps, rounds["mean_accuracy_pct"], marker=".", label="Client test")
        axes[0, 0].plot(steps, rounds["public_prediction_accuracy_pct"],
                        marker=".", label="Public prediction")
        axes[0, 0].set_ylabel("Accuracy (%)")
        axes[0, 0].legend()
        axes[0, 1].plot(steps, rounds["mean_test_loss"], marker=".")
        axes[0, 1].set_ylabel("Mean test loss")
        axes[1, 0].plot(steps, adaptive["lambda_value"], marker=".", label="Selected lambda")
        axes[1, 0].plot(steps, adaptive["lambda_lower"], linestyle="--", label="Lower endpoint")
        axes[1, 0].set_ylabel("Entropy weight")
        gap_axis = axes[1, 0].twinx()
        gap_axis.plot(steps, adaptive["representative_gap"], color="tab:red",
                      marker=".", label="Entropy gap")
        gap_axis.set_ylabel("Representative entropy gap")
        axes[1, 0].legend(loc="upper left")
        axes[1, 1].plot(steps, rounds["cumulative_total_mib"], marker=".")
        axes[1, 1].set_ylabel("Cumulative communication (MiB)")
        for axis in axes.flat:
            axis.set_xlabel("Communication round")
            axis.grid(alpha=0.2)
        title = f"{self.config.dataset}, alpha={self.config.dirichlet_alpha}, seed={self.config.seed}"
        figure.suptitle(f"Synthetic smoke test: {title}" if self.config.smoke_test else title)
        figure.savefig(self.directory / "training_curves.png", dpi=180)
        plt.close(figure)


def write_seed_summary(prefix, rows):
    frame = pd.DataFrame(rows)
    frame.to_csv(str(prefix) + ".csv", index=False)
    summary = dict(
        seeds=frame["seed"].tolist(),
        smoke_test=bool(frame["smoke_test"].any()),
        mean_final_accuracy_pct=float(frame["mean_accuracy_pct"].mean()),
        std_final_accuracy_pct=float(frame["mean_accuracy_pct"].std(ddof=1)),
        mean_cumulative_total_mib=float(frame["cumulative_total_mib"].mean()),
    )
    Path(str(prefix) + ".json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
