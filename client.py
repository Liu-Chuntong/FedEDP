
import math

import torch
from torch import nn
from torch.nn import functional as F

from data import DistillationDataset, make_loader


def legacy_noise_std(config):
    
    nominal_sensitivity = (
        config.lipschitz_constant * config.delta_theta / (2 * config.temperature)
    )
    return (
        nominal_sensitivity * math.sqrt(2 * math.log(1.25 / config.delta))
        / config.epsilon
    )


def perturb_soft_labels(probabilities, config):
    sigma = legacy_noise_std(config)
    noise = torch.normal(
        mean=0.0, std=sigma, size=probabilities.shape,
        device=probabilities.device, dtype=probabilities.dtype,
    )
    return torch.softmax((probabilities + noise) / config.temperature, dim=-1)


def prediction_entropy(probabilities):
    values = probabilities.clamp_min(1e-10)
    return -(values * values.log()).sum(dim=-1)


def current_learning_rate(config, round_number):
    
    exponent = round_number - 1
    return max(config.initial_lr * config.lr_decay_rate ** exponent, config.min_lr)


def adjust_dropout(losses, probability, config):
    if len(losses) < config.dropout_window:
        return probability
    window = config.dropout_window
    recent = sum(losses[-window:]) / window
    previous = sum(losses[-window - 1:-1]) / window
    if recent > previous * (1 + config.dropout_loss_tolerance):
        return min(probability + config.dropout_step, config.dropout_max)
    if recent < previous * (1 - config.dropout_loss_tolerance):
        return max(probability - config.dropout_step, config.dropout_min)
    return probability


class FederatedClient:
    def __init__(self, client_id, model, private_dataset, config):
        self.client_id = client_id
        self.model = model.to(config.device)
        self.private_dataset = private_dataset
        self.config = config

    def _optimizer(self, round_number):
        config = self.config
        kwargs = {
            "lr": current_learning_rate(config, round_number),
            "weight_decay": config.weight_decay,
        }
        if config.optimizer == "adam":
            return torch.optim.Adam(
                self.model.parameters(), betas=config.adam_betas,
                eps=config.adam_epsilon, **kwargs,
            )
        return torch.optim.SGD(
            self.model.parameters(), momentum=config.momentum, **kwargs
        )

    def _train(self, dataset, round_number, distillation=False):
        config = self.config
        loader = make_loader(dataset, config, shuffle=True)
        optimizer = self._optimizer(round_number)
        history = []
        dropout_probability = config.dropout_initial
        self.model.train()
        for _ in range(config.local_epochs):
            if config.dynamic_dropout:
                for module in self.model.modules():
                    if isinstance(module, (nn.Dropout, nn.Dropout2d)):
                        module.p = dropout_probability
            total_loss = 0.0
            for batch in loader:
                images = batch[0].to(config.device)
                optimizer.zero_grad(set_to_none=True)
                logits = self.model(images)
                if distillation:
                    soft = batch[1].to(config.device)
                    hard = batch[2].to(config.device)
                    if config.distillation_loss == "kl":
                        kd_loss = F.kl_div(
                            F.log_softmax(logits / config.temperature, dim=1),
                            soft, reduction="batchmean",
                        ) * config.temperature ** 2
                    else:
                        kd_loss = F.cross_entropy(
                            logits / config.temperature, soft
                        ) * config.temperature ** 2
                    loss = (
                        (1 - config.hard_label_weight) * kd_loss
                        + config.hard_label_weight * F.cross_entropy(logits, hard)
                    )
                else:
                    loss = F.cross_entropy(logits, batch[1].to(config.device))
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            epoch_loss = total_loss / len(loader)
            history.append(epoch_loss)
            if config.dynamic_dropout:
                dropout_probability = adjust_dropout(history, dropout_probability, config)
        return history[-1]

    def train_private(self, round_number):
        return self._train(self.private_dataset, round_number)

    @torch.no_grad()
    def predict_public(self, public_loader):
        
        self.model.eval()
        predictions, entropies = [], []
        for images, _ in public_loader:
            probabilities = torch.softmax(
                self.model(images.to(self.config.device)) / self.config.temperature,
                dim=1,
            )
            entropies.append(prediction_entropy(probabilities).cpu())
            predictions.append(perturb_soft_labels(probabilities, self.config).cpu())
        return torch.cat(predictions), torch.cat(entropies)

    def distill(self, public_dataset, soft_labels, hard_labels, round_number):
        dataset = DistillationDataset(public_dataset, soft_labels, hard_labels)
        return self._train(dataset, round_number, distillation=True)

    @torch.no_grad()
    def evaluate(self, loader):
        self.model.eval()
        correct, count, loss_sum = 0, 0, 0.0
        for images, labels in loader:
            labels = labels.to(self.config.device)
            logits = self.model(images.to(self.config.device))
            loss_sum += F.cross_entropy(logits, labels).item()
            correct += (logits.argmax(dim=1) == labels).sum().item()
            count += len(labels)
        return 100 * correct / count, loss_sum / len(loader)
