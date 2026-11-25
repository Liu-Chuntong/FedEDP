"""Dataset-specific CNN architectures without attention modules."""

import torch
from torch import nn


class ClientModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        channels = [config.input_channels, 32, 64, 128]
        if config.dataset == "svhn":
            channels.append(256)
        stages = []
        for in_channels, out_channels in zip(channels[:-1], channels[1:]):
            layers = [
                nn.Conv2d(in_channels, out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            ]
            layers.extend([
                nn.MaxPool2d(2),
                nn.Dropout2d(config.conv_dropout),
            ])
            stages.append(nn.Sequential(*layers))
        self.stages = nn.Sequential(*stages)
        spatial_size = config.image_size // (2 ** len(stages))
        self.classifier = nn.Sequential(
            nn.Linear(channels[-1] * spatial_size ** 2, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(config.classifier_dropout),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, config.num_classes),
        )

    def forward(self, images):
        """Return logits; temperature scaling is handled by the client."""
        features = torch.flatten(self.stages(images), 1)
        return self.classifier(features)
