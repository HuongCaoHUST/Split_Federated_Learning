"""ResNet-18 partitions for dynamic-cut split federated learning."""

import torch.nn as nn

from model.staged_classification import (
    StagedClassifier,
    StagedClassifierDynamicServer,
    StagedClassifierEdge,
)


def _conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False,
    )


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = _conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = _conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.relu(out + identity)


def _residual_stage(in_channels, out_channels, stride):
    return nn.Sequential(
        BasicBlock(in_channels, out_channels, stride=stride),
        BasicBlock(out_channels, out_channels),
    )


class ResNet18(StagedClassifier):
    """Standard ResNet-18 grouped into seven globally addressable stages."""

    LAYER_NAMES = (
        "stem",
        "layer1",
        "layer2",
        "layer3",
        "layer4",
        "avgpool_flatten",
        "classifier",
    )
    SUPPORTED_CUT_LAYERS = tuple(range(len(LAYER_NAMES) - 1))

    @classmethod
    def build_stage(cls, layer_index, num_classes):
        stages = {
            0: lambda: nn.Sequential(
                nn.Conv2d(
                    3, 64, kernel_size=7, stride=2, padding=3, bias=False
                ),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            ),
            1: lambda: _residual_stage(64, 64, stride=1),
            2: lambda: _residual_stage(64, 128, stride=2),
            3: lambda: _residual_stage(128, 256, stride=2),
            4: lambda: _residual_stage(256, 512, stride=2),
            5: lambda: nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(start_dim=1)
            ),
            6: lambda: nn.Linear(512, num_classes),
        }
        try:
            return stages[layer_index]()
        except KeyError as exc:
            raise ValueError(f"Unknown ResNet18 stage {layer_index}.") from exc


class ResNet18Edge(StagedClassifierEdge):
    MODEL_CLASS = ResNet18


class ResNet18DynamicServer(StagedClassifierDynamicServer):
    MODEL_CLASS = ResNet18
