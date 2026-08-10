"""MobileNetV2 partitions for dynamic-cut split federated learning."""

import torch.nn as nn

from model.staged_classification import (
    StagedClassifier,
    StagedClassifierDynamicServer,
    StagedClassifierEdge,
)


def _conv_bn_relu6(in_channels, out_channels, stride=1):
    return nn.Sequential(
        nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        ),
        nn.BatchNorm2d(out_channels),
        nn.ReLU6(inplace=True),
    )


class InvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, stride, expand_ratio):
        super().__init__()
        hidden_channels = int(round(in_channels * expand_ratio))
        self.use_residual = stride == 1 and in_channels == out_channels
        layers = []
        if expand_ratio != 1:
            layers.extend(
                [
                    nn.Conv2d(in_channels, hidden_channels, 1, bias=False),
                    nn.BatchNorm2d(hidden_channels),
                    nn.ReLU6(inplace=True),
                ]
            )
        layers.extend(
            [
                nn.Conv2d(
                    hidden_channels,
                    hidden_channels,
                    3,
                    stride=stride,
                    padding=1,
                    groups=hidden_channels,
                    bias=False,
                ),
                nn.BatchNorm2d(hidden_channels),
                nn.ReLU6(inplace=True),
                nn.Conv2d(hidden_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
            ]
        )
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv(x)
        return x + out if self.use_residual else out


def _inverted_stage(in_channels, out_channels, repeats, stride, expand_ratio):
    blocks = [
        InvertedResidual(in_channels, out_channels, stride, expand_ratio)
    ]
    blocks.extend(
        InvertedResidual(out_channels, out_channels, 1, expand_ratio)
        for _ in range(1, repeats)
    )
    return nn.Sequential(*blocks)


class MobileNetV2(StagedClassifier):
    """MobileNetV2 grouped by inverted-residual resolution stages."""

    LAYER_NAMES = (
        "stem",
        "bottleneck_16",
        "bottleneck_24",
        "bottleneck_32",
        "bottleneck_64",
        "bottleneck_96",
        "bottleneck_160",
        "bottleneck_320",
        "last_conv",
        "avgpool_flatten",
        "classifier",
    )
    SUPPORTED_CUT_LAYERS = tuple(range(len(LAYER_NAMES) - 1))

    @classmethod
    def build_stage(cls, layer_index, num_classes):
        stages = {
            0: lambda: _conv_bn_relu6(3, 32, stride=2),
            1: lambda: _inverted_stage(32, 16, 1, 1, 1),
            2: lambda: _inverted_stage(16, 24, 2, 2, 6),
            3: lambda: _inverted_stage(24, 32, 3, 2, 6),
            4: lambda: _inverted_stage(32, 64, 4, 2, 6),
            5: lambda: _inverted_stage(64, 96, 3, 1, 6),
            6: lambda: _inverted_stage(96, 160, 3, 2, 6),
            7: lambda: _inverted_stage(160, 320, 1, 1, 6),
            8: lambda: nn.Sequential(
                nn.Conv2d(320, 1280, kernel_size=1, bias=False),
                nn.BatchNorm2d(1280),
                nn.ReLU6(inplace=True),
            ),
            9: lambda: nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(start_dim=1)
            ),
            10: lambda: nn.Sequential(
                nn.Dropout(p=0.2), nn.Linear(1280, num_classes)
            ),
        }
        try:
            return stages[layer_index]()
        except KeyError as exc:
            raise ValueError(f"Unknown MobileNetV2 stage {layer_index}.") from exc


class MobileNetV2Edge(StagedClassifierEdge):
    MODEL_CLASS = MobileNetV2


class MobileNetV2DynamicServer(StagedClassifierDynamicServer):
    MODEL_CLASS = MobileNetV2
