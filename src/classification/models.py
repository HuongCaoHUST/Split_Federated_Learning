"""Registry and construction helpers for classification split models.

A registered model must expose compatible full, edge and dynamic-server
implementations. The full model stores global stages in ``layers``; edge and
server state dictionaries use local ``layers`` indices so the generic
aggregation code can reconstruct the global model.
"""

from __future__ import annotations

from dataclasses import dataclass

from model.Alexnet import AlexNet, AlexNetDynamicServer, AlexNetEdge
from model.MobilenetV2 import (
    MobileNetV2,
    MobileNetV2DynamicServer,
    MobileNetV2Edge,
)
from model.Resnet18 import ResNet18, ResNet18DynamicServer, ResNet18Edge
from src.classification.data import get_class_names, get_dataset_name


@dataclass(frozen=True)
class ClassificationModelSpec:
    name: str
    full_class: type
    edge_class: type
    server_class: type
    input_channels: int
    minimum_input_size: int

    @property
    def layer_names(self) -> tuple[str, ...]:
        return tuple(self.full_class.LAYER_NAMES)

    @property
    def supported_cut_layers(self) -> tuple[int, ...]:
        return tuple(self.full_class.SUPPORTED_CUT_LAYERS)

    def validate_cut_layer(self, cut_layer: int) -> int:
        return self.full_class.validate_cut_layer(cut_layer)


MODEL_REGISTRY = {
    "ALEXNET": ClassificationModelSpec(
        name="AlexNet",
        full_class=AlexNet,
        edge_class=AlexNetEdge,
        server_class=AlexNetDynamicServer,
        input_channels=3,
        minimum_input_size=63,
    ),
    "RESNET18": ClassificationModelSpec(
        name="ResNet18",
        full_class=ResNet18,
        edge_class=ResNet18Edge,
        server_class=ResNet18DynamicServer,
        input_channels=3,
        minimum_input_size=32,
    ),
    "MOBILENETV2": ClassificationModelSpec(
        name="MobileNetV2",
        full_class=MobileNetV2,
        edge_class=MobileNetV2Edge,
        server_class=MobileNetV2DynamicServer,
        input_channels=3,
        minimum_input_size=32,
    ),
}


def _model_config(config):
    model_config = config.get("model", {})
    if not isinstance(model_config, dict):
        raise TypeError(
            "Classification model config must be a mapping, for example "
            "model: {name: AlexNet}."
        )
    return model_config


def _normalize_model_name(name) -> str:
    normalized = str(name).strip().upper().replace("-", "").replace("_", "")
    aliases = {
        "ALEXNET": "ALEXNET",
        "RESNET18": "RESNET18",
        "MOBILENETV2": "MOBILENETV2",
    }
    try:
        return aliases[normalized]
    except KeyError as exc:
        supported = ", ".join(spec.name for spec in MODEL_REGISTRY.values())
        raise ValueError(
            f"Classification model '{name}' is not registered; choose {supported}."
        ) from exc


def get_model_spec(config) -> ClassificationModelSpec:
    model_config = _model_config(config)
    return MODEL_REGISTRY[
        _normalize_model_name(model_config.get("name", "AlexNet"))
    ]


def get_model_name(config) -> str:
    return get_model_spec(config).name


def validate_classification_config(config) -> None:
    """Validate model/dataset compatibility before any workers are started."""
    model_spec = get_model_spec(config)
    model_config = _model_config(config)
    dataset_config = config.get("dataset", {})
    class_names = get_class_names(config)
    num_classes = int(model_config.get("num_classes", len(class_names)))
    if num_classes != len(class_names):
        raise ValueError(
            f"{get_dataset_name(config)} has {len(class_names)} classes, but "
            f"model.num_classes={num_classes}."
        )

    channels = int(dataset_config.get("channels", 3))
    if channels != model_spec.input_channels:
        raise ValueError(
            f"{model_spec.name} expects {model_spec.input_channels} input channels, "
            f"but dataset.channels={channels}."
        )
    input_size = int(dataset_config.get("input_size", 224))
    if input_size < model_spec.minimum_input_size:
        raise ValueError(
            f"{model_spec.name} requires dataset.input_size >= "
            f"{model_spec.minimum_input_size}; received {input_size}."
        )


def _model_options(config):
    model_config = _model_config(config)
    return {
        "num_classes": int(
            model_config.get("num_classes", len(get_class_names(config)))
        ),
        "seed": int(model_config.get("seed", 42)),
    }


def build_full_model(config):
    spec = get_model_spec(config)
    return spec.full_class(**_model_options(config))


def build_edge_model(config, cut_layer, checkpoint=None):
    spec = get_model_spec(config)
    return spec.edge_class(
        cut_layer=cut_layer,
        checkpoint=checkpoint,
        **_model_options(config),
    )


def build_server_model(config, supported_cut_layers, checkpoint=None):
    spec = get_model_spec(config)
    return spec.server_class(
        supported_cut_layers=supported_cut_layers,
        checkpoint=checkpoint,
        **_model_options(config),
    )
