"""Reusable full/edge/dynamic-server wrappers for staged classifiers."""

from __future__ import annotations

from collections import OrderedDict

import torch
import torch.nn as nn


def load_checkpoint_state(path):
    checkpoint = torch.load(path, map_location="cpu")
    if isinstance(checkpoint, nn.Module):
        state = checkpoint.state_dict()
    elif isinstance(checkpoint, dict):
        state = checkpoint
        for key in ("model_state_dict", "state_dict", "model"):
            if key not in checkpoint:
                continue
            state = checkpoint[key]
            if isinstance(state, nn.Module):
                state = state.state_dict()
            break
    else:
        raise TypeError(f"Unsupported checkpoint type: {type(checkpoint)}")

    if not isinstance(state, dict):
        raise TypeError("Checkpoint does not contain a model state dictionary.")
    return OrderedDict(
        (key.removeprefix("module."), value) for key, value in state.items()
    )


def build_seeded_stage(model_class, layer_index, num_classes, seed):
    if seed is None:
        return model_class.build_stage(layer_index, num_classes)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed) + layer_index)
        return model_class.build_stage(layer_index, num_classes)


class StagedClassifier(nn.Module):
    """Base for a classifier whose global parameter names live under layers."""

    LAYER_NAMES = ()
    SUPPORTED_CUT_LAYERS = ()

    def __init__(self, num_classes=10, seed=None):
        super().__init__()
        self.layers = nn.ModuleList(
            build_seeded_stage(type(self), index, num_classes, seed)
            for index in range(len(self.LAYER_NAMES))
        )

    @classmethod
    def build_stage(cls, layer_index, num_classes):
        raise NotImplementedError

    @property
    def num_layers(self):
        return len(self.layers)

    @classmethod
    def _validate_layer_index(cls, layer_index, name):
        if isinstance(layer_index, bool) or not isinstance(layer_index, int):
            raise TypeError(f"{name} must be an integer.")
        last_layer = len(cls.LAYER_NAMES) - 1
        if not 0 <= layer_index <= last_layer:
            raise ValueError(
                f"{name} must be between 0 and {last_layer}; "
                f"received {layer_index}."
            )
        return layer_index

    @classmethod
    def validate_cut_layer(cls, cut_layer):
        cls._validate_layer_index(cut_layer, "cut_layer")
        if cut_layer not in cls.SUPPORTED_CUT_LAYERS:
            raise ValueError(
                "cut_layer must leave at least one stage on the server; "
                f"supported values: {list(cls.SUPPORTED_CUT_LAYERS)}."
            )
        return cut_layer

    def forward_range(self, x, start_layer=0, end_layer=None):
        if end_layer is None:
            end_layer = self.num_layers - 1
        start_layer = self._validate_layer_index(start_layer, "start_layer")
        end_layer = self._validate_layer_index(end_layer, "end_layer")
        if start_layer > end_layer:
            raise ValueError("start_layer must be less than or equal to end_layer.")
        for layer_index in range(start_layer, end_layer + 1):
            x = self.layers[layer_index](x)
        return x

    def forward_to(self, x, cut_layer):
        return self.forward_range(x, 0, self.validate_cut_layer(cut_layer))

    def forward_from(self, x, cut_layer):
        cut_layer = self.validate_cut_layer(cut_layer)
        return self.forward_range(x, cut_layer + 1, self.num_layers - 1)

    def forward(self, x, cut_layer=None):
        if cut_layer is None:
            return self.forward_range(x)
        return self.forward_to(x, cut_layer)


class StagedClassifierEdge(nn.Module):
    """Memory-efficient edge partition for one inclusive global cut."""

    MODEL_CLASS = None

    def __init__(self, cut_layer, num_classes=10, seed=42, checkpoint=None):
        super().__init__()
        if self.MODEL_CLASS is None:
            raise TypeError("MODEL_CLASS must be set by the edge implementation.")
        self.cut_layer = self.MODEL_CLASS.validate_cut_layer(cut_layer)
        self.layers = nn.ModuleList(
            build_seeded_stage(self.MODEL_CLASS, index, num_classes, seed)
            for index in range(self.cut_layer + 1)
        )
        if checkpoint is not None:
            self.load_global_checkpoint(checkpoint)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def load_global_checkpoint(self, path):
        state = load_checkpoint_state(path)
        target_keys = set(self.state_dict())
        edge_state = {key: value for key, value in state.items() if key in target_keys}
        self.load_state_dict(edge_state, strict=True)


class StagedClassifierDynamicServer(nn.Module):
    """Server partition that accepts activations from several configured cuts."""

    MODEL_CLASS = None

    def __init__(
        self,
        supported_cut_layers,
        num_classes=10,
        seed=42,
        checkpoint=None,
    ):
        super().__init__()
        if self.MODEL_CLASS is None:
            raise TypeError("MODEL_CLASS must be set by the server implementation.")
        self.supported_cut_layers = self._normalize_cut_layers(
            supported_cut_layers
        )
        self.min_cut_layer = min(self.supported_cut_layers)
        self.global_layer_indices = list(
            range(self.min_cut_layer + 1, len(self.MODEL_CLASS.LAYER_NAMES))
        )
        self.layers = nn.ModuleList(
            build_seeded_stage(self.MODEL_CLASS, index, num_classes, seed)
            for index in self.global_layer_indices
        )
        if checkpoint is not None:
            self.load_global_checkpoint(checkpoint)

    def _normalize_cut_layers(self, cut_layers):
        if isinstance(cut_layers, int) and not isinstance(cut_layers, bool):
            cut_layers = [cut_layers]
        if not isinstance(cut_layers, (list, tuple)) or not cut_layers:
            raise ValueError("supported_cut_layers must contain at least one cut.")
        return sorted(
            {self.MODEL_CLASS.validate_cut_layer(cut) for cut in cut_layers}
        )

    def _local_index(self, global_layer_index):
        return global_layer_index - (self.min_cut_layer + 1)

    def forward(self, x, cut_layer):
        cut_layer = self.MODEL_CLASS.validate_cut_layer(cut_layer)
        if cut_layer not in self.supported_cut_layers:
            raise ValueError(
                f"cut_layer={cut_layer} was not configured for this server; "
                f"configured values: {self.supported_cut_layers}."
            )
        for global_index in range(
            cut_layer + 1, len(self.MODEL_CLASS.LAYER_NAMES)
        ):
            x = self.layers[self._local_index(global_index)](x)
        return x

    def load_global_checkpoint(self, path):
        state = load_checkpoint_state(path)
        server_state = {}
        for local_index, global_index in enumerate(self.global_layer_indices):
            global_prefix = f"layers.{global_index}."
            local_prefix = f"layers.{local_index}."
            for key, value in state.items():
                if key.startswith(global_prefix):
                    server_state[local_prefix + key[len(global_prefix):]] = value
        self.load_state_dict(server_state, strict=True)
