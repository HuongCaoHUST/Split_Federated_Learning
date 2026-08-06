from collections import OrderedDict

import torch.nn as nn


class AlexNet(nn.Module):
    """AlexNet with runtime-selectable split points.

    Layers are grouped into logical stages and numbered from 0 to 8.  A cut
    at stage ``k`` means that the edge executes stages ``0..k`` and the
    server executes stages ``k + 1..8``.  The cut is inclusive, matching the
    global-layer convention used by the split-learning pipeline.

    Stage layout:
        0: conv1 + relu + pool
        1: conv2 + relu + pool
        2: conv3 + relu
        3: conv4 + relu
        4: conv5 + relu + pool
        5: adaptive average pool + flatten
        6: dropout + fc1 + relu
        7: dropout + fc2 + relu
        8: classification layer
    """

    LAYER_NAMES = (
        "conv1",
        "conv2",
        "conv3",
        "conv4",
        "conv5",
        "avgpool_flatten",
        "fc1",
        "fc2",
        "classifier",
    )
    SUPPORTED_CUT_LAYERS = tuple(range(len(LAYER_NAMES) - 1))
    _LEGACY_STATE_PREFIXES = {
        "features.0.": "layers.0.0.",
        "features.3.": "layers.1.0.",
        "features.6.": "layers.2.0.",
        "features.8.": "layers.3.0.",
        "features.10.": "layers.4.0.",
        "classifier.1.": "layers.6.1.",
        "classifier.4.": "layers.7.1.",
        "classifier.6.": "layers.8.",
    }

    def __init__(self, num_classes=10):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=3, stride=2),
                ),
                nn.Sequential(
                    nn.Conv2d(96, 256, kernel_size=5, padding=2),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=3, stride=2),
                ),
                nn.Sequential(
                    nn.Conv2d(256, 384, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                ),
                nn.Sequential(
                    nn.Conv2d(384, 384, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                ),
                nn.Sequential(
                    nn.Conv2d(384, 256, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=3, stride=2),
                ),
                nn.Sequential(
                    nn.AdaptiveAvgPool2d((6, 6)),
                    nn.Flatten(start_dim=1),
                ),
                nn.Sequential(
                    nn.Dropout(),
                    nn.Linear(256 * 6 * 6, 4096),
                    nn.ReLU(inplace=True),
                ),
                nn.Sequential(
                    nn.Dropout(),
                    nn.Linear(4096, 4096),
                    nn.ReLU(inplace=True),
                ),
                nn.Linear(4096, num_classes),
            ]
        )

    @property
    def num_layers(self):
        return len(self.layers)

    @classmethod
    def _upgrade_legacy_state_dict(cls, state_dict):
        """Translate checkpoints created by the previous AlexNet layout."""
        upgraded = OrderedDict()
        for key, value in state_dict.items():
            upgraded_key = key
            for old_prefix, new_prefix in cls._LEGACY_STATE_PREFIXES.items():
                if key.startswith(old_prefix):
                    upgraded_key = new_prefix + key[len(old_prefix):]
                    break
            upgraded[upgraded_key] = value

        if hasattr(state_dict, "_metadata"):
            upgraded._metadata = state_dict._metadata
        return upgraded

    def load_state_dict(self, state_dict, strict=True, assign=False):
        """Load both the current layout and legacy ``features`` checkpoints."""
        if any(
            key.startswith(tuple(self._LEGACY_STATE_PREFIXES))
            for key in state_dict
        ):
            state_dict = self._upgrade_legacy_state_dict(state_dict)
        return super().load_state_dict(state_dict, strict=strict, assign=assign)

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
        """Validate and return an edge/server cut-layer index."""
        cls._validate_layer_index(cut_layer, "cut_layer")
        if cut_layer not in cls.SUPPORTED_CUT_LAYERS:
            raise ValueError(
                f"cut_layer must leave at least one stage on the server; "
                f"supported values: {list(cls.SUPPORTED_CUT_LAYERS)}."
            )
        return cut_layer

    def forward_range(self, x, start_layer=0, end_layer=None):
        """Run an inclusive range of global AlexNet stages.

        This is the primitive used by both sides of a dynamic split.  The
        input must have the shape produced immediately before ``start_layer``.
        """
        if end_layer is None:
            end_layer = self.num_layers - 1

        start_layer = self._validate_layer_index(start_layer, "start_layer")
        end_layer = self._validate_layer_index(end_layer, "end_layer")
        if start_layer > end_layer:
            raise ValueError(
                "start_layer must be less than or equal to end_layer; "
                f"received {start_layer} and {end_layer}."
            )

        for layer_index in range(start_layer, end_layer + 1):
            x = self.layers[layer_index](x)
        return x

    def forward_to(self, x, cut_layer):
        """Run the edge path through ``cut_layer`` (inclusive)."""
        cut_layer = self.validate_cut_layer(cut_layer)
        return self.forward_range(x, start_layer=0, end_layer=cut_layer)

    def forward_from(self, x, cut_layer):
        """Run the server path using the activation produced at a cut."""
        cut_layer = self.validate_cut_layer(cut_layer)
        return self.forward_range(
            x,
            start_layer=cut_layer + 1,
            end_layer=self.num_layers - 1,
        )

    def forward(self, x, cut_layer=None):
        """Run the full model, or the edge path when ``cut_layer`` is set."""
        if cut_layer is None:
            return self.forward_range(x)
        return self.forward_to(x, cut_layer)
