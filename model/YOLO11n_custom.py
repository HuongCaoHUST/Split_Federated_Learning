import torch
import torch.nn as nn
import os
from ultralytics import YOLO
from ultralytics.nn.modules import (
    C2PSA,
    SPPF,
    C3k2,
    Conv,
    Concat,
    Detect
)

class YOLO11_EDGE(nn.Module):
    def __init__(self, pretrained = None):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(Conv(c1=3, c2=16, k=3, s=2))
        self.layers.append(Conv(c1=16, c2=32, k=3, s=2))
        self.layers.append(C3k2(c1=32, c2=64, n=1, c3k=False, e=0.25))
        self.layers.append(Conv(c1=64, c2=64, k=3, s=2))
        self.layers.append(C3k2(c1=64, c2=128, n=1, c3k=False, e=0.25))
        self.layers.append(Conv(c1=128, c2=128, k=3, s=2))
        self.layers.append(C3k2(c1=128, c2=128, n=1, c3k=True))
        self.layers.append(Conv(c1=128, c2=256, k=3, s=2))
        self.layers.append(C3k2(c1=256, c2=256, n=1, c3k=True))
        self.layers.append(SPPF(c1=256, c2=256, k=5))
        self.layers.append(C2PSA(c1=256, c2=256, n=1))
        self.output_indices = [4, 6, 10]
        if pretrained:
            self.load_pretrained_weights(pretrained)

    def forward(self, x):
        outputs = []
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i in self.output_indices:
                outputs.append(x)
        return outputs
    
    def load_pretrained_weights(self, pt_path):
        print(f"Loading weights from {pt_path}...")
        yolo_model = YOLO(pt_path) 
        pretrained_model = yolo_model.model
        loaded_count = 0

        for i in range(len(self.layers)):
            try:
                source_layer = pretrained_model.model[i]
                target_layer = self.layers[i]
                target_layer.load_state_dict(source_layer.state_dict())
                # print(f"Layer {i}: Loaded {type(target_layer).__name__}")
                loaded_count += 1
            except Exception as e:
                print(f"Layer {i}: Failed to load. Error: {e}")
                break
        print(f"Load pretrained model success {loaded_count}/{len(self.layers)} layers")
    
class YOLO11_SERVER(nn.Module):
    def __init__(self, nc=80, pretrained = None):
        super().__init__()
        self.nc = nc
    
        self.layers = nn.ModuleList()
        self.layers.append(nn.Upsample(scale_factor=2, mode='nearest'))
        self.layers.append(Concat(dimension=1))
        self.layers.append(C3k2(384, 128, n=1, c3k=False))
        self.layers.append(nn.Upsample(scale_factor=2, mode='nearest'))
        self.layers.append(Concat(dimension=1))
        self.layers.append(C3k2(256, 64, n=1, c3k=False))
        self.layers.append(Conv(64, 64, k=3, s=2))
        self.layers.append(Concat(dimension=1))
        self.layers.append(C3k2(192, 128, n=1, c3k=False))
        self.layers.append(Conv(128, 128, k=3, s=2))
        self.layers.append(Concat(dimension=1))
        self.layers.append(C3k2(384, 256, n=1, c3k=True))
        self.layers.append(Detect(nc=nc, ch=[64, 128, 256]))

        self.model = self.layers
        detect_layer = self.layers[-1]
        if isinstance(detect_layer, Detect):
            detect_layer.stride = torch.tensor([8., 16., 32.])
            detect_layer.bias_init()

        if pretrained:
            self.load_pretrained_weights(pretrained)

    def forward(self, client_outputs):
        p3, p4, p5 = client_outputs

        # Block 1
        x = self.layers[0](p5)           # Up
        x = self.layers[1]([x, p4])      # Concat
        x = self.layers[2](x)            # C3k2
        f13 = x                          # Lưu lại feature map layer 13 (index 2)

        # Block 2 (P3 branch)
        x = self.layers[3](x)            # Up
        x = self.layers[4]([x, p3])      # Concat
        x = self.layers[5](x)            # C3k2
        head_p3 = x                      # Output cho Head P3

        # Block 3 (P4 branch)
        x = self.layers[6](head_p3)      # Conv
        x = self.layers[7]([x, f13])     # Concat với f13
        x = self.layers[8](x)            # C3k2
        head_p4 = x                      # Output cho Head P4

        # Block 4 (P5 branch)
        x = self.layers[9](head_p4)      # Conv
        x = self.layers[10]([x, p5])     # Concat với P5 gốc
        x = self.layers[11](x)           # C3k2
        head_p5 = x                      # Output cho Head P5

        return self.layers[12]([head_p3, head_p4, head_p5])
    
    def load_pretrained_weights(self, pt_path):
        print(f"Loading weights from {pt_path}...")
        yolo_model = YOLO(pt_path) 
        pretrained_model = yolo_model.model.model
        loaded_count = 0
        offset = 11

        for i in range(len(self.layers)):
            try:
                source_layer = pretrained_model[i + offset]
                target_layer = self.layers[i]
                target_layer.load_state_dict(source_layer.state_dict())
                # print(f"Layer {i + offset}: Loaded {type(target_layer).__name__}")
                loaded_count += 1
            except Exception as e:
                print(f"Layer {i}: Failed to load. Error: {e}")
                break
        print(f"Load pretrained model success {loaded_count}/{len(self.layers)} layers")
    
class YOLO11_Full(nn.Module):
    def __init__(self, nc=80, pretrained=None):
        super().__init__()
        self.nc = nc
        self.layers = nn.ModuleList()

        self.layers.append(Conv(c1=3, c2=16, k=3, s=2))
        self.layers.append(Conv(c1=16, c2=32, k=3, s=2))
        self.layers.append(C3k2(c1=32, c2=64, n=1, c3k=False, e=0.25))
        self.layers.append(Conv(c1=64, c2=64, k=3, s=2))
        self.layers.append(C3k2(c1=64, c2=128, n=1, c3k=False, e=0.25))
        self.layers.append(Conv(c1=128, c2=128, k=3, s=2))
        self.layers.append(C3k2(c1=128, c2=128, n=1, c3k=True))
        self.layers.append(Conv(c1=128, c2=256, k=3, s=2))
        self.layers.append(C3k2(c1=256, c2=256, n=1, c3k=True))
        self.layers.append(SPPF(c1=256, c2=256, k=5))
        self.layers.append(C2PSA(c1=256, c2=256, n=1))
        self.layers.append(nn.Upsample(scale_factor=2, mode='nearest'))
        self.layers.append(Concat(dimension=1))
        self.layers.append(C3k2(c1=384, c2=128, n=1, c3k=False)) # 256(P5_up) + 128(P4) = 384
        self.layers.append(nn.Upsample(scale_factor=2, mode='nearest'))
        self.layers.append(Concat(dimension=1))
        self.layers.append(C3k2(c1=256, c2=64, n=1, c3k=False)) # 128(up) + 128(P3) = 256
        self.layers.append(Conv(c1=64, c2=64, k=3, s=2))
        self.layers.append(Concat(dimension=1))
        self.layers.append(C3k2(c1=192, c2=128, n=1, c3k=False)) # 64(down) + 128(L13) = 192
        self.layers.append(Conv(c1=128, c2=128, k=3, s=2))
        self.layers.append(Concat(dimension=1))
        self.layers.append(C3k2(c1=384, c2=256, n=1, c3k=True)) # 128(down) + 256(P5) = 384
        self.layers.append(Detect(nc=nc, ch=[64, 128, 256]))

        self.model = self.layers
        detect_layer = self.layers[-1]
        if isinstance(detect_layer, Detect):
            detect_layer.stride = torch.tensor([8., 16., 32.])
            detect_layer.bias_init()

        if pretrained:
            self.load_pretrained_weights(pretrained)

    def forward(self, x):
        x = self.layers[0](x)
        x = self.layers[1](x)
        x = self.layers[2](x)
        x = self.layers[3](x)
        p3 = self.layers[4](x)   # Save P3
        x = self.layers[5](p3)

        return self.forward_from_cut5([p3, x])

    def forward_from_cut5(self, client_outputs):
        """Run layers 6..23 from the two cut-5 boundary activations.

        The instance still owns the *full* canonical model. Layers 0..5 are
        deliberately not executed here: their activations were produced by an
        edge replica. This lets a server own one optimizer state for every
        canonical layer while keeping split forward/backward computation.
        """
        if not isinstance(client_outputs, (list, tuple)) or len(client_outputs) != 2:
            raise ValueError("cut-5 execution expects [layer_4_output, layer_5_output].")

        p3, x = client_outputs
        p4 = self.layers[6](x)   # Save P4
        x = self.layers[7](p4)
        x = self.layers[8](x)
        x = self.layers[9](x)
        p5 = self.layers[10](x)  # Save P5

        x = self.layers[11](p5)          # Up
        x = self.layers[12]([x, p4])     # Concat with P4
        f13 = self.layers[13](x)         # C3k2 (Save feature map 13)

        x = self.layers[14](f13)         # Up
        x = self.layers[15]([x, p3])     # Concat with P3
        head_p3 = self.layers[16](x)     # Output P3

        x = self.layers[17](head_p3)     # Down
        x = self.layers[18]([x, f13])    # Concat with f13
        head_p4 = self.layers[19](x)     # Output P4

        # Block Down 2 (P5 Branch)
        x = self.layers[20](head_p4)     # Down
        x = self.layers[21]([x, p5])     # Concat with P5
        head_p5 = self.layers[22](x)     # Output P5

        # Head
        return self.layers[23]([head_p3, head_p4, head_p5])

    def load_pretrained_weights(self, pt_path):
        print(f"Loading weights from {pt_path}...")
        model_container = YOLO(pt_path)
        pretrained_layers = model_container.model.model
        
        loaded_count = 0
        
        for i in range(len(self.layers)):
            try:
                source_layer = pretrained_layers[i]
                target_layer = self.layers[i]
                target_layer.load_state_dict(source_layer.state_dict())
                # print(f"Layer {i}: Loaded {type(target_layer).__name__}")
                loaded_count += 1
            except Exception as e:
                print(f"Layer {i}: Failed to load. Error: {e}")
                break
        print(f"Load pretrained model success {loaded_count}/{len(self.layers)} modules")

class YOLO11_EDGE_5(nn.Module):
    def __init__(self, pretrained=None):
        super().__init__()
        self.layers = nn.ModuleList()
        # Dựa trên log:
        self.layers.append(Conv(c1=3, c2=16, k=3, s=2))                 # 0
        self.layers.append(Conv(c1=16, c2=32, k=3, s=2))                # 1
        self.layers.append(C3k2(c1=32, c2=64, n=1, c3k=False, e=0.25))  # 2
        self.layers.append(Conv(c1=64, c2=64, k=3, s=2))                # 3
        self.layers.append(C3k2(c1=64, c2=128, n=1, c3k=False, e=0.25)) # 4
        self.layers.append(Conv(c1=128, c2=128, k=3, s=2))              # 5
        
        if pretrained:
            self.load_pretrained_weights(pretrained)

    def forward(self, x):
        for i in range(4):
            x = self.layers[i](x)
        p3 = self.layers[4](x)
        x = self.layers[5](p3)
        
        return [p3, x]
    
    def load_pretrained_weights(self, pt_path):
        print(f"EDGE: Loading weights from {pt_path}...")
        yolo_model = YOLO(pt_path) 
        source_model = yolo_model.model.model
        loaded_count = 0

        for i in range(len(self.layers)):
            try:
                self.layers[i].load_state_dict(source_model[i].state_dict())
                loaded_count += 1
            except Exception as e:
                print(f"EDGE Layer {i}: Failed. {e}")
                break
        print(f"EDGE: Loaded {loaded_count}/{len(self.layers)} layers.")

class YOLO11_SERVER_5(nn.Module):
    def __init__(self, nc=80, pretrained=None):
        super().__init__()
        self.nc = nc
        self.layers = nn.ModuleList()

        self.layers.append(C3k2(c1=128, c2=128, n=1, c3k=True)) 
        self.layers.append(Conv(c1=128, c2=256, k=3, s=2))     
        self.layers.append(C3k2(c1=256, c2=256, n=1, c3k=True)) 
        self.layers.append(SPPF(c1=256, c2=256, k=5))
        self.layers.append(C2PSA(c1=256, c2=256, n=1))

        self.layers.append(nn.Upsample(scale_factor=2, mode='nearest'))
        self.layers.append(Concat(dimension=1))                        
        self.layers.append(C3k2(384, 128, n=1, c3k=False))       
        
        self.layers.append(nn.Upsample(scale_factor=2, mode='nearest')) 
        self.layers.append(Concat(dimension=1))                
        self.layers.append(C3k2(256, 64, n=1, c3k=False))        
        
        self.layers.append(Conv(64, 64, k=3, s=2))             
        self.layers.append(Concat(dimension=1))    
        self.layers.append(C3k2(192, 128, n=1, c3k=False))  
        
        self.layers.append(Conv(128, 128, k=3, s=2))        
        self.layers.append(Concat(dimension=1))  
        self.layers.append(C3k2(384, 256, n=1, c3k=True))  
        
        self.layers.append(Detect(nc=nc, ch=[64, 128, 256])) 

        self.model = self.layers

        detect_layer = self.layers[-1]
        if isinstance(detect_layer, Detect):
            detect_layer.stride = torch.tensor([8., 16., 32.])
            detect_layer.bias_init()

        if pretrained:
            self.load_pretrained_weights(pretrained)

    def forward(self, client_outputs):
        p3, x = client_outputs

        x = self.layers[0](x) 
        p4 = x 

        x = self.layers[1](x)
        x = self.layers[2](x)

        x = self.layers[3](x)
        x = self.layers[4](x)
        p5 = x

        f_backbone_end = x 

        x = self.layers[5](f_backbone_end) 
        x = self.layers[6]([x, p4])      
        x = self.layers[7](x)        
        f13 = x                 

        x = self.layers[8](x)          
        x = self.layers[9]([x, p3])   
        x = self.layers[10](x)         
        head_p3 = x                  

        x = self.layers[11](head_p3) 
        x = self.layers[12]([x, f13]) 
        x = self.layers[13](x)        
        head_p4 = x         

        x = self.layers[14](head_p4)  
        x = self.layers[15]([x, p5])
        x = self.layers[16](x)         
        head_p5 = x    

        # Detect
        return self.layers[17]([head_p3, head_p4, head_p5])

    def load_pretrained_weights(self, pt_path):
        print(f"SERVER: Loading weights from {pt_path}...")
        yolo_model = YOLO(pt_path) 
        source_model = yolo_model.model.model
        loaded_count = 0
        
        offset = 6

        for i in range(len(self.layers)):
            try:
                source_layer = source_model[i + offset]
                target_layer = self.layers[i]
                target_layer.load_state_dict(source_layer.state_dict())
                loaded_count += 1
            except Exception as e:
                print(f"SERVER Layer {i} (Source {i+offset}): Failed. {e}")
                
        print(f"SERVER: Loaded {loaded_count}/{len(self.layers)} layers.")


class YOLO11_DYNAMIC_SERVER(nn.Module):
    """YOLO11 server that accepts intermediate features from multiple cuts.

    The module owns every YOLO layer after the earliest configured cut. A
    payload produced at a later cut skips the layers that have already run on
    that edge, while all paths converge on the same remaining server weights.

    Supported edge payloads match the existing edge classes in this module:

    - cut 5: ``[p3, layer_5_output]``
    - cut 10: ``[p3, p4, p5]``
    - cut 15: ``[layer_15_output, f13, p5]``
    - cut 20: ``[layer_20_output, p5, head_p3, head_p4]``
    """

    SUPPORTED_CUT_LAYERS = (5, 10, 15, 20)
    OUTPUT_COUNT_BY_CUT = {5: 2, 10: 3, 15: 3, 20: 4}
    LAST_MODEL_LAYER = 23

    def __init__(self, supported_cut_layers, nc=80, pretrained=None):
        super().__init__()
        self.nc = nc
        self.supported_cut_layers = self._normalize_cut_layers(supported_cut_layers)
        self.min_cut_layer = min(self.supported_cut_layers)

        all_server_layers = self._build_server_layers(nc)
        first_server_layer = self.min_cut_layer + 1
        first_available_layer = min(self.SUPPORTED_CUT_LAYERS) + 1
        start = first_server_layer - first_available_layer

        self.global_layer_indices = list(
            range(first_server_layer, self.LAST_MODEL_LAYER + 1)
        )
        self.layers = nn.ModuleList(all_server_layers[start:])
        self.model = self.layers

        detect_layer = self._layer(self.LAST_MODEL_LAYER)
        if isinstance(detect_layer, Detect):
            detect_layer.stride = torch.tensor([8., 16., 32.])
            detect_layer.bias_init()

        if pretrained:
            self.load_pretrained_weights(pretrained)

    @classmethod
    def _normalize_cut_layers(cls, cut_layers):
        if isinstance(cut_layers, int) and not isinstance(cut_layers, bool):
            cut_layers = [cut_layers]
        if not isinstance(cut_layers, (list, tuple)) or not cut_layers:
            raise ValueError("supported_cut_layers must contain at least one cut.")

        normalized = []
        for cut_layer in cut_layers:
            if isinstance(cut_layer, bool):
                raise ValueError("Each cut layer must be an integer.")
            try:
                cut_layer = int(cut_layer)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Invalid cut layer {cut_layer!r}; expected an integer."
                ) from exc
            if cut_layer not in cls.SUPPORTED_CUT_LAYERS:
                raise ValueError(
                    f"Unsupported cut layer {cut_layer}; supported values: "
                    f"{list(cls.SUPPORTED_CUT_LAYERS)}."
                )
            if cut_layer not in normalized:
                normalized.append(cut_layer)
        return sorted(normalized)

    @staticmethod
    def _build_server_layers(nc):
        """Build canonical YOLO layers 6..23 in their global order."""
        return [
            C3k2(c1=128, c2=128, n=1, c3k=True),       # 6
            Conv(c1=128, c2=256, k=3, s=2),             # 7
            C3k2(c1=256, c2=256, n=1, c3k=True),        # 8
            SPPF(c1=256, c2=256, k=5),                  # 9
            C2PSA(c1=256, c2=256, n=1),                 # 10
            nn.Upsample(scale_factor=2, mode='nearest'), # 11
            Concat(dimension=1),                         # 12
            C3k2(c1=384, c2=128, n=1, c3k=False),       # 13
            nn.Upsample(scale_factor=2, mode='nearest'), # 14
            Concat(dimension=1),                         # 15
            C3k2(c1=256, c2=64, n=1, c3k=False),        # 16
            Conv(c1=64, c2=64, k=3, s=2),               # 17
            Concat(dimension=1),                         # 18
            C3k2(c1=192, c2=128, n=1, c3k=False),       # 19
            Conv(c1=128, c2=128, k=3, s=2),             # 20
            Concat(dimension=1),                         # 21
            C3k2(c1=384, c2=256, n=1, c3k=True),        # 22
            Detect(nc=nc, ch=[64, 128, 256]),            # 23
        ]

    def _layer(self, global_layer_index):
        local_index = global_layer_index - (self.min_cut_layer + 1)
        if local_index < 0 or local_index >= len(self.layers):
            raise ValueError(
                f"Layer {global_layer_index} is not owned by a server whose "
                f"minimum cut is {self.min_cut_layer}."
            )
        return self.layers[local_index]

    def _validate_inputs(self, client_outputs, cut_layer):
        if isinstance(cut_layer, bool):
            raise ValueError("cut_layer must be an integer.")
        try:
            cut_layer = int(cut_layer)
        except (TypeError, ValueError) as exc:
            raise ValueError("cut_layer must be an integer.") from exc

        if cut_layer not in self.supported_cut_layers:
            raise ValueError(
                f"cut_layer={cut_layer} was not configured for this server; "
                f"configured values: {self.supported_cut_layers}."
            )
        if not isinstance(client_outputs, (list, tuple)):
            raise TypeError("client_outputs must be a list or tuple of tensors.")

        expected_count = self.OUTPUT_COUNT_BY_CUT[cut_layer]
        if len(client_outputs) != expected_count:
            raise ValueError(
                f"cut_layer={cut_layer} expects {expected_count} tensors, "
                f"received {len(client_outputs)}."
            )
        return cut_layer

    def forward(self, client_outputs, cut_layer):
        cut_layer = self._validate_inputs(client_outputs, cut_layer)

        if cut_layer == 5:
            p3, x = client_outputs
            p4 = self._layer(6)(x)
            x = self._layer(7)(p4)
            x = self._layer(8)(x)
            x = self._layer(9)(x)
            p5 = self._layer(10)(x)
        elif cut_layer == 10:
            p3, p4, p5 = client_outputs

        if cut_layer <= 10:
            x = self._layer(11)(p5)
            x = self._layer(12)([x, p4])
            f13 = self._layer(13)(x)
            x = self._layer(14)(f13)
            x_at_15 = self._layer(15)([x, p3])
        elif cut_layer == 15:
            x_at_15, f13, p5 = client_outputs

        if cut_layer <= 15:
            head_p3 = self._layer(16)(x_at_15)
            x = self._layer(17)(head_p3)
            x = self._layer(18)([x, f13])
            head_p4 = self._layer(19)(x)
            x_at_20 = self._layer(20)(head_p4)
        else:  # cut_layer == 20
            x_at_20, p5, head_p3, head_p4 = client_outputs

        x = self._layer(21)([x_at_20, p5])
        head_p5 = self._layer(22)(x)
        return self._layer(23)([head_p3, head_p4, head_p5])

    def load_pretrained_weights(self, pt_path):
        print(f"DYNAMIC SERVER: Loading weights from {pt_path}...")
        yolo_model = YOLO(pt_path)
        source_model = yolo_model.model.model
        loaded_count = 0

        for local_index, global_index in enumerate(self.global_layer_indices):
            try:
                self.layers[local_index].load_state_dict(
                    source_model[global_index].state_dict()
                )
                loaded_count += 1
            except Exception as exc:
                print(
                    f"DYNAMIC SERVER Layer {local_index} "
                    f"(Source {global_index}): Failed. {exc}"
                )

        print(
            f"DYNAMIC SERVER: Loaded {loaded_count}/{len(self.layers)} layers."
        )

class YOLO11_EDGE_15(nn.Module):
    def __init__(self, pretrained=None):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(Conv(c1=3, c2=16, k=3, s=2))
        self.layers.append(Conv(c1=16, c2=32, k=3, s=2))
        self.layers.append(C3k2(c1=32, c2=64, n=1, c3k=False, e=0.25))
        self.layers.append(Conv(c1=64, c2=64, k=3, s=2))
        self.layers.append(C3k2(c1=64, c2=128, n=1, c3k=False, e=0.25))
        self.layers.append(Conv(c1=128, c2=128, k=3, s=2))
        self.layers.append(C3k2(c1=128, c2=128, n=1, c3k=True))
        self.layers.append(Conv(c1=128, c2=256, k=3, s=2))
        self.layers.append(C3k2(c1=256, c2=256, n=1, c3k=True))
        self.layers.append(SPPF(c1=256, c2=256, k=5))
        self.layers.append(C2PSA(c1=256, c2=256, n=1))
        self.layers.append(nn.Upsample(scale_factor=2, mode='nearest'))
        self.layers.append(Concat(dimension=1))
        self.layers.append(C3k2(384, 128, n=1, c3k=False))
        self.layers.append(nn.Upsample(scale_factor=2, mode='nearest'))
        self.layers.append(Concat(dimension=1))

        if pretrained:
            self.load_pretrained_weights(pretrained)

    def forward(self, x):
        x = self.layers[0](x)
        x = self.layers[1](x)
        x = self.layers[2](x)
        x = self.layers[3](x)
        p3 = self.layers[4](x)
        x = self.layers[5](p3)
        p4 = self.layers[6](x)
        x = self.layers[7](p4)
        x = self.layers[8](x)
        x = self.layers[9](x)
        p5 = self.layers[10](x)
        
        x = self.layers[11](p5)
        x = self.layers[12]([x, p4])
        f13 = self.layers[13](x)
        
        x = self.layers[14](f13)
        x_out = self.layers[15]([x, p3])
        
        return [x_out, f13, p5]
    
    def load_pretrained_weights(self, pt_path):
        print(f"EDGE: Loading weights from {pt_path}...")
        yolo_model = YOLO(pt_path) 
        source_model = yolo_model.model.model
        loaded_count = 0

        for i in range(len(self.layers)):
            try:
                self.layers[i].load_state_dict(source_model[i].state_dict())
                loaded_count += 1
            except Exception as e:
                print(f"EDGE Layer {i}: Failed. {e}")
                break
        print(f"EDGE: Loaded {loaded_count}/{len(self.layers)} layers.")

class YOLO11_SERVER_15(nn.Module):
    def __init__(self, nc=80, pretrained=None):
        super().__init__()
        self.nc = nc
        self.layers = nn.ModuleList()
        
        self.layers.append(C3k2(256, 64, n=1, c3k=False))
        self.layers.append(Conv(64, 64, k=3, s=2))
        self.layers.append(Concat(dimension=1))
        self.layers.append(C3k2(192, 128, n=1, c3k=False))
        self.layers.append(Conv(128, 128, k=3, s=2))
        self.layers.append(Concat(dimension=1))
        self.layers.append(C3k2(384, 256, n=1, c3k=True))
        self.layers.append(Detect(nc=nc, ch=[64, 128, 256]))

        self.model = self.layers

        detect_layer = self.layers[-1]
        if isinstance(detect_layer, Detect):
            detect_layer.stride = torch.tensor([8., 16., 32.])
            detect_layer.bias_init()

        if pretrained:
            self.load_pretrained_weights(pretrained)

    def forward(self, client_outputs):
        x, f13, p5 = client_outputs

        head_p3 = self.layers[0](x)
        
        x = self.layers[1](head_p3)
        x = self.layers[2]([x, f13])
        head_p4 = self.layers[3](x)
        
        x = self.layers[4](head_p4)
        x = self.layers[5]([x, p5])
        head_p5 = self.layers[6](x)
        
        return self.layers[7]([head_p3, head_p4, head_p5])

    def load_pretrained_weights(self, pt_path):
        print(f"SERVER: Loading weights from {pt_path}...")
        yolo_model = YOLO(pt_path) 
        source_model = yolo_model.model.model
        loaded_count = 0
        offset = 16 

        for i in range(len(self.layers)):
            try:
                source_layer = source_model[i + offset]
                target_layer = self.layers[i]
                target_layer.load_state_dict(source_layer.state_dict())
                loaded_count += 1
            except Exception as e:
                print(f"SERVER Layer {i} (Source {i+offset}): Failed. {e}")
                
        print(f"SERVER: Loaded {loaded_count}/{len(self.layers)} layers.")

class YOLO11_EDGE_20(nn.Module):
    def __init__(self, pretrained=None):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(Conv(c1=3, c2=16, k=3, s=2))
        self.layers.append(Conv(c1=16, c2=32, k=3, s=2))
        self.layers.append(C3k2(c1=32, c2=64, n=1, c3k=False, e=0.25))
        self.layers.append(Conv(c1=64, c2=64, k=3, s=2))
        self.layers.append(C3k2(c1=64, c2=128, n=1, c3k=False, e=0.25))
        self.layers.append(Conv(c1=128, c2=128, k=3, s=2))
        self.layers.append(C3k2(c1=128, c2=128, n=1, c3k=True))
        self.layers.append(Conv(c1=128, c2=256, k=3, s=2))
        self.layers.append(C3k2(c1=256, c2=256, n=1, c3k=True))
        self.layers.append(SPPF(c1=256, c2=256, k=5))
        self.layers.append(C2PSA(c1=256, c2=256, n=1))
        self.layers.append(nn.Upsample(scale_factor=2, mode='nearest'))
        self.layers.append(Concat(dimension=1))
        self.layers.append(C3k2(384, 128, n=1, c3k=False))
        self.layers.append(nn.Upsample(scale_factor=2, mode='nearest'))
        self.layers.append(Concat(dimension=1))
        self.layers.append(C3k2(256, 64, n=1, c3k=False))
        self.layers.append(Conv(64, 64, k=3, s=2))
        self.layers.append(Concat(dimension=1))
        self.layers.append(C3k2(192, 128, n=1, c3k=False))
        self.layers.append(Conv(128, 128, k=3, s=2))

        if pretrained:
            self.load_pretrained_weights(pretrained)

    def forward(self, x):
        x = self.layers[0](x)
        x = self.layers[1](x)
        x = self.layers[2](x)
        x = self.layers[3](x)
        p3 = self.layers[4](x)
        x = self.layers[5](p3)
        p4 = self.layers[6](x)
        x = self.layers[7](p4)
        x = self.layers[8](x)
        x = self.layers[9](x)
        p5 = self.layers[10](x)
        
        x = self.layers[11](p5)
        x = self.layers[12]([x, p4])
        f13 = self.layers[13](x)
        
        x = self.layers[14](f13)
        x = self.layers[15]([x, p3])
        head_p3 = self.layers[16](x)
        
        x = self.layers[17](head_p3)
        x = self.layers[18]([x, f13])
        head_p4 = self.layers[19](x)
        
        x_20 = self.layers[20](head_p4)
        
        return [x_20, p5, head_p3, head_p4]
    
    def load_pretrained_weights(self, pt_path):
        print(f"EDGE: Loading weights from {pt_path}...")
        yolo_model = YOLO(pt_path) 
        source_model = yolo_model.model.model
        loaded_count = 0

        for i in range(len(self.layers)):
            try:
                self.layers[i].load_state_dict(source_model[i].state_dict())
                loaded_count += 1
            except Exception as e:
                print(f"EDGE Layer {i}: Failed. {e}")
                break
        print(f"EDGE: Loaded {loaded_count}/{len(self.layers)} layers.")

class YOLO11_SERVER_20(nn.Module):
    def __init__(self, nc=80, pretrained=None):
        super().__init__()
        self.nc = nc
        self.layers = nn.ModuleList()
        
        self.layers.append(Concat(dimension=1))
        self.layers.append(C3k2(384, 256, n=1, c3k=True))
        self.layers.append(Detect(nc=nc, ch=[64, 128, 256]))

        self.model = self.layers

        detect_layer = self.layers[-1]
        if isinstance(detect_layer, Detect):
            detect_layer.stride = torch.tensor([8., 16., 32.])
            detect_layer.bias_init()

        if pretrained:
            self.load_pretrained_weights(pretrained)

    def forward(self, client_outputs):
        x_20, p5, head_p3, head_p4 = client_outputs

        x = self.layers[0]([x_20, p5])
        head_p5 = self.layers[1](x)
        
        return self.layers[2]([head_p3, head_p4, head_p5])

    def load_pretrained_weights(self, pt_path):
        print(f"SERVER: Loading weights from {pt_path}...")
        yolo_model = YOLO(pt_path) 
        source_model = yolo_model.model.model
        loaded_count = 0
        offset = 21 

        for i in range(len(self.layers)):
            try:
                source_layer = source_model[i + offset]
                target_layer = self.layers[i]
                target_layer.load_state_dict(source_layer.state_dict())
                loaded_count += 1
            except Exception as e:
                print(f"SERVER Layer {i} (Source {i+offset}): Failed. {e}")
                
        print(f"SERVER: Loaded {loaded_count}/{len(self.layers)} layers.")

if __name__ == "__main__":
    edge_model = YOLO11_EDGE(pretrained='yolo11n.pt')
    server_model = YOLO11_SERVER(pretrained='yolo11n.pt')
    full_model = YOLO11_Full(pretrained='yolo11n.pt')
