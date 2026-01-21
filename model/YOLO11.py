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

if __name__ == "__main__":
    full_model = YOLO11_Full(pretrained='yolo11n.pt')