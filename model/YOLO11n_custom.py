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
                print(f"Layer {i}: Loaded {type(target_layer).__name__}")
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

        self.head = Detect(nc=nc, ch=[64, 128, 256])

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

        return self.head([head_p3, head_p4, head_p5])
    
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
                print(f"Layer {i + offset}: Loaded {type(target_layer).__name__}")
                loaded_count += 1
            except Exception as e:
                print(f"Layer {i}: Failed to load. Error: {e}")
                break

        source_head = pretrained_model[-1]
        if (source_head.cv2[0][0].conv.weight.shape == 
            self.head.cv2[0][0].conv.weight.shape):
            self.head.load_state_dict(source_head.state_dict())
            loaded_count += 1
        else:
            print("Head not loaded (NC mismatch).")

        print(f"Load pretrained model success {loaded_count}/{len(self.layers) + 1} layers")
    

if __name__ == "__main__":
    my_model = YOLO11_SERVER(pretrained='yolo11n.pt')