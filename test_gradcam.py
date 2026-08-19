from model.Alexnet import AlexNet, AlexNetEdge
from src.classification.grad_cam import SplitGradCAM

gradcam = SplitGradCAM(
    full_model_class=AlexNet,
    edge_model_class=AlexNetEdge,
    best_weights="best.pt",
    cut_layer=2,
    num_classes=10
)

heatmap, overlay, class_id = gradcam.explain(
    "0010.png",
    save_path="gradcam_cut_2.png",
)

print("Predicted class:", class_id)
