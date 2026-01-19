from ultralytics import YOLO
model = YOLO("results/run_25/cifar_net.pt")
metrics = model.val(data="./data/livingroom_4_1.yaml")