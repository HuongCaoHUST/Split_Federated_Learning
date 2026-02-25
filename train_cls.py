from ultralytics.nn.tasks import ClassificationModel
model = ClassificationModel("yolo11n-cls.yaml", nc=10)
print(model)