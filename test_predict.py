import torch
import cv2
import numpy as np
import random
from ultralytics.utils import nms, ops
from ultralytics.data.augment import LetterBox
from model.YOLO11n_custom import YOLO11_Full

random.seed(42)
colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(80)]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = YOLO11_Full(nc=20) 
ckpt = torch.load("train_results/2-2/Livingroom_Non-IID_cut_10.pt", map_location=device, weights_only=False)
names = ckpt.get('names', {i: f"class_{i}" for i in range(80)})
model.load_state_dict(ckpt['model'])
model.to(device).eval()

img_path = "test_2.jpg"
ori_img = cv2.imread(img_path)

img_resized = LetterBox(new_shape=(640, 640), auto=False, stride=32)(image=ori_img)
img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
img_tensor = img_rgb.transpose((2, 0, 1))
img_tensor = np.ascontiguousarray(img_tensor)
img_tensor = torch.from_numpy(img_tensor).float() / 255.0
img_tensor = img_tensor.unsqueeze(0).to(device)

with torch.no_grad():
    results = model(img_tensor) 
    if isinstance(results, list):
        results = results[0] 

preds = nms.non_max_suppression(results, conf_thres=0.7, iou_thres=0.7)

for i, det in enumerate(preds):
    if len(det):
        det[:, :4] = ops.scale_boxes(img_tensor.shape[2:], det[:, :4], ori_img.shape).round()
        
        for *xyxy, conf, cls in det:
            class_id = int(cls)
            class_name = names.get(class_id, f"Class {class_id}")
            label = f"{class_name} {conf:.2f}"
            color = colors[class_id % len(colors)]
            
            c1, c2 = (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3]))
            cv2.rectangle(ori_img, c1, c2, color, thickness=2, lineType=cv2.LINE_AA)
            
            tf = max(2 - 1, 1)
            t_size = cv2.getTextSize(label, 0, fontScale=0.5, thickness=tf)[0]
            label_c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
            cv2.rectangle(ori_img, c1, label_c2, color, -1, cv2.LINE_AA)  
            cv2.putText(ori_img, label, (c1[0], c1[1] - 2), 0, 0.5, 
                        [255, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

cv2.imshow("Result", ori_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
for i in range(5):
    cv2.waitKey(1)