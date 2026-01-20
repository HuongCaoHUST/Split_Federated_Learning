import torch
import torchvision
import numpy as np
import time

def xywh2xyxy(x):
    """Chuyển đổi box từ (x_center, y_center, w, h) sang (x1, y1, x2, y2)"""
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # top left x
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # top left y
    y[..., 2] = x[..., 0] + x[..., 2] / 2  # bottom right x
    y[..., 3] = x[..., 1] + x[..., 3] / 2  # bottom right y
    return y

def box_iou(box1, box2, eps=1e-7):
    """Tính IoU giữa 2 tập boxes."""
    (a1, a2), (b1, b2) = box1.unsqueeze(1).chunk(2, 2), box2.unsqueeze(0).chunk(2, 2)
    inter = (torch.min(a2, b2) - torch.max(a1, b1)).clamp(0).prod(2)
    area1 = (a2 - a1).prod(2)
    area2 = (b2 - b1).prod(2)
    return inter / (area1 + area2 - inter + eps)

def scale_boxes(img1_shape, boxes, img0_shape, ratio_pad=None):
    """Scale box từ kích thước ảnh input về ảnh gốc."""
    if ratio_pad is None:
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    boxes[..., [0, 2]] -= pad[0]
    boxes[..., [1, 3]] -= pad[1]
    boxes[..., :4] /= gain
    boxes[..., [0, 2]] = boxes[..., [0, 2]].clamp(0, img0_shape[1])
    boxes[..., [1, 3]] = boxes[..., [1, 3]].clamp(0, img0_shape[0])
    return boxes

def non_max_suppression(
    prediction,
    conf_thres=0.25,
    iou_thres=0.45,
    classes=None,
    agnostic=False,
    multi_label=False,
    labels=(),
    max_det=300,
    nc=0, 
):
    """
    Hàm NMS xử lý output của YOLO (v8/v11).
    Input: prediction (Batch_Size, 4 + Num_Classes, Num_Anchors)
    """
    bs = prediction.shape[0]
    if nc == 0:
        nc = prediction.shape[1] - 4

    # Transpose nếu shape là (Batch, 4+nc, Anchors) -> (Batch, Anchors, 4+nc)
    if prediction.shape[1] == 4 + nc:
        prediction = prediction.transpose(-1, -2)

    # --- SỬA LỖI TẠI ĐÂY ---
    # Kiểm tra max confidence của các class (index 4 trở đi) cho từng anchor
    # amax(-1): Lấy giá trị lớn nhất theo chiều class
    xc = prediction[..., 4:].amax(-1) > conf_thres 

    # Settings
    max_wh = 7680
    max_nms = 30000
    time_limit = 0.5 + 0.05 * bs
    
    t = time.time()
    output = [torch.zeros((0, 6), device=prediction.device)] * bs
    
    for xi, x in enumerate(prediction):
        # Apply constraints
        x = x[xc[xi]]  # Lọc bỏ các anchor có conf thấp

        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 4:] *= x[:, 4:]  # conf = obj_conf * cls_conf

        # Box conversion
        box = xywh2xyxy(x[:, :4])

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = (x[:, 4:] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, 4 + j, None], j[:, None].float()), 1)
        else:
            # best class only
            conf, j = x[:, 4:].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]

        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        n = x.shape[0]
        if not n:
            continue
        elif n > max_nms:
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)
        boxes, scores = x[:, :4] + c, x[:, 4]
        
        i = torchvision.ops.nms(boxes, scores, iou_thres)
        if i.shape[0] > max_det:
            i = i[:max_det]

        output[xi] = x[i]
        
        if (time.time() - t) > time_limit:
            break

    return output