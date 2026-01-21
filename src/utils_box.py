import torch
import torchvision
import numpy as np
import time
from ultralytics.utils.ops import xywh2xyxy

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
    bs = prediction.shape[0]
    if nc == 0:
        nc = prediction.shape[1] - 4

    if prediction.shape[1] == 4 + nc:
        prediction = prediction.transpose(-1, -2)
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

        # Box conversion
        box = xywh2xyxy(x[:, :4])

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