import torch
import math
import os

from torchvision.utils import draw_bounding_boxes
from torchvision.ops import nms


def intersection_over_union(boxes_preds, boxes_labels, box_format="midpoint"):
    """
    Video explanation of this function:
    https://youtu.be/XXYG5ZWtjj0

    This function calculates intersection over union (iou) given pred boxes
    and target boxes.

    Parameters:
        boxes_preds (tensor): Predictions of Bounding Boxes (BATCH_SIZE, 4)
        boxes_labels (tensor): Correct labels of Bounding Boxes (BATCH_SIZE, 4)
        box_format (str): midpoint/corners, if boxes (x,y,w,h) or (x1,y1,x2,y2)

    Returns:
        tensor: Intersection over union for all examples
    """

    if box_format == "midpoint":
        box1_x1 = boxes_preds[..., 0:1] - boxes_preds[..., 2:3] / 2
        box1_y1 = boxes_preds[..., 1:2] - boxes_preds[..., 3:4] / 2
        box1_x2 = boxes_preds[..., 0:1] + boxes_preds[..., 2:3] / 2
        box1_y2 = boxes_preds[..., 1:2] + boxes_preds[..., 3:4] / 2
        box2_x1 = boxes_labels[..., 0:1] - boxes_labels[..., 2:3] / 2
        box2_y1 = boxes_labels[..., 1:2] - boxes_labels[..., 3:4] / 2
        box2_x2 = boxes_labels[..., 0:1] + boxes_labels[..., 2:3] / 2
        box2_y2 = boxes_labels[..., 1:2] + boxes_labels[..., 3:4] / 2

    if box_format == "corners":
        box1_x1 = boxes_preds[..., 0:1]
        box1_y1 = boxes_preds[..., 1:2]
        box1_x2 = boxes_preds[..., 2:3]
        box1_y2 = boxes_preds[..., 3:4]
        box2_x1 = boxes_labels[..., 0:1]
        box2_y1 = boxes_labels[..., 1:2]
        box2_x2 = boxes_labels[..., 2:3]
        box2_y2 = boxes_labels[..., 3:4]

    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    x2 = torch.min(box1_x2, box2_x2)
    y2 = torch.min(box1_y2, box2_y2)

    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)
    box1_area = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
    box2_area = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))

    return intersection / (box1_area + box2_area - intersection + 1e-6)


def get_learning_rate(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def save_checkpoint(model, optimizer, filename="my_checkpoint.pth.tar"):
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)


def save_model(model, optimizer, args):
    save_dir = args.logs_path + "/save"
    save_dir_exists = os.path.exists(save_dir)
    if not save_dir_exists:
        os.makedirs(save_dir)
    filename = str(save_dir) + "/" + "checkpoint.pth.tar"
    save_checkpoint(model, optimizer, filename=filename)


def load_checkpoint(checkpoint_path, model, optimizer, device):
    print("Loading checkpoint...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])


def unletterbox(x, yp_boxes, z):

    batch, channel, tensor_height, tensor_width = x.shape  # (16, 3, 416, 416)
    b = 0
    img_width = z[b][1]  # 640
    img_height = z[b][2]  # 426
    device = yp_boxes.device

    # Un-letterbox
    if img_width > img_height:
        letter_height = tensor_height * img_height / img_width  # e.g. 321.3
        letter_width = tensor_width  # 416
        letter_x = 1
        letter_y = letter_width / letter_height
        offset_x = 0
        offset_y = ((letter_width - letter_height) / 2) * letter_width / letter_height

        scale_x = img_width / tensor_width
        scale_y = img_height / tensor_height
        scale_w = img_width / tensor_width
        scale_h = img_width / tensor_height

    elif img_height > img_width:
        letter_height = tensor_height  # 416
        letter_width = tensor_width * img_width / img_height  # e.g. 321.3
        letter_x = letter_height / letter_width
        letter_y = 1
        offset_x = ((letter_height - letter_width) / 2) * letter_height / letter_width
        offset_y = 0

        scale_x = img_width / tensor_width
        scale_y = img_height / tensor_height
        scale_w = img_height / tensor_width
        scale_h = img_height / tensor_height

    else:
        letter_x, letter_y = 1, 1
        offset_x, offset_y = 0, 0
        scale_x = img_width / tensor_width
        scale_y = img_height / tensor_height
        scale_w = img_width / tensor_width
        scale_h = img_height / tensor_height

    letter = torch.tensor([1, letter_x, letter_y, 1, 1, 1, 1], device=device)
    scaling = torch.tensor([1, scale_x, scale_y, scale_w, scale_h, 1, 1], device=device)
    offset = torch.tensor([0, offset_x, offset_y, 0, 0, 0, 0], device=device)

    yp_boxes = scaling * (letter * yp_boxes - offset)
    return yp_boxes


def get_yp_boxes(x, yp, z, confidence_threshold, iou_threshold, anchors):

    batch, channel, tensor_height, tensor_width = x.shape  # (16, 3, 416, 416)
    s = len(yp)
    boxes = []
    b = 0
    img_id = z[b][0]
    device = yp[0].device

    for s in range(s):

        g_w, g_h = yp[s].shape[2], yp[s].shape[3]

        # Normalize p, x, y
        yp[s][b, :, :, :, 0:2] = torch.sigmoid(yp[s][b, :, :, :, 0:2])
        yp[s][b, :, :, :, 4] = torch.sigmoid(yp[s][b, :, :, :, 4])

        # Decode x, y
        yp[s][b, :, :, :, 0] = yp[s][b, :, :, :, 0] + torch.arange(0, g_w, device=device).unsqueeze(0).repeat(g_w, 1).repeat(3, 1, 1)
        yp[s][b, :, :, :, 1] = yp[s][b, :, :, :, 1] + torch.arange(0, g_h, device=device).unsqueeze(0).repeat(g_h, 1).permute(1, 0).repeat(3, 1, 1)
        yp[s][b, :, :, :, 0] = yp[s][b, :, :, :, 0] * tensor_width / g_w
        yp[s][b, :, :, :, 1] = yp[s][b, :, :, :, 1] * tensor_height / g_h

        # Decode w, h
        yp[s][b, :, :, :, 2:4] = torch.pow(math.e, yp[s][b, :, :, :, 2:4])
        a = anchors[s].reshape(3, 1, 1, 2).to(device=device)
        yp[s][b, :, :, :, 2:4] = yp[s][b, :, :, :, 2:4] * a

        # Ignore detections under confidence threshold
        mask = (yp[s][b, :, :, :, 4] >= confidence_threshold).unsqueeze(-1).repeat(1, 1, 1, 96)
        boxes_tensor = yp[s][b][mask].reshape(-1, 96)

        box_object = torch.zeros((boxes_tensor.shape[0], 7), device=device)
        box_object[:, 0] = img_id
        box_object[:, 1:6] = boxes_tensor[:, 0:5]
        box_object[:, 6] = torch.argmax(boxes_tensor[:, 5:], dim=1)
        boxes.append(box_object)

    boxes = torch.cat(boxes, dim=0)

    # Per class NMS
    cls_boxes_all = []
    for cls in boxes[:, 6].unique().tolist():
        idx = torch.where(boxes[:, 6] == int(cls))
        cls_boxes = boxes[idx[0], :]
        corner_boxes = torch.zeros(cls_boxes.shape[0], 4)
        corner_boxes[:, 0:2] = cls_boxes[:, 1:3] - cls_boxes[:, 3:5] * 0.5
        corner_boxes[:, 2:4] = cls_boxes[:, 1:3] + cls_boxes[:, 3:5] * 0.5
        idx = nms(corner_boxes, cls_boxes[:, 5], iou_threshold)
        cls_boxes_all.append(cls_boxes[idx, :])

    boxes = torch.concat(cls_boxes_all, dim=0) if cls_boxes_all else None
    return boxes


def denormalize(x, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    # 3, H, W, B
    ten = x.clone().permute(1, 2, 3, 0)
    for t, m, s in zip(ten, mean, std):
        t.mul_(s).add_(m)
    # B, 3, H, W
    return torch.clamp(ten, 0, 1).permute(3, 0, 1, 2)


def draw_y_on_x(x, y, classes):

    b = 0
    n = (y.sum(dim=2) > 0).sum()

    y[b, :n, 1:3] = y[b, :n, 1:3] - y[b, :n, 3:5] * 0.5
    y[b, :n, 3:5] = y[b, :n, 1:3] + y[b, :n, 3:5]

    boxes = y[b, :n, 1:5] * x.shape[-1]
    class_ids = y[b, :n, 0].tolist()

    labels = [classes[int(x)] for x in class_ids]
    x[0, ...] = draw_bounding_boxes(image=x[0, ...].type(torch.uint8), boxes=boxes, colors=(0, 255, 255), labels=labels, width=1)


def draw_yp_on_x(x, yp, z, probability_threshold, iou_threshold, anchors, classes):

    batch, channel, tensor_height, tensor_width = x.shape  # (16, 3, 416, 416)
    boxes = []
    b = 0
    img_id = z[b][0]
    device = yp[0].device

    for scale in range(len(yp)):

        g_w, g_h = yp[scale].shape[2], yp[scale].shape[3]

        # Normalize p, x, y
        yp[scale][b, :, :, :, 0:2] = torch.sigmoid(yp[scale][b, :, :, :, 0:2])
        yp[scale][b, :, :, :, 4] = torch.sigmoid(yp[scale][b, :, :, :, 4])

        # Decode x, y
        yp[scale][b, :, :, :, 0] = yp[scale][b, :, :, :, 0] + torch.arange(0, g_w).unsqueeze(0).repeat(g_w, 1).repeat(3, 1, 1)
        yp[scale][b, :, :, :, 1] = yp[scale][b, :, :, :, 1] + torch.arange(0, g_h).unsqueeze(0).repeat(g_h, 1).permute(1, 0).repeat(3, 1, 1)
        yp[scale][b, :, :, :, 0] = yp[scale][b, :, :, :, 0] * tensor_width / g_w
        yp[scale][b, :, :, :, 1] = yp[scale][b, :, :, :, 1] * tensor_height / g_h

        yp[scale][b, :, :, :, 2:4] = torch.pow(math.e, yp[scale][b, :, :, :, 2:4])
        a = anchors[scale].reshape(3, 1, 1, 2).cpu()
        yp[scale][b, :, :, :, 2:4] = yp[scale][b, :, :, :, 2:4] * a

        mask = (yp[scale][b, :, :, :, 4] >= probability_threshold).unsqueeze(-1).repeat(1, 1, 1, 96)
        boxes_tensor = yp[scale][b][mask].reshape(-1, 96)

        box_object = torch.zeros((boxes_tensor.shape[0], 7), device=device)
        box_object[:, 0] = img_id
        box_object[:, 1:6] = boxes_tensor[:, 0:5]
        box_object[:, 6] = torch.argmax(boxes_tensor[:, 5:], dim=1)
        boxes.append(box_object)

    boxes = torch.cat(boxes, dim=0)

    # Per class NMS
    cls_boxes_all = []
    for cls in boxes[:, 6].unique().tolist():
        idx = torch.where(boxes[:, 6] == int(cls))
        cls_boxes = boxes[idx[0], :]
        corner_boxes = torch.zeros(cls_boxes.shape[0], 4)
        corner_boxes[:, 0:2] = cls_boxes[:, 1:3] - cls_boxes[:, 3:5] * 0.5
        corner_boxes[:, 2:4] = cls_boxes[:, 1:3] + cls_boxes[:, 3:5] * 0.5
        idx = nms(corner_boxes, cls_boxes[:, 5], iou_threshold)
        cls_boxes_all.append(cls_boxes[idx, :])

    if cls_boxes_all:
        boxes = torch.cat(cls_boxes_all, dim=0)
        box = boxes[:, 1:5]
        corner_box = torch.zeros_like(box)
        corner_box[:, 0:2] = box[:, 0:2] - box[:, 2:4] * 0.5
        corner_box[:, 2:4] = box[:, 0:2] + box[:, 2:4] * 0.5
        labels = [classes[int(x[6].item())] + " " + str(int(x[5].item() * 100)) + "%" for x in boxes]
        x[b, ...] = draw_bounding_boxes(image=x[b, ...].type(torch.uint8), boxes=corner_box, colors=(255, 0, 255), labels=labels, width=1)
