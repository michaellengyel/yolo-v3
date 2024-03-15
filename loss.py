import torch
import torch.nn as nn
import numpy as np


def bboxes_iou(bboxes_a, bboxes_b, xyxy=True):

    if bboxes_a.shape[1] != 4 or bboxes_b.shape[1] != 4:
        raise IndexError

    if xyxy:
        tl = torch.max(bboxes_a[:, None, :2], bboxes_b[:, :2])
        br = torch.min(bboxes_a[:, None, 2:], bboxes_b[:, 2:])
        area_a = torch.prod(bboxes_a[:, 2:] - bboxes_a[:, :2], 1)
        area_b = torch.prod(bboxes_b[:, 2:] - bboxes_b[:, :2], 1)
    else:
        tl = torch.max((bboxes_a[:, None, :2] - bboxes_a[:, None, 2:] / 2), (bboxes_b[:, :2] - bboxes_b[:, 2:] / 2))
        br = torch.min((bboxes_a[:, None, :2] + bboxes_a[:, None, 2:] / 2), (bboxes_b[:, :2] + bboxes_b[:, 2:] / 2))
        area_a = torch.prod(bboxes_a[:, 2:], 1)
        area_b = torch.prod(bboxes_b[:, 2:], 1)
    en = (tl < br).type(tl.type()).prod(dim=2)
    area_i = torch.prod(br - tl, 2) * en  # * ((tl < br).all())
    return area_i / (area_a[:, None] + area_b - area_i)


class YoloScaleLoss(nn.Module):

    def __init__(self, scale, ignore_threshold):
        super(YoloScaleLoss, self).__init__()
        self.l2_loss = nn.MSELoss(reduction='sum')
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='sum')
        strides = [32, 16, 8]
        self.anchors = [[10, 13], [16, 30], [33, 23], [30, 61], [62, 45], [59, 119], [116, 90], [156, 198], [373, 326]]
        self.anch_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]][scale]
        self.n_anchors = len(self.anch_mask)
        self.n_classes = 91
        self.ignore_threshold = ignore_threshold
        self.stride = strides[scale]
        self.all_anchors_grid = [(w / self.stride, h / self.stride) for w, h in self.anchors]
        self.masked_anchors = [self.all_anchors_grid[i] for i in self.anch_mask]
        self.ref_anchors = np.zeros((len(self.all_anchors_grid), 4))
        self.ref_anchors[:, 2:] = np.array(self.all_anchors_grid)

    def forward(self, output, labels):

        device = output.device
        batch_size = output.shape[0]
        grid_size = output.shape[2]
        n_ch = 5 + self.n_classes
        ref_anchors = torch.tensor(self.ref_anchors, device=device)

        x_shift = torch.arange(0, grid_size, device=device).unsqueeze(0).repeat(grid_size, 1).repeat(batch_size, 3, 1, 1)
        y_shift = torch.arange(0, grid_size, device=device).unsqueeze(0).repeat(grid_size, 1).permute(1, 0).repeat(batch_size, 3, 1, 1)

        masked_anchors = torch.tensor(self.masked_anchors, device=device)
        w_anchors = masked_anchors[:, 0].reshape(1, 3, 1, 1)
        h_anchors = masked_anchors[:, 1].reshape(1, 3, 1, 1)

        pred = output.detach().clone()
        pred[..., np.r_[:2, 4:n_ch]] = torch.sigmoid(pred[..., np.r_[:2, 4:n_ch]])
        pred[..., 0] += x_shift
        pred[..., 1] += y_shift
        pred[..., 2] = torch.exp(pred[..., 2]) * w_anchors
        pred[..., 3] = torch.exp(pred[..., 3]) * h_anchors

        pred = pred[..., :4]

        # target assignment
        tgt_mask = torch.zeros(batch_size, self.n_anchors, grid_size, grid_size, 4 + self.n_classes, device=device)
        obj_mask = torch.ones(batch_size, self.n_anchors, grid_size, grid_size, device=device)
        tgt_scale = torch.zeros(batch_size, self.n_anchors, grid_size, grid_size, 2, device=device)
        target = torch.zeros(batch_size, self.n_anchors, grid_size, grid_size, n_ch, device=device)

        n_label = (labels.sum(dim=2) > 0).sum(dim=1)  # number of objects

        truth_x_all = labels[:, :, 1] * grid_size
        truth_y_all = labels[:, :, 2] * grid_size
        truth_w_all = labels[:, :, 3] * grid_size
        truth_h_all = labels[:, :, 4] * grid_size
        truth_i_all = truth_x_all.to(torch.int16)
        truth_j_all = truth_y_all.to(torch.int16)

        # Creating the object mask and the target mask
        for b in range(batch_size):
            n = int(n_label[b])
            if n == 0:
                continue
            truth_box = torch.zeros(n, 4, device=device)
            truth_box[:n, 2] = truth_w_all[b, :n]
            truth_box[:n, 3] = truth_h_all[b, :n]
            truth_i = truth_i_all[b, :n]
            truth_j = truth_j_all[b, :n]

            # calculate iou between truth and reference anchors
            anchor_ious_all = bboxes_iou(truth_box, ref_anchors, xyxy=False)
            best_n_all = torch.argmax(anchor_ious_all, dim=1)
            best_n = best_n_all % 3
            best_n_mask = ((best_n_all == self.anch_mask[0]) | (best_n_all == self.anch_mask[1]) | (best_n_all == self.anch_mask[2]))

            truth_box[:n, 0] = truth_x_all[b, :n]
            truth_box[:n, 1] = truth_y_all[b, :n]

            pred_ious = bboxes_iou(pred[b].reshape(-1, 4), truth_box, xyxy=False)
            pred_best_iou, _ = pred_ious.max(dim=1)
            pred_best_iou = (pred_best_iou > self.ignore_threshold)
            pred_best_iou = pred_best_iou.view(pred[b].shape[:3])
            # set mask to zero (ignore) if pred matches truth
            obj_mask[b] = ~pred_best_iou

            if best_n_mask.sum() == 0:
                continue

            # Creating targets from labels:
            for ti in range(best_n.shape[0]):
                if best_n_mask[ti] == 1:
                    i, j = truth_i[ti], truth_j[ti]
                    a = best_n[ti]
                    obj_mask[b, a, j, i] = 1
                    tgt_mask[b, a, j, i, :] = 1
                    target[b, a, j, i, 0] = truth_x_all[b, ti] - truth_x_all[b, ti].to(torch.int16).to(torch.float)
                    target[b, a, j, i, 1] = truth_y_all[b, ti] - truth_y_all[b, ti].to(torch.int16).to(torch.float)
                    target[b, a, j, i, 2] = torch.log(truth_w_all[b, ti] / masked_anchors[best_n[ti], 0] + 1e-16)
                    target[b, a, j, i, 3] = torch.log(truth_h_all[b, ti] / masked_anchors[best_n[ti], 1] + 1e-16)
                    target[b, a, j, i, 4] = 1
                    target[b, a, j, i, 5 + labels[b, ti, 0].to(torch.int16)] = 1
                    tgt_scale[b, a, j, i, :] = torch.sqrt(2 - truth_w_all[b, ti] * truth_h_all[b, ti] / grid_size / grid_size)

        # loss calculation
        obj_mask = obj_mask.bool()
        tgt_mask = tgt_mask.bool()

        output[..., 2:4] *= tgt_scale  # Scale with anchors
        out_obj = output[..., 4][obj_mask]
        out_xy = output[..., 0:2][tgt_mask[..., 0:2]]
        out_wh = output[..., 2:4][tgt_mask[..., 2:4]]
        out_cls = output[..., np.r_[5:n_ch]][tgt_mask[..., 4:n_ch]]

        target[..., 2:4] *= tgt_scale  # Scale with anchors
        targ_obj = target[..., 4][obj_mask]
        targ_xy = target[..., 0:2][tgt_mask[..., 0:2]]
        targ_wh = target[..., 2:4][tgt_mask[..., 2:4]]
        targ_cls = target[..., np.r_[5:n_ch]][tgt_mask[..., 4:n_ch]]

        loss_xy = self.bce_loss(out_xy, targ_xy)
        loss_wh = self.l2_loss(out_wh, targ_wh) / 2
        loss_obj = self.bce_loss(out_obj, targ_obj)
        loss_cls = self.bce_loss(out_cls, targ_cls)

        loss = loss_xy + loss_wh + loss_obj + loss_cls
        return loss


class YoloLoss(nn.Module):

    def __init__(self, ignore_threshold):
        super(YoloLoss, self).__init__()
        self.yolo_loss_s0 = YoloScaleLoss(scale=0, ignore_threshold=ignore_threshold)
        self.yolo_loss_s1 = YoloScaleLoss(scale=1, ignore_threshold=ignore_threshold)
        self.yolo_loss_s2 = YoloScaleLoss(scale=2, ignore_threshold=ignore_threshold)

    def forward(self, yp, y):
        loss_s0 = self.yolo_loss_s0(yp[0], y)
        loss_s1 = self.yolo_loss_s1(yp[1], y)
        loss_s2 = self.yolo_loss_s2(yp[2], y)
        loss = loss_s0 + loss_s1 + loss_s2
        return loss
