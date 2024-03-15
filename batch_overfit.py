import torch.nn
import numpy as np
import time
from tqdm import tqdm
import os
import sys
import argparse
import shutil

import torch.optim as optim
from utils.dataset import CustomDataset
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image
from yolo import YoloV3
from loss import YoloLoss

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from utils.utils import save_model
from utils.utils import get_yp_boxes
from utils.utils import unletterbox
from utils.utils import denormalize
from utils.utils import draw_y_on_x
from utils.utils import draw_yp_on_x

from utils import arguments
from utils import custom_transforms


def train(epoch, model, train_loader, scaler, optimizer, writer, args):

    test_cycles = 50
    loss_train_sum = 0.0
    model.train()

    yolo_loss = YoloLoss(ignore_threshold=args.ignore_threshold).to(args.device)

    for idx in tqdm(range(test_cycles)):

        model.zero_grad()
        x, y, z = next(iter(train_loader))
        x = x.to(args.device)
        y = y.to(args.device)

        # with torch.cuda.amp.autocast():
        yp = model(x)
        loss = yolo_loss(yp, y)
        loss.backward()
        optimizer.step()
        # scaler.scale(loss).backward()
        # scaler.step(optimizer)
        # scaler.update()

        loss_train_sum += loss.item()

    loss_train = loss_train_sum / test_cycles
    writer.add_scalar("Loss/Train", loss_train, epoch)


def validate(epoch, model, val_loader, scheduler, writer, args):

    yolo_loss = YoloLoss(ignore_threshold=args.ignore_threshold).to(args.device)

    test_cycles = args.batch_size
    loss_val_sum = 0.0
    model.eval()
    with torch.no_grad():
        for i, (x, y, z) in enumerate(tqdm(val_loader)):

            if i == test_cycles:
                break

            x = x.to(args.device)
            y = y.to(args.device)
            yp = model(x)
            loss = yolo_loss(yp, y)
            loss_val_sum += loss.item()

    loss_val = loss_val_sum / test_cycles
    # scheduler.step(loss_val)
    scheduler.step(epoch)
    writer.add_scalar("Loss/Val", loss_val, epoch)
    writer.add_scalar("Loss/Lr", scheduler.optimizer.param_groups[0]['lr'], epoch)


def metrics(epoch, model, val_loader, writer, anchors, args):

    coco_gt = COCO(args.val_annotations_path)
    test_cycles = args.batch_size
    model.eval()
    all_yp_boxes = []

    with torch.no_grad():
        for i, (x, y, z) in enumerate(tqdm(val_loader)):
            if i == test_cycles:
                break
            x = x.to(args.device)
            yp = model(x)
            yp = (yp[0].cpu().detach().clone(), yp[1].cpu().detach().clone(), yp[2].cpu().detach().clone())
            yp_boxes = get_yp_boxes(x, yp, z, args.confidence_threshold, args.iou_threshold, anchors)
            if yp_boxes is None:
                continue
            yp_boxes = unletterbox(x, yp_boxes, z)
            yp_boxes[:, 1:3] = yp_boxes[:, 1:3] - yp_boxes[:, 3:5] * 0.5  # Formate change
            all_yp_boxes.append(yp_boxes)

    all_yp_boxes = torch.cat(all_yp_boxes, dim=0)
    bboxes = all_yp_boxes.cpu().numpy()

    bboxes = np.array(bboxes)
    coco_dt = coco_gt.loadRes(bboxes)
    imgIds = sorted(coco_gt.getImgIds())
    imgIds = imgIds[0:args.batch_size]
    coco_eval = COCOeval(coco_gt, coco_dt, iouType='bbox')
    coco_eval.params.imgIds = imgIds
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    writer.add_scalar("Precision/(AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ]", coco_eval.stats[0], epoch)
    writer.add_scalar("Precision/(AP) @[ IoU=0.50      | area=   all | maxDets=100 ]", coco_eval.stats[1], epoch)
    writer.add_scalar("Precision/(AP) @[ IoU=0.75      | area=   all | maxDets=100 ]", coco_eval.stats[2], epoch)
    writer.add_scalar("Precision/(AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ]", coco_eval.stats[3], epoch)
    writer.add_scalar("Precision/(AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ]", coco_eval.stats[4], epoch)
    writer.add_scalar("Precision/(AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ]", coco_eval.stats[5], epoch)
    writer.add_scalar("Recall/(AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ]", coco_eval.stats[6], epoch)
    writer.add_scalar("Recall/(AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ]", coco_eval.stats[7], epoch)
    writer.add_scalar("Recall/(AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ]", coco_eval.stats[8], epoch)
    writer.add_scalar("Recall/(AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ]", coco_eval.stats[9], epoch)
    writer.add_scalar("Recall/(AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ]", coco_eval.stats[10], epoch)
    writer.add_scalar("Recall/(AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ]", coco_eval.stats[11], epoch)

    time.sleep(1)

    return coco_eval.stats[1]


def visualize(model, val_loader, scaled_anchors, args):

    test_cycles = args.batch_size
    model.eval()

    with torch.no_grad():
        for i, (x, y, z) in enumerate(tqdm(val_loader)):

            if i == test_cycles:
                break

            x = x.to(args.device)
            yp = model(x)

            yp_render = (yp[0].cpu().detach().clone(), yp[1].cpu().detach().clone(), yp[2].cpu().detach().clone())
            x_render = x.cpu().detach().clone()
            x_render = denormalize(x_render) * 255
            draw_y_on_x(x_render, y, args.classes)
            draw_yp_on_x(x_render, yp_render, z, args.visualization_threshold, args.iou_threshold, scaled_anchors, args.classes)
            # Save batch grid as image
            image_dir = args.logs_path + "/images"
            image_dir_exists = os.path.exists(image_dir)
            if not image_dir_exists:
                os.makedirs(image_dir)
            img_name = str(image_dir) + "/" + str(int(z[0][0].item())) + ".png"
            save_image(x_render / 255, img_name)


def main(args):

    val_transforms = custom_transforms.get_val_transforms(image_size=416)
    train_dataset = CustomDataset(root=args.val_images_path, annFile=args.val_annotations_path, transforms=val_transforms)
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, drop_last=True, pin_memory=True)
    val_transforms = custom_transforms.get_val_transforms(image_size=416)
    val_dataset = CustomDataset(root=args.val_images_path, annFile=args.val_annotations_path, transforms=val_transforms)
    val_loader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=False, num_workers=0, drop_last=True, pin_memory=True)

    # Personal
    model = YoloV3(args=args.yolo).to(args.device)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 60], gamma=0.1)
    scaler = torch.cuda.amp.GradScaler()
    writer = SummaryWriter(log_dir=args.logs_path)
    anchors = torch.tensor(args.anchors)
    current_best_metric = 0.0

    for epoch in range(args.epoch_num, args.epochs, 1):

        train(epoch, model, train_loader, scaler, optimizer, writer, args)
        validate(epoch, model, val_loader, scheduler, writer, args)
        visualize(model, val_loader, anchors, args)
        checkpoint_metric = metrics(epoch, model, val_loader, writer, anchors, args)
        if checkpoint_metric > current_best_metric:
            print("Saving model at map:", checkpoint_metric)
            current_best_metric = checkpoint_metric
            save_model(model, optimizer, args)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-gpu", "--gpu", help="Used GPU Id")
    real_args = parser.parse_args()
    print("Using GPU:", real_args.gpu)

    os.chdir(sys.path[0])
    args = arguments.parse_config("./args.yaml")
    logs_dir = os.path.exists(args.logs_path)
    if not logs_dir:
        os.makedirs(args.logs_path)
    shutil.copyfile("./args.yaml", args.logs_path + "/args.yaml")
    main(args)
