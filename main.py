import torch.nn
import numpy as np
import time
from tqdm import tqdm
import os
import sys
import argparse
import random
import shutil

import torch.optim as optim

from utils.dataset import CustomDataset
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image
from yolo import YoloV3
from loss import YoloLoss
from utils.schedulers import get_lr_scheduler

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from utils.utils import save_model
from utils.utils import get_yp_boxes
from utils.utils import unletterbox
from utils.utils import denormalize
from utils.utils import draw_y_on_x
from utils.utils import draw_yp_on_x
from utils.utils import get_learning_rate

from utils import arguments
from utils import custom_transforms


def validate(epoch, model, yolo_loss, val_loader, writer, args):
    print("Validate...")

    model.eval()
    loss_val_sum = 0.0
    cycle_limit = max(int(len(val_loader) * args.val_ratio), 1)
    val_iter = iter(val_loader)
    with torch.no_grad():
        for _ in range(cycle_limit):
            x, y, z = next(val_iter)
            x = x.to(args.device)
            y = y.to(args.device)
            yp = model(x)
            loss = yolo_loss(yp, y)
            loss_val_sum += loss.item()

    loss_val = loss_val_sum / cycle_limit
    writer.add_scalar("Loss/Val", loss_val, epoch)


def metrics(epoch, model, val_loader, writer, anchors, args):
    print("Metrics...")

    coco_gt = COCO(args.val_annotations_path)
    model.eval()
    cycle_limit = max(int(len(val_loader) * args.metric_ratio), 1)
    val_iter = iter(val_loader)
    all_yp_boxes = []

    with torch.no_grad():
        for _ in range(cycle_limit):
            x, y, z = next(val_iter)
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
    imgIds = imgIds[0:cycle_limit]
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


def visualize(model, val_loader, anchors, args):
    print("Visualize...")

    model.eval()
    cycle_limit = max(int(len(val_loader) * args.visualize_ratio), 1)
    val_iter = iter(val_loader)
    with torch.no_grad():
        for _ in range(cycle_limit):
            x, y, z = next(val_iter)
            x = x.to(args.device)
            yp = model(x)

            yp_render = (yp[0].cpu().detach().clone(), yp[1].cpu().detach().clone(), yp[2].cpu().detach().clone())
            x_render = x.cpu().detach().clone()
            x_render = denormalize(x_render) * 255
            draw_y_on_x(x_render, y, args.classes)
            draw_yp_on_x(x_render, yp_render, z, args.confidence_threshold, args.iou_threshold, anchors, args.classes)
            # Save batch grid as image
            image_dir = args.logs_path + "/images"
            image_dir_exists = os.path.exists(image_dir)
            if not image_dir_exists:
                os.makedirs(image_dir)
            img_name = str(image_dir) + "/" + str(z[0][0].item()) + ".png"
            save_image(x_render / 255, img_name)


def main(args):

    train_transforms = custom_transforms.get_train_transforms(image_size=args.train_img_size)
    train_dataset = CustomDataset(root=args.train_images_path, annFile=args.train_annotations_path, transforms=train_transforms)
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True, pin_memory=True)
    val_transforms = custom_transforms.get_val_transforms(image_size=args.val_img_size)
    val_dataset = CustomDataset(root=args.val_images_path, annFile=args.val_annotations_path, transforms=val_transforms)
    val_loader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=False, num_workers=0, drop_last=True, pin_memory=True)

    model = YoloV3(args=args.yolo).to(args.device)
    yolo_loss = YoloLoss(ignore_threshold=args.ignore_threshold).to(args.device)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    scaler = torch.cuda.amp.GradScaler()
    scheduler = get_lr_scheduler(optimizer=optimizer, args=args.scheduler)
    writer = SummaryWriter(log_dir=args.logs_path)
    anchors = torch.tensor(args.anchors)
    current_best_metric = 0.0

    model.train()
    train_iter = iter(train_loader)
    for idx in tqdm(range(1, args.steps + 1)):

        model.zero_grad()
        for minibatch in range(args.subdivisions):

            try:
                x, y, z = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                x, y, z = next(train_iter)

            x = x.to(args.device)
            y = y.to(args.device)

            with torch.cuda.amp.autocast():
                yp = model(x)
                loss = yolo_loss(yp, y)
                scaler.scale(loss).backward()

        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        writer.add_scalar("Loss/Train", loss.item(), idx)
        writer.add_scalar("Loss/Lr", get_learning_rate(optimizer), idx)

        if args.resize_frequency != 0 and idx % args.resize_frequency == 0:
            imgsize = (random.randint(0, 9) % 10 + 10) * 32
            train_transforms = custom_transforms.get_train_transforms(image_size=imgsize)
            train_dataset.transforms = train_transforms
            train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True, pin_memory=True)
            train_iter = iter(train_loader)

        if idx % args.evaluate_frequency == 0:
            model.eval()
            validate(idx, model, yolo_loss, val_loader, writer, args)
            visualize(model, val_loader, anchors, args)
            checkpoint_metric = metrics(idx, model, val_loader, writer, anchors, args)
            if checkpoint_metric > current_best_metric:
                print("Saving model at map:", checkpoint_metric)
                current_best_metric = checkpoint_metric
                save_model(model, optimizer, args)
            model.train()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-gpu", "--gpu", default="cuda:0", help="Used GPU Id")
    real_args = parser.parse_args()
    print("Using GPU:", real_args.gpu)
    os.chdir(sys.path[0])
    args = arguments.parse_config("./args.yaml")
    logs_dir = os.path.exists(args.logs_path)
    if not logs_dir:
        os.makedirs(args.logs_path)
    shutil.copyfile("./args.yaml", args.logs_path + "/args.yaml")
    main(args)
