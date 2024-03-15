import torch.nn
import time
from tqdm import tqdm
import os
import sys

import torch.optim as optim
from utils.dataset import CustomDataset
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from yolo import YoloV3

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from utils.utils import get_yp_boxes
from utils.utils import unletterbox
from utils.utils import load_checkpoint

from utils import arguments
from utils import custom_transforms


def metrics(epoch, model, val_loader, writer, coco_gt, anchors, args):

    model.eval()
    cycle_limit = len(val_loader)
    val_iter = iter(val_loader)
    all_yp_boxes = []

    with torch.no_grad():
        for _ in tqdm(range(cycle_limit)):
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


def main(args):

    log_path = "./logs/Benchmark_LambdaLR_0_01"
    weights_path = "./logs/240224-21:17:13_LambdaLR/save/checkpoint.pth.tar"

    val_transforms = custom_transforms.get_val_transforms(image_size=416)
    val_dataset = CustomDataset(root=args.val_images_path, annFile=args.val_annotations_path, transforms=val_transforms)
    val_loader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=False, num_workers=0, drop_last=True, pin_memory=True)

    coco_gt = COCO(args.val_annotations_path)
    model = YoloV3(args=args.yolo).to(args.device)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    writer = SummaryWriter(log_dir=log_path)
    anchors = torch.tensor(args.anchors).to(args.device)

    load_checkpoint(weights_path, model, optimizer, args.device)
    metrics(0, model, val_loader, writer, coco_gt, anchors, args)


if __name__ == "__main__":

    os.chdir(sys.path[0])
    args = arguments.parse_config("./args.yaml")
    main(args)
