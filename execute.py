import copy

import torch.nn
from tqdm import tqdm
import os
import sys
from PIL import Image
import numpy as np

import torch.optim as optim
from utils.dataset import CustomDataset
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from yolo import YoloV3

from utils.utils import denormalize
from utils.utils import draw_y_on_x
from utils.utils import draw_yp_on_x
from utils.utils import get_yp_boxes
from utils.utils import unletterbox

from utils import arguments
from utils import custom_transforms

import albumentations as A
import cv2
from albumentations.pytorch import ToTensorV2

from torchvision.utils import draw_bounding_boxes


def get_image_paths(input_path):
    image_paths = []
    file_list = os.listdir(input_path)
    for filename in file_list:
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
            image_path = os.path.join(input_path, filename)
            image_paths.append(image_path)
    return image_paths


def preprocess_image(image_path, image_size):

    image = np.array(Image.open(image_path).convert('RGB'))

    tf = A.Compose([
        A.LongestMaxSize(max_size=image_size),
        A.PadIfNeeded(min_height=int(image_size), min_width=int(image_size), border_mode=cv2.BORDER_CONSTANT, value=(124, 116, 104)),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0),
        ToTensorV2()
    ])

    x = tf(image=image)['image'].unsqueeze(0)
    z = torch.tensor((0, image.shape[1], image.shape[0])).unsqueeze(0)
    image = torch.tensor(image)
    return x, image, z


def load_checkpoint(checkpoint_path, model, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    # Remapping variable name from older models
    for key in list(checkpoint['state_dict'].keys()):
        checkpoint['state_dict'][key.replace('darknet', "encoder")] = checkpoint['state_dict'].pop(key)
    model.load_state_dict(checkpoint["state_dict"])


def main(args):

    input_path = '/t/home/peter.lengyel/private/yolo_inputs'
    output_path = '/t/home/peter.lengyel/private/yolo_outputs'
    weights_path = "/t/home/peter.lengyel/private/yolo_v3/logs/230515-12:57:20_less_aug_cutout_cos_mult_2/save/checkpoint.pth.tar"

    image_paths = get_image_paths(input_path)
    model = YoloV3(args=args.yolo).to(args.device).eval()
    load_checkpoint(weights_path, model, args.device)
    anchors = torch.tensor(args.anchors).to(args.device)

    with torch.no_grad():
        for idx, image_path in tqdm(enumerate(image_paths)):
            x, image, z = preprocess_image(image_path=image_path, image_size=args.val_img_size)
            x = x.to(args.device)
            yp = model(x)
            yp = (yp[0].cpu().detach().clone(), yp[1].cpu().detach().clone(), yp[2].cpu().detach().clone())
            yp_boxes = get_yp_boxes(x, yp, z, args.visualization_threshold, args.iou_threshold, anchors)
            yp_boxes = unletterbox(x, yp_boxes, z)
            yp_boxes[:, 1:3] = yp_boxes[:, 1:3] - yp_boxes[:, 3:5] * 0.5
            yp_boxes[:, 3:5] = yp_boxes[:, 1:3] + yp_boxes[:, 3:5]

            boxes = yp_boxes.tolist()
            labels = [args.classes[int(x[6])] + " " + str(int(x[5] * 100)) + "%" for x in boxes]
            x_render = draw_bounding_boxes(image=image.type(torch.uint8).permute(2, 0, 1), boxes=yp_boxes[:, 1:5], labels=labels, width=2)

            image_dir_exists = os.path.exists(output_path)
            if not image_dir_exists:
                os.makedirs(output_path)
            img_name = str(output_path) + "/" + image_path.split('/')[-1][:-4] + ".png"
            save_image(x_render / 255, img_name)


if __name__ == "__main__":

    os.chdir(sys.path[0])
    args = arguments.parse_config("./args.yaml")
    main(args)
