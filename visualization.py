import torch.nn
from tqdm import tqdm
import os
import sys

import torch.optim as optim
from utils.dataset import CustomDataset
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from yolo import YoloV3

from utils.utils import denormalize
from utils.utils import draw_y_on_x
from utils.utils import draw_yp_on_x
from utils.utils import load_checkpoint

from utils import arguments
from utils import custom_transforms


def visualize(model, val_loader, anchors, args, path):

    model.eval()
    cycle_limit = 1000
    val_iter = iter(val_loader)
    with torch.no_grad():
        for _ in tqdm(range(cycle_limit)):
            x, y, z = next(val_iter)
            x = x.to(args.device)
            yp = model(x)

            yp_render = (yp[0].cpu().detach().clone(), yp[1].cpu().detach().clone(), yp[2].cpu().detach().clone())
            x_render = x.cpu().detach().clone()
            x_render = denormalize(x_render) * 255
            draw_y_on_x(x_render, y, args.classes)
            draw_yp_on_x(x_render, yp_render, z, args.visualization_threshold, args.iou_threshold, anchors, args.classes)
            # Save batch grid as image
            image_dir = path
            image_dir_exists = os.path.exists(image_dir)
            if not image_dir_exists:
                os.makedirs(image_dir)
            img_name = str(image_dir) + "/" + str(z[0][0].item()) + ".png"
            save_image(x_render / 255, img_name)


def main(args):

    log_path = "/t/home/peter.lengyel/private/logs/images"
    weights_path = "/t/home/peter.lengyel/private/logs/230411-22:57:14_baseline_rgb_and_reshape/save/checkpoint_0_5022.pth.tar"

    val_transforms = custom_transforms.get_val_transforms(image_size=416)
    val_dataset = CustomDataset(root=args.val_images_path, annFile=args.val_annotations_path, transforms=val_transforms)
    val_loader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=False, num_workers=0, drop_last=True, pin_memory=True)

    model = YoloV3(args=args.yolo).to(args.device)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    anchors = torch.tensor(args.anchors).to(args.device)
    load_checkpoint(weights_path, model, optimizer, args.device)
    visualize(model, val_loader, anchors, args, log_path)


if __name__ == "__main__":

    os.chdir(sys.path[0])
    args = arguments.parse_config("./args.yaml")
    main(args)
