from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.dataset import CustomDataset
from torchvision.utils import make_grid
from torchvision.utils import draw_bounding_boxes
import matplotlib.pyplot as plt
from utils.utils import denormalize
from utils import custom_transforms
import torch

print(plt.get_backend())


def plot_batch(x, y, imgsize):
    x = denormalize(x)
    imgs = [x[i] for i in range(x.shape[0])]

    boxes = torch.zeros((y.shape[0], 50, 4))
    boxes[:, :, 0] = y[:, :, 1] - y[:, :, 3] / 2
    boxes[:, :, 1] = y[:, :, 2] - y[:, :, 4] / 2
    boxes[:, :, 2] = y[:, :, 1] + y[:, :, 3] / 2
    boxes[:, :, 3] = y[:, :, 2] + y[:, :, 4] / 2
    boxes = boxes * imgsize
    boxes = [boxes[i] for i in range(boxes.shape[0])]
    imgs_boxes = [draw_bounding_boxes((imgs[i] * 255).to(torch.uint8), boxes[i]) for i in range(y.shape[0])]

    imgs = make_grid(imgs_boxes, nrow=4)
    fig, ax = plt.subplots(figsize=(16, 16))
    ax.imshow(imgs.numpy().transpose((1, 2, 0)))
    plt.tight_layout()
    plt.show()


def main():

    coco = "/t/home/peter.lengyel/private/coco_2017/"
    root_train = coco + "images/train2017/"
    annFile_train = coco + "annotations/instances_train2017.json"
    root_val = coco + "images/val2017/"
    annFile_val = coco + "annotations/instances_val2017.json"

    train_transforms = custom_transforms.get_train_transforms(image_size=416)
    train_dataset = CustomDataset(root=root_train, annFile=annFile_train, transforms=train_transforms)
    train_loader = DataLoader(dataset=train_dataset, batch_size=16, shuffle=False, num_workers=0, drop_last=True, pin_memory=True)

    val_transforms = custom_transforms.get_val_transforms(image_size=416)
    val_dataset = CustomDataset(root=root_val, annFile=annFile_val, transforms=val_transforms)
    val_loader = DataLoader(dataset=val_dataset, batch_size=4, shuffle=False, num_workers=0, drop_last=True, pin_memory=True)

    for _ in range(100):
        x, y, z = next(iter(train_loader))
        imgsize = 416
        plot_batch(x, y, imgsize)

    for x, y, z in tqdm(val_loader):
        imgsize = 416
        plot_batch(x, y, imgsize)


if __name__ == "__main__":
    main()