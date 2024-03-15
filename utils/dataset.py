from torchvision.datasets import CocoDetection
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image, ImageFile
import os
import torch
from tqdm import tqdm

ImageFile.LOAD_TRUNCATED_IMAGES = True

scale = [13, 26, 52]
num_scales = 3
num_anchors_per_scale = 3
ignore_iou_threshold = 0.5


class CustomDataset(CocoDetection):

    def __init__(self, root: str, annFile, transform=None, target_transform=None, transforms=None):
        super(CocoDetection, self).__init__(root, transforms, transform, target_transform)
        from pycocotools.coco import COCO
        self.coco = COCO(annFile)
        self.ids = []
        self.ids = list(sorted(self.coco.imgs.keys()))

    def __getitem__(self, index):

        img_id = self.ids[index]
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        labels = self.coco.loadAnns(ann_ids)
        path = self.coco.loadImgs(img_id)[0]['file_name']

        image = np.array(Image.open(os.path.join(self.root, path)).convert('RGB'))
        bboxes = [x['bbox'] + [x['category_id']] for x in labels if x['area'] != 0.0]
        img_w, img_h = float(image.shape[0]), float(image.shape[1])

        transforms = self.transforms(image=image, bboxes=bboxes)
        x = transforms['image']
        bboxes = transforms['bboxes']

        y = torch.zeros(50, 5)
        num_boxes = y.shape[0] if y.shape[0] <= len(bboxes) else len(bboxes)
        if bboxes:
            bboxes = torch.tensor(bboxes)
            bboxes[:, 0:2] = bboxes[:, 0:2] + (bboxes[:, 2:4] * 0.5)
            bboxes[:, 0:4] = bboxes[:, 0:4] / x.shape[1]  # Normalize using image size
            bboxes = bboxes.roll(1, 1)
            y[:num_boxes] = bboxes[:num_boxes]

        meta = torch.tensor((img_id, img_h, img_w))

        return x, y, meta


def main():

    coco = "/t/home/peter.lengyel/private/coco_2017/"
    root_train = coco + "images/train2017/"
    annFile_train = coco + "annotations/instances_train2017.json"

    import custom_transforms
    train_transforms = custom_transforms.get_train_transforms(image_size=416)
    train_dataset = CustomDataset(root=root_train, annFile=annFile_train, transforms=train_transforms)
    train_loader = DataLoader(dataset=train_dataset, batch_size=4, shuffle=True, num_workers=0, drop_last=True, pin_memory=True)

    for x, y, z in tqdm(train_loader):
        pass


if __name__ == "__main__":
    main()
