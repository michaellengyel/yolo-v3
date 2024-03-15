import albumentations as A
import cv2
from albumentations.pytorch import ToTensorV2


def get_train_transforms(image_size):

    train_transforms = [

        A.MultiplicativeNoise(p=0.1),
        A.CLAHE(p=0.1),
        A.Posterize(p=0.1),
        A.ToGray(p=0.05),
        A.RandomBrightnessContrast(p=0.1),
        A.Cutout(num_holes=8, max_h_size=40, max_w_size=40, fill_value=(124, 116, 104), always_apply=False, p=0.1),

        A.HorizontalFlip(p=0.5),
        A.Affine(translate_percent=(-0.25, 0.25), rotate=(-5, 5), scale=(0.5, 1.5), shear=(-5, 5), interpolation=cv2.INTER_AREA, fit_output=False, cval=(124, 116, 104), p=1.0),
        A.Resize(height=image_size, width=image_size),

        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0),
        ToTensorV2()
    ]
    return A.Compose(train_transforms, bbox_params=A.BboxParams(format="coco", min_visibility=0.5, label_fields=[], min_area=1))


def get_val_transforms(image_size):

    val_transforms = [
        A.LongestMaxSize(max_size=image_size),
        A.PadIfNeeded(min_height=int(image_size), min_width=int(image_size), border_mode=cv2.BORDER_CONSTANT, value=(124, 116, 104)),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0),
        ToTensorV2(),
    ]
    return A.Compose(val_transforms, bbox_params=A.BboxParams(format="coco", min_visibility=0.0, label_fields=[], min_area=1))
