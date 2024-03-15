import yaml
import torch.nn
from dataclasses import dataclass
from datetime import date
import time


@dataclass
class Args:
    data_path: None
    train_images_path: None
    train_annotations_path: None
    val_images_path: None
    val_annotations_path: None
    output_path: None
    logs_path: None
    batch_size: None
    subdivisions: None
    num_workers: None
    epochs: None
    steps: None
    device: None
    train_ratio: None
    val_ratio: None
    metric_ratio: None
    visualize_ratio: None
    learning_rate: None
    weight_decay: None
    unfreeze_step: None
    continue_training: None
    weights_path: None
    epoch_num: None
    visualization_threshold: None
    confidence_threshold: None
    iou_threshold: None
    evaluate_frequency: None
    resize_frequency: None
    train_img_size: None
    val_img_size: None
    num_classes: None
    ignore_threshold: None
    scheduler: None
    yolo: None
    anchors: None
    classes: None


def parse_config(path):
    with open(path, "r") as stream:
        try:
            data_loaded = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    args = Args
    args.data_path = data_loaded['data_path']
    args.train_images_path = data_loaded['data_path'] + "/images/train2017/"
    args.train_annotations_path = data_loaded['data_path'] + "/annotations/instances_train2017.json"
    args.val_images_path = data_loaded['data_path'] + "/images/val2017/"
    args.val_annotations_path = data_loaded['data_path'] + "/annotations/instances_val2017.json"
    args.output_path = data_loaded['output_path']
    args.continue_training = data_loaded['continue_training']

    if data_loaded['continue_training']:
        args.logs_path = data_loaded['checkpoint']
        args.weights_path = data_loaded['checkpoint'] + "/save/checkpoint.pth.tar"
        args.epoch_num = data_loaded['epoch_num']
    else:
        args.logs_path = data_loaded['logs_path'] + "/" + str(date.today().strftime("%y%m%d")) + str(time.strftime("-%H:%M:%S", time.localtime())) + "_" + data_loaded['name']
        args.epoch_num = 0

    args.batch_size = data_loaded['batch_size']
    args.subdivisions = data_loaded['subdivisions']
    args.num_workers = data_loaded['num_workers']
    args.epochs = data_loaded['epochs']
    args.steps = data_loaded['steps']
    args.device = data_loaded['device'] if torch.cuda.is_available() else "cpu"
    args.train_ratio = data_loaded['train_ratio']
    args.val_ratio = data_loaded['val_ratio']
    args.metric_ratio = data_loaded['metric_ratio']
    args.visualize_ratio = data_loaded['visualize_ratio']
    args.learning_rate = data_loaded['learning_rate']
    args.weight_decay = data_loaded['weight_decay']
    args.unfreeze_step = data_loaded['unfreeze_step']
    args.visualization_threshold = data_loaded['visualization_threshold']
    args.confidence_threshold = data_loaded['confidence_threshold']
    args.iou_threshold = data_loaded['iou_threshold']
    args.evaluate_frequency = data_loaded['evaluate_frequency']
    args.resize_frequency = data_loaded['resize_frequency']
    args.train_img_size = data_loaded['train_img_size']
    args.val_img_size = data_loaded['val_img_size']
    args.num_classes = data_loaded['num_classes']
    args.ignore_threshold = data_loaded['ignore_threshold']
    args.scheduler = data_loaded['scheduler']
    args.scheduler["learning_rate"] = data_loaded['learning_rate']
    args.yolo = data_loaded['yolo']
    args.anchors = data_loaded['anchors']
    args.classes = data_loaded['classes']

    return args
