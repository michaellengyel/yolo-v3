import torch.nn
import torch.onnx

import config
from yolo import YoloV3


def main():

    # model = YoloV3(num_classes=config.NUM_CLASSES)

    from yolov3_delete import YOLOv3
    import yaml
    with open("../yolov3_default_delete.cfg", 'r') as f:
        cfg = yaml.load(f)
    model = YOLOv3(cfg['MODEL'], ignore_thre=0.7)

    x = torch.randn((1, 3, 416, 416))
    torch.onnx.export(model, x,
                      "super_resolution_them.onnx",
                      export_params=True,
                      opset_version=11,
                      do_constant_folding=True,
                      input_names = ['input'],
                      output_names = ['output'],
                      dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}})


if __name__ == "__main__":

    main()
