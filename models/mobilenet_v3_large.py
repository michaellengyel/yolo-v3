import torch
import torch.nn as nn
import torchvision.models as models


class MobileNetLarge(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = models.mobilenet_v3_large(pretrained=True)
        self.modules = list(self.model.children())[0:1]  # delete the last fc layer.
        self.backbone_nn = nn.Sequential(*self.modules[0])
        self.branch_channels = (960, 112, 40)  # s_13, s_26, s_52

    def forward(self, x):

        s_26, s_52 = None, None

        for i, module in enumerate(self.backbone_nn.children()):
            x = module(x)
            if i == 6:
                s_52 = x.clone()
            elif i == 12:
                s_26 = x.clone()

        return x, s_26, s_52


def main():

    model = MobileNetLarge()
    model.eval()
    x = torch.randn((16, 3, 416, 416))
    print(model.parameters())

    yp = model(x)
    print(len(yp))
    print(yp[0].shape)
    print(yp[1].shape)
    print(yp[2].shape)


if __name__ == '__main__':
    main()
