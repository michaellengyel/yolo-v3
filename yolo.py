import torch
import torch.nn as nn

from models.darknet import Darknet
from models.mobilenet_v3_small import MobileNetSmall
from models.mobilenet_v3_large import MobileNetLarge


class DBL(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(DBL, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.leakyrelu = nn.LeakyReLU(0.1)

    def forward(self, x):
        return self.leakyrelu(self.batchnorm(self.conv(x)))


class YoloV3(nn.Module):
    def __init__(self, args):
        super().__init__()

        # Initialize encoder based on config
        if args["encoder"] == "darknet":
            self.encoder = Darknet("models/config/darknet.cfg")
            self.encoder.load_darknet_weights("models/weights/darknet53.conv.74")
        elif args["encoder"] == "mobilenet_v3_small":
            self.encoder = MobileNetSmall()
        elif args["encoder"] == "mobilenet_v3_large":
            self.encoder = MobileNetLarge()
        else:
            assert False, "Invalid Encoder set in config file!"

        self.num_classes = args["num_classes"]
        s_13_c, s_26_c, s_52_c = self.encoder.branch_channels

        # Conv set 1
        self.dbl_1_1 = DBL(s_13_c, s_13_c//2, kernel_size=1, stride=1, padding=0)
        self.dbl_1_2 = DBL(s_13_c//2, s_13_c, kernel_size=3, stride=1, padding=1)
        self.dbl_1_3 = DBL(s_13_c, s_13_c//2, kernel_size=1, stride=1, padding=0)
        self.dbl_1_4 = DBL(s_13_c//2, s_13_c, kernel_size=3, stride=1, padding=1)
        self.dbl_1_5 = DBL(s_13_c, s_13_c//2, kernel_size=1, stride=1, padding=0)

        # Scale branch
        self.dbl_s1 = DBL(s_13_c//2, s_13_c, kernel_size=3, stride=1, padding=1)
        self.conv_s1 = nn.Conv2d(s_13_c, (self.num_classes + 5) * 3, kernel_size=1, stride=1, padding=0)

        # Main branch
        self.dbl_m_1 = DBL(s_13_c//2, s_13_c//4, kernel_size=1, stride=1, padding=0)
        self.upsample_1 = nn.Upsample(scale_factor=2)

        # Conv set 2
        self.dbl_2_1 = DBL(s_13_c//4 + s_26_c, s_13_c//4, kernel_size=1, stride=1, padding=0)
        self.dbl_2_2 = DBL(s_13_c//4, s_13_c//2, kernel_size=3, stride=1, padding=1)
        self.dbl_2_3 = DBL(s_13_c//2, s_13_c//4, kernel_size=1, stride=1, padding=0)
        self.dbl_2_4 = DBL(s_13_c//4, s_13_c//2, kernel_size=3, stride=1, padding=1)
        self.dbl_2_5 = DBL(s_13_c//2, s_13_c//4, kernel_size=1, stride=1, padding=0)

        # Scale branch
        self.dbl_s2 = DBL(s_13_c//4, s_13_c//2, kernel_size=3, stride=1, padding=1)
        self.conv_s2 = nn.Conv2d(s_13_c//2, (self.num_classes + 5) * 3, kernel_size=3, stride=1, padding=1)

        # Main branch
        self.dbl_m_2 = DBL(s_13_c//4, s_13_c//8, kernel_size=1, stride=1, padding=0)
        self.upsample_2 = nn.Upsample(scale_factor=2)

        # Conv set 3
        self.dbl_3_1 = DBL(s_13_c//8 + s_52_c, s_13_c//8, kernel_size=1, stride=1, padding=0)
        self.dbl_3_2 = DBL(s_13_c//8, s_13_c//4, kernel_size=3, stride=1, padding=1)
        self.dbl_3_3 = DBL(s_13_c//4, s_13_c//8, kernel_size=1, stride=1, padding=0)
        self.dbl_3_4 = DBL(s_13_c//8, s_13_c//4, kernel_size=3, stride=1, padding=1)
        self.dbl_3_5 = DBL(s_13_c//4, s_13_c//8, kernel_size=1, stride=1, padding=0)

        # Scale branch
        self.dbl_s3 = DBL(s_13_c//8, s_13_c//4, kernel_size=3, stride=1, padding=1)
        self.conv_s3 = nn.Conv2d(s_13_c//4, (self.num_classes + 5) * 3, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        rc_13, rc_26, rc_52 = self.encoder(x)

        x = self.dbl_1_1(rc_13)
        x = self.dbl_1_2(x)
        x = self.dbl_1_3(x)
        x = self.dbl_1_4(x)
        x = self.dbl_1_5(x)

        # Scale branch
        s_13 = self.dbl_s1(x)
        s_13 = self.conv_s1(s_13)
        s_13 = s_13.reshape(s_13.shape[0], 3, self.num_classes + 5, s_13.shape[2], s_13.shape[3]).permute(0, 1, 3, 4, 2)

        # Main branch
        x = self.dbl_m_1(x)
        x = self.upsample_1(x)
        x = torch.cat([x, rc_26], dim=1)

        x = self.dbl_2_1(x)
        x = self.dbl_2_2(x)
        x = self.dbl_2_3(x)
        x = self.dbl_2_4(x)
        x = self.dbl_2_5(x)

        # Scale branch
        s_26 = self.dbl_s2(x)
        s_26 = self.conv_s2(s_26)
        s_26 = s_26.reshape(s_26.shape[0], 3, self.num_classes + 5, s_26.shape[2], s_26.shape[3]).permute(0, 1, 3, 4, 2)

        # Main branch
        x = self.dbl_m_2(x)
        x = self.upsample_2(x)
        x = torch.cat([x, rc_52], dim=1)

        x = self.dbl_3_1(x)
        x = self.dbl_3_2(x)
        x = self.dbl_3_3(x)
        x = self.dbl_3_4(x)
        x = self.dbl_3_5(x)

        # Scale branch
        s_52 = self.dbl_s3(x)
        s_52 = self.conv_s3(s_52)
        s_52 = s_52.reshape(s_52.shape[0], 3, self.num_classes + 5, s_52.shape[2], s_52.shape[3]).permute(0, 1, 3, 4, 2)

        return s_13, s_26, s_52

    def encoder_requires_grad(self, requires_grad=True):
        print("Encoder Requires Grad: True") if requires_grad else print("Encoder Requires Grad: False")
        for param in self.encoder.parameters():
            param.requires_grad = requires_grad


def main():

    yolo = {
        "pretrained": True,
        "num_classes": 91,
        "encoder": "darknet",
        "darknet_params": {"config_path": "models/config/darknet.cfg",
                           "weights_path": "models/weights/darknet53.conv.74"}
    }

    model = YoloV3(args=yolo)
    model.encoder_requires_grad(False)
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
