import torch.nn
import numpy as np
import time
from tqdm import tqdm
import os
import sys
import argparse

import torch.optim as optim
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from utils.utils import get_learning_rate

from utils import arguments
from utils.schedulers import get_lr_scheduler


class MockNet(nn.Module):
    def __init__(self):
        super(MockNet, self).__init__()
        self.conv = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3)
        self.batchnorm = nn.BatchNorm2d(num_features=3)
        self.leakyrelu = nn.LeakyReLU(0.1)

    def forward(self, x):
        return self.leakyrelu(self.batchnorm(self.conv(x)))


def main(args):

    model = MockNet()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = get_lr_scheduler(optimizer=optimizer, args=args.scheduler)
    writer = SummaryWriter(log_dir=args.logs_path)

    for i in tqdm(range(1, args.steps + 1)):

        writer.add_scalar("Loss/Lr", get_learning_rate(optimizer), i)
        scheduler.step()
        if i % args.evaluate_frequency == 0:
            print("MODEL EVAL")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-gpu", "--gpu", help="Used GPU Id")
    real_args = parser.parse_args()
    print("Using GPU:", real_args.gpu)

    os.chdir(sys.path[0])
    args = arguments.parse_config("../args.yaml")
    main(args)
