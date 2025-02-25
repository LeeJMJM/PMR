#!/usr/bin/python
# -*- coding:utf-8 -*-
from torch import nn
import warnings

# ----------------------------inputsize >=28----------------------------------


class Identity(nn.Module):  # Construct a layer for CAM operations
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class DCNN(nn.Module):
    def __init__(self, pretrained=False, in_channel=1, out_channel=10):
        super(DCNN, self).__init__()
        if pretrained is True:
            warnings.warn("Pretrained model is not available")

        self.layer1 = nn.Sequential(
            nn.Conv1d(in_channel, 16, kernel_size=60, stride=4),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True))

        self.layer2 = nn.Sequential(
            nn.Conv1d(16, 32, kernel_size=36, stride=2),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2))

        self.layer3 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=3),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True))

        self.layer4 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),

            Identity(),  # do nothing, only for easy CAM operations

            nn.AdaptiveMaxPool1d(4))

        self.layer5 = nn.Sequential(
            nn.Linear(128 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True))
        self.fc = nn.Linear(64, out_channel)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = x.view(x.size(0), -1)
        x = self.layer5(x)
        x = self.fc(x)

        return x
