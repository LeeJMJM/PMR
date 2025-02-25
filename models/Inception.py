# coding=utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicConv1d(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv1d, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, **kwargs, bias=False)
        self.bn = nn.BatchNorm1d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)


class InceptionA(nn.Module):

    def __init__(self, in_channels, pool_features):
        super(InceptionA, self).__init__()
        self.branch1x1 = BasicConv1d(in_channels, 64, kernel_size=1)

        self.branch5x5_1 = BasicConv1d(in_channels, 48, kernel_size=1)
        self.branch5x5_2 = BasicConv1d(48, 64, kernel_size=5, padding=2)

        self.branch3x3dbl_1 = BasicConv1d(in_channels, 64, kernel_size=1)
        self.branch3x3dbl_2 = BasicConv1d(64, 96, kernel_size=3, padding=1)
        self.branch3x3dbl_3 = BasicConv1d(96, 96, kernel_size=3, padding=1)

        self.branch_pool = BasicConv1d(in_channels,
                                       pool_features,
                                       kernel_size=1)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        branch_pool = F.avg_pool1d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch5x5, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)


class InceptionB(nn.Module):

    def __init__(self, in_channels):
        super(InceptionB, self).__init__()
        self.branch3x3 = BasicConv1d(in_channels, 384, kernel_size=3, stride=2)
        self.branch3x3dbl_1 = BasicConv1d(in_channels, 64, kernel_size=1)
        self.branch3x3dbl_2 = BasicConv1d(64, 96, kernel_size=3, padding=1)
        self.branch3x3dbl_3 = BasicConv1d(96, 96, kernel_size=3, stride=2)

    def forward(self, x):
        branch3x3 = self.branch3x3(x)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        branch_pool = F.max_pool1d(x, kernel_size=3, stride=2)

        outputs = [branch3x3, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)


class Identity(nn.Module):  # Construct a layer for CAM operations
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class Inception(nn.Module):
    def __init__(self, in_channel, out_channel=10):
        super(Inception, self).__init__()
        self.conv = self.conv1 = nn.Conv1d(in_channel, 32,
                                           kernel_size=61,
                                           stride=3,
                                           padding=1,
                                           bias=False)
        self.bn = nn.BatchNorm1d(32, eps=0.001)
        self.Conv1d_2a_3x3 = BasicConv1d(32, 32, kernel_size=14)
        self.Conv1d_2b_3x3 = BasicConv1d(32, 64, kernel_size=3, padding=1)
        self.Conv1d_3b_1x1 = BasicConv1d(64, 80, kernel_size=1)
        self.Conv1d_4a_3x3 = BasicConv1d(80, 192, kernel_size=3)
        self.Mixed_5b = InceptionA(192, pool_features=32)
        self.Mixed_5c = InceptionA(256, pool_features=64)
        self.Mixed_5d = InceptionA(288, pool_features=64)
        self.Mixed_6a = InceptionB(288)

        self.Iden = Identity()  # do nothing, only for easy CAM operations

        self.fc = nn.Linear(768, out_channel)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.Conv1d_2a_3x3(x)
        x = self.Conv1d_2b_3x3(x)
        x = F.max_pool1d(x, kernel_size=3, stride=2)
        x = self.Conv1d_3b_1x1(x)
        x = self.Conv1d_4a_3x3(x)
        x = F.max_pool1d(x, kernel_size=3, stride=2)
        x = self.Mixed_5b(x)
        x = self.Mixed_5c(x)
        x = self.Mixed_5d(x)
        x = self.Mixed_6a(x)
        x = self.Iden(x)
        x = nn.AdaptiveMaxPool1d(1)(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
