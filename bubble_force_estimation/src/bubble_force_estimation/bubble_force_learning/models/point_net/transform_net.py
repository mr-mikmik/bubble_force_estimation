import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from bubble_force_estimation.bubble_force_learning.models.point_net.t_net import Tnet


class Transform(nn.Module):
    def __init__(self, num_in_features=3):
        super().__init__()
        self.num_in_features = num_in_features
        self.input_transform = Tnet(k=3)
        self.feature_transform = Tnet(k=64)

        self.conv1 = nn.Conv1d(self.num_in_features, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)

    def forward(self, input):
        matrix3x3 = self.input_transform(input)
        # batch matrix multiplication
        mod = torch.bmm(torch.transpose(input, 1, 2), matrix3x3).transpose(1, 2)

        mod = F.relu(self.bn1(self.conv1(mod)))

        matrix64x64 = self.feature_transform(mod)
        mod = torch.bmm(torch.transpose(mod, 1, 2), matrix64x64).transpose(1, 2)

        mod = F.relu(self.bn2(self.conv2(mod)))
        mod = self.bn3(self.conv3(mod))
        mod = nn.MaxPool1d(mod.size(-1))(mod)
        output = nn.Flatten(1)(mod)
        return output, matrix3x3, matrix64x64
