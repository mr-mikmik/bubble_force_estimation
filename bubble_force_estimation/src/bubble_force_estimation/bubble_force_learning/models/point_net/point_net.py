import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from bubble_force_estimation.bubble_force_learning.models.point_net.t_net import Tnet
from bubble_force_estimation.bubble_force_learning.models.point_net.transform_net import Transform


class PointNet(nn.Module):
    def __init__(self, classes=10):
        super().__init__()
        self.transform = Transform()
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, classes)
        

        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.dropout = nn.Dropout(p=0.3)
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, input):
        mod, matrix3x3, matrix64x64 = self.transform(input)
        mod = F.relu(self.bn1(self.fc1(mod)))
        mod = F.relu(self.bn2(self.dropout(self.fc2(mod))))
        output = self.fc3(mod)
        return self.logsoftmax(output), matrix3x3, matrix64x64

    def pointnetloss(outputs, labels, m3x3, m64x64, alpha=0.0001):
        criterion = torch.nn.NLLLoss()
        new_output=outputs.size(0)
        id3x3 = torch.eye(3, requires_grad=True).repeat(new_output, 1, 1)
        id64x64 = torch.eye(64, requires_grad=True).repeat(new_output, 1, 1)
        if outputs.is_cuda:
          id3x3 = id3x3.cuda()
          id64x64 = id64x64.cuda()
        diff3x3 = id3x3-torch.bmm(m3x3, m3x3.transpose(1, 2))
        diff64x64 = id64x64-torch.bmm(m64x64, m64x64.transpose(1, 2))
        return criterion(outputs, labels) + alpha * (torch.norm(diff3x3)+torch.norm(diff64x64)) / float(bs)
