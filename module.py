import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class Tnet(nn.Module):
   def __init__(self, k=3):
      super(Tnet, self).__init__()
      self.conv1 = nn.Conv1d(3, 64, 1)
      self.conv2 = nn.Conv1d(64, 128, 1)
      self.conv3 = nn.Conv1d(128, 1024, 1)
      self.fc1 = nn.Linear(1024, 512)
      self.fc2 = nn.Linear(512, 256)
      self.fc3 = nn.Linear(256, 3*3)
      self.relu = nn.ReLU()


      self.bn1 = nn.BatchNorm1d(64)
      self.bn2 = nn.BatchNorm1d(128)
      self.bn3 = nn.BatchNorm1d(1024)
      self.bn4 = nn.BatchNorm1d(512)
      self.bn5 = nn.BatchNorm1d(256)
       

   def forward(self, input):
      # input.shape == (output,n,3)
      output = input.size(0)
      mod = F.relu(self.bn1(self.conv1(input)))
      mod = F.relu(self.bn2(self.conv2(mod)))
      mod = F.relu(self.bn3(self.conv3(mod)))
      pool = nn.MaxPool1d(mod.size(-1))(mod)
      flat = nn.Flatten(1)(pool)
      mod = F.relu(self.bn4(self.fc1(flat)))
      mod = F.relu(self.bn5(self.fc2(mod)))
      
      #initialize as identity
      init = torch.eye(self.k, requires_grad=True).repeat(bs,1,1)
      if mod.is_cuda:
        init=init.cuda()
      matrix = self.fc3(mod).view(-1,self.k,self.k) + init
      return matrix


class Transform(nn.Module):
   def __init__(self):
        super().__init__()
        self.input_transform = Tnet(k=3)
        self.feature_transform = Tnet(k=64)
        self.conv1 = nn.Conv1d(3,64,1)

        self.conv2 = nn.Conv1d(64,128,1)
        self.conv3 = nn.Conv1d(128,1024,1)
       

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
       
   def forward(self, input):
        matrix3x3 = self.input_transform(input)
        # batch matrix multiplication
        mod = torch.bmm(torch.transpose(input,1,2), matrix3x3).transpose(1,2)

        mod = F.relu(self.bn1(self.conv1(mod)))

        matrix64x64 = self.feature_transform(mod)
        mod = torch.bmm(torch.transpose(mod,1,2), matrix64x64).transpose(1,2)

        mod = F.relu(self.bn2(self.conv2(mod)))
        mod = self.bn3(self.conv3(mod))
        mod = nn.MaxPool1d(mod.size(-1))(mod)
        output = nn.Flatten(1)(mod)
        return output, matrix3x3, matrix64x64

class PointNet(nn.Module):
    def __init__(self, classes = 10):
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

    def pointnetloss(outputs, labels, m3x3, m64x64, alpha = 0.0001):
        criterion = torch.nn.NLLLoss()
        new_output=outputs.size(0)
        id3x3 = torch.eye(3, requires_grad=True).repeat(new_output,1,1)
        id64x64 = torch.eye(64, requires_grad=True).repeat(new_output,1,1)
        if outputs.is_cuda:
          id3x3=id3x3.cuda()
          id64x64=id64x64.cuda()
        diff3x3 = id3x3-torch.bmm(m3x3,m3x3.transpose(1,2))
        diff64x64 = id64x64-torch.bmm(m64x64,m64x64.transpose(1,2))
        return criterion(outputs, labels) + alpha * (torch.norm(diff3x3)+torch.norm(diff64x64)) / float(bs)
