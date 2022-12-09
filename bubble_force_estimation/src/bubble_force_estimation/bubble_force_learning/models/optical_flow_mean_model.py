import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import matplotlib.pyplot as plt
from matplotlib import cm
import torchvision
import numpy as np
import os
import sys

from bubble_drawing.bubble_learning.models.aux.fc_module import FCModule


class OpticalFlowMeanModel(pl.LightningModule):
    """Model composed by a fc network that takes the mean of the optical flow and predicts the wrench"""
    def __init__(self, input_sizes, num_fcs=3, fc_h_dim=100, skip_layers=None, lr=1e-4, dataset_params=None, activation='relu'):
        super().__init__()
        self.input_sizes = input_sizes
        self.num_fcs = num_fcs
        self.fc_h_dim = fc_h_dim
        self.skip_layers = skip_layers
        self.lr = lr
        self.dataset_params = dataset_params
        self.activation = activation

        self.spring_model = self._get_spring_model()
        self.mse_loss = nn.MSELoss()

        self.save_hyperparameters()

    @classmethod
    def get_name(cls):
        return 'optical_flow_mean_model'

    @property
    def name(self):
        return self.get_name()

    def _get_spring_model(self):
        sizes = self._get_sizes()
        in_size = sizes['optical_flow_mean_r'] + sizes['optical_flow_mean_l']
        out_size = sizes['def_wrench_ext']
        model_sizes = [in_size] + [self.fc_h_dim] * self.num_fcs + [out_size]
        spring_model = FCModule(sizes=model_sizes, skip_layers=self.skip_layers, activation=self.activation)
        return spring_model

    def forward(self, optical_flow_mean_r, optical_flow_mean_l):
        optical_flow_mean = torch.cat([optical_flow_mean_r, optical_flow_mean_l], dim=-1)
        predicted_wrench = self.spring_model(optical_flow_mean)
        return predicted_wrench

    def _get_sizes(self):
        sizes = self.input_sizes
        sizes['def_wrench_ext'] = np.prod(sizes['def_wrench_ext'])
        return sizes

    def _step(self, batch, batch_idx, phase='train'):
        optical_flow_mean_r = batch['optical_flow_mean_r']
        optical_flow_mean_l = batch['optical_flow_mean_l']
        wrench_ext_gth = batch['def_wrench_ext']
        # Query the model:
        wrench_ext_pred = self.forward(optical_flow_mean_r, optical_flow_mean_l)
        # Compute loss:
        loss = self._compute_loss(wrench_ext_pred, wrench_ext_gth)

        self.log('{}_batch'.format(phase), batch_idx)
        self.log('{}_loss'.format(phase), loss)

        return loss

    def training_step(self, train_batch, batch_idx):
        loss = self._step(train_batch, batch_idx, phase='train')
        return loss

    def validation_step(self, val_batch, batch_idx):
        loss = self._step(val_batch, batch_idx, phase='val')
        return loss

    def _compute_loss(self, wrench_ext_pred, wrench_ext_gth):
        wrench_prediction_loss = self.mse_loss(wrench_ext_pred, wrench_ext_gth.squeeze())
        loss = wrench_prediction_loss
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

