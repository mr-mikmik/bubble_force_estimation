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
from bubble_drawing.bubble_learning.models.aux.img_encoder import ImageEncoder


class BubbleImage2ForceModel(pl.LightningModule):
    """Model composed by a fc network that takes the mean of the optical flow and predicts the wrench"""
    def __init__(self, input_sizes, num_fcs=3, num_convs=3, conv_hidden_sizes=None, fc_h_dim=100, ks=3, skip_layers=None, lr=1e-4, dataset_params=None, activation='relu'):
        super().__init__()
        self.input_sizes = input_sizes
        self.num_fcs = num_fcs
        self.num_convs = num_convs
        self.conv_hidden_sizes = conv_hidden_sizes
        self.fc_h_dim = fc_h_dim
        self.ks = ks
        self.skip_layers = skip_layers
        self.lr = lr
        self.dataset_params = dataset_params
        self.activation = activation

        self.spring_model = self._get_spring_model()

        self.mse_loss = nn.MSELoss()

        self.save_hyperparameters()

    @classmethod
    def get_name(cls):
        return 'image2force_model'

    @property
    def name(self):
        return self.get_name()

    def _get_spring_model(self):
        sizes = self._get_sizes()
        img_size = sizes['bubble_comb_img']  # (C_in, W_in, H_in)
        out_size = sizes['def_wrench_ext']
        spring_model = ImageEncoder(input_size=img_size,
                                   latent_size=out_size,
                                   num_convs=self.num_convs,
                                   conv_h_sizes=self.conv_hidden_sizes,
                                   ks=self.ks,
                                   num_fcs=self.num_fcs,
                                   fc_hidden_size=self.fc_h_dim,
                                   activation=self.activation)
        return spring_model

    def forward(self, img_r, img_l):
        bubble_comb_img = torch.stack([img_r, img_l], dim=-3)
        predicted_wrench = self.spring_model(bubble_comb_img)
        return predicted_wrench

    def _get_sizes(self):
        sizes = self.input_sizes
        sizes['def_wrench_ext'] = np.prod(sizes['def_wrench_ext'])
        r_sizes = sizes['def_color_img_r']
        l_sizes = sizes['def_color_img_l']
        combined_channels = 2
        sizes['bubble_comb_img'] = np.insert(r_sizes[-2:], 0, combined_channels)
        return sizes

    def _step(self, batch, batch_idx, phase='train'):
        def_img_r = batch['def_color_img_r']
        def_img_l = batch['def_color_img_l']
        wrench_ext_gth = batch['def_wrench_ext']
        # Query the model:
        wrench_ext_pred = self.forward(def_img_r, def_img_l)
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


