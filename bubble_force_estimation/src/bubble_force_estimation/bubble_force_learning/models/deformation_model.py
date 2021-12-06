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

from bubble_control.bubble_learning.models.aux.fc_module import FCModule
from bubble_force_estimation.bubble_force_learning.models.point_net.point_net import RegressionPointNet


class DeformationModelBase(pl.LightningModule):
    """Model composed by a fc network that takes the mean of the optical flow and predicts the wrench"""
    def __init__(self, input_sizes, lr=1e-4, dataset_params=None, activation='relu'):
        super().__init__()
        self.input_sizes = input_sizes
        self.lr = lr
        self.dataset_params = dataset_params
        self.activation = activation

        self.spring_model = self._get_spring_model()

        self.mse_loss = nn.MSELoss()

        self.save_hyperparameters()

    @classmethod
    def get_name(cls):
        return 'deformation_model'

    @property
    def name(self):
        return self.get_name()

    def _get_spring_model(self):
        sizes = self._get_sizes()
        spring_model_pointnet = RegressionPointNet(num_in_features=sizes['deformations'][-1],
                                                  out_size=sizes['def_wrench_ext'])
        return spring_model_pointnet

    def forward(self, deformations):
        predicted_wrench = self.spring_model(deformations)
        return predicted_wrench

    def _get_sizes(self):
        sizes = self.input_sizes
        sizes['def_wrench_ext'] = np.prod(sizes['def_wrench_ext'])
        return sizes

    def _get_deformations(self, batch):
        raise NotImplementedError('Not implemented. You need to extend this method')
        pass

    def _step(self, batch, batch_idx, phase='train'):
        wrench_ext_gth = batch['def_wrench_ext']
        # Query the model:
        deformations = self._get_deformations(batch)
        wrench_ext_pred = self.forward(deformations)
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


class DeformationOnlyModel(DeformationModelBase):
    """
    Only takes the deformations, without refernce points
    """
    def __init__(*args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def get_name(cls):
        return 'deformation_only_model'

    def _get_deformations(self, batch):
        deformations_r = batch['deformations_r']
        deformations_l = batch['deformations_l']
        deformations = torch.concatenate([deformations_r, deformations_l], axis=-2)
        return deformations

    def _get_sizes(self):
        sizes = super()._get_sizes()
        sizes['deformations'] = self.input_sizes['deformations_r']
        return sizes


class DeformationWithReferenceModel(DeformationModelBase):
    """
    Only takes the deformations, without refernce points
    """
    def __init__(*args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def get_name(cls):
        return 'deformation_with_reference_model'

    def _get_deformations(self, batch):
        deformations_r = batch['deformations_r']
        deformations_l = batch['deformations_l']
        points_ref_r = batch['points_ref_r']
        points_ref_l = batch['points_ref_l']
        def_r = torch.concatenate([deformations_r, points_ref_r], axis=-1)
        def_l = torch.concatenate([deformations_l, points_ref_l], axis=-1)
        deformations = torch.concatenate([def_r, def_l], axis=-2)
        return deformations

    def _get_sizes(self):
        sizes = super()._get_sizes()
        sizes['deformations'] = self.input_sizes['deformations_r'] + self.input_sizes['points_ref_r']
        return sizes


class DeformationAndPointsModel(DeformationModelBase):
    """
    Only takes the deformations, without refernce points
    """
    def __init__(*args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def get_name(cls):
        return 'deformation_and_points_model'

    def _get_deformations(self, batch):
        deformations_r = batch['deformations_r']
        deformations_l = batch['deformations_l']
        points_ref_r = batch['points_ref_r']
        points_ref_l = batch['points_ref_l']
        points_def_r = batch['points_def_r']
        points_def_l = batch['points_def_l']
        def_r = torch.concatenate([deformations_r, points_ref_r, points_def_r], axis=-1)
        def_l = torch.concatenate([deformations_l, points_ref_l, points_def_l], axis=-1)
        deformations = torch.concatenate([def_r, def_l], axis=-2)
        return deformations

    def _get_sizes(self):
        sizes = super()._get_sizes()
        sizes['deformations'] = self.input_sizes['deformations_r'] + self.input_sizes['points_ref_r'] + self.input_sizes['points_def_r']
        return sizes