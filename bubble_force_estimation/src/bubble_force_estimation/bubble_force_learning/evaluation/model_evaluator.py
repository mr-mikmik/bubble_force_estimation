import numpy as np
import os
import torch
from torch.utils.data import DataLoader

from bubble_force_estimation.bubble_force_learning.models.deformation_mean_model import DeformationMeanModel
from bubble_force_estimation.bubble_force_learning.models.deformation_model import DeformationOnlyModel, DeformationWithReferenceModel, DeformationAndPointsModel
from bubble_force_estimation.bubble_force_learning.models.img2force_model import BubbleImage2ForceModel
from bubble_force_estimation.bubble_force_learning.models.optical_flow_model import OpticalFlowModel
from bubble_force_estimation.bubble_force_learning.models.optical_flow_mean_model import OpticalFlowMeanModel


class DatasetEvaluator(object):

    def __init__(self, dataset, data_name, Models, versions):
        self.dataset = dataset
        self.data_name = data_name
        self.Models = Models
        self.versions = versions

        self.models = self._load_models()


    def _load_models(self):
        models = []
        for i, Model in enumerate(self.Models):
            version_i = self.versions[i]
            model_i = self._load_model(Model, version_i, self.data_name)
            models.append(model_i)
        return models

    def _load_model(self, Model, load_version, data_path):
        model_name = Model.get_name()
        version_chkp_path = os.path.join(data_path, 'tb_logs', '{}'.format(model_name),
                                         'version_{}'.format(load_version), 'checkpoints')
        checkpoints_fs = [f for f in os.listdir(version_chkp_path) if
                          os.path.isfile(os.path.join(version_chkp_path, f))]
        checkpoint_path = os.path.join(version_chkp_path, checkpoints_fs[0])
        model = Model.load_from_checkpoint(checkpoint_path)
        model.freeze()
        return model

    def _evaluate_model(self, model):
        evaluate_loader = DataLoader(self.dataset, batch_size=len(self.dataset),
                                     num_workers=8)  # use all dataset to evaluate it
        # TODO: Check if the model is logging, avoid to log anything
        losses = []
        for b_i, eval_batch in enumerate(evaluate_loader):
            loss_i = model.validation_step(eval_batch, batch_idx=0)
            losses.append(loss_i.item())
        loss = np.mean(losses)
        return loss

    def evaluate(self):
        scores = {}
        for i, model_i in enumerate(self.models):
            name_i = model_i.get_name()
            score_i = self._evaluate_model(model_i)
            scores[name_i] = score_i
        return scores


# Test:

if __name__ == '__main__':
    from bubble_force_estimation.bubble_force_learning.datasets.bubble_force_dataset import BubbleForceDataset2StatesWithFixedNumberDeformations
    data_name = '/home/mik/Desktop/bubble_force_generalization_data'
    model_load_path = '/home/mik/Desktop/bubble_force_data' 

    scene_names = ['r10', 'r7p5', 'r5', 'r2p5', 'cr2p5r10', 'cr2p5r15', 'cr5r15'] # TODO: Add more scene names
    scene_scores = {}
    for scene_name in scene_names:
        dataset = BubbleForceDataset2StatesWithFixedNumberDeformations(data_name=data_name, num_deformations=100, scene_name=scene_name, dtype=torch.float32)# TODO Fill the values
        model_dict = {
            BubbleImage2ForceModel: 0,
            OpticalFlowMeanModel: 3,
            OpticalFlowModel: 0,
            DeformationMeanModel: 2,
            DeformationOnlyModel: 0,
            DeformationWithReferenceModel: 4,
            DeformationAndPointsModel: 0,
        }

        de = DatasetEvaluator(dataset, model_load_path, list(model_dict.keys()), list(model_dict.values()))
        scores = de.evaluate()
        print('\n{} SCORES:'.format(scene_name))
        for k, v in scores.items():
            print('\t{}: {}'.format(k, v))
        scene_scores[scene_name] = scores

