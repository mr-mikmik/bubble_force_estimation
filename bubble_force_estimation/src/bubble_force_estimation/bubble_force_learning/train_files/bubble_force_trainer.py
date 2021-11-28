import torch


from bubble_control.bubble_learning.train_files.parsed_trainer import ParsedTrainer


from bubble_force_estimation.bubble_force_learning.models.optical_flow_mean_model import OpticalFlowMeanModel
from bubble_force_estimation.bubble_force_learning.datasets.bubble_force_dataset import BubbleForceDataset2States


if __name__ == '__main__':

    # params:
    default_params = {
        'data_name' : '/home/mmint/Desktop/bubble_force_data',
        'max_epochs' : 1500,
        'num_fcs' : 2,
        'fc_h_dim' : 50,
        'dtype': torch.float32,
        'model': OpticalFlowMeanModel.get_name()
    }
    default_types = {
        'batch_size': int,
        'val_batch_size': int
    }
    Model = [OpticalFlowMeanModel]
    Dataset = [BubbleForceDataset2States]
    parsed_trainer = ParsedTrainer(Model, Dataset, default_args=default_params, default_types=default_types)

    parsed_trainer.train()
