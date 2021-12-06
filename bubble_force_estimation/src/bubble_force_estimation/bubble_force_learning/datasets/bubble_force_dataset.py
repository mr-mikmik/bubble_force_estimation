import numpy as np
import torch

from bubble_utils.bubble_datasets.bubble_dataset_base import BubbleDatasetBase
from bubble_force_estimation.optical_flow.optical_flow import optical_flow, mean_optical_flow, optical_flow_pyr
from bubble_force_estimation.optical_flow.compute_deformations import compute_deformations


class BubbleForceDatasetBase(BubbleDatasetBase):

    def __init__(self, *args, wrench_frame='med_base', flow_method='pyramidal', **kwargs):
        self.wrench_frame = wrench_frame
        self.flow_method = flow_method
        super().__init__(*args, **kwargs)

    @classmethod
    def get_name(self):
        return 'bubble_force_dataset'

    def _load_bubble_sensor(self, fc, scene_name, key=''):
        if key != '':
            key = '{}_'.format(key)

        bubble_sample = {
            '{}color_img_r'.format(key) : self._load_color_img(fc=fc, scene_name=scene_name, camera_name='right'),
            '{}color_img_l'.format(key) : self._load_color_img(fc=fc, scene_name=scene_name, camera_name='left'),
            '{}depth_img_r'.format(key) : self._load_depth_img(fc=fc, scene_name=scene_name, camera_name='right'),
            '{}depth_img_l'.format(key) : self._load_depth_img(fc=fc, scene_name=scene_name, camera_name='left'),
            '{}wrench_ext'.format(key) : self._get_wrench(fc=fc, scene_name=scene_name, frame_id=self.wrench_frame, wrench_name='external_wrenches'),
            '{}wrench_robot'.format(key) : self._get_wrench(fc=fc, scene_name=scene_name, frame_id=self.wrench_frame, wrench_name='wrenches')
        # We could add more things like tf information
        }
        return bubble_sample

    def _add_flow(self, sample):
        flow_functions = {
            'pyramidal': optical_flow_pyr,
            'basic': optical_flow
        }
        if not self.flow_method in flow_functions:
            raise NotImplementedError('Flow method {} not yet implemented. Only: {}'.format(self.flow_method, flow_functions.keys()))
        flow_function = flow_functions[self.flow_method]
        optical_flow_r = np.stack(flow_function(sample['def_color_img_r'], sample['undef_color_img_r']))
        optical_flow_l = np.stack(flow_function(sample['def_color_img_l'], sample['undef_color_img_l']))
        sample['optical_flow_r'] = optical_flow_r
        sample['optical_flow_mean_r'] = np.mean(optical_flow_r, axis=(1, 2))
        sample['optical_flow_l'] = optical_flow_l
        sample['optical_flow_mean_l'] = np.mean(optical_flow_l, axis=(1, 2))
        return sample

    def _add_deformations(self, sample):
        # compute deformations
        deformations_r, points_ref_r, points_def_r = compute_deformations(
            img_1=sample['undef_color_img_r'],
            img_2=sample['def_color_img_r'],
            depth_1=sample['undef_depth_img_r'],
            depth_2=sample['def_depth_img_r'],
            flow=sample['optical_flow_r'],
            camera_matrix=sample['camera_info_depth_r']['K'],
            return_point_coordinates=True,
        )
        deformations_l, points_ref_l, points_def_l = compute_deformations(
            img_1=sample['undef_color_img_l'],
            img_2=sample['def_color_img_l'],
            depth_1=sample['undef_depth_img_l'],
            depth_2=sample['def_depth_img_l'],
            flow=sample['optical_flow_l'],
            camera_matrix=sample['camera_info_depth_l']['K'],
            return_point_coordinates=True,
        )

        # add the deformations to the sample
        sample['deformations_r'] = deformations_r
        sample['points_ref_r'] = points_ref_r
        sample['points_def_r'] = points_def_r
        sample['deformations_mean_r'] = np.mean(deformations_r, axis=0)
        sample['deformations_l'] = deformations_l
        sample['points_ref_l'] = points_ref_l
        sample['points_def_l'] = points_def_l
        sample['deformations_mean_l'] = np.mean(deformations_l, axis=0)
        return sample


class BubbleForceDataset3States(BubbleForceDatasetBase):
    """
    A sample contains reference state (undeformed), initial state, and final state.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def get_name(self):
        return 'bubble_force_dataset_3states'

    def _get_sample(self, indx):
        # fc: index of the line in the datalegend (self.dl) of the sample
        dl_line = self.dl.iloc[indx]
        scene_name = dl_line['Scene']
        undef_fc = dl_line['UndeformedFC']
        init_fc = dl_line['InitialStateFC']
        final_fc = dl_line['FinalStateFC']

        undef_sample = self._load_bubble_sensor(undef_fc, scene_name=scene_name, key='undef')
        init_sample = self._load_bubble_sensor(init_fc, scene_name=scene_name, key='init')
        final_sample = self._load_bubble_sensor(final_fc, scene_name=scene_name, key='final')

        sample = {
            'camera_info_depth_r': self._load_camera_info_depth(scene_name, camera_name='right'),
            'camera_info_depth_l': self._load_camera_info_depth(scene_name, camera_name='left'),
        }
        sample.update(undef_sample)
        sample.update(init_sample)
        sample.update(final_sample)

        return sample


class BubbleForceDataset2States(BubbleForceDatasetBase):
    """
    A sample contains reference state (undeformed), and deformed.
    Here we split the initial state, and final state. as two seperate samples.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _get_filecodes(self):
        # duplicate the filecodes:
        fcs = np.arange(2*len(super()._get_filecodes()))
        return fcs

    @classmethod
    def get_name(cls):
        return 'bubble_force_dataset_2states'

    def _get_sample(self, indx):
        # fc: index of the line in the datalegend (self.dl) of the sample
        true_indx = indx // 2
        dl_line = self.dl.iloc[true_indx]
        scene_name = dl_line['Scene']
        undef_fc = dl_line['UndeformedFC']
        if indx % 2 == 0:
            def_fc = dl_line['InitialStateFC']
        else:
            def_fc = dl_line['FinalStateFC']

        undef_sample = self._load_bubble_sensor(undef_fc, scene_name=scene_name, key='undef')
        def_sample = self._load_bubble_sensor(def_fc, scene_name=scene_name, key='def')

        sample = {
            'camera_info_depth_r': self._load_camera_info_depth(scene_name, camera_name='right'),
            'camera_info_depth_l': self._load_camera_info_depth(scene_name, camera_name='left'),
        }
        sample.update(undef_sample)
        sample.update(def_sample)
        # Add optical flow:
        sample = self._add_flow(sample)
        sample = self._add_deformations(sample)
        return sample





# DEBUG:
if __name__ == '__main__':
    data_name = '/home/mmint/Desktop/bubble_force_data'
    dataset = BubbleForceDataset2States(data_name=data_name, dtype=torch.float)
    print(len(dataset))
    d0 = dataset[0]