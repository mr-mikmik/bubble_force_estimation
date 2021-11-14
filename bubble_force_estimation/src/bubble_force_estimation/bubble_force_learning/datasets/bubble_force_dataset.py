import numpy as np

from bubble_utils.bubble_datasets.bubble_dataset_base import BubbleDatasetBase


class BubbleForceDatset(BubbleDatasetBase):

    def __init__(self, *args, wrench_frame='med_base', **kwargs):
        self.wrench_frame = wrench_frame
        super().__init__(*args, **kwargs)

    @property
    def name(self):
        return 'bubble_force_dataset'

    def _get_sample(self, fc):
        # fc: index of the line in the datalegend (self.dl) of the sample
        dl_line = self.dl.iloc[fc]
        scene_name = dl_line['Scene']
        undef_fc = dl_line['UndeformedFC']
        init_fc = dl_line['InitialStateFC']
        final_fc = dl_line['FinalStateFC']

        # TODO: Load the remaining values

        camera_info_depth_r = self._load_camera_info_depth(scene_name, camera_name='right')
        camera_info_depth_l = self._load_camera_info_depth(scene_name, camera_name='left')

        init_color_img_r = self._load_color_img(fc=init_fc, scene_name=scene_name, camera_name='right')
        init_color_img_l = self._load_color_img(fc=init_fc, scene_name=scene_name, camera_name='left')
        init_depth_img_r = self._load_depth_img(fc=init_fc, scene_name=scene_name, camera_name='right')
        init_depth_img_l = self._load_depth_img(fc=init_fc, scene_name=scene_name, camera_name='left')

        init_wrench_ext = self._get_wrench(fc=init_fc, scene_name=scene_name, frame_id=self.wrench_frame, wrench_name='external_wrenches')
        init_wrench_robot = self._get_wrench(fc=init_fc, scene_name=scene_name, frame_id=self.wrench_frame, wrench_name='wrenches')


        # TODO: Fill the sample
        sample = {

        }

        return sample