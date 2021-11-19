import numpy as np

from bubble_utils.bubble_datasets.bubble_dataset_base import BubbleDatasetBase


class BubbleForceDatasetBase(BubbleDatasetBase):

    def __init__(self, *args, wrench_frame='med_base', **kwargs):
        self.wrench_frame = wrench_frame
        super().__init__(*args, **kwargs)

    @property
    def name(self):
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


class BubbleForceDataset3States(BubbleForceDatasetBase):
    """
    A sample contains reference state (undeformed), initial state, and final state.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def name(self):
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

    def __len__(self):
        return super().__len__() * 2

    @property
    def name(self):
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

        return sample

