import abc

import numpy as np
import rospy
from collections import OrderedDict
import gym
import copy
import tf.transformations as tr
import time

from bubble_utils.bubble_data_collection.bubble_data_collection_base import BubbleDataCollectionBase
from bubble_control.bubble_drawer.bubble_drawer import BubbleDrawer
from bubble_control.aux.action_sapces import ConstantSpace
from netft_utils.ft_sensor import FTSensor
from bubble_utils.bubble_data_collection.wrench_recorder import WrenchRecorder


class BubbleForceDataCollection(BubbleDataCollectionBase):

    def __init__(self, *args, sensor_name='netft', grasp_area_size=(.01, .01), move_length_limits=(0.001, 0.005), grasp_width_limits=(20,35), **kwargs):
        self.sensor_name = sensor_name
        self.grasp_area_size = np.asarray(grasp_area_size)
        self.move_length_limits = move_length_limits
        self.grasp_width_limits = grasp_width_limits
        self.action_space = self._get_action_space()
        super().__init__(*args, **kwargs)
        self.ft_sensor = FTSensor(ns=self.sensor_name)
        self.ft_sensor_wrench_recorder = WrenchRecorder(wrench_topic=self.ft_sensor.topic_name, scene_name=self.scene_name, save_path=self.save_path, wrench_name='external_wrenches')

    def _get_action_space(self):
        action_space_dict = OrderedDict()
        action_space_dict['start_point'] = gym.spaces.Box(-self.grasp_area_size, self.grasp_area_size, (2,), dtype=np.float64)  # random uniform (y, z) axes
        action_space_dict['start_theta'] = ConstantSpace(0.)
        action_space_dict['direction'] = gym.spaces.Box(low=0, high=2*np.pi, shape=())
        action_space_dict['length'] = gym.spaces.Box(low=self.move_length_limits[0], high=self.move_length_limits[1], shape=())
        action_space_dict['grasp_width'] = gym.spaces.Box(low=self.grasp_width_limits[0], high=self.grasp_width_limits[1], shape=())
        action_space = gym.spaces.Dict(action_space_dict)
        return action_space

    def _check_valid_action(self, action):
        # TODO: Check if
        valid_action = True
        return valid_action

    def _get_wrench_recording_frames(self):
        wrench_recording_frames = super()._get_wrench_recording_frames() + [
                                   '{}_sensor'.format(self.sensor_name),
                                   '{}_tool_frame'.format(self.sensor_name),
                                   '{}_tool_tip_frame'.format(self.sensor_name)]
        return wrench_recording_frames

    def _get_recording_frames(self):
        child_frames = super()._get_recording_frames() + [
                        '{}_sensor'.format(self.sensor_name),
                        '{}_tool_frame'.format(self.sensor_name),
                        '{}_tool_tip_frame'.format(self.sensor_name)]
        return child_frames

    def _record(self, fc=None):
        # add the wrench of the FTSensor to the recorded topics
        self.ft_sensor_wrench_recorder.record(fc=fc, frame_names=self._get_wrench_recording_frames())
        super()._record(fc=fc)

    def _get_legend_column_names(self):
        action_keys = self._sample_action().keys()
        column_names = ['Scene', 'Time', 'UndeformedFC', 'InitialStateFC',  'FinalStateFC', 'GraspForce'] + list(action_keys)
        return column_names

    def _get_legend_lines(self, data_params):
        legend_lines = []
        undef_fc_i = data_params['undeformed_fc']
        init_fc_i = data_params['initial_fc']
        final_fc_i = data_params['final_fc']
        grasp_force_i = data_params['grasp_force']
        action_i = data_params['action']
        scene_i = self.scene_name
        action_keys = self._sample_action().keys()
        action_values = [action_i[k] for k in action_keys]
        line_i = [scene_i, undef_fc_i, init_fc_i, final_fc_i,  grasp_force_i] + action_values
        legend_lines.append(line_i)
        return legend_lines

    def _sample_action(self):
        action_i = self.action_space.sample()
        return action_i

    def _get_sensor_tool_pose(self, ref_frame='med_base'):
        tool_frame_name = '{}_tool_frame'.format(self.sensor_name)
        rf_T_tf = self.tf2_listener.get_transform(ref_frame, tool_frame_name)
        tool_pose_rf = self.med._matrix_to_pose(rf_T_tf)
        return tool_pose_rf

    def _get_sensor_tool_tip_pose(self, ref_frame='med_base'):
        tool_frame_name = '{}_tool_tip_frame'.format(self.sensor_name)
        rf_T_tf = self.tf2_listener.get_transform(ref_frame, tool_frame_name)
        tool_pose_rf = self.med._matrix_to_pose(rf_T_tf)
        return tool_pose_rf

    def _set_robot_position_sensor_tool_frame(self, y, z, theta, ref_frame=None):
        if ref_frame is None:
            ref_frame = '{}_tool_frame'.format(self.sensor_name)
        target_position = np.array([0, y, z]) # in ref_frame
        # compute orientation -- theta is the rotation along the x axis of the sensor
        # base_quat = np.array([0, 0, 0, 1.0])
        base_quat = tr.quaternion_from_euler(np.pi, 0, np.pi, axes='sxyz') # rpy
        import pdb; pdb.set_trace()
        rotation_quat = tr.quaternion_about_axis(theta, axis=np.array([1, 0, 0]))
        target_quat = tr.quaternion_multiply(rotation_quat, base_quat)                          # in ref_frame # TODO: Account for the grasp frame orientation
        target_pose = np.concatenate([target_position, target_quat])
        self.med.plan_to_pose(self.med.arm_group, ee_link_name='grasp_frame', target_pose=list(target_pose), frame_id=ref_frame)

    def _cartesian_delta_motion_sensor_tool_frame(self, delta_y, delta_z, ref_frame=None):
        if ref_frame is None:
            ref_frame = '{}_tool_frame'.format(self.sensor_name)
        delta_pos = np.array([0, delta_y, delta_z])
        self.med.cartesian_delta_motion(delta_pos, ref_frame=ref_frame)

    def _collect_data_sample(self, params=None):
        """
        Adjust the robot so the object has a constant pose (target_pose) in the reference ref_frame
        returns:
            data_params: <dict> containing the parameters and information of the collected data
        """
        import pdb; pdb.set_trace()
        data_params = {}

        # Sample drawing parameters:
        is_valid_action = False
        action_i = None
        while not is_valid_action:
            action_i = self._sample_action()
            is_valid_action = self._check_valid_action(action_i)

        # Sample the fcs:
        undef_fc = self.get_new_filecode()
        init_fc = self.get_new_filecode()
        final_fc = self.get_new_filecode()

        grasp_force_i = 0  # TODO: read grasp force

        # Move to initial configuration
        self.med.gripper.move(100, 80)
        start_point = action_i['start_point']
        self._set_robot_position_sensor_tool_frame(action_i['start_point'][0], action_i['start_point'][0], action_i['start_theta'])

        # calibrate
        self.ft_sensor.zero()
        print('sleeping ') # TODO: Remove
        time.sleep(15.0)
        self._record(fc=undef_fc)
        # grasp
        grasp_width = action_i['grasp_width'] # TODO Consider using a force instead of a width
        self.med.gripper.move(grasp_width, 20)

        # record init state:
        self._record(fc=init_fc)

        # move
        final_point = action_i['length'] * np.array([np.cos(action_i['direction']), np.sin(action_i['direction'])])
        self._cartesian_delta_motion_sensor_tool_frame(final_point[0], final_point[1])

        # record final_state
        self._record(fc=final_fc)

        # reset motion
        self.med.gripper.open_gripper()

        data_params['undeformed_fc'] = undef_fc
        data_params['initial_fc'] = init_fc
        data_params['final_fc'] = final_fc
        data_params['grasp_force'] = grasp_force_i
        data_params['action'] = action_i

        return data_params

