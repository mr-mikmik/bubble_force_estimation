#! /usr/bin/env python
import argparse

from bubble_force_estimation.bubble_force_data_collection.bubble_force_data_collection import BubbleForceDataCollection

# TEST THE CODE: ------------------------------------------------------------------------------------------------------
import rospy


def collect_data_force_test(save_path, scene_name, num_data=10, grasp_area_size=(.25, .25), move_length_limits=(0.01, 0.05), grasp_width_limits=(20,30)):
    rospy.init_node('data_collection')
    dc = BubbleForceDataCollection(data_path=save_path, scene_name=scene_name, grasp_area_size=grasp_area_size, move_length_limits=move_length_limits, grasp_width_limits=grasp_width_limits)
    dc.collect_data(num_data=num_data)


if __name__ == '__main__':

    parser = argparse.ArgumentParser('Collect Data Drawing')
    parser.add_argument('save_path', type=str, help='path to save the data')
    parser.add_argument('num_data', type=int, help='Number of data samples to be collected')
    parser.add_argument('--scene_name', type=str, default='drawing_data', help='scene name for the data. For organization purposes')
    parser.add_argument('--grasp_area_size', type=float, default=(.01, .01), nargs=2, help='size of the y and z semiaxis of the inital grasp area')
    parser.add_argument('--move_length_limits', type=float, default=(.001, .002), nargs=2, help='(minimum_length, maximum_length) of the move')
    parser.add_argument('--grasp_width_limits', type=float, default=(20, 35), nargs=2, help='(minimum_width, maximum_width) of the grasp')

    args = parser.parse_args()

    save_path = args.save_path
    scene_name = args.scene_name
    num_data = args.num_data
    grasp_area_size = args.grasp_area_size
    move_length_limits = args.move_length_limits
    grasp_width_limits = args.grasp_width_limits


    collect_data_force_test(save_path, scene_name, num_data=num_data, grasp_area_size=grasp_area_size, move_length_limits=move_length_limits, grasp_width_limits=grasp_width_limits)