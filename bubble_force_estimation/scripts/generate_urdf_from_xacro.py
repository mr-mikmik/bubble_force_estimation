import argparse
import os

if __name__ == '__main__':

    parser = argparse.ArgumentParser('URDF_from_XACRO')
    parser.add_argument('xacro_file_path', type=str, help='path from the xacro_path')
    parser.add_argument('urdf_file_path', type=str, help='path to the urdf generated file')
    parser.add_argument('xacro_args', type=str, nargs='+', default='')

    args = parser.parse_args()

    os.system('rosrun xacro') # Generate the urdf