import rospy
import roslaunch
import os
import rospkg
import subprocess
import time
from std_srvs.srv import Trigger, TriggerResponse


class FTURDFGenerator(object):
    """
    Class to automate the call of bubble_force_setup.launch.
    This launches the urdf collision geometry of the tool
    """
    # TODO: Add option to handle multiple tools

    def __init__(self, tool_name='r7p5mm_ati_cylinder', output='log'):
        self.package_name = 'bubble_force_estimation'
        self.launch_file = 'ft_urdf.launch' # name of the file to be launched inside self.package_name
        self.node_name = 'ft_urdf_generator'
        self.tool_name = tool_name
        self.output = output # 'log' for file logging, 'screen' for terminal logging
        self.proc = None

    def _start_node(self):
        command = f'roslaunch {self.package_name} {self.launch_file} tool_name:={self.tool_name}  --{self.output}'
        self.proc = subprocess.Popen(command, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT) # we hide all kind of output
        time.sleep(2.0) # Give it some time to execute the launch file
    def _close_node(self):
        if self.proc is not None:
            self.proc.kill()

    def generate(self):
        self._close_node()
        self._start_node()

