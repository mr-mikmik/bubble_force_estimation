import rospy
import roslaunch
import os
import rospkg
import subprocess
from std_srvs.srv import Trigger, TriggerResponse


class FTURDFGenerator(object):
    """
    Class to automate the call of bubble_force_setup.launch.
    This launches the urdf collision geometry of the tool
    """
    # TODO: Add option to handle multiple tools

    def __init__(self, tool_name='r7p5mm_ati_cylinder'):
        self.package_name = 'bubble_force_estimation'
        self.launch_file = 'ft_urdf.launch' # name of the file to be launched inside self.package_name
        self.node_name = 'ft_urdf_generator'
        self.tool_name = tool_name
        self.proc = None

    def _start_node(self):
        command = 'roslaunch {0} {1} tool_name:={2}'.format(self.package_name, self.launch_file, self.tool_name)
        self.proc = subprocess.Popen(command, shell=True)

    def _close_node(self):
        if self.proc is not None:
            self.proc.kill()

    def generate(self):
        self._close_node()
        self._start_node()

