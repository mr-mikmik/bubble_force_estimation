import os
import sys

package_name = 'bubble_force_estimation'
package_path = project_path = os.path.join(os.path.dirname(os.path.abspath(__file__)).split('/'+package_name)[0], package_name)

meshes_path = os.path.join(package_path, package_name, 'meshes', 'visual')

def get_mesh_path(mesh_name):
    return os.path.join(meshes_path, mesh_name)
