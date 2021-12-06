import numpy as np

from bubble_force_estimation.optical_flow.optical_flow import optical_flow, mean_optical_flow, optical_flow_pyr
from bubble_force_estimation.optical_flow.dot_detection import dot_detection


def estimate_correspondences(img1, img2, flow):
    """

    :param img1: reference image (undeformed) in grayscale
    :param img2: deformed image in grayscale
    :param flow: (2,W,H) numpy array
    :return: points_1, points_2. Two numpy array of size (N,2) containing pixel coordinates.
        Given p_1_i and p_2_i the points in the n_i position for points_1 and points_2, they are the image coordinates
        of the same corresponent point before and after the deformation.

    """
    dots_1 = dot_detection(img1)
    dots_2 = dot_detection(img2)
    # TODO: Estima
    

    return points_1, points_2



