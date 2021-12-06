import numpy as np
import matplotlib.pyplot as plt

from bubble_force_estimation.optical_flow.optical_flow import optical_flow, mean_optical_flow, optical_flow_pyr
from bubble_force_estimation.optical_flow.dot_detection import dot_detection


# Possible improvement: include distance threshold
def estimate_correspondences(I1g, I2g, flow, cos_sim_th=0.8, visualize=False):
    """
    :param I1g:       reference image (undeformed) in grayscale
    :param I2g:       deformed image in grayscale
    :param cos_sim:   minimum cosine similarity between the estimated and true flow
    :return: points_1, points_2. Two numpy array of size (N,2) containing pixel coordinates.
        Given p_1_i and p_2_i the points in the n_i position for points_1 and points_2, they are the image coordinates
        of the same corresponent point before and after the deformation.
    """
    points1 = np.array([])
    points2 = np.array([])
    
    # Detect dots
    dots1, threshed1 = dot_detection(I1g)
    dots2, threshed2 = dot_detection(I2g)
    
    # Find flow
    u, v = flow
    
    # Find correspondence
    for dot1 in dots1:
        i = int(dot1[0])
        j = int(dot1[1])
        est_flow = [u[i, j], v[i, j]]
        dest = dot1 + est_flow
        dist = [np.linalg.norm(dest-dot2) for dot2 in dots2]
        closest = dots2[np.argmin(dist)]
        true_flow = closest - dot1
        
        # When the correspondence is itself
        if np.linalg.norm(true_flow) == 0 or np.linalg.norm(est_flow) == 0:
            cos_sim = 1
        else:
            cos_sim = np.dot(est_flow, true_flow)/(np.linalg.norm(est_flow)*np.linalg.norm(true_flow))
        if cos_sim > cos_sim_th:
            points1 = np.append(points1, dot1).reshape(-1,2)
            points2 = np.append(points2, closest).reshape(-1,2)
    
    # Optional visualization
    if visualize:
        plt.figure(figsize=(12, 16), dpi=80)
        for dot in points1:
            plt.plot(dot[0], dot[1], 'bo', markersize=2)
        for dot in points2:
            plt.plot(dot[0], dot[1], 'ro', markersize=2)
        for i in range(len(points1)):
            i, j = int(points1[i][0]), int(points1[i][1])
            plt.arrow(i, j, u[i, j], v[i, j], head_width=1, head_length=2, color='blue')
        plt.show()
        
    return points1, points2




