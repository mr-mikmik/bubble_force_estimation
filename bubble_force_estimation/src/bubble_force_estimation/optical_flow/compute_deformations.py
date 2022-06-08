
from bubble_force_estimation.optical_flow.estimate_correspondences import estimate_correspondences
from mmint_camera_utils.camera_utils.camera_utils import project_depth_image


def compute_deformations(img_1, img_2, depth_1, depth_2, flow, camera_matrix, return_point_coordinates=False):
    pixel_points_1, pixel_points_2 = estimate_correspondences(img_1, img_2, flow)
    # project points
    points_1 = project_depth_image(depth_1, K=camera_matrix) # (w,h,3) array of the (x,y,z) coordiantes for each pixel in the image
    points_2 = project_depth_image(depth_2, K=camera_matrix) # (w,h,3) array of the (x,y,z) coordiantes for each pixel in the image
    # select points:
    projected_pixel_points_1 = points_1[list(pixel_points_1.T)]
    projected_pixel_points_2 = points_2[list(pixel_points_2.T)]
    # compute deformations
    deformations = projected_pixel_points_2 - projected_pixel_points_1
    if return_point_coordinates:
        return deformations, projected_pixel_points_1, projected_pixel_points_2
    else:
        return deformations
