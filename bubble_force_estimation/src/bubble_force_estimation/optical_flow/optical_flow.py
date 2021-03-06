import numpy as np
import cv2
import os
import sys
from PIL import Image
import matplotlib.pyplot as plt
from scipy import signal

project_path = os.path.join(os.path.dirname(os.path.abspath(__file__)).split('/bubble_force_estimation')[0], 'bubble_force_estimation')
package_path = os.path.join(project_path, 'bubble_force_estimation', 'src', 'bubble_force_estimation', 'optical_flow')
sys.path.append(project_path)


def optical_flow(I1g, I2g, window_size=5, normalize=False):
    """
    Takes two images and window_size, returns the optical flow by Lucas Kanade algorithm
    :params I1g:          grayscale image of the first frame
    :params I2g:          grayscale image of the second frame
    :params window_size:  size of the window of pixels that are assumed to have the same displacement
    """
    kernel_x = np.array([[-1., 1.], [-1., 1.]])
    kernel_y = np.array([[-1., -1.], [1., 1.]])
    kernel_t1 = np.array([[-1., -1.], [-1., -1.]])
    kernel_t2 = np.array([[1., 1.], [1., 1.]])

    # Normalize
    if normalize:
        I1g = I1g / 255.
        I2g = I2g / 255.
    
    mode = 'same'
    fx = signal.convolve2d(I1g, kernel_x, boundary='symm', mode=mode) + signal.convolve2d(I2g, kernel_x, boundary='symm', mode=mode)
    fy = signal.convolve2d(I1g, kernel_y, boundary='symm', mode=mode) + signal.convolve2d(I2g, kernel_y, boundary='symm', mode=mode)
    ft = signal.convolve2d(I1g, kernel_t1, boundary='symm', mode=mode) + signal.convolve2d(I2g, kernel_t2, boundary='symm', mode=mode)
    u = np.zeros(I1g.shape)
    v = np.zeros(I1g.shape)
    
    # within window window_size * window_size
    w = window_size
    for i in range(0, I1g.shape[0]-w+1):
        for j in range(0, I1g.shape[1]-w+1):
            Ix = fx[i:i+w, j:j+w].flatten()
            Iy = fy[i:i+w, j:j+w].flatten()
            It = ft[i:i+w, j:j+w].flatten()
            b = -1*np.reshape(It, (It.shape[0], 1))
            A = np.vstack((Ix, Iy)).T 
            nu = np.matmul(np.linalg.pinv(A), b) 
            u[i+int(w/2), j+int(w/2)] = nu[0]
            v[i+int(w/2), j+int(w/2)] = nu[1]
    return (u,v)

def optical_flow_pyr(I1g, I2g, window_size=15, normalize=False):
    """
    Takes two images and window_size, returns the optical flow by Gunnar Farneback's algorithm
    :params I1g:          grayscale image of the first frame
    :params I2g:          grayscale image of the second frame
    :params window_size:  size of the window of pixels that are assumed to have the same displacement
    """
    flow = cv2.calcOpticalFlowFarneback(I1g, I2g, None, 0.5, 3, window_size, 3, 5, 1.2, 0)
    u = flow[:,:,0]
    v = flow[:,:,1]
    return (u,v)


def mean_optical_flow(I1g, I2g, window_size=5, normalize=False):
    x_flow, y_flow = optical_flow(I1g, I2g, window_size=window_size, normalize=normalize)
    x_flow_mean = np.mean(x_flow)
    y_flow_mean = np.mean(y_flow)
    mean_flow = np.array([x_flow_mean, y_flow_mean])
    return mean_flow


# DEBUG:
def load_img(img_path):
    imm = Image.open(img_path)
    img_ar = np.asarray(imm.getdata()).reshape(imm.size[1], imm.size[0])
    return img_ar

if __name__ == '__main__':
    test_path = os.path.join(package_path, 'test_imgs')
    img1 = load_img(os.path.join(test_path, 'img_1.png'))
    img2 = load_img(os.path.join(test_path, 'img_2.png'))
    import pdb; pdb.set_trace()
    flow = optical_flow(img1, img2)
