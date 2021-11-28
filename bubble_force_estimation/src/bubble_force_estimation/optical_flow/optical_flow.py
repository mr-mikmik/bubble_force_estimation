import numpy as np
from scipy import signal


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
