import cv2
import numpy as np
from matplotlib import pyplot as plt

def dot_detection(I1g, block_size=15, C=7, visualize=False):
    """
        Takes one grayscale image, block_size and C
        Returns the center coordinates of each dot in the image
        :params I1g:          grayscale image of the image containing dots
        :params block_size:   size of a pixel neighborhood that is used to calculate a threshold value for the pixel (must be odd number >1)
        :params C:            constant subtracted from the mean or weighted mean (normally should be positive)
        :params visualize:    if True, output a visualization of the dots
    """
    
    # Adaptive threshold the image and find dot contours
    threshed = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,block_size,C)
    cnts = cv2.findContours(threshed, cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)[-2]
    
    # Filter by dot area (usually not needed)
    upper_limit = 100
    xcnts = []
    for cnt in cnts:
        if cv2.contourArea(cnt) < upper_limit:
            xcnts.append(cnt)
    
    # Calculate dot center coordinates
    dots = np.array([])
    for contour in xcnts:
        dot = np.mean(contour.reshape(-1,2), axis=0).astype(int)
        dots = np.append(dots, dot).reshape(-1,2)
        
    # Optional visualization
    if visualize:
        plt.imshow(I1g)
        for dot in dots:
            plt.plot(dot[0], dot[1], 'ro', markersize=2)
            
    return dots


def dot_detection_gs(I1g, bs1=3, bs2=30, C1=2, C2=10, save_figure=False):
    """
        Takes one grayscale image, upper and lower limit of block_size and C
        Runs grid search and visualize the results
        :params I1g:           grayscale image of the image containing dots
        :params bs1:           Lower limit of block size
        :params bs2:           Upper limit of block size
        :params C1:            Lower limit of C
        :params C2:            Upper limit of C
        :params save_figure:   if True, save the result as dot_detection_gs.png
    """
    
    num_rows = np.ceil(np.ceil((bs2-bs1)/2)*(C2-C1)/3)
    figure, axis = plt.subplots(int(num_rows), 3)
    figure.set_size_inches(18.5, 6*num_rows)

    curr_time = time.time()
    count = 0
    for block_size in range(bs1,bs2,2):
        for C in range(C1,C2):
            cur_axis = axis[int(np.floor(count/3)),count%3]
            cur_axis.set_title('bs={}, C={}, count={}'.format(block_size, C, len(xcnts)))
            cur_axis.imshow(cv2.resize(cv2.imread('data/1undeformed_image.png'), (200,200)))
            dots = dot_detection(gray, block_size, C)
            for dot in dots:
                cur_axis.plot(dot[0], dot[1], 'ro', markersize=2)
            count = count+1
    
    # Save figure or display results
    if save_figure:
        plt.savefig("dot_detection_gs.png")
    else:
        plt.show()

    print("It took {:.2f} seconds".format(time.time()-curr_time)) 
    
    

