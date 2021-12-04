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
