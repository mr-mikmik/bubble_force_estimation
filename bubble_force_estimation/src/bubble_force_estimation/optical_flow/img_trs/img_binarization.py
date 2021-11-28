import numpy as np
import cv2
import os
import sys
from PIL import Image
import matplotlib.pyplot as plt

project_path = os.path.join(os.path.dirname(os.path.abspath(__file__)).split('/bubble_force_estimation')[0], 'bubble_force_estimation')
package_path = os.path.join(project_path, 'bubble_force_estimation', 'src', 'bubble_force_estimation', 'optical_flow')
sys.path.append(project_path)


class BubbleImgBinarization(object):
    """
    binarize the image so latex is white (255) and dots are black (0).
    """

    def __call__(self, img, th=128):
        bin_img = img.copy() # TODO: binarize the image
        bin_img[np.where(img > th)] = 255.
        bin_img[np.where(img <= th)] = 0.
        return bin_img







# DEBUG:

def serach_threshold(img, th_min=None, th_max=None):
    w = 5
    h = 5
    if th_min is None:
        th_min = np.min(img)
    if th_max is None:
        th_max = np.max(img)
    ths = np.linspace(th_min, th_max, w*h)
    fig, axs = plt.subplots(nrows=h, ncols=h, sharex=True, sharey=True)
    img_binarization = BubbleImgBinarization()
    axs_flattened = axs.flatten()
    for i, th_i in enumerate(ths):
        ax_i = axs_flattened[i]
        bin_img_i = img_binarization(img, th_i)
        ax_i.imshow(bin_img_i)
        ax_i.set_title('th: {:.2f}'.format(th_i))
    fig = plt.figure(2)
    plt.imshow(img)
    plt.show()


def gird_binarization(img):
    k_size = 10
    tiles = []
    img_width, img_height = img.shape
    img_binarization = BubbleImgBinarization()
    bin_img = img.copy()
    for i in range(0, img_height, k_size):
        for j in range(0, img_width, k_size):
            tile_i = img[j:j+k_size, i:i+k_size]
            bin_tile_i = img_binarization(tile_i/np.max(tile_i), th=0.5)
            bin_img[j:j+k_size, i:i+k_size] = bin_tile_i
    plt.imshow(bin_img)
    plt.show()
    return bin_img

def avg_filter(img):
    k_size = 30
    img_width, img_height = img.shape
    avg_img = img.copy()
    for i in range(img_height):
        for j in range(img_width):
            tile_i = img[j:j+k_size, i:i+k_size]
            tile_i_avg = np.mean(tile_i)
            avg_img[j,i] = tile_i_avg
    return avg_img

def blob_detector(img):
    color_img = np.stack([img]*3, axis=-1)
    import pdb; pdb.set_trace()
    im = Image.fromarray(img)
    detector = cv2.SimpleBlobDetector()
    keypoints = detector.detect(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY))
    im_with_keypoints = cv2.drawKeypoints(im, keypoints, np.array([]), (0, 0, 255),
                                          cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    # Show keypoints
    cv2.imshow("Keypoints", im_with_keypoints)
    cv2.waitKey(0)


def load_img(img_path):
    imm = Image.open(img_path)
    img_ar = np.asarray(imm.getdata()).reshape(imm.size[1],imm.size[0])
    return img_ar

def reverse_img(img):
    img_rev = -1*(img.astype('int32')-255)
    return img_rev.astype('uint8')

if __name__ == '__main__':
    img_name = 'img_1.png'
    img_path = os.path.join(package_path, 'test_imgs', img_name)
    camera_info_path = os.path.join(package_path, 'test_imgs', 'camera_info.npy')
    camera_info = np.load(camera_info_path, allow_pickle=True)
    img_binarization = BubbleImgBinarization()
    img_ar = load_img(img_path)
    img_avg = avg_filter(img_ar)
    img_d = img_ar - img_avg

    # fig = plt.figure('img1')
    # plt.imshow(img_ar)
    # fig = plt.figure('img2')
    # plt.imshow(img_avg)
    #
    # fig = plt.figure('img3')
    # plt.imshow(img_d)
    # plt.show()
    # import pdb; pdb.set_trace()
    # plt.hist(img_d.flatten())
    # plt.show()

    # bin_img = img_binarization(img_ar)
    # search threshold
    # gird_binarization(img_d)
    # serach_threshold(img_ar, th_min=500, th_max=700)
    # serach_threshold(img_d, th_max=500, th_min=-500)
    # plt.imshow(bin_img)
    # plt.show()

    img_bin = img_binarization(img_d, th=-250).astype('uint8') # image binarized
    fig = plt.figure(1)
    plt.imshow(img_bin)
    # serach for countours
    img_bin_rev = reverse_img(img_bin)
    import pdb; pdb.set_trace()
    cnts = cv2.findContours(img_bin_rev, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]
    img_center = img_d.copy()/np.max(img_ar)*255
    for cnt in cnts:
        cnt_avg = np.mean(cnt, axis=0).flatten().astype('uint8')
        img_center[cnt_avg[0], cnt_avg[1]] = 255.0
    fig = plt.figure(2)
    plt.imshow(img_center)
    plt.show()
    import pdb; pdb.set_trace()
    # blob_detector(img_bin)
