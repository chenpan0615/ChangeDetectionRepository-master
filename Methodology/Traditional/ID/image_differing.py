# image differencing CD
import cv2
import numpy as np
from Methodology.util.data_prepro import stad_img
from Methodology.util.cluster_util import otsu

def run_id(pre_data, post_data, output, stad=True, open=False):
    """image differencing CD

    Args:
        pre_data (str): path to pre-temporal image
        post_data (str): path to pre-temporal image
        output (str): path to output file
        stad (bool, optional): normalize data. Defaults to True.
        open (bool): use open operation to remove noises. Default: False
    """
    # read bi-temporal images
    img_a = cv2.imread(pre_data).astype(np.float32)
    img_b = cv2.imread(post_data).astype(np.float32)
    
    img_height, img_width = img_a.shape[0], img_a.shape[1]
    # normalize the bi-temporal images
    if stad:
        img_a = stad_img(img_a)
        img_b = stad_img(img_b)
        
    # convert to gray scale
    if len(img_a.shape) == 3:
        img_a = cv2.cvtColor(img_a, cv2.COLOR_BGR2GRAY)
        img_b = cv2.cvtColor(img_b, cv2.COLOR_BGR2GRAY)
        
    # get difference
    img_diff = img_a - img_b
    img_diff = np.abs(img_diff)
    # get a threshold by "Otsu’s Thresholding Method"
    thre = otsu(img_diff.reshape(1, -1))
    
    # pixels with value large than thre are set to 1, and the rest are set to 0
    _, change_binary = cv2.threshold(img_diff, thre, 1, cv2.THRESH_BINARY)
    # open operation
    if open:
        kernel = np.ones((3, 3), np.uint8)
        change_binary = cv2.morphologyEx(change_binary, cv2.MORPH_OPEN, kernel)
        # change_binary = cv2.medianBlur(change_binary, 3)
    change_binary = np.reshape(change_binary, (img_height, img_width))
    # write result
    cv2.imwrite(output, change_binary.astype(np.uint8)*255)