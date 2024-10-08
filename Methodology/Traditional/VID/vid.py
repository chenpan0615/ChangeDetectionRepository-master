# image differencing CD
import cv2
import numpy as np
from Methodology.util.data_prepro import stad_img
from Methodology.util.cluster_util import otsu
# from osgeo import gdal

def run_vid(pre_data, post_data, output, stad=True, open=True):
    """Vegetation Index Differencing  CD

    Args:
        pre_data (str): path to pre-temporal image
        post_data (str): path to pre-temporal image
        output (str): path to output file
        stad (bool, optional): normalize data. Defaults to True.
        open (bool): use open operation to remove noises. Default: False
    """
    # read bi-temporal images
    img_a = cv2.imread(pre_data, -1).astype(np.float32)
    img_b = cv2.imread(post_data, -1).astype(np.float32)
    img_height, img_width = img_a.shape[0], img_a.shape[1]
    # calculate NDVI
    ndvi_1 = (img_a[:,:,3] - img_a[:,:,2]) / (img_a[:,:,3] + img_a[:,:,2])
    ndvi_2 = (img_b[:,:,3] - img_b[:,:,2]) / (img_b[:,:,3] + img_b[:,:,2])
        
    # get difference
    img_diff = ndvi_1 - ndvi_2
    img_diff = np.abs(img_diff)*255
    # get a threshold by "Otsu’s Thresholding Method"
    thre = otsu(img_diff.astype(np.uint8).reshape(1, -1))
    # pixels with value large than thre are set to 1, and the rest are set to 0
    _, change_binary = cv2.threshold(img_diff, 50, 1, cv2.THRESH_BINARY)
    # open operation
    if open:
        kernel = np.ones((3, 3), np.uint8)
        change_binary = cv2.morphologyEx(change_binary, cv2.MORPH_OPEN, kernel)
        # change_binary = cv2.medianBlur(change_binary, 3)
    change_binary = np.reshape(change_binary, (img_height, img_width))
    # write result
    cv2.imwrite(output, change_binary.astype(np.uint8)*255)