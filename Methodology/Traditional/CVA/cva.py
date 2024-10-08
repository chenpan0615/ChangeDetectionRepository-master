import numpy as np
import cv2
import time
from Methodology.util.cluster_util import otsu
from Methodology.util.data_prepro import stad_img


def CVA(img_X, img_Y, stad=True):
    # CVA has not affinity transformation consistency, so it is necessary to normalize
    # multi-temporal images to eliminate the radiometric inconsistency between them
    # normalize the bi-temporal images
    if stad:
        img_X = stad_img(img_X)
        img_Y = stad_img(img_Y)
        
    img_diff = img_X - img_Y
    L2_norm = np.sqrt(np.sum(np.square(img_diff), axis=0))
    return L2_norm

def run_cva(pre_data, post_data, output):
    """CVA CD

    Args:
        pre_data (str): path to pre-temporal image
        post_data (str): path to pre-temporal image
        output (str): path to output file
        stad (bool, optional): normalize data. Defaults to True.
    """
    img_X = cv2.imread(pre_data).transpose(2,0,1)  # data set X
    img_Y = cv2.imread(post_data).transpose(2,0,1)  # data set Y

    channel, img_height, img_width = img_X.shape
    tic = time.time()
    L2_norm = CVA(img_X, img_Y)
    # get a threshold by "Otsu’s Thresholding Method"
    thre = otsu(L2_norm.reshape(1, -1))
    
    _, bcm = cv2.threshold(L2_norm, thre, 255, cv2.THRESH_BINARY) 
    bcm = np.reshape(bcm, (img_height, img_width))
    cv2.imwrite(output, bcm)
    toc = time.time()
    print(toc - tic)

