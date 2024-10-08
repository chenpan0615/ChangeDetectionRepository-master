import numpy as np
from Methodology.Traditional.PCAKmeans.algorithm import pca_k_means
from Methodology.Traditional.PCAKmeans.util import diff_image
import cv2

def run_pcakmean(pre_data, post_data, output):
    before_img = cv2.imread(pre_data)[:, :, 0:3]
    after_img = cv2.imread(post_data)[:, :, 0:3]
    eig_dim = 10
    block_sz = 4

    diff_img = diff_image(before_img, after_img, is_abs=True, is_multi_channel=True)
    change_img = pca_k_means(diff_img, block_size=block_sz,
                             eig_space_dim=eig_dim)
    cv2.imwrite(output, change_img)
