# post-classification CD
import os
import cv2
import numpy as np

def run_pc(pre_data, post_data, output_path, open=True):
    """post-classification CD

    Args:
        pre_data (str): path to pre-temporal classification map
        post_data (str): path to pre-temporal classification map
        output (str): path to output file
        open (bool): use open operation to remove noises. Default: False
    """
    # read bi-temporal images
    img_a = cv2.imread(pre_data, -1).astype(np.uint8)
    img_b = cv2.imread(post_data, -1).astype(np.uint8)

    # get difference
    img_diff = np.abs(img_a - img_b)
    change_binary = np.zeros(img_a.shape, np.uint8)
    change_binary[img_diff!=0] = 1
    
    # open operation
    if open:
        kernel = np.ones((3, 3), np.uint8)
        # change_binary = cv2.morphologyEx(change_binary, cv2.MORPH_OPEN, kernel)
        change_binary = cv2.medianBlur(change_binary, 3)

    # get change direction from bi-temporal images
    img_a[change_binary==0] = 0
    img_b[change_binary==0] = 0
    
    # write result
    cv2.imwrite(os.path.join(output_path, "pc_binary.png"), change_binary.astype(np.uint8)*255)
    cv2.imwrite(os.path.join(output_path, "pc_pre.png"), img_a.astype(np.uint8))
    cv2.imwrite(os.path.join(output_path, "pc_post.png"), img_b.astype(np.uint8))
