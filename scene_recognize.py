import sys
import cv2
import numpy as np
import skimage.measure

"""
    Response to scene changes using main quare error between two image with the same dimension.
    e.x. python scene_recognize.py frame0.jpg frame12.jpg frame16.jpg
    >>>True
    e.x. python scene_recognize.py frame0.jpg frame8.jpg frame12.jpg
    >>>False
    e.x. python scene_recognize.py frame16.jpg frame28.jpg frame68.jpg
    >>>True
"""

def identify_scene(pre_err_diff, curr_err_diff, pre_err, curr_err, cutThreshold, aroundCutThreshold):
#    print(pre_err_diff, curr_err_diff)
    if curr_err_diff>=aroundCutThreshold*pre_err_diff and\
       cutThreshold*curr_err_diff>=min(pre_err, curr_err) and pre_err!=0:
        return True # new scene
    else:
        return False

def mse_maxpool(img1, img2):
    # Max pooling: since a individual shot is defined as a sequence of smooth camera motion.
    # Consecutive frames have similar max pooling results and thus small mse.
    img1_maxpool = skimage.measure.block_reduce(img1, (3,3), np.max)
    img2_maxpool = skimage.measure.block_reduce(img2, (3,3), np.max)
    
    err = np.sum((img1_maxpool.astype("float") - img2_maxpool.astype("float")) ** 2)
    err /= float(img1_maxpool.shape[0] * img1_maxpool.shape[1])
    return err


if __name__ == '__main__':
    # THRESHOLD
    shrink = 5
    cut_signal = 5

    img1 = cv2.imread(sys.argv[1], 0)
    img1_resize = cv2.resize(img1, (int(img1.shape[1]/shrink), int(img1.shape[0]/shrink)))
    img2 = cv2.imread(sys.argv[2], 0)
    img2_resize = cv2.resize(img2, (int(img2.shape[1]/shrink), int(img2.shape[0]/shrink)))
    img3 = cv2.imread(sys.argv[3], 0)
    img3_resize = cv2.resize(img3, (int(img3.shape[1]/shrink), int(img3.shape[0]/shrink)))

    pre_esm_err = mse_maxpool(img1_resize, img2_resize)
    curr_esm_err = mse_maxpool(img2_resize, img3_resize)
    print(pre_esm_err, curr_esm_err)
    print(identify_scene(pre_esm_err, curr_esm_err, cut_signal))
