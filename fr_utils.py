'''
This file is a fraction of code within
https://github.com/shahariarrabby/deeplearning.ai/blob/master/COURSE%204%20Convolutional
%20Neural%20Networks/Week%2004/Face%20Recognition/fr_utils.py
It's for using face net pretrained-models.
'''

import tensorflow as tf
import numpy as np
import os
import cv2


def img_path_to_encoding(image_path, model):
    img1 = cv2.imread(image_path, 1)
    return img_to_encoding(img1, model)
    

def img_to_encoding(image, model):
    image = cv2.resize(image, (96, 96)) 
    img = image[...,::-1]
    img = np.around(np.transpose(img, (2,0,1))/255.0, decimals=12)
    x_train = np.array([img])
    embedding = model.predict_on_batch(x_train)
    return embedding

