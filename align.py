'''
This file is to aligned detected face patches, to standardize input patches and
database patches, to gaurentee the accuracy of recognition.
Author: Ruichen Li
'''

import cv2
import matplotlib.pyplot as plt
import numpy as np
from mtcnn.mtcnn import MTCNN
import detection
#%matplotlib inline


def find_patch(img, method):
    '''
    This function finds the patch of faces within img.
    
    Args:
        img: The image template for finding faces.
        method: String indicate method for detection, can be 'mtcnn' or 'haar'.
        
    Returns:
        patches: A list of numpy array of faces.
        faces: A list of coordinate of each faces within img.
    '''
    
    # detect faces and eyes in image using method
    faces, eyes = detection.face_detection(img, method)
    
    # no faces found
    if(len(faces) == 0):
        return -1, -1
    
    # found all faces within image, get all aligned patches
    patches = []
    for i in range(len(faces)):
        curr_face = faces[i]
        curr_eyes = eyes[i]
        curr_patch = find_single_patch(img, curr_face, curr_eyes, method)
        patches.append(curr_patch)
        
    return patches, faces


def find_single_patch(img, faces, eyes, method):
    '''
    This function finds an aligned face patch within img, with coordinate of faces
    and eyes detected using method.
    
    Agrs:
        img: The image template where face is find.
        faces: The coordinate of such detected face within img.
        eyes: The coordinate of such detected eyes within img.
        method: The method use to find find faces and eyes.
        
    Returns:
        aligned_patch: The aligned patch of such face (i.e. front face, same size etc).
    '''
    
    # parse input
    x, y, w, h = faces
    rows = img.shape[0]
    cols = img.shape[1]

    # find center of eyes based on different methods
    # for mtcnn detector
    if(method == 'mtcnn'):
        e1_center = eyes[0]
        e2_center = eyes[1]
        
    # for haar detector
    else:
        # haar detector's eyes location are bounding box coordinates
        ex1, ey1, ew1, eh1 = eyes[0]
        ex2, ey2, ew2, eh2 = eyes[1]

        # if eyes not detected, choose the detected face as patch
        if(ex1 == -1 or ex2 == -1):
            patch = img[y:y+h, x:x+w]
            
        # if detected, change eye coords with respect to image coordiante
        else:
            ex1 += x
            ex2 += x
            ey1 += y
            ey2 += y

            # find center point for each eye
            e1_center = (ey1 + np.floor(eh1 / 2), ex1 + np.floor(ew1 / 2))
            e2_center = (ey2 + np.floor(eh2 / 2), ex2 + np.floor(ew2 / 2))

    # calculate angle between 2 eyes centers, this is angle we need to rotate image
    d1 = e1_center[0] - e2_center[0]
    d2 = e1_center[1] - e2_center[1]
    if(d2 < 0):
        d1 = -d1
        d2 = -d2
    angle = np.degrees(np.arctan2(d1, d2))

    # rotate image based on angle
    trans_center = (np.floor((e1_center[1] + e2_center[1]) / 2), \
                    np.floor((e1_center[0] + e2_center[0]) / 2))

    # get rotation matrix and do affine transformation
    M = cv2.getRotationMatrix2D(trans_center, angle, 1)
    trans_img = cv2.warpAffine(img , M, (cols,rows))

    # visualized rotated image, with eyes aligned
#    plt.imshow(trans_img)

    # padding is 2% of image's width and length
    padding = min(cols * 0.02, rows * 0.02)
    # new coordinate to extract patch
    x_start = int(max(0, np.floor(x - padding)))
    x_end = int(min(cols, np.floor(x + w + padding)))
    y_start = int(max(0, np.floor(y - padding)))
    y_end = int(min(rows, np.floor(y + h + padding)))
    
    # extract patch
    patch = trans_img[y_start:y_end, x_start:x_end]

    # standardize all patches to 300 x 300 pixel
    dim = (300, 300)
    aligned_patch = cv2.resize(patch, dim, interpolation = cv2.INTER_AREA)

    # visualize extracted patch
#    plt.imshow(aligned_patch)

    return aligned_patch

