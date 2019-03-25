'''
This file is for detect faces and facial landmarks within image.
Autor: Ruichen Li
'''

import cv2
import matplotlib.pyplot as plt
import numpy as np
from mtcnn.mtcnn import MTCNN
#%matplotlib inline


def face_detection(img, method):
    '''
    This function finds faces and eyes within image using different detector.
    
    Args:
        img: Image template for finding faces and eyes.
        method: String indicate method for detection, can be 'mtcnn' or 'haar'.
    
    Returns:
        faces: A list of coordinate of founded faces.
        eyes: A list of pair coordinate of eyes respective to faces.
    '''
    
    # find faces and eyes using MTCNN detectpr
    if(method == 'mtcnn'):
        faces, eyes = mtcnn_detect(img)
    # find faces and eyes using haar detector
    else:
        faces, eyes = haar_detect(img)
        
    return faces, eyes


def mtcnn_detect(img):
    '''
    This function detects faces and eyes within img using MTCNN detector.
    
    Args:
        img: Image template for finding faces and eyes.
    
    Returns:
        faces: A list of coordinate of founded faces using MTCNN detector.
        eyes: A list of pair coordinate of eyes respective to faces using MTCNN detector.
    '''
    
    # initialized detecor
    detector = MTCNN()
    
    # detect faces
    res = detector.detect_faces(img)
    
    # parse result
    faces = []
    eyes = []
    for next_face in res:
        # force faces detection within boundary
        coords = next_face['box']
        if(coords[0] < 0):
            coords[0] = 0
        if(coords[1] < 0):
            coords[1] = 0        
        faces.append(coords)
        
        # find detected coordinates of left eye and right eye
        kps = next_face['keypoints']
        leye = kps['left_eye']
        reye = kps['right_eye']
        eyes.append([[reye[1], reye[0]], [leye[1], leye[0]]])
        
    return faces, eyes


def haar_detect(img):
    '''
    This function detects faces and eyes within img using haar detector.
    
    Args:
        img: Image template for finding faces and eyes.
    
    Returns:
        faces: A list of coordinate of founded faces using haar detector.
        eyes: A list of pair coordinate of eyes respective to faces using haar detector.
    '''
    
    # turn img into grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # load cascade classifier pre-trained file
    haar_face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
    eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
    
    # detect faces using multiple scale factor
    scale = 1.3
    faces = haar_face_cascade.detectMultiScale(gray, scale, 5)
    while(len(faces) == 0 and scale != 1.05):
        scale = max(1.05, scale - 0.05)
        faces = haar_face_cascade.detectMultiScale(gray, scale, 5)

    # visualize detection
#     for next_face in faces:
#         visualize_detection(gray, next_face)

    # detect eyes
    eyes = []
    for (x, y, w, h) in faces:
        # detect within face subarea
        roi = gray[y:y+h, x:x+w]
        pair_eyes = eye_cascade.detectMultiScale(roi)
        
        # if no eyes detected, we set coords to -1
        if(len(pair_eyes) == 0 or len(pair_eyes) == 1):
            pair_eyes = [[-1, -1, -1, -1], [-1, -1, -1, -1]]
        eyes.append(pair_eyes)
        
        # visualize detection
#         for next_eye in pair_eyes:
#             next_eye[0] += x
#             next_eye[1] += y
#             visualize_detection(gray, next_eye)

    return faces, eyes


def visualize_detection(gray, coords):
    '''
    This function helps visualize detection of faces and eyes specified by coords
    by drawing a rectangle by coords.
    
    Args:
        gray: The grayscale image template for detection.
        coords: The coordinate of resulted detection within gray image.
    
    Returns:
        None
    '''
    
    # parse input
    x, y, w, h = coords
    
    # draw rectangle by coordinates and visualize
    cv2.rectangle(gray, (x,y), (x+w,y+h),(255,0,0),5)
    plt.imshow(gray, cmap='gray')
    
    return

