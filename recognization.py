'''
This file recognize faces within image, and further verify such prediction and 
classify unknowm person.
Author: Ruichen Li
'''

import tensorflow as tf
from keras import backend as K
from fr_utils import *
import cv2
import os
import matplotlib.pyplot as plt
import align
import shutil
from sklearn import preprocessing
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from keras.models import model_from_json

K.set_image_data_format('channels_first')


def standard_embeddings(FRmodel, align_method):
    '''
    This function find embeddings of selected representative images of each class
    in database. Selected images are subset of database.
    
    Args:
        FRmodel: The pretrained model finds embeddings of an image.
        align_method: Method used to align face patch, can be 'MTCNN' or 'haar'.
    
    Returns:
        ebds: A list of embeddings of selected images.
        std_class: A list of class names coordinate to ebds.
    '''
    
    # initialization
    std_path = 'standard'
    ebds = []
    std_classes = []
    
    for fname in os.listdir(std_path):
        # find class of current image
        std_class = fname.split('_')[0]
        fpath = os.path.join(std_path, fname)
        img = cv2.imread(fpath)
        
        if(img is not None):
            # find embeddings of such image
            find, coords = align.find_patch(img, align_method)
            patch = find[0]
            curr_ebd = img_to_encoding(patch, FRmodel)[0]
            ebds.append(curr_ebd)
            std_classes.append(std_class)
    
    return ebds, std_classes


def verify(prediction, ebds, FRmodel, align_method):
    '''
    This function classifies whether current perdiction has unknown person.
    
    Args:
        prediction: Prediction of class gathered by machine learning models.
        ebds: Embeddings of query image face patch.
        FRmodel: Pre-trained model for calculating image embeddings.
        align_method: Method used to align face patch, can be 'MTCNN' or 'haar'.
    
    Returns:
        prediction: The updated predictions including classifed unknown person.
    '''
    
    # get embeddings and class name of standard images
    std_ebd, std_classes = standard_embeddings(FRmodel, align_method)
    
    # for each prediction get from machine learning models
    num = len(prediction)
    std_num = len(std_classes)
    for i in range(num):
        curr_pred = prediction[i]        
        curr_ebd = ebds[i]
        distance = []
        
        # find its distance between each standard image patch's embedding
        for j in range(std_num):
            curr_std_ebd = std_ebd[j]
            dist = np.sqrt(np.sum((curr_std_ebd - curr_ebd)**2))
            distance.append(dist)
            
        # when have close match with standard, trust this result
        min_dist = min(distance)
        if(min_dist <= 0.4):
            prediction[i] = std_classes[distance.index(min_dist)]
            
        # otherwise, check if prediction from ml model contradict with distances
        else:
            idx = [index for index, value in enumerate(std_classes) if value == curr_pred]
            # set such prediction 'None' if distance too large
            if (distance[idx[0]] >= 0.6 and distance[idx[1]] >= 0.6):
                prediction[i] = 'None'
                
    return prediction


def predict(img, ml_model, encoder, align_method, FRmodel):
    '''
    This function predict the classes of faces within query img.
    
    Args:
        img: The query image for recognize people.
        ml_model: The trained machine learning model to get prediction.
        encoder: The encoder used along machine learning model, for translate prediction
                from integer to string.
        align_method: Method used to align face patch, can be 'MTCNN' or 'haar'.
        FRmodel: Pre-trained model for calculating image embeddings.
    
    Returns:
        res_predictions: A list of string of class names predict by ml_model and
                        translate using encoder.
        queries: A list of patches of detected and aligned faces in img using 
                align_method.
        coords: A list of coordinate represents queries' locations within image.
    '''
    
    # detect and align patches of faces within image, along with location coordinates
    queries, coords = align.find_patch(img, align_method)
    
    # no face detected in query image
    if(queries == -1):
        return [], [], []
    
    # initialization
    num = len(queries)
    ebds = []
    res_predictions = []
    
    # for each face patch within image, recognize its class
    for i in range(num):
        # find its embedding and predict using ml_model
        patch = queries[i]
        ebd = img_to_encoding(patch, FRmodel)
        ebds.append(ebd)
        query_predict = ml_model.predict(ebd)
        # translate prediction back into string
        res = encoder.inverse_transform(query_predict)[0]
        res_predictions.append(res)
    
    # report raw prediction result and result after consider unknown
    print('Prediction before find unknown: ', res_predictions)
    verified_pred = verify(res_predictions, ebds, FRmodel, align_method)
    print('Prediction after find unknown: ', verified_pred)
    
    return  res_predictions, queries, coords

