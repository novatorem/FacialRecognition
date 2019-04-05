'''
This file is doing preparation of face recognition, it prepares database, load facenet
model and train specified machine learning model.
This file is expected to run only once in an recognition process.
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


def set_up(align_method, ml_method):
    '''
    This is a high-level function that prepare everything needed for later recognition
    system use. It finds number of samples for each database class, load the facenet
    model for encoding faces, encoding faces within database, and train specified
    machine learning model.
    
    Args:
        align_method: Method used to align face patch, can be 'MTCNN' or 'haar'.
        ml_method: Specifed machine learning model used for classification.
    
    Returns:
        FRmodel: The loaded pretrained model finds embeddings of an image.
        ml_model: The model just trained on database ready for recognition.
        encoder: The encoder used along machine learning model, for translate prediction
                from integer to string.
    '''
    
    # load database by aligning detected faces in image directory
    db_num = load_to_database(align_method)
    
    # load facenet model
    FRmodel = load_facenet()
    
    # find all embeddings of database's face patches
    db_embeddings = find_database_embeddings(FRmodel)
    
    # train specified machine learning model and encoder for translate labels
    ml_model, encoder = train_db(db_num, db_embeddings, FRmodel, ml_method)
    
    return FRmodel, ml_model, encoder


def load_to_database(method):
    '''
    This function load database by aligning detected faces in imaeg directory.
    
    Args:
        method: Method used to align face patch, can be 'MTCNN' or 'haar'.
    
    Returns:
        db_num: A dictionary of numbers of samples for each database class.
    '''
    
    # number of samples for each class in database, in dictionary form
    db_num = {}
    
    # initialize images and database directory paths
    database_dir = 'database'
    sample_dir = 'images'
    
    # if database already exist, remove it
    if(os.path.exists(database_dir)):
        shutil.rmtree(database_dir)
    # create a new empty database directory
    os.mkdir(database_dir)
    
    # get all classes of sample, recursively find patch for each class's each sample
    classes = os.listdir(sample_dir)
    for i in classes:
        if(i.startswith('.') == False):
            # path for a specific class
            dir_path = os.path.join(sample_dir, i)
            os.mkdir(os.path.join(database_dir, i))

            n = 0
            for fname in os.listdir(dir_path):
                if(fname.startswith('.') == False):
                    # path for a specific sample within class
                    fpath = os.path.join(dir_path, fname)
                    img = cv2.imread(fpath)

                    # if is image file, find aligned patch and save to database
                    if(img is not None):
                        # detect faces and coordinate
                        find, coords = align.find_patch(img, method)

                        # no face found in this sample, drop it
                        if(find == -1):
                            print('No face found, drop image ' + fname)
                        else:
                            patch = find[0]
                            # save aligned patch in database/class name/image filename
                            patch_path = os.path.join(database_dir, i, fname)

                            # end when write to database fails
                            write = cv2.imwrite(patch_path, patch)
                            if(write is False):
                                print('ERROR: Fail to write an aligned patch.')
                                return 0
                            else:
                                n += 1
            # there are n samples in such class
            db_num[i] = n
        
    return db_num


def find_database_embeddings(model):
    '''
    This function find embeddings of patches in database using facenet model.
    
    Args:
        model: The loaded pretrained model finds embeddings of an image.
    
    Returns:
        db_embeddings: A dictionary of all embeddings of database's face patches.
    '''
    
    # database embeddings as dictionary
    db_embeddings = {}
    # initialization
    db_path = 'database'
    
    # find embeddings for each class
    classes = os.listdir(db_path)
    for next_class in classes:
        class_path = os.path.join(db_path, next_class)
        class_embeddings = []
        
        # find embeddings for each sample in class
        for fname in os.listdir(class_path):
            # path for a specific image within class
            fpath = os.path.join(class_path, fname)
            img = cv2.imread(fpath)
            
            # if is image file, find and add its embedding
            if(img is not None):
                curr_embedding = img_to_encoding(img, model)
                class_embeddings.append(curr_embedding)
        
        # add current class's embeddings into dictionary
        db_embeddings[next_class] = class_embeddings
                
    return db_embeddings


# This function was took from provided code in Assignment 5, of file facenet.py
def load_facenet():
    """
    loads a saved pretrained model from a json file
    :return:
    """
    # load json and create model
    json_file = open('FRmodel.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    FRmodel = model_from_json(loaded_model_json)

    # load weights into new model
    FRmodel.load_weights("FRmodel.h5")
    print("Loaded model from disk")

    return FRmodel


def train_db(db_num, db_embeddings, FRmodel, method):
    '''
    This function train specified machine learning model for recognition.
    
    Args:
        db_num: A dictionary of numbers of samples for each database class.
        db_embeddings: A dictionary of all embeddings of database's face patches.
        FRmodel: The loaded pretrained model finds embeddings of an image.
        method: Specifed machine learning model used for classification.
    
    Returns:
        model: The model just trained on database ready for recognition.
        encoder: The encoder used along machine learning model, for translate prediction
                from integer to string.
    '''
    
    # split database into 70% training set and 30% test set
    frac = 0.7
    # initialize training and testing data and labels
    x_train = []
    class_train = []
    x_test = []
    class_test = []

    # split on next class in database
    for next_class in os.listdir('database'):
        ebs = db_embeddings[next_class]
        # randomize order of samples
        np.random.shuffle(ebs)

        # find index based on 70%~30% ratio
        num = len(ebs)
        idx = int(np.floor(num * frac))
        # split dataabse
        train = ebs[:idx]
        test = ebs[idx:]
        
        # add to overall training and testing data and labels
        for next_ebd in train:
            x_train.append(next_ebd[0])
            class_train.append(next_class)
        for next_ebd in test:
            x_test.append(next_ebd[0])
            class_test.append(next_class)
    
    # initialize encoder to encode labels, models only accept integer value labels
    encoder = preprocessing.LabelEncoder()
    encoder.fit(class_train + class_test)
    # transform string labels into integers
    y_train = encoder.transform(class_train)
    y_test = encoder.transform(class_test)
    
    # using linear svm model if specified
    if(method == 'svm'):
        model = LinearSVC()
        model.fit(x_train, y_train)
    
    # using knn model if specified
    else:        
        model = KNeighborsClassifier(n_neighbors=1, metric='euclidean')
        model.fit(x_train, y_train)

    # get prediction on test set, calculate and report test accuracy
    prediction = model.predict(x_test)
    accuracy = np.sum(prediction == y_test) / len(prediction)
    print('Machine Learning model is: ', method, ', test accuracy is:', accuracy)
    
    return model, encoder

