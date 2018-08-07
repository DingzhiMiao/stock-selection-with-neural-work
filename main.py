import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy
from PIL import Image
from scipy import ndimage
from pre_utils import model, predict, create_dataset, load_dataset, mkdir, loginfo
import tensorflow as tf
from tensorflow.python.framework import ops
import logging
import os

start_month = 83
end_month = 83

for end_month in np.arange(100, 113):
    for start_month in np.arange(end_month, end_month-13, -3):
        retlabel = 'rel_ret'
        initialize = False
        
        train_label = '%d-%d' % (start_month, end_month)
        
        for i_month in range(start_month, end_month + 1):
            # create and loading the dataset
            
            if not os.path.exists("datasets/train_stock_month_"+str(i_month)+"_"+ retlabel +".h5"):
                create_dataset(i_month, retlabel)
            train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, train_set_stock, test_set_stock = load_dataset(i_month, retlabel)
            
            
            # flatten the training and test images
            train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
            test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T
            
            if not initialize:
                # normalize image vectors
                train_set_x = np.copy(train_set_x_flatten/255)
                test_set_x = np.copy(test_set_x_flatten/255)
                
                # get training and test labels (without process)
                train_set_y = np.copy(train_set_y_orig)
                test_set_y = np.copy(test_set_y_orig)
                
                initialize = True
            else:
                train_set_x = np.concatenate((train_set_x, train_set_x_flatten/255), axis = 1)
                test_set_x = np.concatenate((test_set_x, test_set_x_flatten/255), axis = 1)
                
                train_set_y = np.concatenate((train_set_y, train_set_y_orig), axis = 1)
                test_set_y = np.concatenate((test_set_y, test_set_y_orig), axis = 1)
        
        # output to log and screen
        mkdir('log')
        
        loginfo(train_label, "Number of examples in training set: %d" % (train_set_x.shape[1]))
        
        parameters = model(train_label, train_set_x, train_set_y, test_set_x, test_set_y, learning_rate= 0.0001)
        
        loginfo(train_label, "Train on %s Month Data, test behavior of next 3 month:" % (train_label))
        
        for i_month in range(end_month+1, end_month+4):
            
            if not os.path.exists("datasets/train_stock_month_"+str(i_month)+"_"+ retlabel +".h5"):
                create_dataset(i_month, retlabel)
            train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, train_set_stock, test_set_stock = load_dataset(i_month)
            
            All_test_x_orig = np.concatenate((train_set_x_orig, test_set_x_orig), axis = 0)
            All_test_y_orig = np.concatenate((train_set_y_orig, test_set_y_orig), axis = 1)
            
            All_test_x_flatten = All_test_x_orig.reshape(All_test_x_orig.shape[0], -1).T
            
            All_test_x = All_test_x_flatten/255
            All_test_y = All_test_y_orig
            
            thisAccuracy = predict(train_label, All_test_x, All_test_y)
            
            loginfo(train_label, 'Test Accuracy for MONTH %d: %f' % (i_month, thisAccuracy))
