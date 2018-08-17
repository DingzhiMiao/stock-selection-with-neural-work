# -*- coding: utf-8 -*-
"""
Created on Thu Aug 16 18:19:21 2018

@author: Yulab
"""

import numpy as np
from keras import layers
from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D, Dropout
from keras.models import Model, load_model
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
import pydot
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
from keras.initializers import glorot_uniform
import scipy.misc
from matplotlib.pyplot import imshow

import keras.backend as K
K.set_image_data_format('channels_last')
K.set_learning_phase(1)


def VGG16_Model(input_shape):
    """
    Implementation of the HappyModel.
    
    Arguments:
    input_shape -- shape of the images of the dataset

    Returns:
    model -- a Model() instance in Keras
    """
    
    ### START CODE HERE ###
    # Feel free to use the suggested outline in the text above to get started, and run through the whole
    # exercise (including the later portions of this notebook) once. The come back also try out other
    # network architectures as well. 
    
    # Define the input placeholder as a tensor with shape input_shape. Think of this as your input image!
    X_input = Input(input_shape)
    
    # 1st layer, 2 Conv layers, (3, 3) * 64
    X = Conv2D(64, (3, 3), strides=(1,1), padding='same', name='conv_1a', kernel_initializer = glorot_uniform())(X_input)
    X = Activation('relu')(X)
    X = Conv2D(64, (3, 3), strides=(1,1), padding='same', name='conv_1b', kernel_initializer = glorot_uniform())(X)
    X = Activation('relu')(X)
    
    X = MaxPooling2D((2,2), strides=(2,2))(X)
    
    # 2nd layer, 2 Conv layers, (3, 3) * 128
    X = Conv2D(128, (3, 3), strides=(1,1), padding='same', name='conv_2a', kernel_initializer = glorot_uniform())(X)
    X = Activation('relu')(X)
    X = Conv2D(128, (3, 3), strides=(1,1), padding='same', name='conv_2b', kernel_initializer = glorot_uniform())(X)
    X = Activation('relu')(X)
    
    X = MaxPooling2D((2,2), strides=(2,2))(X)
    
    # 3rd layer, 3 Conv layers, (3, 3) * 256
    X = Conv2D(256, (3, 3), strides=(1,1), padding='same', name='conv_3a', kernel_initializer = glorot_uniform())(X)
    X = Activation('relu')(X)
    X = Conv2D(256, (3, 3), strides=(1,1), padding='same', name='conv_3b', kernel_initializer = glorot_uniform())(X)
    X = Activation('relu')(X)
    X = Conv2D(256, (3, 3), strides=(1,1), padding='same', name='conv_3c', kernel_initializer = glorot_uniform())(X)
    X = Activation('relu')(X)
    
    X = MaxPooling2D((2,2), strides=(2,2))(X)
    
    # 4th layer, 3 Conv layers, (3, 3) * 512
    X = Conv2D(512, (3, 3), strides=(1,1), padding='same', name='conv_4a', kernel_initializer = glorot_uniform())(X)
    X = Activation('relu')(X)
    X = Conv2D(512, (3, 3), strides=(1,1), padding='same', name='conv_4b', kernel_initializer = glorot_uniform())(X)
    X = Activation('relu')(X)
    X = Conv2D(512, (3, 3), strides=(1,1), padding='same', name='conv_4c', kernel_initializer = glorot_uniform())(X)
    X = Activation('relu')(X)
    
    X = MaxPooling2D((2,2), strides=(2,2))(X)
    
    # 5th layer, 3 Conv layers, (3, 3) * 512
    X = Conv2D(512, (3, 3), strides=(1,1), padding='same', name='conv_5a', kernel_initializer = glorot_uniform())(X)
    X = Activation('relu')(X)
    X = Conv2D(512, (3, 3), strides=(1,1), padding='same', name='conv_5b', kernel_initializer = glorot_uniform())(X)
    X = Activation('relu')(X)
    X = Conv2D(512, (3, 3), strides=(1,1), padding='same', name='conv_5c', kernel_initializer = glorot_uniform())(X)
    X = Activation('relu')(X)
    
    X = MaxPooling2D((2,2), strides=(2,2))(X)
    
    # 6th layer, two flatten layer, 4096
    X = Flatten()(X)
    X = Dense(4096, activation='relu', name = 'fc_6a', kernel_initializer=glorot_uniform())(X)
    X = Dropout(0.5)(X)
    X = Dense(4096, activation='relu', name = 'fc_6b', kernel_initializer=glorot_uniform())(X)
    X = Dropout(0.5)(X)
    
    # output layer, softmax
    X = Dense(1, activation='sigmoid', name = 'fc_out', kernel_initializer=glorot_uniform())(X)
    
    # Create model. This creates your Keras model instance, you'll use this instance to train/test the model.
    model = Model(inputs = X_input, outputs = X, name='VGG16_Model')
    
    
    return model