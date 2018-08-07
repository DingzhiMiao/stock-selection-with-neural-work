# -*- coding: utf-8 -*-
"""
Created on Tue Jul 24 14:14:48 2018

@author: Dingzhi Hu
"""

import xlrd
import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy
from PIL import Image
from scipy import ndimage
import pandas as pd
import random
import os
import tensorflow as tf
import math
from tensorflow.python.framework import ops
import logging


def mkdir(path):
    
    folder = os.path.exists(path)
 
    if not folder:                   #判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(path)            #makedirs 创建文件时如果路径不存在会创建这个路径
        print("---  new folder...  ---")
        print("---  OK  ---")
    else:
        print("---  There is this folder!  ---")
    
    return folder

def create_dataset(i_month, retlabel = 'rel_ret'):
    """
    Create dataset for i_month, exclude inactive stock, and relabel from return to binary classes.
    
    Arguments:
        i_month - the month data to process. (add multi months data in future)
        retlabel - the return you want to use, 'rel_ret' as default. Other choices: abs_ret ...
    
    Returns:
        ALL DATA IS SAVED!!!! 
        USE load_dataset() TO LOAD!!!
        data was saved in dataset/(test|train)_stock_month_(n)_(retlabel).h5
    """
    
    if os.path.exists("datasets/train_stock_month_"+str(i_month)+"_"+ retlabel +".h5"):
        raw_input_A = input("dataset already exist, want to recreate? (y/n)")
        if raw_input_A == "n":
            return
    
    
    # Get excel data and process the ret
    #   Rule: +30% = 1; -30% = 0
    fname = "pic/month_%d/data/retinfo_%d.xlsx" % (i_month, i_month)
    df = pd.read_excel(fname)
    
    # exclude useless data
    usedata = df.loc[df['status'] == 1]
    
    # change from dataframework to np data
    raw_y = usedata[retlabel].values
    
    # positive and negative 30 percentile point
    pos_point = np.percentile(raw_y[raw_y >= 0],40)
    neg_point = np.percentile(raw_y[raw_y < 0],60)
    
    # get labels
    processdata = df.loc[((df[retlabel] >= pos_point) | (df[retlabel] <=neg_point)) & (df['status'] == 1)]
    final_stock = processdata['id'].values
    final_y = processdata[retlabel].values
    final_label = np.copy(final_y)
    final_label[final_y >= 0] = 1
    final_label[final_y < 0] = 0
    
    # random data to create training and dev dataset
    r = np.random.permutation(len(final_label))
    rand_label = final_label[r]
    rand_stock = final_stock[r]
    
    test_size = 120
    test_label = rand_label[0:test_size]
    test_stock = rand_stock[0:test_size]
    
    train_label = rand_label[test_size:len(rand_label)]
    train_stock = rand_stock[test_size:len(rand_stock)]
    
    # create test img and train img
    initialize = False
    for i_stock in test_stock:
        fname = "pic/month_%d/img/img_%d_%d.png" % (i_month, i_month, i_stock)
        image = np.array(Image.open(fname))
        if initialize:
            im_test = np.concatenate((im_test, image), axis = 0)
        else:
            im_test = np.copy(image)
            initialize = True
    im_test = im_test.reshape(len(test_label), 224, 224, 3)
    
    initialize = False
    for i_stock in train_stock:
        fname = "pic/month_%d/img/img_%d_%d.png" % (i_month, i_month, i_stock)
        image = np.array(Image.open(fname))
        if initialize:
            im_train = np.concatenate((im_train, image), axis = 0)
        else:
            im_train = np.copy(image)
            initialize = True
    im_train = im_train.reshape(len(train_label), 224, 224, 3)  
    
    mkdir('datasets')
    
    # create h5 file
    h5file = h5py.File("datasets/train_stock_month_"+str(i_month)+"_"+ retlabel +".h5", "w")
    h5file.create_dataset("train_set_x", data = im_train)
    h5file.create_dataset("train_set_y", data = train_label)
    h5file.create_dataset("stock_name", data = train_stock)
    h5file.close()
    
    h5file = h5py.File("datasets/test_stock_month_"+str(i_month)+"_"+ retlabel +".h5", "w")
    h5file.create_dataset("test_set_x", data = im_test)
    h5file.create_dataset("test_set_y", data = test_label)
    h5file.create_dataset("stock_name", data = test_stock)
    h5file.close()
    
    print("Month %d data using label %s is save in datasets/ directory, please load" % (i_month, retlabel))
    
def load_dataset(i_month, retlabel = 'rel_ret'):
    train_dataset = h5py.File("datasets/train_stock_month_"+str(i_month)+"_"+ retlabel +".h5", "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # your train set labels
    train_set_stock = np.array(train_dataset["stock_name"][:])

    test_dataset = h5py.File("datasets/test_stock_month_"+str(i_month)+"_"+ retlabel +".h5", "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:]) # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:]) # your test set labels
    test_set_stock = np.array(test_dataset["stock_name"][:])
    
    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))
    
    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, train_set_stock, test_set_stock

# GRADED FUNCTION: create_placeholders


def create_placeholders(n_x, n_y):
    """
    Creates the placeholders for the tensorflow session.

    Arguments:
    n_x -- scalar, size of an image vector (num_px * num_px = 64 * 64 * 3 = 12288)
    n_y -- scalar, number of classes (from 0 to 5, so -> 6)

    Returns:
    X -- placeholder for the data input, of shape [n_x, None] and dtype "float"
    Y -- placeholder for the input labels, of shape [n_y, None] and dtype "float"

    Tips:
    - You will use None because it let's us be flexible on the number of examples you will for the placeholders.
      In fact, the number of examples during test/train is different.

    from coursera.org
    """

    X = tf.placeholder(dtype=tf.float32, shape=[n_x, None])
    Y = tf.placeholder(dtype=tf.float32, shape=[n_y, None])

    return X, Y

# GRADED FUNCTION: initialize_parameters


def initialize_parameters():
    """
    Initializes parameters to build a neural network with tensorflow. The shapes are:
                        W1 : [25, 150528]
                        b1 : [25, 1]
                        W2 : [12, 25]
                        b2 : [12, 1]
                        W3 : [1, 12]
                        b3 : [1, 1]

    Returns:
    parameters -- a dictionary of tensors containing W1, b1, W2, b2, W3, b3
    
    from cousera.org
    """

    W1 = tf.get_variable("W1", [25, 150528], initializer=tf.contrib.layers.xavier_initializer())
    b1 = tf.get_variable("b1", [25, 1], initializer=tf.zeros_initializer())
    W2 = tf.get_variable("W2", [12, 25], initializer=tf.contrib.layers.xavier_initializer())
    b2 = tf.get_variable("b2", [12, 1], initializer=tf.zeros_initializer())
    W3 = tf.get_variable("W3", [1, 12], initializer=tf.contrib.layers.xavier_initializer())
    b3 = tf.get_variable("b3", [1, 1], initializer=tf.zeros_initializer())

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2,
                  "W3": W3,
                  "b3": b3}

    return parameters

# GRADED FUNCTION: forward_propagation

def forward_propagation(X, parameters):
    """
    Implements the forward propagation for the model: LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SOFTMAX
    
    Arguments:
    X -- input dataset placeholder, of shape (input size, number of examples)
    parameters -- python dictionary containing your parameters "W1", "b1", "W2", "b2", "W3", "b3"
                  the shapes are given in initialize_parameters

    Returns:
    A3 -- the output of the last LINEAR unit
    
    from coursera.org
    """
    
    # Retrieve the parameters from the dictionary "parameters" 
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    W3 = parameters['W3']
    b3 = parameters['b3']
                                            # Numpy Equivalents:
    Z1 = tf.add(tf.matmul(W1, X), b1)       # Z1 = np.dot(W1, X) + b1
    A1 = tf.nn.relu(Z1)                     # A1 = relu(Z1)
    Z2 = tf.add(tf.matmul(W2, A1), b2)      # Z2 = np.dot(W2, a1) + b2
    A2 = tf.nn.relu(Z2)                     # A2 = relu(Z2)
    Z3 = tf.add(tf.matmul(W3, A2), b3)      # Z3 = np.dot(W3,Z2) + b3
    A3 = tf.sigmoid(Z3)                     # A3 = np.sigmoid(Z3)
    
    return A3

# GRADED FUNCTION: compute_cost 

def compute_cost(A3, Y):
    """
    Computes the cost
    
    Arguments:
    A3 -- output of forward propagation (output of the last LINEAR unit), of shape (6, number of examples)
    Y -- "true" labels vector placeholder, same shape as Z3
    
    Returns:
    cost - Tensor of the cost function
    
    from coursera.org
    """
    
    # to fit the tensorflow requirement for tf.nn.sigmoid_cross_entropy_with_logits(...,...)
    logits = tf.transpose(A3)
    labels = tf.transpose(Y)
    
    costraw = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = logits, labels = labels))
    
    return costraw

def random_mini_batches(X, Y, mini_batch_size = 64):
    """
    Creates a list of random minibatches from (X, Y)
    
    Arguments:
    X -- input data, of shape (input size, number of examples)
    Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples)
    mini_batch_size - size of the mini-batches, integer
    seed -- this is only for the purpose of grading, so that you're "random minibatches are the same as ours.
    
    Returns:
    mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)
    
    from coursera.org
    """
    
    m = X.shape[1]                  # number of training examples
    mini_batches = []
    
    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation].reshape((Y.shape[0],m))

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = math.floor(m/mini_batch_size) # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[:, k * mini_batch_size : k * mini_batch_size + mini_batch_size]
        mini_batch_Y = shuffled_Y[:, k * mini_batch_size : k * mini_batch_size + mini_batch_size]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[:, num_complete_minibatches * mini_batch_size : m]
        mini_batch_Y = shuffled_Y[:, num_complete_minibatches * mini_batch_size : m]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    return mini_batches


def model(train_label, X_train, Y_train, X_test, Y_test, learning_rate = 0.0001,
          num_epochs = 1500, minibatch_size = 32, print_cost = True, beta = 0.01):
    """
    Implements a three-layer tensorflow neural network: LINEAR->RELU->LINEAR->RELU->LINEAR->SOFTMAX.
    
    Arguments:
    X_train -- training set, of shape (input size = 12288, number of training examples = 1080)
    Y_train -- test set, of shape (output size = 6, number of training examples = 1080)
    X_test -- training set, of shape (input size = 12288, number of training examples = 120)
    Y_test -- test set, of shape (output size = 6, number of test examples = 120)
    learning_rate -- learning rate of the optimization
    num_epochs -- number of epochs of the optimization loop
    minibatch_size -- size of a minibatch
    print_cost -- True to print the cost every 100 epochs
    
    Returns:
    parameters -- parameters learnt by the model. They can then be used to predict.
    
    from coursera.org
    """
    
    ops.reset_default_graph()                         # to be able to rerun the model without overwriting tf variables
    (n_x, m) = X_train.shape                          # (n_x: input size, m : number of examples in the train set)
    n_y = Y_train.shape[0]                            # n_y : output size
    costs = []                                        # To keep track of the cost
    
    # Create Placeholders of shape (n_x, n_y)
    X, Y = create_placeholders(n_x, n_y)
    
    # Initialize parameters
    parameters = initialize_parameters()
    
    # Forward propagation: Build the forward propagation in the tensorflow graph
    A3 = forward_propagation(X, parameters)
    
    # Cost function: Add cost function to tensorflow graph
    costraw = compute_cost(A3, Y)
    regularizers = tf.nn.l2_loss(parameters["W1"]) + tf.nn.l2_loss(parameters["W2"]) + tf.nn.l2_loss(parameters["W3"])
    
    cost = tf.reduce_mean(costraw + beta * regularizers)
    
    # Backpropagation: Define the tensorflow optimizer. Use an AdamOptimizer.
    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)
    
    # Initialize all the variables
    init = tf.global_variables_initializer()
    
    # Save parameters for later use
    saver = tf.train.Saver()
    
    # Start the session to compute the tensorflow graph
    with tf.Session() as sess:
        
        # Run the initialization
        sess.run(init)
        
        # to save time, if there is model, use it for output
        if os.path.exists('model/month_%s.ckpt.index' % (train_label)):
            load_path = saver.restore(sess, 'model/month_%s.ckpt' % (train_label))
            loginfo(train_label, "Model restored from file: %s\n" % (load_path))
        else:
            # Do the training loop
            for epoch in range(num_epochs):
        
                epoch_cost = 0.                       # Defines a cost related to an epoch
                num_minibatches = int(m / minibatch_size) # number of minibatches of size minibatch_size in the train set
                minibatches = random_mini_batches(X_train, Y_train, minibatch_size)
        
                for minibatch in minibatches:
        
                    # Select a minibatch
                    (minibatch_X, minibatch_Y) = minibatch
                    
                    # IMPORTANT: The line that runs the graph on a minibatch.
                    # Run the session to execute the "optimizer" and the "cost", the feedict should contain a minibatch for (X,Y).
                    _ , minibatch_cost = sess.run([optimizer, cost], feed_dict = {X: minibatch_X, Y: minibatch_Y})
                    
                    epoch_cost += minibatch_cost / num_minibatches
        
                # Print the cost every epoch
                if print_cost == True and epoch % 100 == 0:
                    loginfo(train_label, "Cost after epoch %i: %f" % (epoch, epoch_cost))
                if print_cost == True and epoch % 5 == 0:
                    costs.append(epoch_cost)
                    
            # plot the cost
            plt.plot(np.squeeze(costs))
            plt.ylabel('cost')
            plt.xlabel('iterations (per tens)')
            plt.title("Learning rate =" + str(learning_rate))
            plt.show()
            
            # save the model
            mkdir('model')
            save_path = saver.save(sess, 'model/month_%s.ckpt' % (train_label))
            print("Model saved in file: %s" % (save_path))
            loginfo(train_label, "Parameters have been trained!\n")
        
        # lets save the parameters in a variable
        parameters = sess.run(parameters)
    
        # Calculate the correct predictions
        correct_prediction = tf.equal(tf.round(A3), Y)
    
        # Calculate accuracy on the test set
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        
        loginfo(train_label, "Train on %s Month Data" % (train_label))
        loginfo(train_label, "Train Accuracy: %f" % accuracy.eval({X: X_train, Y: Y_train}))
        loginfo(train_label, "Test Accuracy: %f" % accuracy.eval({X: X_test, Y: Y_test}))
        
        
        
        return parameters

def predict(train_label, X_test, Y_test):
    
    ops.reset_default_graph()                         # to be able to rerun the model without overwriting tf variables
    (n_x, m) = X_test.shape                          # (n_x: input size, m : number of examples in the train set)
    n_y = Y_test.shape[0]                            # n_y : output size
    
    # Create Placeholders of shape (n_x, n_y)
    X, Y = create_placeholders(n_x, n_y)
    
    # Initialize parameters
    parameters = initialize_parameters()
    
    # Forward propagation: Build the forward propagation in the tensorflow graph
    A3 = forward_propagation(X, parameters)

    # Initialize all the variables
    init = tf.global_variables_initializer()
    
    # to load parameters
    saver = tf.train.Saver()
    
    # Start the session to compute the tensorflow graph
    with tf.Session() as sess:
        
        # Run the initialization
        sess.run(init)
        
        # Restore Model
        load_path = saver.restore(sess, 'model/month_%s.ckpt' % (train_label))
        print("Model restored from file: %s" % (load_path))
        
        # Calculate the correct predictions
        correct_prediction = tf.equal(tf.round(A3), Y)
    
        # Calculate accuracy on the test set
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        
        thisAccuracy = accuracy.eval({X: X_test, Y: Y_test})
        
        return thisAccuracy

def loginfo(train_label, message):
    
    logging.basicConfig(level=logging.DEBUG)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s: - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S')
    
    logger = logging.getLogger('mylog')
    
    filehandler = logging.FileHandler('log/%s.log' % (train_label))
    filehandler.setLevel(level=logging.DEBUG)
    filehandler.setFormatter(formatter)
    logger.addHandler(filehandler)
        
    logger.info(message)
    logger.removeHandler(filehandler)
    