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

def mkdir(path):
    
    folder = os.path.exists(path)
 
    if not folder:                   #判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(path)            #makedirs 创建文件时如果路径不存在会创建这个路径
        print("---  new folder...  ---")
        print("---  OK  ---")
    else:
        print("---  There is this folder!  ---")
    
    return folder

def create_dataset(i_month, retlabel = 'abs_ret'):
    """
    Create dataset for i_month, exclude inactive stock, and relabel from return to binary classes.
    
    Arguments:
        i_month - the month data to process. (add multi months data in future)
        retlabel - the return you want to use, 'abs_ret' as default. Other choices: ret_ret ...
    
    Returns:
        ALL DATA IS SAVED!!!! 
        USE load_dataset() TO LOAD!!!
        data was saved in dataset/(test|train)_stock_month_(n).h5
    """
    
    if os.path.exists("datasets/train_stock_month_"+str(i_month)+".h5"):
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
    h5file = h5py.File("datasets/train_stock_month_"+str(i_month)+".h5", "w")
    h5file.create_dataset("train_set_x", data = im_train)
    h5file.create_dataset("train_set_y", data = train_label)
    h5file.create_dataset("stock_name", data = train_stock)
    h5file.close()
    
    h5file = h5py.File("datasets/test_stock_month_"+str(i_month)+".h5", "w")
    h5file.create_dataset("test_set_x", data = im_test)
    h5file.create_dataset("test_set_y", data = test_label)
    h5file.create_dataset("stock_name", data = test_stock)
    h5file.close()
    
    print("Month %d data is save in datasets/ directory, please load" % (i_month))
    
def load_dataset(i_month):
    train_dataset = h5py.File("datasets/train_stock_month_"+str(i_month)+".h5", "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # your train set labels
    train_set_stock = np.array(train_dataset["stock_name"][:])

    test_dataset = h5py.File("datasets/test_stock_month_"+str(i_month)+".h5", "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:]) # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:]) # your test set labels
    test_set_stock = np.array(test_dataset["stock_name"][:])
    
    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))
    
    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, train_set_stock, test_set_stock
