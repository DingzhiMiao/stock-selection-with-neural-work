# -*- coding: utf-8 -*-
"""
Created on Fri Aug 31 23:54:51 2018

@author: Yulab
"""

import os
import os.path

import numpy as np
import tensorflow as tf
from pre_utils import create_dataset, load_dataset, mkdir, loginfo
import VGG16_tf_model
import VGG_utils

#%%
IMG_W = 224
IMG_H = 224
N_CLASSES = 2
BATCH_SIZE = 32
LEARNING_RATE = 0.01
MAX_STEP = 15000   # it took me about one hour to complete the training.
IS_PRETRAIN = False
RET_LABEL = 'rel_ret'

def train(train_label, X_train, Y_train, X_test, Y_test, keep_prob = 0.5):
    
    train_log_dir = './/logs//train//'+train_label+'//'
    val_log_dir = './/logs//val//'+train_label+'//'
    
    # Create Placeholders of shape (n_x, n_y)
    X = tf.placeholder(tf.float32, shape=[None, IMG_W, IMG_H, 3])
    Y = tf.placeholder(tf.int16, shape=[None, N_CLASSES])
    KP = tf.placeholder(tf.float32)
    
    logits = VGG16_tf_model.VGG16(X, N_CLASSES, IS_PRETRAIN, keep_prob = KP)
    loss = VGG_utils.loss(logits, Y)
    accuracy = VGG_utils.accuracy(logits, Y)
    
    my_global_step = tf.Variable(0, name='global_step', trainable=False)    # mark which step it is
    train_op = VGG_utils.optimize(loss, LEARNING_RATE, my_global_step)
    
    saver = tf.train.Saver(tf.global_variables())
    summary_op = tf.summary.merge_all()
    
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    tra_summary_writer = tf.summary.FileWriter(train_log_dir, sess.graph)
    val_summary_writer = tf.summary.FileWriter(val_log_dir, sess.graph)
    
    try:
        step = 0
        while step < MAX_STEP:
            if coord.should_stop():
                break
            minibatches = VGG_utils.random_mini_batches(X_train, Y_train, BATCH_SIZE)
            
            for minibatch in minibatches:
                    # Select a minibatch
                    (minibatch_X, minibatch_Y) = minibatch
                    _, tra_loss, tra_acc = sess.run([train_op, loss, accuracy], 
                                                    feed_dict={X:minibatch_X, Y:minibatch_Y, KP:keep_prob})
                    if step % 50 == 0 or (step + 1) == MAX_STEP:
                        print('Step: %d, loss: %.4f, accuracy: %.4f%%' % (step, tra_loss, tra_acc))
                        summary_str = sess.run(summary_op)
                        tra_summary_writer.add_summary(summary_str, step)
                    
                    if step % 200 == 0 or (step + 1) == MAX_STEP:
                        val_loss, val_acc = sess.run([loss, accuracy],
                                                     feed_dict={X:X_test, Y:Y_test, KP:1})
                        print('**  Step %d, val loss = %.2f, val accuracy = %.2f%%  **' %(step, val_loss, val_acc))
                        
                        summary_str = sess.run(summary_op)
                        val_summary_writer.add_summary(summary_str, step)
                    
                    if step % 2000 == 0 or (step + 1) == MAX_STEP:
                        checkpoint_path = os.path.join(train_log_dir, 'model.ckpt')
                        saver.save(sess, checkpoint_path, global_step=step)
                    
                    step += 1
                        
    except tf.errors.OutOfRangeError:
        print('Done training --- epoch limit reached')
    finally:
        coord.request_stop()
    
    coord.join(threads)
    sess.close()
                        
#%%
def convert_to_one_hot(y, C):
    y = y.astype(dtype = int)
    return np.eye(C)[y.reshape(-1)]
        
#%%
def model(start_month=100, end_month=101):
    
    retlabel = RET_LABEL
    initialize = False  
    train_label = '%d-%d' % (start_month, end_month)
    
    for i_month in range(start_month, end_month + 1):
        # create and loading the dataset
        
        if not os.path.exists("../old/datasets/train_stock_month_"+str(i_month)+"_"+ retlabel +".h5"):
            create_dataset(i_month, retlabel)
        train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, train_set_stock, test_set_stock = load_dataset(i_month, retlabel)
        
        
        # flatten the training and test images
        train_set_x_flatten = train_set_x_orig
        test_set_x_flatten = test_set_x_orig
        
        if not initialize:
            # normalize image vectors
            train_set_x = np.copy(train_set_x_flatten/255)
            test_set_x = np.copy(test_set_x_flatten/255)
            
            # get training and test labels (without process)
            train_set_y = np.copy(train_set_y_orig)
            test_set_y = np.copy(test_set_y_orig)
            
            initialize = True
        else:
            train_set_x = np.concatenate((train_set_x, train_set_x_flatten/255), axis = 0)
            test_set_x = np.concatenate((test_set_x, test_set_x_flatten/255), axis = 0)
            
            train_set_y = np.concatenate((train_set_y, train_set_y_orig), axis = 1)
            test_set_y = np.concatenate((test_set_y, test_set_y_orig), axis = 1)
    
    
    train_set_y = convert_to_one_hot(train_set_y, N_CLASSES)
    test_set_y = convert_to_one_hot(test_set_y, N_CLASSES)
    train(train_label, train_set_x, train_set_y, test_set_x, test_set_y, keep_prob=0.5)
    