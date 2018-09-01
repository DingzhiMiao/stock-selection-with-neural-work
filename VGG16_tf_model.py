# -*- coding: utf-8 -*-
"""
Created on Fri Aug 31 23:27:51 2018

@author: Yulab
"""

import tensorflow as tf
import VGG_utils

#%%
def VGG16(x, n_classes, is_pretrain=True, keep_prob=0.5, seed=1):
    
    x = VGG_utils.conv('conv1_1', x, 64, is_pretrain=is_pretrain, seed=seed)
    x = VGG_utils.conv('conv1_2', x, 64, is_pretrain=is_pretrain, seed=seed)
    with tf.name_scope('pool1'):
        x = VGG_utils.pool('pool1', x)
    
    x = VGG_utils.conv('conv2_1', x, 128, is_pretrain=is_pretrain, seed=seed)
    x = VGG_utils.conv('conv2_2', x, 128, is_pretrain=is_pretrain, seed=seed)
    with tf.name_scope('pool2'):
        x = VGG_utils.pool('pool2', x)
    
    x = VGG_utils.conv('conv3_1', x, 256, is_pretrain=is_pretrain, seed=seed)
    x = VGG_utils.conv('conv3_2', x, 256, is_pretrain=is_pretrain, seed=seed)
    x = VGG_utils.conv('conv3_3', x, 256, is_pretrain=is_pretrain, seed=seed)    
    with tf.name_scope('pool3'):
        x = VGG_utils.pool('pool3', x)
    
    x = VGG_utils.conv('conv4_1', x, 512, is_pretrain=is_pretrain, seed=seed)
    x = VGG_utils.conv('conv4_2', x, 512, is_pretrain=is_pretrain, seed=seed)
    x = VGG_utils.conv('conv4_3', x, 512, is_pretrain=is_pretrain, seed=seed)    
    with tf.name_scope('pool4'):
        x = VGG_utils.pool('pool4', x)
    
    x = VGG_utils.conv('conv5_1', x, 512, is_pretrain=is_pretrain, seed=seed)
    x = VGG_utils.conv('conv5_2', x, 512, is_pretrain=is_pretrain, seed=seed)
    x = VGG_utils.conv('conv5_3', x, 512, is_pretrain=is_pretrain, seed=seed)    
    with tf.name_scope('pool5'):
        x = VGG_utils.pool('pool5', x)
        
    x = VGG_utils.FC_layer('fc6', x, out_nodes=4096, keep_prob=1, seed=seed)
    x = VGG_utils.FC_layer('fc7', x, out_nodes=4096, keep_prob=keep_prob, seed=seed)
    x = VGG_utils.FC_layer('fc8', x, out_nodes=n_classes, keep_prob=keep_prob, seed=seed)
    
    return x
