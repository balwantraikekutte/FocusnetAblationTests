#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 31 02:13:40 2018

@author: ck807
"""

import os
import glob
import numpy as np
#import cv2

import tensorflow as tf

from keras.models import Model
import matplotlib.pyplot as plt

import keras
from keras import layers
from keras.layers.convolutional import Conv2D, UpSampling2D, SeparableConv2D
from keras.layers.pooling import MaxPooling2D, GlobalAveragePooling2D, AveragePooling2D
from keras.layers import Input, Dropout, Dense, BatchNormalization, Flatten
from keras.regularizers import l2
from keras.optimizers import SGD, Adam
from keras.metrics import mean_absolute_error
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.preprocessing.image import img_to_array

from frozenresidualblockwithlayernames import ResidualR
from frozenresidualblockwithlayernames import initial_conv_block1, Residual2, Residual3, Residual4, Residual5, Residual6
from frozenresidualblockwithlayernames import Residual7, Residual8, Residual9, Residual10, Residual11, Residual12
from frozenresidualblockwithlayernames import Residual13, Residual14, Residual15, Residual16, Residual17, Residual18, Residual19

from se import squeeze_excite_block

from layers import initial_conv_block, bottleneck_block_with_se

from resnet import _conv_bn_relu, _residual_block, basic_block, _bn_relu

import keras.backend as K

CHANNEL_AXIS = 3

print('Loading the data..')
trainData = np.load('trainData.npy')
trainMask = np.load('trainMask.npy')
valData = np.load('valData.npy')
valMask = np.load('valMask.npy')

print('Building and compiling the model..')
smooth=1
def dice_coef(y_true, y_pred, smooth=1):
    intersection = K.sum(y_true * y_pred, axis=[1,2,3])
    union = K.sum(y_true, axis=[1,2,3]) + K.sum(y_pred, axis=[1,2,3])
    return K.mean( (2. * intersection + smooth) / (union + smooth), axis=0)

def dice_coef_loss(y_true, y_pred):
    return 1-dice_coef(y_true, y_pred)

def jaccard_coef(y_true, y_pred):
    intersection = K.sum(y_true * y_pred, axis=[0, -1, -2])
    sum_ = K.sum(y_true + y_pred, axis=[0, -1, -2])
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return K.mean(jac)

with tf.device('/device:GPU:0'):
    input = Input((192, 192, 3), name='Input')
    
    conv1 = _conv_bn_relu(filters=32, kernel_size=(7, 7), strides=(1, 1))(input)
    conv1 = _residual_block(basic_block, filters=32, repetitions=1, is_first_layer=True)(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2), name='MaxPool1')(conv1)
    
    conv2 = _residual_block(basic_block, filters=64, repetitions=1, is_first_layer=True)(pool1)
    conv2 = _residual_block(basic_block, filters=64, repetitions=1, is_first_layer=True)(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2), name='MaxPool2')(conv2)
    
    conv3 = _residual_block(basic_block, filters=128, repetitions=1, is_first_layer=True)(pool2)
    conv3 = _residual_block(basic_block, filters=128, repetitions=1, is_first_layer=True)(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2), name='MaxPool3')(conv3)
    
    conv4 = _residual_block(basic_block, filters=256, repetitions=1, is_first_layer=True)(pool3)
    conv4 = _residual_block(basic_block, filters=256, repetitions=1, is_first_layer=True)(conv4)
    drop4 = Dropout(0.2, name='Dropout1')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2), name='MaxPool4')(drop4)
    
    conv5 = _residual_block(basic_block, filters=256, repetitions=1, is_first_layer=True)(pool4)
    conv5 = _residual_block(basic_block, filters=256, repetitions=1, is_first_layer=True)(conv5)
    drop5 = Dropout(0.2, name='Dropout2')(conv5)
    
    up6 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', name='UpConv1')(UpSampling2D(size = (2,2), name='Up1')(drop5))
    merge6 = keras.layers.Concatenate(name='Concat1')([drop4,up6])
    conv6 = _residual_block(basic_block, filters=128, repetitions=1, is_first_layer=True)(merge6)
    conv6_1 = _residual_block(basic_block, filters=128, repetitions=1, is_first_layer=True)(conv6)
    
    up7 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', name='UpConv2')(UpSampling2D(size = (2,2), name='Up2')(conv6_1))
    merge7 = keras.layers.Concatenate(name='Concat2')([conv3,up7])
    conv7 = _residual_block(basic_block, filters=64, repetitions=1, is_first_layer=True)(merge7)
    conv7_1 = _residual_block(basic_block, filters=64, repetitions=1, is_first_layer=True)(conv7)
    
    up8 = Conv2D(32, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', name='UpConv3')(UpSampling2D(size = (2,2), name='Up3')(conv7_1))
    merge8 = keras.layers.Concatenate(name='Concat3')([conv2,up8])
    conv8 = _residual_block(basic_block, filters=32, repetitions=1, is_first_layer=True)(merge8)
    conv8_1 = _residual_block(basic_block, filters=32, repetitions=1, is_first_layer=True)(conv8)
    
    up9 = Conv2D(16, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', name='UpConv4')(UpSampling2D(size = (2,2), name='Up4')(conv8_1))
    merge9 = keras.layers.Concatenate(name='Concat4')([conv1,up9])
    conv9 = _residual_block(basic_block, filters=16, repetitions=1, is_first_layer=True)(merge9)
    conv10 = _residual_block(basic_block, filters=4, repetitions=1, is_first_layer=True)(conv9)
    conv10 = _residual_block(basic_block, filters=1, repetitions=1, is_first_layer=True)(conv10)
    conv11 = Conv2D(1, 1, activation = 'sigmoid', name='Output')(conv10)
    
    model = Model(input, conv11)
    
model.compile(loss=dice_coef_loss, optimizer=SGD(lr=0.03, momentum=0.9, nesterov=True), metrics=[dice_coef,jaccard_coef])

model.summary()
    
callbacks = [
    EarlyStopping(monitor='val_loss', patience=10, verbose=1, min_delta=0.001),
    ModelCheckpoint("resnet_faster_val_loss_ldmk_checkpoint.h5", monitor='val_loss', verbose=1, save_best_only=True, mode='auto'),
    ReduceLROnPlateau(monitor='val_loss', patience=5, verbose=1, mode='auto', epsilon=0.001, cooldown=0, min_lr=0.5e-7)
            ]

history = model.fit(trainData, trainMask, validation_data=(valData, valMask), batch_size=32, epochs=200, verbose=1, shuffle=True, callbacks=callbacks)
