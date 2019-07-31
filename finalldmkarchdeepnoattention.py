#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 21 16:00:48 2018

@author: ck807
"""

import numpy as np

import tensorflow as tf

from keras.models import Model
import matplotlib.pyplot as plt

import keras
from keras import layers
from keras.layers.convolutional import Conv2D, UpSampling2D, SeparableConv2D
from keras.layers.pooling import MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Input, Dropout, Dense, BatchNormalization
from keras.regularizers import l2
from keras.optimizers import SGD, RMSprop, Adam
from keras.metrics import mean_absolute_error
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from frozenresidualblockwithlayernames import ResidualR
from frozenresidualblockwithlayernames import initial_conv_block1, Residual2, Residual3, Residual4, Residual5, Residual6
from frozenresidualblockwithlayernames import Residual7, Residual8, Residual9, Residual10, Residual11, Residual12
from frozenresidualblockwithlayernames import Residual13, Residual14, Residual15, Residual16, Residual17, Residual18, Residual19

from se import squeeze_excite_block

from layers import initial_conv_block, bottleneck_block_with_se

import keras.backend as K


trainData = np.load('trainDataRegressor.npy')
trainLabel = np.load('trainLabelRegressor.npy')
valData = np.load('valDataRegressor.npy')
valLabel = np.load('valLabelRegressor.npy')

#trainData = trainData[0:3500,:,:,:]
#trainLabel = trainLabel[0:3500,:]
#valData = valData[0:500,:,:,:]
#valLabel = valLabel[0:500,:]


def huber_loss(y_true, y_pred, clip_delta=1.0):
  error = y_true - y_pred
  cond  = K.abs(error) < clip_delta

  squared_loss = 0.5 * K.square(error)
  linear_loss  = clip_delta * (K.abs(error) - 0.5 * clip_delta)

  return tf.where(cond, squared_loss, linear_loss)

def huber_loss_mean(y_true, y_pred, clip_delta=1.0):
  return K.mean(huber_loss(y_true, y_pred, clip_delta))

w = 10.0
e = 2.0
c = w - w * K.log(1 + (w/e))
#print('Wing Loss Parameters:')
#print('w = ', w)
#print('e = ', e)
#sess=tf.Session()
#print('c = ', sess.run(c))

def wingLoss(y_true, y_pred, w=w, e=e, c=c):
    error = y_true - y_pred
    cond = K.abs(error) < w
    true = w * (K.log(1 + (K.abs(error)/e)))
    otherwise = K.abs(error) - c
    return tf.where(cond, true, otherwise)


with tf.device('/device:GPU:3'):
    inputs = Input((192, 192, 3), name='Input')
    init = initial_conv_block(inputs, weight_decay=5e-4)
    
    #x1 = ResidualR(32, 64, init)    #192x192x64
    #x1 = ResidualR(64, 64, x1)
    #x1 = ResidualR(64, 64, x1)    #192x192x64
    x1 = Conv2D(64, (3,3), padding='same', kernel_initializer='he_normal')(init)
    x1 = BatchNormalization()(x1)
    x1 = layers.LeakyReLU()(x1)
    x1 = Conv2D(64, (3,3), padding='same', kernel_initializer='he_normal')(x1)
    x1 = BatchNormalization()(x1)
    x1 = layers.LeakyReLU()(x1)
    x1pool = MaxPooling2D(pool_size=(2,2))(x1)
    
    #x2 = ResidualR(64, 96, x1pool)   #96x96x96
    #x2 = ResidualR(96, 96, x2)
    #x2 = ResidualR(96, 96, x2)   #96x96x96
    x2 = Conv2D(96, (3,3), padding='same', kernel_initializer='he_normal')(x1pool)
    x2 = BatchNormalization()(x2)
    x2 = layers.LeakyReLU()(x2)
    x2 = Conv2D(96, (3,3), padding='same', kernel_initializer='he_normal')(x2)
    x2 = BatchNormalization()(x2)
    x2 = layers.LeakyReLU()(x2)
    x2pool = MaxPooling2D(pool_size=(2,2))(x2)

#with tf.device('/device:GPU:2'):   
    #x3 = ResidualR(96, 128, x2pool)   #48x48x128
    #x3 = ResidualR(128, 128, x3)
    #x3 = ResidualR(128, 128, x3)   #48x48x128
    x3 = Conv2D(128, (3,3), padding='same', kernel_initializer='he_normal')(x2pool)
    x3 = BatchNormalization()(x3)
    x3 = layers.LeakyReLU()(x3)
    x3 = Conv2D(128, (3,3), padding='same', kernel_initializer='he_normal')(x3)
    x3 = BatchNormalization()(x3)
    x3 = layers.LeakyReLU()(x3)
    x3pool = MaxPooling2D(pool_size=(2,2))(x3)
    
    #x4 = ResidualR(128, 256, x3pool)   #24x24x256
    #x4 = ResidualR(256, 256, x4)
    #x4 = ResidualR(256, 256, x4)   #24x24x256
    x4 = Conv2D(256, (3,3), padding='same', kernel_initializer='he_normal')(x3pool)
    x4 = BatchNormalization()(x4)
    x4 = layers.LeakyReLU()(x4)
    x4 = Conv2D(256, (3,3), padding='same', kernel_initializer='he_normal')(x4)
    x4 = BatchNormalization()(x4)
    x4 = layers.LeakyReLU()(x4)
    x4pool = MaxPooling2D(pool_size=(2,2))(x4)

with tf.device('/device:GPU:2'):    
    #x5 = ResidualR(256, 256, x4pool)   #12x12x256
    #x5 = ResidualR(256, 256, x5)
    #x5 = ResidualR(256, 256, x5)   #12x12x256
    x5 = Conv2D(512, (3,3), padding='same', kernel_initializer='he_normal')(x4pool)
    x5 = BatchNormalization()(x5)
    x5 = layers.LeakyReLU()(x5)
    x5 = Conv2D(512, (3,3), padding='same', kernel_initializer='he_normal')(x5)
    x5 = BatchNormalization()(x5)
    x5 = layers.LeakyReLU()(x5)
    x5pool = MaxPooling2D(pool_size=(2,2))(x5)
    
    #x6 = ResidualR(256, 512, x5pool)   #6x6x512
    #x6 = ResidualR(512, 512, x6)
    #x6 = ResidualR(512, 512, x6)   #6x6x512
    
    #xpool = GlobalAveragePooling2D()(x6)
    #xpool = MaxPooling2D(pool_size=(2,2))(x6)
    
    flatten = layers.Flatten()(x5pool)
    
    dense1 = layers.Dense(512)(flatten)
    dense1 = layers.LeakyReLU()(dense1)
    dense1 = layers.Dropout(0.5)(dense1)

    #dense2 = layers.Dense(1024)(dense1)
    #dense2 = layers.LeakyReLU()(dense2)
    #dense2 = layers.Dropout(0.5)(dense2)
    
    output = Dense(20, use_bias=False, kernel_regularizer=l2(5e-4), kernel_initializer='he_normal', activation='linear')(dense1)
    
model = Model(inputs, output)

model.summary()

model.compile(loss=wingLoss, optimizer=SGD(lr=0.001, momentum=0.9, nesterov=True), metrics=[mean_absolute_error])
#SGD(lr=0.03, momentum=0.9, nesterov=True)
#'RMSprop'
#Adam(lr=0.001)

callbacks = [
    EarlyStopping(monitor='val_loss', patience=10, verbose=1),
    ModelCheckpoint("val_loss__final_ldmk_deep_noatt_checkpoint.h5", monitor='val_loss', verbose=1, save_best_only=True, mode='auto'),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1, mode='auto', epsilon=0.01, cooldown=0, min_lr=1e-7)
            ]

history = model.fit(trainData, trainLabel, validation_data=(valData, valLabel), batch_size=16, epochs=100, verbose=1, shuffle=True, callbacks=callbacks)

model.save('final_ldmk_deep_noatt_model.h5')

plt.figure(0)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Loss')
plt.xlabel('epoch')
plt.legend(['loss', 'val_loss'], loc='upper right')
plt.savefig('final_ldmk_deep_noatt_loss.png')

plt.figure(1)
plt.plot(history.history['lr'])
plt.title('Learning Rate')
plt.xlabel('epoch')
plt.savefig('final_ldmk_deep_noatt_lr.png')

plt.figure(2)
plt.plot(history.history['mean_absolute_error'])
plt.plot(history.history['val_mean_absolute_error'])
plt.title('Mean Absolute Error Accuracy')
plt.xlabel('epoch')
plt.legend(['mean_absolute_error', 'val_mean_absolute_error'], loc='upper right')
plt.savefig('final_ldmk_deep_noatt_metric.png')
