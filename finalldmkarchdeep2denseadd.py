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


with tf.device('/device:GPU:0'):
    inputs = Input((192, 192, 3), name='Input')
    
    conv1 = initial_conv_block1(inputs)
    conv1 = Residual2(16, 32, conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2), name='MaxPool1')(conv1)
    
    conv2 = Residual3(32, 32, pool1)
    conv2 = Residual4(32, 64, conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2), name='MaxPool2')(conv2)
    
    conv3 = Residual5(64, 64, pool2)
    conv3 = Residual6(64, 128, conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2), name='MaxPool3')(conv3)
    
    conv4 = Residual7(128, 128, pool3)
    conv4 = Residual8(128, 256, conv4)
    drop4 = Dropout(0.2, name='Dropout1')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2), name='MaxPool4')(drop4)
    
    conv5 = Residual9(256, 256, pool4)
    conv5 = Residual10(256, 128, conv5)
    drop5 = Dropout(0.2, name='Dropout2')(conv5)
    
    up6 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', name='UpConv1')(UpSampling2D(size = (2,2), name='Up1')(drop5))
    merge6 = keras.layers.Concatenate(name='Concat1')([drop4,up6])
    conv6 = Residual11(384, 128, merge6)
    conv6_1 = Residual12(128, 64, conv6)
    
    up7 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', name='UpConv2')(UpSampling2D(size = (2,2), name='Up2')(conv6_1))
    merge7 = keras.layers.Concatenate(name='Concat2')([conv3,up7])
    conv7 = Residual13(192, 64, merge7)
    conv7_1 = Residual14(64, 32, conv7)
    
    up8 = Conv2D(32, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', name='UpConv3')(UpSampling2D(size = (2,2), name='Up3')(conv7_1))
    merge8 = keras.layers.Concatenate(name='Concat3')([conv2,up8])
    conv8 = Residual15(96, 32, merge8)
    conv8_1 = Residual16(32, 16, conv8)
    
    up9 = Conv2D(16, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', name='UpConv4')(UpSampling2D(size = (2,2), name='Up4')(conv8_1))
    merge9 = keras.layers.Concatenate(name='Concat4')([conv1,up9])
    conv9 = Residual17(48, 16, merge9)
    conv10 = Residual18(16, 2, conv9)
    conv10 = Residual19(2, 1, conv10)
    conv11 = Conv2D(1, 1, activation = 'sigmoid', name='Output')(conv10)
    
with tf.device('/device:GPU:1'):
    init = Conv2D(16, (3,3), padding='same', kernel_initializer='he_normal')(inputs)
    init = BatchNormalization()(init)
    init = layers.LeakyReLU()(init)
    
    #x1 = ResidualR(32, 64, init)    #192x192x64
    #x1 = ResidualR(64, 64, x1)
    #x1 = ResidualR(64, 64, x1)    #192x192x64
    x1 = Conv2D(16, (3,3), padding='same', kernel_initializer='he_normal')(init)
    x1 = BatchNormalization()(x1)
    x1 = layers.LeakyReLU()(x1)
    x1 = Conv2D(16, (3,3), padding='same', kernel_initializer='he_normal')(x1)
    x1 = BatchNormalization()(x1)
    x1 = layers.LeakyReLU()(x1)
    x1concat = keras.layers.Add()([x1, conv9]) #192x192x80
    x1se = squeeze_excite_block(x1concat)
    x1conv1 = Conv2D(32, (1,1), padding = 'same', kernel_initializer = 'he_normal')(x1se)
    x1conv1 = layers.LeakyReLU()(x1conv1)
    x1conv2 = Conv2D(32, (1,1), padding = 'same', kernel_initializer = 'he_normal')(x1conv1)
    x1conv2 = layers.LeakyReLU()(x1conv2)
    x1pool = MaxPooling2D(pool_size=(2,2))(x1conv2)
    
    #x2 = ResidualR(64, 96, x1pool)   #96x96x96
    #x2 = ResidualR(96, 96, x2)
    #x2 = ResidualR(96, 96, x2)   #96x96x96
    x2 = Conv2D(32, (3,3), padding='same', kernel_initializer='he_normal')(x1pool)
    x2 = BatchNormalization()(x2)
    x2 = layers.LeakyReLU()(x2)
    x2 = Conv2D(32, (3,3), padding='same', kernel_initializer='he_normal')(x2)
    x2 = BatchNormalization()(x2)
    x2 = layers.LeakyReLU()(x2)
    x2concat = keras.layers.Add()([x2, conv8]) #96x96x128
    x2se = squeeze_excite_block(x2concat)
    x2conv1 = Conv2D(64, (1,1), padding = 'same', kernel_initializer = 'he_normal')(x2se)
    x2conv1 = layers.LeakyReLU()(x2conv1)
    x2conv2 = Conv2D(64, (1,1), padding = 'same', kernel_initializer = 'he_normal')(x2conv1)
    x2conv2 = layers.LeakyReLU()(x2conv2)
    x2pool = MaxPooling2D(pool_size=(2,2))(x2conv2)

#with tf.device('/device:GPU:2'):   
    #x3 = ResidualR(96, 128, x2pool)   #48x48x128
    #x3 = ResidualR(128, 128, x3)
    #x3 = ResidualR(128, 128, x3)   #48x48x128
    x3 = Conv2D(64, (3,3), padding='same', kernel_initializer='he_normal')(x2pool)
    x3 = BatchNormalization()(x3)
    x3 = layers.LeakyReLU()(x3)
    x3 = Conv2D(64, (3,3), padding='same', kernel_initializer='he_normal')(x3)
    x3 = BatchNormalization()(x3)
    x3 = layers.LeakyReLU()(x3)
    x3concat = keras.layers.Add()([x3, conv7]) #48x48x192
    x3se = squeeze_excite_block(x3concat)
    x3conv1 = Conv2D(128, (1,1), padding = 'same', kernel_initializer = 'he_normal')(x3se)
    x3conv1 = layers.LeakyReLU()(x3conv1)
    x3conv2 = Conv2D(128, (1,1), padding = 'same', kernel_initializer = 'he_normal')(x3conv1)
    x3conv2 = layers.LeakyReLU()(x3conv2)
    x3pool = MaxPooling2D(pool_size=(2,2))(x3conv2)
    
    #x4 = ResidualR(128, 256, x3pool)   #24x24x256
    #x4 = ResidualR(256, 256, x4)
    #x4 = ResidualR(256, 256, x4)   #24x24x256
    x4 = Conv2D(128, (3,3), padding='same', kernel_initializer='he_normal')(x3pool)
    x4 = BatchNormalization()(x4)
    x4 = layers.LeakyReLU()(x4)
    x4 = Conv2D(128, (3,3), padding='same', kernel_initializer='he_normal')(x4)
    x4 = BatchNormalization()(x4)
    x4 = layers.LeakyReLU()(x4)
    x4concat = keras.layers.Add()([x4, conv6]) #24x24x384
    x4se = squeeze_excite_block(x4concat)
    x4conv1 = Conv2D(256, (1,1), padding = 'same', kernel_initializer = 'he_normal')(x4se)
    x4conv1 = layers.LeakyReLU()(x4conv1)
    x4conv2 = Conv2D(256, (1,1), padding = 'same', kernel_initializer = 'he_normal')(x4conv1)
    x4conv2 = layers.LeakyReLU()(x4conv2)
    x4pool = MaxPooling2D(pool_size=(2,2))(x4conv2)

#with tf.device('/device:GPU:3'):    
    #x5 = ResidualR(256, 256, x4pool)   #12x12x256
    #x5 = ResidualR(256, 256, x5)
    #x5 = ResidualR(256, 256, x5)   #12x12x256
    x5 = Conv2D(256, (3,3), padding='same', kernel_initializer='he_normal')(x4pool)
    x5 = BatchNormalization()(x5)
    x5 = layers.LeakyReLU()(x5)
    x5 = Conv2D(256, (3,3), padding='same', kernel_initializer='he_normal')(x5)
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

    dense2 = layers.Dense(512)(dense1)
    dense2 = layers.LeakyReLU()(dense2)
    dense2 = layers.Dropout(0.5)(dense2)
    
    output = Dense(20, use_bias=False, kernel_regularizer=l2(5e-4), kernel_initializer='he_normal', activation='linear')(dense2)
    
model = Model(inputs, output)

model.load_weights('val_loss_Residual_checkpoint.h5', by_name=True)

model.summary()

model.compile(loss=wingLoss, optimizer=SGD(lr=0.001, momentum=0.9, nesterov=True), metrics=[mean_absolute_error])
#SGD(lr=0.03, momentum=0.9, nesterov=True)
#'RMSprop'
#Adam(lr=0.001)

callbacks = [
    EarlyStopping(monitor='val_loss', patience=10, verbose=1),
    ModelCheckpoint("val_loss__final_ldmk_deep_2dense_checkpoint.h5", monitor='val_loss', verbose=1, save_best_only=True, mode='auto'),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1, mode='auto', epsilon=0.01, cooldown=0, min_lr=1e-7)
            ]

history = model.fit(trainData, trainLabel, validation_data=(valData, valLabel), batch_size=32, epochs=100, verbose=1, shuffle=True, callbacks=callbacks)

model.save('final_ldmk_deep_2dense_model.h5')

plt.figure(0)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Loss')
plt.xlabel('epoch')
plt.legend(['loss', 'val_loss'], loc='upper right')
plt.savefig('final_ldmk_deep_2dense_loss.png')

plt.figure(1)
plt.plot(history.history['lr'])
plt.title('Learning Rate')
plt.xlabel('epoch')
plt.savefig('final_ldmk_deep_2dense_lr.png')

plt.figure(2)
plt.plot(history.history['mean_absolute_error'])
plt.plot(history.history['val_mean_absolute_error'])
plt.title('Mean Absolute Error Accuracy')
plt.xlabel('epoch')
plt.legend(['mean_absolute_error', 'val_mean_absolute_error'], loc='upper right')
plt.savefig('final_ldmk_deep_2dense_metric.png')
