#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 20 17:53:10 2018

@author: ck807
"""

import numpy as np

import tensorflow as tf

from keras.models import Model
import matplotlib.pyplot as plt

import keras
from keras.layers.convolutional import Conv2D, UpSampling2D
from keras.layers.pooling import MaxPooling2D
from keras.layers import Input, Dropout
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from residualblockwithlayernames import initial_conv_block1, Residual2, Residual3, Residual4, Residual5, Residual6
from residualblockwithlayernames import Residual7, Residual8, Residual9, Residual10, Residual11, Residual12
from residualblockwithlayernames import Residual13, Residual14, Residual15, Residual16, Residual17, Residual18, Residual19

import keras.backend as K


print('Loading the data..')
#trainData = np.load('trainDataAttention.npy')
#trainMask = np.load('trainMaskAttention.npy')
#valData = np.load('valDataAttention.npy')
#valMask = np.load('valMaskAttention.npy')

trainData = np.load('frgctrain.npy')
trainMask = np.load('frgclabel.npy')

trainData = trainData.astype('float32')
trainDataMean = np.mean(trainData)
trainDataStd = np.std(trainData)

trainData -= trainDataMean
trainData /= trainDataStd

trainMak = trainMask.astype('float32')
trainMask /= 255.

trainD = trainData[0:5200,:,:,:]
trainM = trainMask[0:5200,:,:,:]

valData = trainData[5200:6500,:,:,:]
valMask = trainMask[5200:6500,:,:,:]

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


def get_unet():
    inputs = Input((192, 192, 3), name='Input')
           
    conv1 = initial_conv_block1(inputs)
    conv1 = Residual2(32, 32, conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2), name='MaxPool1')(conv1)
    
    conv2 = Residual3(32, 64, pool1)
    conv2 = Residual4(64, 64, conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2), name='MaxPool2')(conv2)
    
    conv3 = Residual5(64, 128, pool2)
    conv3 = Residual6(128, 128, conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2), name='MaxPool3')(conv3)
    
    conv4 = Residual7(128, 256, pool3)
    conv4 = Residual8(256, 256, conv4)
    drop4 = Dropout(0.2, name='Dropout1')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2), name='MaxPool4')(drop4)

    conv5 = Residual9(256, 256, pool4)
    conv5 = Residual10(256, 256, conv5)
    drop5 = Dropout(0.2, name='Dropout2')(conv5)
    
    up6 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', name='UpConv1')(UpSampling2D(size = (2,2), name='Up1')(drop5))
    merge6 = keras.layers.Concatenate(name='Concat1')([drop4,up6])
    conv6 = Residual11(384, 128, merge6)
    conv6 = Residual12(128, 128, conv6)
    
    up7 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', name='UpConv2')(UpSampling2D(size = (2,2), name='Up2')(conv6))
    merge7 = keras.layers.Concatenate(name='Concat2')([conv3,up7])
    conv7 = Residual13(192, 64, merge7)
    conv7 = Residual14(64, 64, conv7)
    
    up8 = Conv2D(32, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', name='UpConv3')(UpSampling2D(size = (2,2), name='Up3')(conv7))
    merge8 = keras.layers.Concatenate(name='Concat3')([conv2,up8])
    conv8 = Residual15(96, 32, merge8)
    conv8 = Residual16(32, 32, conv8)
    
    up9 = Conv2D(16, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', name='UpConv4')(UpSampling2D(size = (2,2), name='Up4')(conv8))
    merge9 = keras.layers.Concatenate(name='Concat4')([conv1,up9])
    conv9 = Residual17(48, 16, merge9)
    conv10 = Residual18(16, 4, conv9)
    conv10 = Residual19(2, 1, conv10)
    conv11 = Conv2D(1, 1, activation = 'sigmoid', name='Output')(conv10)
    
    model = Model(inputs, conv11)

    model.compile(loss=dice_coef_loss, optimizer=SGD(lr=0.03, momentum=0.9, nesterov=True), metrics=[dice_coef,jaccard_coef])
        
    return model

model = get_unet()

model.load_weights('resunet_attention_checkpoint.h5')

model.compile(loss=dice_coef_loss, optimizer=SGD(lr=0.03, momentum=0.9, nesterov=True), metrics=[dice_coef,jaccard_coef])

model.summary()
    
callbacks = [
    EarlyStopping(monitor='val_loss', patience=10, verbose=1, min_delta=0.001),
    ModelCheckpoint("resunet2_attention_checkpoint.h5", monitor='val_loss', verbose=1, save_best_only=True, mode='auto'),
    ReduceLROnPlateau(monitor='val_loss', patience=5, verbose=1, mode='auto', epsilon=0.001, cooldown=0, min_lr=0.5e-7)
            ]

history = model.fit(trainD, trainM, validation_data=(valData, valMask), batch_size=32, epochs=200, verbose=1, shuffle=True, callbacks=callbacks)

model.save('resunet_atention2.h5')

plt.figure(0)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Loss')
plt.xlabel('epoch')
plt.legend(['loss', 'val_loss'], loc='upper right')
plt.savefig('2lossattention.png')

plt.figure(1)
plt.plot(history.history['dice_coef'])
plt.plot(history.history['val_dice_coef'])
plt.title('Dice Coef Accuracy')
plt.xlabel('epoch')
plt.legend(['dice_coef', 'val_dice_coef'], loc='upper left')
plt.savefig('2dice_coef_attention.png')

plt.figure(2)
plt.plot(history.history['jaccard_coef'])
plt.plot(history.history['val_jaccard_coef'])
plt.title('Jaccard Coef Accuracy')
plt.xlabel('epoch')
plt.legend(['jaccard_coef', 'val_jaccard_coef'], loc='upper left')
plt.savefig('2jaccard_coef_attention.png')

plt.figure(3)
plt.plot(history.history['lr'])
plt.title('Learning Rate')
plt.xlabel('epoch')
plt.savefig('2lr_attention.png')
